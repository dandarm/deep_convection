from argparse import Namespace
import json

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from emma_gpm.forecasting import ForecastWindow
from emma_gpm.ssl_model import history_config
from emma_gpm.seviri_clips import VIDEO_MAE_CHANNELS, SeviriZarrClipReader
from emma_gpm.seviri_frame_cache import build_frame_cache
from emma_gpm.seviri_frame_dataset import FrameFileDataset
from emma_gpm.seviri_clip_files import ClipBatchLoader
from scripts.build_seviri_ssl_extension import merge_candidate_caches
from scripts.run_seviri_ssl_selected import selected_stages


def test_base_architecture_and_unchanged_small_default():
    small = history_config(ForecastWindow())
    explicit = history_config(ForecastWindow(), model_size='small')
    assert small.to_dict() == explicit.to_dict()
    assert (small.hidden_size, small.num_hidden_layers, small.num_attention_heads) == (384,12,6)
    base = history_config(ForecastWindow(), model_size='base')
    assert (base.hidden_size, base.num_hidden_layers, base.num_attention_heads, base.intermediate_size) == (768,12,12,3072)
    assert (base.decoder_hidden_size, base.decoder_num_hidden_layers, base.decoder_num_attention_heads) == (384,4,6)
    for config in (base,small):
        assert config.num_frames == 10 and config.num_channels == 7
        assert config.tubelet_size == 2 and config.patch_size == 16
        assert config.norm_pix_loss is False
    with pytest.raises(ValueError, match='model size'):
        history_config(ForecastWindow(), model_size='unknown')


def test_base_queue_preserves_effective_batch_and_requested_budget(tmp_path):
    args=Namespace(zarr_root=tmp_path/'zarr', epochs=150, batch_size=16, workers=4,
        gradient_accumulation=2, seed=20261024, frame_cache=tmp_path/'frames',
        clip_root=None, model_size='base', reference_series=tmp_path/'previous',
        variability_reference_manifest=tmp_path/'reference.csv')
    stages=selected_stages(args,tmp_path/'dataset',tmp_path/'n32768')
    assert [s['name'] for s in stages] == ['future','joint']
    for stage in stages:
        assert stage['epochs'] == 150
        flags=stage['flags']
        for name,value in [('--batch-size','16'),('--gradient-accumulation','2'),
                           ('--model-size','base'),('--epochs','150')]:
            assert flags[flags.index(name)+1] == value
        assert '--frame-cache' in flags and '--clip-root' not in flags


def test_overlapping_candidate_caches_require_consistent_radiometry(tmp_path):
    row=dict(start_time='2020-09-12T00:00Z',origin_y=0,origin_x=0,valid=True,
        mean_k=260.,variance_k2=400.,mean_abs_change_k=1.,p90_abs_change_k=2.,fraction_change_gt2k=.1)
    a,b=tmp_path/'a.csv',tmp_path/'b.csv'
    pd.DataFrame([row]).to_csv(a,index=False)
    pd.DataFrame([dict(row,start_time='2020-09-12T02:00+02:00')]).to_csv(b,index=False)
    assert len(merge_candidate_caches([a,b])) == 1
    pd.DataFrame([dict(row,mean_k=261.)]).to_csv(b,index=False)
    with pytest.raises(ValueError,match='Conflicting'):
        merge_candidate_caches([a,b])


def test_daily_file_loader_preserves_orientation_normalization_and_detects_mutation(tmp_path):
    root,cache=tmp_path/'zarr',tmp_path/'cache'
    # Spatial and temporal variation makes reversed axes and frame order observable.
    values=(240+np.arange(17)[:,None,None,None]+np.arange(7)[None,:,None,None]
            +np.arange(224)[None,None,:,None]/16+np.arange(224)[None,None,None,:]/32).astype(np.float16)
    xr.Dataset({'seviri':(('time','channel','y','x'),values)},coords=dict(
        time=pd.date_range('2020-09-12',periods=17,freq='5min'),channel=list(VIDEO_MAE_CHANNELS),
        y=np.arange(224),x=np.arange(224)[::-1])).to_zarr(
            root/'2020/20200912.zarr',mode='w',consolidated=True,zarr_format=2)
    build_frame_cache(root,cache,['20200912'])
    manifest=tmp_path/'train.csv'
    pd.DataFrame(dict(start_time=['2020-09-12T00:00Z','2020-09-12T00:05Z'],
                      origin_y=[0,0],origin_x=[0,0])).to_csv(manifest,index=False)
    data=FrameFileDataset(cache,manifest,zarr_root=root)
    with SeviriZarrClipReader(root) as reader:
        expected=torch.stack([torch.from_numpy(reader.load(t,origin_y=0,origin_x=0))
                              for t in data.manifest.start_time])
    loader=ClipBatchLoader(data,batch_size=2,workers=2,prefetch=2,mean=264.,std=21.,
                          device='cpu',seed=10,shuffle=False)
    actual=next(iter(loader))['pixel_values']
    assert torch.equal(actual,(expected-264.)/21.)
    ForecastWindow().split(actual)
    path=cache/'20200912.npy'
    with path.open('r+b') as stream:
        stream.seek(-2,2);stream.write(b'\x00\x00')
    with pytest.raises(ValueError,match='changed during use'):
        data.decode_into(0,np.empty((16,7,224,224),dtype=np.float16))
    data.close()
    with pytest.raises(ValueError,match='checksum differs'):
        FrameFileDataset(cache,manifest,zarr_root=root)
