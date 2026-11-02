from argparse import Namespace
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.seviri_clips import SeviriZarrClipReader
from emma_gpm.seviri_frame_cache import build_frame_cache
from emma_gpm.ssl_dataset import inventory_upload
from scripts import build_seviri_ssl_random_dataset as builder
from scripts import build_seviri_ssl_pilot_dataset as pilot


class InlineExecutor:
    def __init__(self, *, initializer, initargs, **kwargs):
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pilot.READER.close()

    def map(self, fn, rows, **kwargs):
        return map(fn, rows)


def test_random_dataset_preserves_dates_and_statistics_and_checks_both_splits(tmp_path, monkeypatch):
    root=tmp_path/'raw';root.mkdir()
    values=np.broadcast_to((250+2*np.arange(7,dtype=np.float16))[None,:,None,None],(17,7,225,226)).copy()
    for day in ('20200912','20200917'):
        path=root/f'2020/{day}.zarr';path.parent.mkdir(exist_ok=True)
        ds=xr.Dataset({'seviri':(('time','channel','y','x'),values)},
            coords={'time':pd.date_range(day,periods=17,freq='5min'),'channel':list(THERMAL_CHANNELS),
                'x':np.linspace(-10,10,226),'y':np.linspace(40,50,225)},
            attrs={'projection_wkt':'EPSG:4326','units':'K','calibration':'brightness_temperature'})
        ds.to_zarr(path,zarr_format=2,consolidated=True,encoding={'seviri':{'chunks':(1,7,224,224)}})
    inventory=inventory_upload(root,0)
    inventory_path=tmp_path/'inventory.json';inventory_path.write_text(json.dumps(inventory))
    cache=tmp_path/'cache';build_frame_cache(root,cache,['20200912','20200917'])
    previous=tmp_path/'previous';previous.mkdir()
    for split,day in [('train','2020-09-12'),('validation','2020-09-17')]:
        pd.DataFrame([dict(start_time=day+'T00:00:00Z',origin_y=0,origin_x=0,
            day=day,time_bin=0,store='2020/'+day.replace('-','')+'.zarr')]).to_csv(previous/f'{split}.csv',index=False)
    out=tmp_path/'out';out.mkdir()
    args=Namespace(output_dir=out,previous_dataset=previous,inventory=inventory_path,
        zarr_root=root,frame_cache=cache,count=4,candidates=8,validation_count=2,
        domain=[-20,40,30,70],seed=123,workers=1)
    monkeypatch.setattr(builder,'ProcessPoolExecutor',InlineExecutor)
    builder.build(args)
    dataset=out/'n4/dataset'
    train=pd.read_csv(dataset/'train.csv');validation=pd.read_csv(dataset/'validation.csv')
    assert len(train)==4 and len(validation)==2
    assert set(train.day)=={'2020-09-12'} and set(validation.day)=={'2020-09-17'}
    stats=json.loads((dataset/'channel_stats.json').read_text())
    with SeviriZarrClipReader(root) as reader:
        raw=np.stack([reader.load(row.start_time,origin_y=row.origin_y,origin_x=row.origin_x)
            for row in train.itertuples()])
    assert stats['global_mean_k']==pytest.approx(raw.mean(dtype=np.float64),abs=1e-10)
    assert stats['global_std_k']==pytest.approx(raw.std(dtype=np.float64),abs=1e-10)
    protocol=json.loads((dataset/'protocol.json').read_text())
    assert protocol['geographic_check_passed'] and protocol['spatial_resolution_pixels']==1
    assert protocol['domain_lonlat']==[-20,40,30,70]
    assert not protocol['strict_nesting'] and not protocol['validation_identical']
    assert len(protocol['cache_sample_verification'])==2
    assert json.loads((out/'dataset_ready.json').read_text())['status']=='completed'
