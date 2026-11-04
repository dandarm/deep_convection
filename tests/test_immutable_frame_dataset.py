import numpy as np
import pandas as pd
import xarray as xr

from emma_gpm.seviri_clips import VIDEO_MAE_CHANNELS
from emma_gpm.seviri_frame_cache import build_frame_cache, file_digest
from emma_gpm.seviri_frame_dataset import FrameFileDataset


def test_four_training_readers_use_prepared_arrays_without_raw_io_or_array_hashes(tmp_path,monkeypatch):
    raw,cache=tmp_path/'raw',tmp_path/'cache'
    values=(220+np.arange(16*7*4*5).reshape(16,7,4,5)/128).astype(np.float16)
    xr.Dataset({'seviri':(('time','channel','y','x'),values)},coords=dict(
        time=pd.date_range('2020-09-12',periods=16,freq='5min'),
        channel=list(VIDEO_MAE_CHANNELS),y=[0,1,2,3],x=[4,3,2,1,0]
    )).to_zarr(raw/'2020/20200912.zarr',zarr_format=2,consolidated=True)
    build_frame_cache(raw,cache,['20200912'])
    manifest=tmp_path/'clips.csv'
    pd.DataFrame([dict(start_time='2020-09-12T00:00:00Z',origin_y=1,origin_x=1)]).to_csv(manifest,index=False)
    raw.rename(tmp_path/'offline_raw')
    def forbidden(*args,**kwargs):
        raise AssertionError('Raw source verification must stay outside training')
    monkeypatch.setattr('emma_gpm.seviri_frame_cache.source_stamp',forbidden)
    def metadata_identity_only(path):
        assert path.name=='metadata.json','Training must not hash NPY arrays'
        return file_digest(path)
    monkeypatch.setattr('emma_gpm.seviri_frame_dataset.file_digest',metadata_identity_only)
    for _ in range(4):
        data=FrameFileDataset(cache,manifest,zarr_root=raw)
        assert not data.verify_integrity
        data._reader.crop_size=3
        out=np.empty((16,7,3,3),dtype=np.float16)
        data.decode_into(0,out)
        np.testing.assert_array_equal(out.view('u2'),values[:,:,1:4,1:4][:,:,::-1,::-1].view('u2'))
        data.close()
