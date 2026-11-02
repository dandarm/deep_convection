import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.seviri_clips import SeviriZarrClipReader,IncompleteSeviriClipError
from emma_gpm.seviri_frame_cache import build_frame_cache,SeviriFrameCacheReader
from emma_gpm.seviri_tiled_cache import build_tiled_cache,TiledFrameCacheReader
from emma_gpm.seviri_frame_dataset import FrameFileDataset


@pytest.mark.parametrize('reverse_x,reverse_y',[(False,False),(True,False),(False,True),(True,True)])
def test_tiled_arbitrary_crops_preserve_bits_orientation_and_normalization(tmp_path,reverse_x,reverse_y):
    root=tmp_path/'raw';store=root/'2020/20200912.zarr';store.parent.mkdir(parents=True)
    t,c,y,x=np.indices((17,7,227,229))
    values=(245+t*.125+c*2+y*.03125+x*.0625).astype(np.float16)
    values[0,0,0,0]=np.nan
    xs=np.arange(229);ys=np.arange(227)
    if reverse_x:xs=xs[::-1]
    if reverse_y:ys=ys[::-1]
    ds=xr.Dataset({'seviri':(('time','channel','y','x'),values)},
        coords=dict(time=pd.date_range('2020-09-12',periods=17,freq='5min'),channel=list(THERMAL_CHANNELS),y=ys,x=xs))
    ds.to_zarr(store,zarr_format=2,consolidated=True,encoding={'seviri':{'chunks':(1,7,224,224)}})
    native=tmp_path/'native';tiled=tmp_path/'tiled'
    build_frame_cache(root,native,['20200912'])
    metadata=build_tiled_cache(native,tiled)
    assert metadata['stores']['2020/20200912.zarr']['native_bits_verified']
    manifest=pd.DataFrame([dict(start_time='2020-09-12T00:05:00Z',origin_y=3,origin_x=5)])
    manifest.to_csv(tmp_path/'manifest.csv',index=False)
    with SeviriZarrClipReader(root) as zarr_reader,SeviriFrameCacheReader(root,native) as original,TiledFrameCacheReader(root,tiled) as reader:
        for oy,ox in [(0,0),(1,2),(3,5)]:
            expected=original.load_native('2020-09-12T00:05:00Z',origin_y=oy,origin_x=ox)
            actual=reader.load_native('2020-09-12T00:05:00Z',origin_y=oy,origin_x=ox)
            np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))
            np.testing.assert_array_equal(reader.load('2020-09-12T00:05:00Z',origin_y=oy,origin_x=ox,global_mean=250,global_std=20),
                zarr_reader.load('2020-09-12T00:05:00Z',origin_y=oy,origin_x=ox,global_mean=250,global_std=20))
        with pytest.raises(IncompleteSeviriClipError,match='bounds'):
            reader.load_native('2020-09-12T00:00:00Z',origin_y=4,origin_x=5)
        with pytest.raises(IncompleteSeviriClipError,match='Missing exact'):
            reader.load_native('2020-09-12T00:10:00Z',origin_y=0,origin_x=0)
        with pytest.raises(IncompleteSeviriClipError,match='NaN'):
            reader.load_native('2020-09-12T00:00:00Z',origin_y=0,origin_x=0)
    data=FrameFileDataset(tiled,tmp_path/'manifest.csv',zarr_root=root)
    try:
        output=data.decode_into(0,np.empty((16,7,224,224),dtype=np.float16))
        np.testing.assert_array_equal(output.view(np.uint16),expected.view(np.uint16))
    finally:data.close()
    # A completed conversion can resume without changing metadata or pixels.
    before=(tiled/'metadata.json').read_bytes();build_tiled_cache(native,tiled)
    assert (tiled/'metadata.json').read_bytes()==before
    with (tiled/'20200912.npy').open('r+b') as f:
        f.seek(-2,2);f.write(b'xx')
    with pytest.raises(ValueError,match='checksum'):
        FrameFileDataset(tiled,tmp_path/'manifest.csv',zarr_root=root)


def test_tiled_reader_preserves_cross_midnight_times(tmp_path):
    root=tmp_path/'raw'
    for day,start,value in [('20200912','2020-09-12T23:50',250),('20200913','2020-09-13T00:00',260)]:
        store=root/f'2020/{day}.zarr';store.parent.mkdir(parents=True,exist_ok=True)
        ds=xr.Dataset({'seviri':(('time','channel','y','x'),np.full((2,7,4,5),value,dtype=np.float16))},
            coords=dict(time=pd.date_range(start,periods=2,freq='5min'),channel=list(THERMAL_CHANNELS),y=np.arange(4),x=np.arange(5)[::-1]))
        ds.to_zarr(store,zarr_format=2,consolidated=True)
    native=tmp_path/'native';tiled=tmp_path/'tiled'
    build_frame_cache(root,native,['20200912','20200913']);build_tiled_cache(native,tiled)
    with SeviriFrameCacheReader(root,native,num_frames=3,crop_size=3) as original,TiledFrameCacheReader(root,tiled,num_frames=3,crop_size=3) as reader:
        kwargs=dict(start_time='2020-09-12T23:55Z',origin_y=1,origin_x=2)
        np.testing.assert_array_equal(reader.load_native(**kwargs).view(np.uint16),original.load_native(**kwargs).view(np.uint16))
