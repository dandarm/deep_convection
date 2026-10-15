import numpy as np
import pandas as pd
import pytest
import xarray as xr
from emma_gpm.seviri import RSS_CRS, THERMAL_CHANNELS
from emma_gpm.rdt_preview import archive_inventory, load_example
from emma_gpm.seviri_clips import IncompleteSeviriClipError


def write_store(root, times, missing=None):
    data=np.full((len(times),7,2,2),250.,dtype=np.float32)
    data[:,:,0,0]=260
    if missing is not None:data[missing,0,0,0]=np.nan
    ds=xr.Dataset({'seviri':(('time','channel','y','x'),data)},
        coords={'time':times,'channel':list(THERMAL_CHANNELS),'x':[3000.,0.], 'y':[3e6,3e6+3000]},
        attrs={'units':'K','calibration':'brightness_temperature','projection_wkt':RSS_CRS.to_wkt()})
    ds.to_zarr(root/'2020/20200101.zarr',mode='w',consolidated=True,zarr_format=2)


def test_inventory_queries_unsorted_storage_but_rejects_duplicate_slots(tmp_path):
    times=pd.date_range('2020-01-01',periods=3,freq='5min')
    write_store(tmp_path,times[[2,0,1]])
    inventory,available,_=archive_inventory(tmp_path)
    assert inventory.source_unsorted.iloc[0]
    pd.testing.assert_index_equal(available,times)
    write_store(tmp_path,times[[0,0,1]])
    with pytest.raises(ValueError,match='chronological'):archive_inventory(tmp_path)


def test_context_and_area_follow_reader_orientation_without_missing_frame_repair(tmp_path):
    times=pd.date_range('2020-01-01',periods=17,freq='5min')
    write_store(tmp_path,times)
    example=load_example(tmp_path,times[1],0,0,warmup_frames=1,crop_size=2)
    assert example['clip'].shape==(16,7,2,2)
    assert example['clip'][0,0,-1,-1]==260
    assert example['area'].shape==(2,2)
    assert example['times'][0]==times[1]
    write_store(tmp_path,times,missing=0)
    with pytest.raises(IncompleteSeviriClipError,match='NaN'):load_example(tmp_path,times[1],0,0,warmup_frames=1,crop_size=2)
