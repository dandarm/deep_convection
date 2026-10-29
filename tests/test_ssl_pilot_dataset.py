import numpy as np
import pandas as pd
import pytest
import xarray as xr

from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.ssl_dataset import (inventory_upload, available_candidates, crop_chunk_names,
    temporal_partition, balanced_candidate_order, spatial_axis, verify_snapshot, file_digest)


def test_physical_chunks_native_time_mapping_and_coordinate_publication(tmp_path):
    store = tmp_path / '2020/20200912.zarr'
    store.parent.mkdir()
    times = pd.date_range('2020-09-12', periods=17, freq='5min')
    native = times[8:].append(times[:8])
    ds = xr.Dataset({'seviri': (('time','channel','y','x'), np.full((17,7,224,224),270,dtype=np.float16))},
        coords=dict(time=native,channel=list(THERMAL_CHANNELS),y=np.arange(224),x=np.arange(224)[::-1]))
    ds.to_zarr(store,zarr_format=2,consolidated=True,encoding={'seviri':{'chunks':(1,7,224,224)}})
    inventory = inventory_upload(tmp_path,0)
    frame = available_candidates(inventory)
    assert len(frame)==2
    names = crop_chunk_names(inventory['stores'][0],frame.iloc[0].start_time,0,0)
    assert names[0]=='9.0.0.0'
    # A missing physical chunk must not be mistaken for a valid fill value.
    (store/'seviri/9.0.0.0').unlink()
    assert len(available_candidates(inventory_upload(tmp_path,0)))==1
    (store/'time/0').unlink()
    assert not inventory_upload(tmp_path,0)['stores']


def test_balanced_sampling_respects_gaps_geography_and_full_clip_embargo():
    frame = pd.DataFrame([dict(start_time=f'2020-09-{day:02d}T{hour:02d}:00:00Z',
        day=f'2020-09-{day:02d}',time_bin=hour//6,origin_y=y,origin_x=x)
        for day in (12,13,16) for hour in (0,6,12,18) for y in (0,224) for x in (0,224)])
    train,val,boundary=temporal_partition(frame)
    assert pd.to_datetime(train.start_time,utc=True).max()+pd.Timedelta(hours=24,minutes=75)<pd.to_datetime(val.start_time,utc=True).min()
    first=balanced_candidate_order(train,42)
    pd.testing.assert_frame_equal(first,balanced_candidate_order(train,42))
    assert len(first.drop_duplicates(['start_time','origin_y','origin_x']))==len(train)
    assert set(first.origin_y)=={0,224}
    assert spatial_axis(763)==[0,112,224,336,448,539]
    assert spatial_axis(1762)[-1]==1538


def test_snapshot_detects_same_length_replacement(tmp_path):
    path=tmp_path/'chunk'
    path.write_bytes(b'abcd')
    hashes={'chunk':file_digest(path)}
    verify_snapshot(tmp_path,hashes)
    path.write_bytes(b'abce')
    with pytest.raises(ValueError,match='source changed'):
        verify_snapshot(tmp_path,hashes)
