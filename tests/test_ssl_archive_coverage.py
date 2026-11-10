import itertools

import numpy as np
import pandas as pd
import pytest

from emma_gpm.ssl_archive_coverage import record_capacity, manifest_frame_counts
from emma_gpm.ssl_dashboard import boundary_crop_origins, crop_origin_locations, crop_footprint


def record(times):
    return dict(shape=[len(times),7,4,4],chunks=[1,7,2,2],times=[t.isoformat() for t in times],
        present=[f'{t}.0.{y}.{x}' for t,y,x in itertools.product(range(len(times)),range(2),range(2))])


def test_capacity_counts_all_integer_origins_and_excludes_a_missing_chunk():
    item=record(pd.date_range('2022-01-01',periods=18,freq='5min'))
    times,windows=record_capacity(item,np.ones((3,3),bool),crop=2)
    assert len(times)==18 and len(windows)==3 and sum(n for _,n in windows)==27
    item['present'].remove('8.0.0.0')
    times,windows=record_capacity(item,np.ones((3,3),bool),crop=2)
    # Four of nine spatial origins intersect the absent 2x2 chunk.
    assert len(times)==18 and len(windows)==3 and sum(n for _,n in windows)==15
    geographic=np.zeros((3,3),bool);geographic[0,0]=True
    times,windows=record_capacity(item,geographic,crop=2)
    assert len(times)==17 and windows==[]


def test_temporal_capacity_respects_real_gaps_and_daily_boundaries():
    item=record(pd.date_range('2022-01-01T23:00',periods=32,freq='5min'))
    times,windows=record_capacity(item,np.ones((3,3),bool),crop=2)
    assert len(times)==32 and len(windows)==5
    assert all(t.day==2 for t,_ in windows)
    item['times'][20]='2022-01-03T00:00:00'
    _,windows=record_capacity(item,np.ones((3,3),bool),crop=2)
    assert windows==[]


def test_selected_frame_count_unions_overlapping_crops_and_crosses_midnight():
    manifest=pd.DataFrame({'start_time':['2022-01-01T23:50Z','2022-01-01T23:50Z','2022-01-01T23:55Z']})
    counts=manifest_frame_counts(manifest)
    assert counts.sum()==17
    assert counts.to_list()==[2,15]


def test_sparse_boundary_examples_select_only_southern_and_eastern_edges():
    points=pd.DataFrame(itertools.product(range(-20,31),range(40,61)),columns=['longitude','latitude'])
    selected=boundary_crop_origins(points)
    assert ((selected.latitude==40)|(selected.longitude==30)).all()
    assert len(selected)==8
    assert not selected.duplicated(['longitude','latitude']).any()
    assert set(selected.loc[selected.edge=='south','longitude'])==set(range(-20,31,10))


@pytest.mark.parametrize('reverse_x,reverse_y',list(itertools.product([False,True],repeat=2)))
def test_crop_outline_begins_at_plotted_origin_and_follows_oriented_pixels(monkeypatch,tmp_path,reverse_x,reverse_y):
    import xarray as xr
    x=np.arange(448)*.05-20
    y=70-np.arange(336)*.05
    x=x[::-1] if reverse_x else x
    y=y[::-1] if reverse_y else y
    ds=xr.Dataset(coords={'x':x,'y':y},attrs={'projection_wkt':'EPSG:4326'})
    monkeypatch.setattr(xr,'open_zarr',lambda *a,**kw:ds)
    manifest=pd.DataFrame({'store':['day.zarr'],'origin_y':[50],'origin_x':[100]})
    point=crop_origin_locations(manifest,tmp_path).iloc[0]
    outline=crop_footprint(point,tmp_path)
    np.testing.assert_allclose(outline[0],point[['longitude','latitude']].to_numpy(float))
    np.testing.assert_allclose(outline[0],outline[-1])
    assert outline[:,0].min()==min(x[100:324]) and outline[:,0].max()==max(x[100:324])
    assert outline[:,1].min()==min(y[50:274]) and outline[:,1].max()==max(y[50:274])
