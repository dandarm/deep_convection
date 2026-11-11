import itertools

import numpy as np
import pandas as pd
import pytest

from emma_gpm.ssl_archive_coverage import record_capacity, manifest_frame_counts, archive_coverage
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


def test_coverage_metadata_preserves_chunk_identity_with_unordered_storage_times():
    item=record(pd.date_range('2022-01-01',periods=18,freq='5min'))
    item['present'].remove('8.0.0.0')
    expected=record_capacity(item,np.ones((3,3),bool),crop=2)
    order=np.roll(np.arange(18),5)
    permuted=dict(item,times=[item['times'][i] for i in order],present=[])
    for i,previous in enumerate(order):
        permuted['present'].extend(f'{i}.0.{y}.{x}' for y,x in itertools.product(range(2),range(2))
            if f'{previous}.0.{y}.{x}' in item['present'])
    actual=record_capacity(permuted,np.ones((3,3),bool),crop=2)
    assert actual[0].equals(expected[0]) and actual[1]==expected[1]


def test_temporal_capacity_crosses_midnight_and_preserves_real_gaps():
    item=record(pd.date_range('2022-01-01T23:00',periods=32,freq='5min'))
    times,windows=record_capacity(item,np.ones((3,3),bool),crop=2)
    assert len(times)==32 and len(windows)==17
    assert sum(t.day==1 for t,_ in windows)==12
    item=record(pd.date_range('2022-01-01T23:00',periods=32,freq='5min').delete(15))
    _,windows=record_capacity(item,np.ones((3,3),bool),crop=2)
    assert len(windows)==1 and windows[0][0]==pd.Timestamp('2022-01-02T00:20Z')


def test_capacity_joins_daily_chunks_across_year_and_rejects_missing_observations():
    first=record(pd.date_range('2022-12-31T23:00',periods=12,freq='5min'))
    second=record(pd.date_range('2023-01-01',periods=20,freq='5min'))
    geographic=np.ones((3,3),bool)
    times,windows=record_capacity(first,geographic,crop=2,following=second)
    assert len(times)==12 and len(windows)==12 and sum(n for _,n in windows)==108
    assert all(t.year==2022 for t,_ in windows)
    second['present'].remove('0.0.0.0')
    _,windows=record_capacity(first,geographic,crop=2,following=second)
    assert sum(n for _,n in windows)==60
    second=record(pd.date_range('2023-01-01T00:05',periods=19,freq='5min'))
    assert record_capacity(first,geographic,crop=2,following=second)[1]==[]


@pytest.mark.parametrize('same_grid',[True,False])
def test_archive_capacity_joins_only_compatible_adjacent_stores(monkeypatch,tmp_path,same_grid):
    import xarray as xr
    import emma_gpm.ssl_archive_coverage as module
    stores=[]
    for day,times in [('first',pd.date_range('2022-12-31T23:00',periods=12,freq='5min')),
                      ('second',pd.date_range('2023-01-01',periods=20,freq='5min'))]:
        stores.append(dict(store=day,shape=[len(times),7,225,226],chunks=[1,7,224,224],
            times=[t.isoformat() for t in times],
            present=[f'{t}.0.{y}.{x}' for t,y,x in itertools.product(range(len(times)),range(2),range(2))]))
    def open_store(path,**kwargs):
        offset=int(not same_grid and path.name=='second')
        return xr.Dataset(coords={'x':np.arange(226)+offset,'y':np.arange(225)},
                          attrs={'projection_wkt':'EPSG:4326'})
    monkeypatch.setattr(xr,'open_zarr',open_store)
    monkeypatch.setattr(module,'geographic_origin_mask',lambda *args:np.ones((2,3),bool))
    daily,summary=archive_coverage(dict(root=str(tmp_path),stores=stores))
    assert summary['frames']==32
    assert summary['temporal_windows']==(17 if same_grid else 5)
    assert summary['cross_midnight_windows']==(12 if same_grid else 0)
    assert summary['crop_candidates']==6*summary['temporal_windows']
    assert daily.temporal_windows.tolist()==([12,5] if same_grid else [0,5])


def test_selected_frame_count_unions_overlapping_crops_and_crosses_midnight():
    manifest=pd.DataFrame({'start_time':['2022-01-01T23:50Z','2022-01-01T23:50Z','2022-01-01T23:55Z']})
    counts=manifest_frame_counts(manifest)
    assert counts.sum()==17
    assert counts.to_list()==[2,15]


def test_sparse_boundary_examples_select_all_four_edges_at_five_degrees():
    points=pd.DataFrame(itertools.product(range(-20,31),range(40,61)),columns=['longitude','latitude'])
    selected=boundary_crop_origins(points)
    assert ((selected.latitude.isin([40,60]))|(selected.longitude.isin([-20,30]))).all()
    assert len(selected)==28
    assert not selected.duplicated(['longitude','latitude']).any()
    for edge in ('south','north'):
        assert set(selected.loc[selected.edge==edge,'longitude'])==set(range(-20,31,5))
    for edge in ('west','east'):
        assert set(selected.loc[selected.edge==edge,'latitude'])==set(range(45,60,5))


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
