import itertools

import numpy as np
import pandas as pd
import pytest

from emma_gpm.ssl_random_sampling import geographic_origin_mask, origin_rectangles, random_candidates


def inventory():
    times = pd.date_range('2020-09-12', periods=20, freq='5min')
    return dict(stores=[dict(store='2020/20200912.zarr', times=[t.isoformat() for t in times],
        shape=[20,7,12,15], chunks=[1,7,4,5],
        present=[f'{t}.0.{y}.{x}' for t,y,x in itertools.product(range(20),range(3),range(3))])])


def test_rectangles_exactly_partition_valid_integer_origins_with_missing_chunks():
    rng=np.random.default_rng(31)
    for height,width,cy,cx,crop in [(12,15,4,5,3),(12,15,4,5,5),(11,14,4,5,6)]:
        complete=rng.random(((height+cy-1)//cy,(width+cx-1)//cx))>.2
        expected={(y,x) for y in range(height-crop+1) for x in range(width-crop+1)
            if complete[y//cy:(y+crop-1)//cy+1,x//cx:(x+crop-1)//cx+1].all()}
        actual=[]
        for y0,y1,x0,x1 in origin_rectangles(complete,height,width,cy,cx,crop):
            actual.extend(itertools.product(range(y0,y1),range(x0,x1)))
        assert len(actual)==len(set(actual)) and set(actual)==expected


def test_random_sampling_is_reproducible_pixel_resolved_and_can_cover_entire_population():
    source=inventory()
    # 19 two-frame windows x 10 y-origins x 13 x-origins.
    all_candidates=random_candidates(source,19*10*13,seed=2,crop=3,frames=2)
    assert set(all_candidates.origin_y)==set(range(10))
    assert set(all_candidates.origin_x)==set(range(13))
    sample=random_candidates(source,200,seed=12,crop=3,frames=2)
    pd.testing.assert_frame_equal(sample,random_candidates(source,200,seed=12,crop=3,frames=2))
    assert not sample.equals(random_candidates(source,200,seed=13,crop=3,frames=2))
    with pytest.raises(ValueError,match='Insufficient'):
        random_candidates(source,19*10*13+1,crop=3,frames=2)


def test_missing_chunks_and_time_gaps_are_excluded_before_pixel_checks():
    source=inventory();record=source['stores'][0]
    record['present'].remove('8.0.1.1')
    candidates=random_candidates(source,500,crop=3,frames=2)
    for row in candidates.itertuples():
        indices=pd.DatetimeIndex(record['times']).get_indexer(pd.date_range(row.start_time,periods=2,freq='5min').tz_localize(None))
        for t in indices:
            for y in range(row.origin_y//4,(row.origin_y+2)//4+1):
                for x in range(row.origin_x//5,(row.origin_x+2)//5+1):
                    assert f'{t}.0.{y}.{x}' in record['present']
    # Keep native chunk indices but introduce a missing observation timestamp.
    record['times'][8]='2020-09-12T04:00:00'
    result=random_candidates(source,500,crop=3,frames=2)
    assert '2020-09-12T00:35:00Z' not in set(result.start_time)
    assert '2020-09-12T00:40:00Z' not in set(result.start_time)


def test_random_candidates_balance_dates_and_six_hour_bins():
    source=inventory();record=source['stores'][0]
    for date,hour in [('2020-09-12',6),('2020-09-13',0),('2020-09-13',6)]:
        times=pd.date_range(f'{date}T{hour:02d}:00:00',periods=20,freq='5min')
        source['stores'].append(dict(record,store=f'{date}/{hour}.zarr',times=[t.isoformat() for t in times]))
    result=random_candidates(source,1001,crop=3,frames=2)
    counts=result.groupby(['day','time_bin']).size()
    assert len(counts)==4 and counts.max()-counts.min()==1


def test_geographic_constraint_checks_every_pixel_and_native_axis_directions():
    x=np.linspace(-24,44,35);y=np.linspace(25,75,26)
    for xs,ys in [(x,y),(x[::-1],y),(x,y[::-1]),(x[::-1],y[::-1])]:
        mask=geographic_origin_mask(xs,ys,'EPSG:4326',(-20,40,30,70),crop=3)
        for iy,ix in itertools.product(range(len(ys)-2),range(len(xs)-2)):
            expected=(np.all((-20<=xs[ix:ix+3])&(xs[ix:ix+3]<=40))
                      and np.all((30<=ys[iy:iy+3])&(ys[iy:iy+3]<=70)))
            assert mask[iy,ix]==expected


def test_random_sampling_respects_full_crop_geographic_support():
    source=inventory();record=source['stores'][0]
    mask=np.zeros((10,13),dtype=bool);mask[2:7,3:9]=True
    result=random_candidates(source,200,crop=3,frames=2,origin_masks={record['store']:mask})
    assert mask[result.origin_y,result.origin_x].all()
    assert set(result.origin_y)==set(range(2,7))
    assert set(result.origin_x)==set(range(3,9))
