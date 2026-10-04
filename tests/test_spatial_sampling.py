import numpy as np
import pandas as pd
import pytest
from emma_gpm.spatial_sampling import CropEvent,crop_iou,shared_frames,choose_crop,verify_overlap


def test_half_open_pixel_boxes_and_inclusive_frame_endpoints():
    start=pd.Timestamp('2020-01-01')
    a=CropEvent(start,0,0)
    assert crop_iou(a,CropEvent(start,0,224),224)==0
    assert crop_iou(a,CropEvent(start,0,112),224)==pytest.approx(1/3)
    assert shared_frames(a,CropEvent(start+pd.Timedelta(minutes=75),0,0))
    assert not shared_frames(a,CropEvent(start+pd.Timedelta(minutes=80),0,0))


def test_infeasible_random_search_rejects_instead_of_falling_back():
    start=pd.Timestamp('2020-01-01')
    coverage=np.zeros((224,224))
    active=[CropEvent(start,0,0)]
    assert choose_crop(np.random.default_rng(42),start,active,coverage,224,attempts=10) is None


def test_sampling_is_reproducible_geographically_filtered_and_strict():
    start=pd.Timestamp('2020-01-01')
    coverage=np.zeros((40,60))
    active=[CropEvent(start,0,0)]
    kwargs=dict(attempts=200,max_iou=0.,center_allowed=lambda y,x:x>=20)
    a=choose_crop(np.random.default_rng(42),start,active,coverage,20,**kwargs)
    b=choose_crop(np.random.default_rng(42),start,active,coverage,20,**kwargs)
    assert a==b and a.x>=20
    assert verify_overlap(active+[a],size=20)['maximum_concurrent_iou']==0
    with pytest.raises(ValueError,match='overlap'):
        verify_overlap(active+[CropEvent(start+pd.Timedelta(minutes=5),0,0)],size=20)
    verify_overlap(active+[CropEvent(start+pd.Timedelta(minutes=80),0,0)],size=20)


def test_exact_pixel_frame_reuse_counts_shared_frames_and_area():
    from emma_gpm.spatial_sampling import pixel_frame_coverage
    frame=pd.DataFrame([dict(start_time='2020-01-01',origin_y=0,origin_x=0),
                        dict(start_time='2020-01-01T00:05',origin_y=0,origin_x=0)])
    result=pixel_frame_coverage(frame,2,4,size=2,frames=2)
    assert result['sampled_pixel_frames']==16
    assert result['unique_pixel_frames']==12
    assert result['repeated_pixel_frame_fraction']==.25
    frame.loc[1,'origin_x']=2
    result=pixel_frame_coverage(frame,2,4,size=2,frames=2)
    assert result['unique_pixel_frames']==16 and result['repeated_pixel_frame_fraction']==0
