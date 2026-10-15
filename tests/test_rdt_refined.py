import numpy as np
import pandas as pd
import pytest
from emma_gpm.rdt_refined import adaptive_cells,correlation_motion,primary_lineage,RefinedRDTConfig,label_refined_clip,association_graph
from emma_gpm.rdt_context import context_window


def test_adaptive_contours_separate_significant_towers_in_warm_envelope():
    ir=np.full((30,30),280.);ir[5:25,5:25]=239.
    ir[8:14,8:14]=220;ir[16:22,16:22]=222
    labels,cells,_=adaptive_cells(ir,np.ones(ir.shape)*10)
    assert len(cells)==2
    assert labels[10,10]!=labels[18,18] and labels[10,10]>0
    assert all(c['threshold_k']<240 for c in cells.values())
    assert labels[5,5]==0


def test_shallow_cloud_not_a_six_kelvin_tower():
    ir=np.full((20,20),280.);ir[5:15,5:15]=237.
    assert not adaptive_cells(ir,np.ones(ir.shape)*10)[1]


def test_correlation_motion_recovers_translation_without_moving_radiances():
    a=np.full((30,30),280.);a[8:14,8:14]=210
    b=np.full((30,30),280.);b[10:16,11:17]=210
    np.testing.assert_array_equal(correlation_motion(a,b,6),[2,3])
    np.testing.assert_array_equal(correlation_motion(a,a,6),[0,0])


def fake_cell(area,seed):return dict(threshold_k=240,area_km2=area,seed=seed)
class Contours:
    def containing(self,threshold,seed):return seed


def test_primary_lineage_preserves_dominant_simple_branches_but_resets_complex():
    old={1:fake_cell(100,100),2:fake_cell(300,300)};now={1:fake_cell(400,400)}
    primary,event=primary_lineage(old,now,{1:[1,2]},{1:[1],2:[1]},Contours(),Contours())
    assert primary=={1:2} and event[1]=='merge_main'
    old={1:fake_cell(400,400)};now={1:fake_cell(100,100),2:fake_cell(300,300)}
    primary,event=primary_lineage(old,now,{1:[1],2:[1]},{1:[1,2]},Contours(),Contours())
    assert primary=={2:1} and event=={1:'split_new',2:'split_main'}
    old={1:fake_cell(100,100),2:fake_cell(100,100)}
    primary,event=primary_lineage(old,now,{1:[1,2],2:[1]},{1:[1,2],2:[1]},Contours(),Contours())
    assert not primary and set(event.values())=={'complex'}


def test_expanded_context_maps_central_native_crop_under_all_orientations():
    for reverse_y in [False,True]:
        for reverse_x in [False,True]:
            data=np.arange(40*50).reshape(40,50)
            y,x,size,roi=context_window(12,18,40,50,crop=10,halo=4,north_reversed=reverse_y,west_reversed=reverse_x)
            tile=data[y:y+size,x:x+size]
            expected=data[12:22,18:28]
            if reverse_y:tile=tile[::-1];expected=expected[::-1]
            if reverse_x:tile=tile[:,::-1];expected=expected[:,::-1]
            np.testing.assert_array_equal(tile[roi],expected)


def test_refined_is_reproducible_two_channel_only_and_rejects_missing_times():
    clip=np.full((3,7,30,30),280.)
    for t in range(3):
        clip[t,4,6:15,6+t:15+t]=235;clip[t,4,9:12,9+t:12+t]=205-t
    times=pd.date_range('2020',periods=3,freq='5min');area=np.ones((30,30))*20
    first=label_refined_clip(clip,times,area)
    other=clip.copy();other[:,[1,2,3,5,6]]=140
    second=label_refined_clip(other,times,area)
    np.testing.assert_array_equal(first['classes'],second['classes'])
    pd.testing.assert_frame_equal(first['objects'],second['objects'])
    assert first['objects'].track_id.nunique()==1
    with pytest.raises(ValueError):label_refined_clip(clip,times[::-1],area)


def test_simple_merge_keeps_main_history_without_fake_area_growth():
    clip=np.full((3,7,40,40),280.)
    # Large main tower develops at t=1; smaller shallow tower is independent.
    clip[0,4,5:12,5:12]=235;clip[0,4,7,7]=213
    clip[1,4,5:14,5:14]=235;clip[1,4,7,7]=210
    clip[:2,4,6:9,18:21]=234
    clip[2,4]=clip[1,4];clip[2,4,7,7]=209
    clip[2,4,6:9,14:21]=230
    result=label_refined_clip(clip,pd.date_range('2020',periods=3,freq='5min'),np.ones((40,40))*20)
    rows=result['objects'];merge=rows[rows.event=='merge_main']
    assert len(merge)==1
    row=merge.iloc[0]
    assert row.temporal_valid and not row.growth_valid
    assert np.isnan(row.growth_km2_15min)
    assert row.class_id==3 and row.reason=='inherited_main_link'
    assert row.track_id==row.primary_parent
    assert not row.phase_valid


def test_advection_recovery_does_not_expand_output_masks():
    a=np.zeros((20,20),bool);a[5:7,5:7]=True
    b=np.zeros_like(a);b[8:10,8:10]=True
    previous={1:dict(mask=a,area_km2=4,motion=np.zeros(2))}
    current={1:dict(mask=b,area_km2=4)}
    parents,_,edges=association_graph(previous,current,np.ones(a.shape),np.array([3,3]),RefinedRDTConfig(min_area_km2=1))
    assert parents=={1:[1]} and edges[1,1]['method']=='correlation'
    assert a.sum()==4 and b.sum()==4
