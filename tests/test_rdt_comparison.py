import pandas as pd
import pytest
from emma_gpm.rdt_comparison import compare_catalogs


def test_comparison_uses_only_common_valid_matching_clips(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    baseline=pd.DataFrame(dict(clip_id=['a','b','c'],status=['ok']*3,start_time=['2020']*3,
        origin_y=[1,2,3],origin_x=[5]*3,convective_pixels=[1,0,0],ot_pixels=[0]*3,
        known_cold_pixels=[2]*3,cold_pixels=[4]*3))
    refined=baseline.iloc[:2].copy();refined.loc[1,'status']='rejected'
    for name in ['single_point_fraction','mean_visible_track_frames','temporal_object_valid_fraction']:
        refined[name]=.2;refined['baseline_'+name]=.5
    baseline.to_csv(old/'catalog.csv',index=False);refined.to_csv(new/'catalog.csv',index=False)
    result=compare_catalogs(new,old)
    assert result['common_valid']==1 and result['baseline_valid']==2
    assert result['baseline_only_valid']==1
    assert result['new_known_cold_share']==.5
    assert result['single_point_fraction']['refined']==.2
    refined.loc[0,'origin_y']=7;refined.to_csv(new/'catalog.csv',index=False)
    with pytest.raises(ValueError,match='identity'):compare_catalogs(new,old)
