import numpy as np
import pandas as pd
from emma_gpm.rdt_dataset import diagnostic_manifest,occupancy_metrics


def test_diagnostic_windows_and_lattice_do_not_overlap_and_are_seeded():
    times=pd.date_range('2020-01-01',periods=1000,freq='5min')
    manifest,design=diagnostic_manifest(times,763,1762,count=400)
    other,_=diagnostic_manifest(times,763,1762,count=400)
    pd.testing.assert_frame_equal(manifest,other)
    starts=pd.DatetimeIndex(manifest.context_start.unique()).sort_values()
    assert np.all(np.diff(starts.asi8)>=95*60*1_000_000_000)
    for _,group in manifest.groupby('context_start'):
        origins=group[['origin_y','origin_x']].to_numpy()
        for i, (y,x) in enumerate(origins):
            for yy,xx in origins[i+1:]:assert abs(y-yy)>=224 or abs(x-xx)>=224
    assert design['clips_per_window']==21


def test_month_balance_and_missing_context_slots_are_respected():
    times=pd.date_range('2020-01-01',periods=100,freq='5min').append(pd.date_range('2020-06-01',periods=100,freq='5min'))
    times=times.delete(22)
    m,_=diagnostic_manifest(times,448,448,count=20)
    assert m.month.value_counts().to_dict()=={1:10,6:10}
    available=set(times.asi8)
    for start in pd.DatetimeIndex(m.context_start):
        assert all((start+pd.Timedelta(minutes=5*i)).value in available for i in range(19))


def test_occupancy_distinguishes_tracks_mean_peak_and_unknown():
    labels=np.zeros((2,2,2),dtype=np.uint8);labels[0,0,0]=2;labels[1,0,:]=4
    result=dict(classes=labels,uncertain=np.zeros_like(labels,dtype=bool),
                objects=pd.DataFrame(dict(track_id=[1,1],class_id=[2,3],ot_pixels=[0,2])))
    values=occupancy_metrics(result,np.ones((2,2))*10)
    assert values['convective_tracks']==1
    assert values['ot_tracks']==1
    assert values['convective_fraction']==3/8
    assert values['convective_peak_fraction']==.5
    assert values['ot_peak_fraction']==.5
    assert values['convective_peak_area_km2']==20
