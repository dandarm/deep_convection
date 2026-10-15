import numpy as np
import pandas as pd
import pytest
import zarr
from emma_gpm.rdt_event_explorer import btd_norm,rank_frames,study_window,study_origin,summarize_frame


def test_zero_center_fixed_btd_and_signed_sides():
    norm=btd_norm()
    np.testing.assert_allclose(norm([-80,-40,0,5,10]),[0,.25,.5,.75,1])
    assert norm(-1)<.5<norm(1)


def test_rank_suppresses_nearby_frames_and_uses_deterministic_ties():
    frame=pd.DataFrame(dict(time=pd.to_datetime(['2020-01-01T00:00','2020-01-01T00:05','2020-01-01T02:00','2020-01-02T00:00']),convective_percent=[10,20,10,0]))
    ranked=rank_frames(frame,'convective_percent')
    assert ranked.time.tolist()==[frame.time[1],frame.time[2]]
    assert rank_frames(frame,'missing').empty


def test_study_uses_complete_window_and_native_orientation():
    times=pd.date_range('2020-01-01',periods=20,freq='5min')
    assert study_window(times,times[10])==times[2]
    assert study_window(times,times[1])==times[0]
    with pytest.raises(ValueError,match='16 frame'):study_window(times.delete(9),times[10])
    tracks=np.zeros((300,500),np.int64);tracks[240:260,390:410]=4294967297
    y,x=study_origin(tracks,4294967297,np.arange(500)[::-1],np.arange(300))
    # Clamped north-up bottom-right crop maps to native top-left indices.
    assert (y,x)==(0,0)
    with pytest.raises(ValueError,match='presente'):study_origin(tracks,123,np.arange(500),np.arange(300))


def test_index_denominator_and_quality_selection(tmp_path):
    directory=tmp_path/'20200101T0000';directory.mkdir()
    group=zarr.open_group(str(directory/'fields.zarr'),mode='w',zarr_format=2)
    labels=np.array([[0,2,3,4],[2,0,0,4]],np.uint8)
    valid=np.array([[1,1,1,1],[1,1,0,0]],bool)
    group.create_array('semantic_label',data=labels);group.create_array('valid_observation',data=valid)
    pd.DataFrame([dict(track_id=i,class_id=2,proxy_confidence=.8,border=(i==2),temporal_valid=True,phase_valid=True,area_km2=100,ot_pixels=1,min_k=210) for i in [1,2]]).to_csv(directory/'objects.csv.gz',index=False)
    summary,events=summarize_frame(directory)
    assert summary['valid_pixels']==6 and summary['convective_pixels']==4 and summary['ot_pixels']==1
    assert summary['convective_percent']==pytest.approx(100*4/6)
    assert [(e['track_id'],e['kind']) for e in events]==[(1,'convection'),(1,'ot')]
