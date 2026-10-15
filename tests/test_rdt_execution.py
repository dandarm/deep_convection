"""Parallel current-frame work must preserve causal diagnostics and snapshots."""
from copy import deepcopy
import pickle
import zlib
import numpy as np
import pandas as pd
import pytest
from emma_gpm.rdt import RDTDomainTeacher
from emma_gpm.rdt.pipeline import prepare_detection
from emma_gpm.rdt.execution import process_frames
from emma_gpm.rdt.storage import restore_latest


def scene(i):
    y, x = np.indices((45, 49))
    ir = np.minimum(290., 215+i+((x-23-i/3)**2+(y-23)**2)/4).astype(np.float32)
    return np.stack([ir+2, ir-1, ir+.5, ir-3, ir, ir-.5, ir-4])


def test_prepared_detection_preserves_fields_tables_ids_history_and_gap():
    area = np.full((45,49),9.)
    serial, parallel = RDTDomainTeacher(area), RDTDomainTeacher(area)
    for i, minutes in enumerate([0,5,10,15,20,30,35]):
        frame = scene(i)
        if i == 2:
            frame[:,0,0] = np.nan
            frame[1,1,1] = -1
        stamp = pd.Timestamp('2020-01-01')+pd.Timedelta(minutes=minutes)
        prepared = prepare_detection(frame,area,parallel.config)
        a = serial.step(frame,stamp,allow_invalid=True)
        b = parallel.step(frame,stamp,allow_invalid=True,prepared=prepared)
        for key in a.fields:
            assert np.array_equal(a.fields[key],b.fields[key],equal_nan=True),key
        pd.testing.assert_frame_equal(a.objects,b.objects)
        pd.testing.assert_frame_equal(a.links,b.links)
        assert a.reset_reason == b.reset_reason
        assert serial.next_track_id == parallel.next_track_id
        assert prepared.consumed
    with pytest.raises(ValueError,match='consumed'):
        parallel.step(frame,stamp+pd.Timedelta(minutes=5),prepared=prepared)


def test_prepared_detection_rejects_different_input_before_advancing_state():
    teacher=RDTDomainTeacher(np.full((45,49),9.))
    frame=scene(0)
    prepared=prepare_detection(frame,teacher.area,teacher.config)
    with pytest.raises(ValueError,match='differs'):
        teacher.step(frame+1,'2020-01-01',prepared=prepared)
    assert teacher.last_time is None and not prepared.consumed


def test_async_writer_commits_snapshots_in_order_and_flushes_before_detection_failure(tmp_path):
    teacher=RDTDomainTeacher(np.full((45,49),9.))
    snapshots=[]
    def frames():
        for i in range(4):
            yield pd.Timestamp('2020-01-01')+pd.Timedelta(minutes=5*i),scene(i),None
        raise ValueError('source failure')
    def committed(record):
        state=pickle.loads(zlib.decompress((tmp_path/'frames'/pd.Timestamp(record['time']).strftime('%Y%m%dT%H%M')/'state.pkl.zlib').read_bytes()))
        snapshots.append((record['time'],state.frame_index,state.last_time))
    with pytest.raises(ValueError,match='source failure'):
        process_frames(teacher,frames(),tmp_path,lambda *args:{},committed,packed=True,async_save=True)
    assert [s[1] for s in snapshots] == [1,2,3,4]
    assert all(pd.Timestamp(stamp)==last for stamp,_,last in snapshots)
    restored,records=restore_latest(tmp_path,teacher.config.config_hash)
    assert len(records)==4 and restored.frame_index==4
    a=teacher.step(scene(4),'2020-01-01T00:20')
    b=restored.step(scene(4),'2020-01-01T00:20')
    for key in a.fields:
        assert np.array_equal(a.fields[key],b.fields[key],equal_nan=True)
    pd.testing.assert_frame_equal(a.objects,b.objects)
