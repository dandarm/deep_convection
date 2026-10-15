"""Parallelization must preserve physical outputs and genuine temporal gaps."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from emma_gpm.rdt import RDTDomainTeacher, RDTTeacherConfig
from emma_gpm.rdt.storage import save_frame, restore_latest, open_fields

spec = importlib.util.spec_from_file_location('parallel_runner', Path(__file__).parents[1]/'scripts/build_rdt_year_parallel.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def frame():
    y,x=np.indices((40,48))
    ir=np.full((40,48),290.,np.float32)
    ir[np.hypot(y-20,x-24)<10]=225
    ir[np.hypot(y-20,x-24)<2]=205
    return np.stack([ir+2,ir-1,ir+.5,ir-3,ir,ir-.5,ir-4])


def test_sequences_preserve_midnight_and_reject_duplicates():
    times=pd.DatetimeIndex(['2020-01-01T23:55','2020-01-02T00:00','2020-01-02T00:10','2020-01-02T00:15'])
    segments=runner.split_sequences(times)
    assert [s['frames'] for s in segments]==[2,2]
    assert segments[1]['previous_time']==times[1].isoformat()
    assert segments[1]['frame_offset']==2
    with pytest.raises(ValueError): runner.split_sequences(times[::-1])
    with pytest.raises(ValueError): runner.split_sequences(times.append(times[-1:]))


def test_namespaced_sequence_matches_serial_after_real_gap():
    cfg=RDTTeacherConfig(smoothing_sigma_pixels=0)
    area=np.full((40,48),9.)
    serial=RDTDomainTeacher(area,cfg)
    a=pd.Timestamp('2020-01-01T23:55')
    serial.step(frame(),a)
    serial.step(frame(),a+pd.Timedelta(minutes=5))
    segmented=RDTDomainTeacher(area,cfg,track_id_offset=2*(1<<32))
    segmented.last_time=serial.last_time
    segmented.frame_index=serial.frame_index
    for stamp in pd.date_range('2020-01-02T00:10',periods=5,freq='5min'):
        expected=serial.step(frame(),stamp)
        actual=segmented.step(frame(),stamp)
        assert actual.reset_reason==expected.reset_reason
        for name in expected.fields:
            if name=='track_id':
                assert np.array_equal(actual.fields[name]>0,expected.fields[name]>0)
            else: np.testing.assert_array_equal(actual.fields[name],expected.fields[name])
        # Local ID values differ, but causal phase, proxy, diagnostics and frame index agree.
        excluded=['track_id','parents','parent_track_ids','child_track_ids','primary_parent']
        pd.testing.assert_frame_equal(actual.objects.drop(columns=excluded),expected.objects.drop(columns=excluded),check_exact=True)
        assert all(actual.objects.track_id >= 2*(1<<32)+1)


def test_zip_roundtrip_resume_checksum_and_legacy(tmp_path):
    cfg=RDTTeacherConfig(smoothing_sigma_pixels=0)
    teacher=RDTDomainTeacher(np.full((40,48),9.),cfg,track_id_offset=1<<32)
    for i in range(3):
        result=teacher.step(frame(),pd.Timestamp('2020-01-01')+pd.Timedelta(minutes=5*i))
        save_frame(tmp_path,result,teacher,{},packed=True)
        directory=tmp_path/'frames'/f'20200101T00{i*5:02d}'
        assert len(list(directory.iterdir()))==5
        with open_fields(directory) as group:
            for name,field in result.fields.items(): np.testing.assert_array_equal(group[name][:],field)
    assert len(list((tmp_path/'frames').glob('*/state.pkl.zlib')))==2
    restored,records=restore_latest(tmp_path,cfg.config_hash)
    assert restored.next_track_id==teacher.next_track_id and len(records)==3
    packed=directory/'fields.zarr.zip'
    with packed.open('ab') as handle: handle.write(b'corruption')
    with pytest.raises(ValueError,match='Corrupt published'): restore_latest(tmp_path,cfg.config_hash)


def test_hexadecimal_hashes_are_never_inferred_as_numbers(tmp_path):
    from emma_gpm.rdt.storage import read_table
    path=tmp_path/'metrics.csv'
    # Native numeric inference on this oversized exponent segfaults pandas.
    digest='7e25247201576e7460a0f43b55b7878617126e4deb1e6a9ee085031cea8539dc'
    path.write_text('fields_sha256,source_frame_sha256,cells\n'+digest+','+digest+',3\n')
    table=read_table(path)
    assert table.fields_sha256[0]==digest and table.source_frame_sha256[0]==digest
    assert table.cells[0]==3
