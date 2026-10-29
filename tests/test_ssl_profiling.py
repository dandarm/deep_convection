import json

from scripts.profile_seviri_ssl import union_duration,trace_activity


def test_cuda_busy_time_counts_overlapping_streams_once():
    assert union_duration([(10,20),(15,30),(4,8),(10,12)])==24
    assert union_duration([])==0


def test_trace_activity_is_clipped_to_measurement_window(tmp_path):
    events=[dict(name='MEASURED_TRAINING_WINDOW',ph='X',ts=100,dur=100),
            dict(name='kernel_a',cat='kernel',ph='X',ts=90,dur=30),
            dict(name='kernel_b',cat='kernel',ph='X',ts=110,dur=20),
            dict(name='copy',cat='gpu_memcpy',ph='X',ts=125,dur=25),
            dict(name='after',cat='kernel',ph='X',ts=210,dur=10)]
    path=tmp_path/'trace.json'
    path.write_text(json.dumps({'traceEvents':events}))
    result=trace_activity(path)
    assert result['kernel_active_fraction']==.3
    assert result['device_activity_active_fraction']==.5
    assert result['kernel_events']==2
