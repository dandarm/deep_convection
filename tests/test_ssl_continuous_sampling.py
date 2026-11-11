import numpy as np
import pandas as pd
import pytest

from emma_gpm.ssl_continuous_sampling import (continuous_windows, event_array,
    preparation_phase, temporal_context, training_time_mask)
from emma_gpm.ssl_incremental_sampling import IncrementalCropSampler, balanced_extension
from emma_gpm.ssl_overlap_audit import overlap_summary
from emma_gpm.ssl_validation_set import assert_day_separation


@pytest.mark.parametrize('date', ['2022-05-06', '2022-12-31', '2024-02-29'])
def test_daily_storage_boundary_does_not_remove_any_complete_start(date):
    times = pd.date_range(date, periods=288+15, freq='5min', tz='UTC')
    windows = continuous_windows(times, date)
    assert len(windows) == 288
    np.testing.assert_array_equal(list(windows.values())[-1], np.arange(287, 303))
    assert preparation_phase(date) != preparation_phase(times[-1])


def test_real_gap_and_heldout_buffer_reject_whole_cross_midnight_windows():
    times = pd.date_range('2022-12-31', periods=303, freq='5min', tz='UTC')
    missing = times.delete(289)  # 00:05 of January 1 is absent.
    windows = continuous_windows(missing, '2022-12-31')
    assert len(windows) == 274  # 22:45 can finish at 00:00; later starts fail.
    allowed = training_time_mask(times, ['2023-01-02'], 24)
    windows = continuous_windows(times, '2022-12-31', allowed=allowed)
    assert len(windows) == 273
    frame = pd.DataFrame({'start_time': [times[ids[0]] for ids in windows.values()]})
    assert_day_separation(frame, ['2023-01-02'], 24)
    with pytest.raises(ValueError, match='reserved'):
        assert_day_separation(pd.DataFrame({'start_time': [times[273]]}), ['2023-01-02'], 24)


def test_cross_midnight_collision_checks_both_neighbors_and_parallel_phases():
    fixed = pd.DataFrame(dict(start_time=['2022-12-31T23:55Z', '2023-01-02T00:00Z'],
        origin_y=[0, 0], origin_x=[0, 0]))
    context = temporal_context(fixed, '2023-01-01')
    assert len(context) == 2
    sampler = IncrementalCropSampler(event_array(context), size=20)
    candidates = fixed.copy()
    candidates.start_time = ['2023-01-01T00:00Z', '2023-01-01T23:55Z']
    assert [sampler.try_add(event) for event in event_array(candidates)] == ['spatial_iou']*2
    candidates.origin_y = 100
    assert [sampler.try_add(event) for event in event_array(candidates)] == ['accepted']*2
    assert preparation_phase('2022-12-31') == preparation_phase('2023-01-02')


def test_continuous_audit_counts_midnight_union_once_and_includes_future_neighbors():
    frame = pd.DataFrame(dict(start_time=['2022-12-31T23:55Z', '2023-01-01T00:00Z'],
        origin_y=[0, 0], origin_x=[0, 1], day=['2022-12-31', '2023-01-01'],
        store=['20221231.zarr', '20230101.zarr']))
    summary, daily, neighbors = overlap_summary(frame, size=2, frames=3)
    # Four distinct timestamps: union areas 4, 6, 6, 4.
    assert summary['unique_pixel_frames'] == 20
    assert summary['sampled_pixel_frames'] == 24
    assert summary['distinct_native_frames'] == 4
    assert daily.distinct_frames.tolist() == [1, 3]
    assert daily.unique_pixel_frames.tolist() == [4, 16]
    np.testing.assert_allclose(neighbors.maximum_active_iou, 1/3)
    np.testing.assert_allclose(neighbors.maximum_shared_observation_fraction, 1/3)
    single, daily, _ = overlap_summary(frame.iloc[:1], size=2, frames=3)
    assert single['unique_pixel_frames'] == 12 and daily.clips.tolist() == [1, 0]


def test_extension_accepts_new_years_and_preserves_existing_selection():
    candidates = pd.DataFrame([dict(clip_id=f'{year}_{i}', day=f'{year}-01-01',
        time_bin=i%4, start_time=f'{year}-01-01T00:00Z', origin_y=i, origin_x=0)
        for year in (2022, 2023) for i in range(20)])
    fixed = candidates.iloc[:8]
    result = balanced_extension(fixed, candidates, seed=4, target=32)
    assert set(fixed.clip_id) <= set(result.clip_id)
    assert len(result) == 32
    assert result.groupby(result.day.str[:7]).size().tolist() == [16, 16]


def test_builder_materializes_native_frames_across_year_boundary(tmp_path, monkeypatch):
    import xarray as xr
    from scripts import build_seviri_ssl_continuous_dataset as builder
    from emma_gpm.seviri_clip_files import codec
    archive, output = tmp_path/'archive', tmp_path/'output'
    (output/'candidate_clips').mkdir(parents=True)
    (output/'daily').mkdir()
    monkeypatch.setattr(builder, 'geographic_origin_mask', lambda *args: np.ones((1, 1), bool))
    records = []
    times = pd.date_range('2022-12-31T23:00', periods=27, freq='5min')
    for start, stop, date in [(0, 12, '20221231'), (12, 27, '20230101')]:
        values = np.broadcast_to(np.arange(start, stop, dtype=np.float16)[:, None, None, None]+250,
                                 (stop-start, 7, 224, 224)).copy()
        values += np.arange(224, dtype=np.float16)[None, None, :, None]/16
        ds = xr.Dataset({'seviri':(('time','channel','y','x'),values)},
            coords=dict(time=times[start:stop], channel=list(builder.VIDEO_MAE_CHANNELS),
                        y=np.arange(224), x=np.arange(224)[::-1]), attrs=dict(projection_wkt='test-grid'))
        store = f'{date[:4]}/{date}.zarr'
        ds.to_zarr(archive/store, zarr_format=2, consolidated=True,
            encoding={'seviri':{'chunks':(1,7,224,224)}})
        records.append(dict(store=store, headers={}, shape=list(values.shape), chunks=[1,7,224,224],
            times=[t.isoformat()+'Z' for t in times[start:stop]],
            present=[f'{i}.0.0.0' for i in range(stop-start)]))
    config = dict(output=str(output), archive=str(archive), domain=[-20,40,30,70],
        reserved_days=[], temporal_separation_hours=24, seed=7, max_iou=.2,
        max_shared_fraction=.2, block_size=32, minimum_acceptance=.1, patience=2, max_proposals=256)
    task = (records[0], records[1], config, [], {})
    result = builder.prepare_day(task)
    assert result['complete_windows'] == result['complete_cross_midnight_windows'] == 12
    assert len(result['clips']) == 1
    row = result['clips'][0]
    raw = np.empty((16,7,224,224), np.float16)
    codec().decode((output/'candidate_clips'/row['file']).read_bytes(), out=raw)
    first = int((pd.Timestamp(row['start_time']).tz_localize(None)-times[0]).total_seconds()/300)
    expected = np.broadcast_to(np.arange(first,first+16,dtype=np.float16)[:,None,None,None]+250,
                               raw.shape).copy()
    expected += np.arange(224,dtype=np.float16)[None,None,:,None]/16
    np.testing.assert_array_equal(raw, expected[...,::-1,::-1])
    assert row['mean_k'] == pytest.approx(raw.astype(float).mean(), abs=1e-9)
    assert row['variance_k2'] == pytest.approx(raw.astype(float).var(), abs=1e-9)
    # Restart reads the committed result without loading native payloads again.
    monkeypatch.setattr(builder, 'load_observations', lambda *a, **k: pytest.fail('Unexpected reread'))
    assert builder.prepare_day(task) == result
