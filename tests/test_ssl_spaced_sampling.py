import numpy as np
import pandas as pd
import pytest

from emma_gpm.spatial_sampling import ActiveCropSampler, verify_overlap
from emma_gpm.ssl_spaced_sampling import (active_clip_rows, balanced_subset,
    window_crop_moments)


def test_chronological_sampler_keeps_full_footprint_mask_and_active_limit():
    mask = np.zeros((21, 41), bool)
    mask[:, 5:35] = True
    first = ActiveCropSampler((40, 60), seed=3, max_iou=.2, size=20,
        proposals_per_start=8, attempts=1000)
    second = ActiveCropSampler((40, 60), seed=3, max_iou=.2, size=20,
        proposals_per_start=8, attempts=1000)
    for time in pd.date_range('2022-01-01', periods=40, freq='5min', tz='UTC'):
        events = first.sample_window(time, mask)
        assert events == second.sample_window(time, mask)
        assert all(mask[event.y, event.x] for event in events)
    assert len(first.events) > 2
    assert verify_overlap(first.events, size=20, max_iou=.2)['maximum_concurrent_iou'] <= .2
    with pytest.raises(ValueError, match='increasing'):
        first.sample_window(first.previous, mask)


def test_strict_sampler_does_not_accept_infeasible_fallback():
    sampler = ActiveCropSampler((2, 2), seed=0, size=2, max_iou=.2)
    mask = np.ones((1, 1), bool)
    assert len(sampler.sample_window('2022-01-01T00:00Z', mask)) == 1
    assert not sampler.sample_window('2022-01-01T01:15Z', mask)
    assert len(sampler.sample_window('2022-01-01T01:20Z', mask)) == 1


def test_rolling_moments_match_direct_population_with_windows_and_time_gaps():
    values = np.random.default_rng(8).uniform(200, 300, (15, 7, 8, 10)).astype(np.float16)
    values[:, :, 7, 9] = np.nan  # Outside all selected crops.
    windows = [(np.arange(i, i+4), [(0, 0), (2, 4)]) for i in (0, 1, 4, 10)]
    for (indices, origins), (means, variances) in zip(windows,
            window_crop_moments(values, windows, crop=3), strict=True):
        for j, (y, x) in enumerate(origins):
            direct = values[indices, :, y:y+3, x:x+3].astype(np.float64)
            assert means[j] == pytest.approx(direct.mean(), abs=1e-10)
            assert variances[j] == pytest.approx(direct.var(), abs=1e-8)


def test_calendar_selection_respects_capacity_equal_months_and_stable_identity():
    rows = []
    for month, capacity in ((1, 2), (2, 10), (3, 20)):
        for day in (1, 2):
            for time_bin in range(4):
                for index in range(capacity):
                    rows.append(dict(day=f'2022-{month:02d}-{day:02d}', time_bin=time_bin,
                        start_time=f'2022-{month:02d}-{day:02d}T{time_bin*6:02d}:00:00Z',
                        origin_y=index, origin_x=0, clip_id=len(rows)))
    candidates = pd.DataFrame(rows)
    selected = balanced_subset(candidates, 100, 5)
    assert len(selected) == 48 + 1
    counts = selected.groupby(selected.day.str[:7]).size()
    assert counts.max() - counts.min() <= 1
    assert selected.clip_id.is_unique
    assert set(selected.clip_id) <= set(candidates.clip_id)
    pd.testing.assert_frame_equal(selected, balanced_subset(candidates, 100, 5))
    assert len(balanced_subset(candidates, 31, 5)) == 31


def test_animation_replays_manifest_at_real_frame_endpoints():
    frame = pd.DataFrame(dict(start_time=['2022-01-01T00:00Z', '2022-01-01T00:05Z'],
        clip_id=['a', 'b']))
    assert active_clip_rows(frame, '2022-01-01T01:15Z').clip_id.tolist() == ['a', 'b']
    assert active_clip_rows(frame, '2022-01-01T01:20Z').clip_id.tolist() == ['b']
    assert active_clip_rows(frame, '2022-01-01T01:22Z').empty


@pytest.mark.parametrize('reverse_y,reverse_x', [(False, False), (True, False), (False, True), (True, True)])
def test_materialized_clip_preserves_native_bits_and_coordinate_orientation(tmp_path, reverse_y, reverse_x):
    from scripts.build_seviri_ssl_spaced_dataset import save_clip
    from emma_gpm.seviri_clip_files import codec, SHAPE
    from emma_gpm.spatial_sampling import CropEvent
    # Different row/column patterns reveal transposes and incorrect flips.
    y, x = np.arange(224, dtype=float)[::-1], np.arange(224, dtype=float)
    values = np.broadcast_to((220 + np.arange(224)[:, None]/4
        + np.arange(224)[None, :]/8).astype(np.float16), SHAPE).copy()
    if reverse_y:
        y, values = y[::-1], values[..., ::-1, :]
    if reverse_x:
        x, values = x[::-1], values[..., ::-1]
    event = CropEvent(pd.Timestamp('2022-01-01', tz='UTC'), 0, 0)
    name, entry = save_clip(values, np.arange(16), event, x, y, tmp_path)
    restored = np.empty(SHAPE, np.float16)
    codec().decode((tmp_path/name).read_bytes(), out=restored)
    assert restored[0, 0, 0, 0] == 220
    assert restored[0, 0, -1, 0] == np.float16(220+223/4)
    assert restored[0, 0, 0, -1] == np.float16(220+223/8)
    assert entry['bytes'] == (tmp_path/name).stat().st_size
