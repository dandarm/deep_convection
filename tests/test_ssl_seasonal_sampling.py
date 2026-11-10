import numpy as np
import pandas as pd
import pytest

from emma_gpm.ssl_seasonal_sampling import (calendar_quotas, crop_moments,
    selected_time_indices, valid_crop_origins, frame_pixel_validity, valid_crop_origins_from_pixels)


def test_calendar_balances_months_with_unequal_day_counts():
    days = ['2022-01-01', '2022-01-02', '2022-02-01']
    quotas = calendar_quotas(days, 100, 8)
    assert sum(sum(q.values()) for q in quotas.values()) == 100
    assert sum(quotas[days[0]].values()) + sum(quotas[days[1]].values()) == 50
    assert sum(quotas[days[2]].values()) == 50
    for daily in quotas.values():
        assert max(daily.values()) - min(daily.values()) <= 1
    assert quotas == calendar_quotas(days, 100, 8)


def test_quality_and_moments_match_direct_pixel_calculation():
    rng = np.random.default_rng(23)
    a = rng.uniform(200, 300, (3, 2, 7, 8)).astype(np.float16)
    origins = [(0, 0), (1, 4), (4, 5)]
    mean, variance = crop_moments(a, origins, crop=3)
    for i, (y, x) in enumerate(origins):
        direct = a[:, :, y:y+3, x:x+3].astype(np.float64)
        assert mean[i] == pytest.approx(direct.mean(), abs=1e-10)
        assert variance[i] == pytest.approx(direct.var(), abs=1e-9)
    geo = np.ones((5, 6), dtype=bool)
    a[2, 1, 1, 2] = np.nan
    a[0, 0, 5, 6] = 99
    geo[4, 0] = False
    physical = np.ones((7, 8), bool)
    physical[6, 7] = False
    valid = valid_crop_origins(a, geo, physical, crop=3)
    for y in range(5):
        for x in range(6):
            patch = a[:, :, y:y+3, x:x+3]
            expected = (geo[y, x] and physical[y:y+3, x:x+3].all()
                and np.isfinite(patch).all() and (patch >= 100).all() and (patch <= 400).all())
            assert valid[y, x] == expected


def test_sparse_frames_preserve_complete_windows_and_reject_gaps():
    times = pd.date_range('2022-01-01', periods=100, freq='5min', tz='UTC')
    indices = selected_time_indices(times, [times[1], times[5], times[70]])
    assert len(indices) == 36
    assert set(range(1, 21)) <= set(indices)
    assert set(range(70, 86)) <= set(indices)
    with pytest.raises(ValueError, match='missing RSS'):
        selected_time_indices(times.delete(10), [times[1]])


def test_missing_pixels_outside_crop_do_not_contaminate_global_statistics():
    a = np.full((2, 2, 5, 6), 250, dtype=np.float16)
    a[0, 0, 0, 0] = np.nan
    a[1, 1, 0, 4] = np.inf
    mean, variance = crop_moments(a, [(2, 2)], crop=3)
    np.testing.assert_array_equal(mean, [250.])
    np.testing.assert_array_equal(variance, [0.])
    assert np.isnan(a[0, 0, 0, 0]) and np.isinf(a[1, 1, 0, 4])
    with pytest.raises(ValueError, match='complete selected'):
        crop_moments(a, [(0, 0)], crop=3)


def test_reused_frame_quality_matches_checks_on_each_window():
    a = np.random.default_rng(2).uniform(200, 300, (12, 7, 8, 9)).astype(np.float16)
    a[3, 2, 1, 5] = np.nan
    a[8, 6, 5, 6] = 500
    quality = frame_pixel_validity(a)
    geo = np.ones((6, 7), bool)
    physical = np.ones((8, 9), bool)
    physical[7, 7] = False
    for indices in ([0, 1, 2], [2, 3, 4], [7, 8, 9], [1, 4, 10]):
        direct = valid_crop_origins(a[indices], geo, physical, crop=3)
        reused = valid_crop_origins_from_pixels(quality[indices].all(axis=0), geo, physical, crop=3)
        np.testing.assert_array_equal(direct, reused)
