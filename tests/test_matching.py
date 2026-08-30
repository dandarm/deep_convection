import numpy as np
import pandas as pd

from emma_gpm.matching import _ellipse_mask_indices, nearest_emma_hour


def test_nearest_hour_keeps_exact_half_hour_at_earlier_hour():
    times = pd.Series(pd.to_datetime(["2020-06-01T12:29:59Z", "2020-06-01T12:30:00Z", "2020-06-01T12:30:01Z"]))
    result = nearest_emma_hour(times)
    expected = pd.Series(pd.to_datetime(["2020-06-01T12:00:00Z", "2020-06-01T12:00:00Z", "2020-06-01T13:00:00Z"]))
    pd.testing.assert_series_equal(result, expected)


def test_ellipse_uses_grid_centres_and_contains_centre():
    lat = np.arange(35.0, 36.1, 0.1)
    lon = np.arange(10.0, 11.1, 0.1)
    iy, ix = _ellipse_mask_indices(35.5, 10.5, 40.0, 20.0, 0.0, lat, lon)
    points = set(zip(iy.tolist(), ix.tolist()))
    centre = (int(np.abs(lat - 35.5).argmin()), int(np.abs(lon - 10.5).argmin()))
    assert centre in points
    assert len(points) > 1


def test_invalid_ellipse_falls_back_to_nearest_grid_centre():
    lat = np.array([30.0, 30.1, 30.2])
    lon = np.array([5.0, 5.1, 5.2])
    iy, ix = _ellipse_mask_indices(30.11, 5.19, np.nan, np.nan, np.nan, lat, lon)
    assert iy.tolist() == [1]
    assert ix.tolist() == [2]
