import numpy as np
import pandas as pd

from emma_gpm.ssl_overlap_audit import rectangle_coverage, overlap_summary
from emma_gpm.spatial_sampling import pixel_frame_coverage


def test_compressed_coordinates_match_full_pixel_raster_union_and_reuse():
    rng = np.random.default_rng(5)
    for count in (1, 4, 25):
        y, x = rng.integers(0, 20, (2, count))
        raster = np.zeros((24, 24), int)
        for a, b in zip(y, x):
            raster[a:a+4, b:b+4] += 1
        area, peak = rectangle_coverage(y, x, 4)
        assert area == np.count_nonzero(raster)
        assert peak == raster.max()


def test_spacetime_audit_matches_explicit_observed_frame_union():
    rng = np.random.default_rng(10)
    frame = pd.DataFrame([dict(start_time=(pd.Timestamp('2022-01-01', tz='UTC')
        +pd.Timedelta(minutes=int(rng.integers(0, 20))*5)).isoformat(),
        origin_y=int(rng.integers(0, 17)), origin_x=int(rng.integers(0, 17)),
        day='2022-01-01', store='2022/20220101.zarr') for _ in range(40)])
    explicit = pixel_frame_coverage(frame, 20, 20, size=4, frames=3)
    summary, _, _ = overlap_summary(frame, size=4, frames=3)
    assert summary['unique_pixel_frames'] == explicit['unique_pixel_frames']
    assert summary['repeated_pixel_frame_fraction'] == explicit['repeated_pixel_frame_fraction']
    assert summary['distinct_native_frames'] == explicit['distinct_frames']
