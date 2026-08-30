from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .emma import EmmaGrid, load_emma_mask


def nearest_emma_hour(times: pd.Series) -> pd.Series:
    """Nearest top-of-hour timestamp; exact +30 min is kept at the earlier hour."""
    floor = times.dt.floor("h")
    delta = (times - floor).dt.total_seconds() / 60.0
    return floor.where(delta <= 30.0, floor + pd.Timedelta(hours=1))


def _ellipse_mask_indices(
    lat0: float,
    lon0: float,
    major_km: float,
    minor_km: float,
    orientation_deg: float,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid-cell centres within the fitted PF ellipse.

    PF documentation calls R_major/R_minor the ellipse axes. We interpret them as
    full axis lengths and therefore divide by two. This geometry is an explicitly
    approximate PF envelope, not the original set of sensor footprints.
    """
    if not np.isfinite(major_km) or not np.isfinite(minor_km) or major_km <= 0 or minor_km <= 0:
        iy = int(np.abs(grid_lat - lat0).argmin())
        ix = int(np.abs(grid_lon - lon0).argmin())
        return np.array([iy]), np.array([ix])
    a = max(major_km / 2.0, 2.5)
    b = max(minor_km / 2.0, 2.5)
    lat_radius = a / 110.574
    lon_radius = a / max(111.320 * np.cos(np.deg2rad(lat0)), 1e-6)
    y_candidates = np.flatnonzero(np.abs(grid_lat - lat0) <= lat_radius)
    x_candidates = np.flatnonzero(np.abs(grid_lon - lon0) <= lon_radius)
    if not len(y_candidates) or not len(x_candidates):
        return np.array([], dtype=int), np.array([], dtype=int)
    yy, xx = np.meshgrid(y_candidates, x_candidates, indexing="ij")
    east = (grid_lon[xx] - lon0) * 111.320 * np.cos(np.deg2rad(lat0))
    north = (grid_lat[yy] - lat0) * 110.574
    theta = np.deg2rad(orientation_deg if np.isfinite(orientation_deg) else 0.0)
    along = east * np.sin(theta) + north * np.cos(theta)
    across = east * np.cos(theta) - north * np.sin(theta)
    inside = (along / a) ** 2 + (across / b) ** 2 <= 1.0
    return yy[inside], xx[inside]


def match_pf_to_emma(
    features: pd.DataFrame,
    emma_paths_by_time: dict[pd.Timestamp, Path],
    grid: EmmaGrid,
    tolerance_minutes: float = 30.0,
) -> pd.DataFrame:
    out = features.copy()
    out["emma_time"] = nearest_emma_hour(out.observation_time)
    out["time_offset_minutes"] = (
        (out.observation_time - out.emma_time).dt.total_seconds() / 60.0
    )
    out["within_time_tolerance"] = out.time_offset_minutes.abs() <= tolerance_minutes
    out["emma_coverage"] = (
        (out.lat >= grid.pilot_lat.min())
        & (out.lat <= grid.pilot_lat.max())
        & (out.lon >= grid.pilot_lon.min())
        & (out.lon <= grid.pilot_lon.max())
    )
    out["mcs_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["robust_mcs_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["overlapping_mcs_ids"] = ""
    out["overlapping_robust_mcs_ids"] = ""
    out["n_overlapped_emma_grid_centres"] = 0
    out["n_overlapped_robust_grid_centres"] = 0
    out["spatial_method"] = "pf_fitted_ellipse_vs_emma_grid_centres"

    eligible = out[out.within_time_tolerance & out.emma_coverage]
    for timestamp, group in eligible.groupby("emma_time"):
        path = emma_paths_by_time.get(timestamp)
        if path is None:
            continue
        mcs_mask, robust_mask = load_emma_mask(path, grid)
        for index, row in group.iterrows():
            lat0 = float(row.get("r_lat", row.lat))
            lon0 = float(row.get("r_lon", row.lon))
            if not np.isfinite(lat0) or not np.isfinite(lon0):
                lat0, lon0 = float(row.lat), float(row.lon)
            yy, xx = _ellipse_mask_indices(
                lat0, lon0,
                float(row.get("r_major", np.nan)),
                float(row.get("r_minor", np.nan)),
                float(row.get("r_orientation", np.nan)),
                grid.pilot_lat, grid.pilot_lon,
            )
            if not len(yy):
                continue
            ids, counts = np.unique(mcs_mask[yy, xx], return_counts=True)
            valid = ids > 0
            ids, counts = ids[valid], counts[valid]
            robust_ids, robust_counts = np.unique(robust_mask[yy, xx], return_counts=True)
            robust_valid = robust_ids > 0
            robust_ids, robust_counts = robust_ids[robust_valid], robust_counts[robust_valid]
            if len(ids):
                primary = int(ids[np.argmax(counts)])
                out.at[index, "mcs_id"] = primary
                out.at[index, "overlapping_mcs_ids"] = ";".join(map(str, sorted(map(int, ids))))
                out.at[index, "n_overlapped_emma_grid_centres"] = int(counts.sum())
            else:
                out.at[index, "mcs_id"] = 0
            if len(robust_ids):
                primary_robust = int(robust_ids[np.argmax(robust_counts)])
                out.at[index, "robust_mcs_id"] = primary_robust
                out.at[index, "overlapping_robust_mcs_ids"] = ";".join(
                    map(str, sorted(map(int, robust_ids)))
                )
                out.at[index, "n_overlapped_robust_grid_centres"] = int(robust_counts.sum())
            else:
                out.at[index, "robust_mcs_id"] = 0

    out["emma_class"] = np.select(
        [
            ~out.emma_coverage,
            ~out.within_time_tolerance,
            out.robust_mcs_id.fillna(0).gt(0),
            out.mcs_id.fillna(0).gt(0),
        ],
        ["outside_emma_coverage", "outside_time_tolerance", "robust_mcs", "mcs_nonrobust_phase"],
        default="outside_emma",
    )
    out["track_uid"] = out.mcs_id.map(
        lambda value: f"2020:{int(value)}" if pd.notna(value) and value > 0 else pd.NA
    )
    return out

