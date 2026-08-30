from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import pandas as pd


TIME_RE = re.compile(r"tracking_(\d{8}T\d{2})\.nc$")


@dataclass(frozen=True)
class EmmaGrid:
    lat: np.ndarray
    lon: np.ndarray
    lat_slice: slice
    lon_slice: slice

    @property
    def pilot_lat(self) -> np.ndarray:
        return self.lat[self.lat_slice]

    @property
    def pilot_lon(self) -> np.ndarray:
        return self.lon[self.lon_slice]


def parse_emma_time(path: str | Path) -> pd.Timestamp:
    match = TIME_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Unrecognized EMMA filename: {path}")
    return pd.Timestamp(
        datetime.strptime(match.group(1), "%Y%m%dT%H").replace(tzinfo=timezone.utc)
    )


def emma_files(root: str | Path) -> list[Path]:
    return sorted(Path(root).rglob("tracking_*.nc"), key=parse_emma_time)


def read_grid(sample_path: str | Path, domain: dict[str, float]) -> EmmaGrid:
    with h5py.File(sample_path) as handle:
        lat = handle["lat"][:]
        lon = handle["lon"][:]
    lat_idx = np.flatnonzero(
        (lat >= domain["lat_min"]) & (lat <= domain["lat_max"])
    )
    lon_idx = np.flatnonzero(
        (lon >= domain["lon_min"]) & (lon <= domain["lon_max"])
    )
    if not len(lat_idx) or not len(lon_idx):
        raise ValueError("Pilot domain does not intersect the EMMA grid")
    return EmmaGrid(
        lat=lat,
        lon=lon,
        lat_slice=slice(int(lat_idx[0]), int(lat_idx[-1]) + 1),
        lon_slice=slice(int(lon_idx[0]), int(lon_idx[-1]) + 1),
    )


def _object_rows(
    timestamp: pd.Timestamp,
    ids: np.ndarray,
    robust_ids: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    full_track_areas_km2: dict[int, float],
    mature_area_threshold_km2: float = 3500.0,
) -> Iterator[dict[str, object]]:
    for mcs_id in np.unique(ids):
        if mcs_id <= 0:
            continue
        yy, xx = np.where(ids == mcs_id)
        robust_here = robust_ids[yy, xx] == mcs_id
        full_area = full_track_areas_km2[int(mcs_id)]
        yield {
            "year": timestamp.year,
            "track_uid": f"{timestamp.year}:{int(mcs_id)}",
            "mcs_id": int(mcs_id),
            "time": timestamp,
            "lat_min": float(lat[yy.min()]),
            "lat_max": float(lat[yy.max()]),
            "lon_min": float(lon[xx.min()]),
            "lon_max": float(lon[xx.max()]),
            "n_emma_grid_cells": int(len(xx)),
            "official_robust_grid_cells": int(robust_here.sum()),
            "official_robust_flag": bool(robust_here.any()),
            "full_track_area_km2": full_area,
            "is_mature_phase_reconstructed": bool(full_area >= mature_area_threshold_km2),
            "centroid_lat_gridmean": float(lat[yy].mean()),
            "centroid_lon_gridmean": float(lon[xx].mean()),
        }


def regular_latlon_cell_area_km2(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Spherical area of each regular lat/lon cell, broadcast over longitude."""
    if len(lat) < 2 or len(lon) < 2:
        raise ValueError("At least two latitude and longitude coordinates are required")
    dlat = float(np.median(np.diff(lat)))
    dlon = float(np.median(np.diff(lon)))
    lat_lo = np.deg2rad(lat - dlat / 2.0)
    lat_hi = np.deg2rad(lat + dlat / 2.0)
    area_by_lat = (6371.0088**2) * np.deg2rad(abs(dlon)) * np.abs(
        np.sin(lat_hi) - np.sin(lat_lo)
    )
    return np.broadcast_to(area_by_lat[:, None], (len(lat), len(lon)))


def prepare_emma_objects(
    root: str | Path,
    domain: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, EmmaGrid]:
    """Create hourly-object and lifecycle tables from official EMMA masks.

    A track is included when at least one EMMA grid-cell centre falls inside the
    pilot domain. The grid-mean centroid is diagnostic only and is never used to
    declare an overpass.
    """
    paths = emma_files(root)
    if not paths:
        raise FileNotFoundError(f"No EMMA tracking files below {root}")
    grid = read_grid(paths[0], domain)
    full_area_map = regular_latlon_cell_area_km2(grid.lat, grid.lon)
    pilot_lat = grid.pilot_lat
    pilot_lon = grid.pilot_lon
    rows: list[dict[str, object]] = []
    for path in paths:
        timestamp = parse_emma_time(path)
        with h5py.File(path) as handle:
            full_ids = handle["mcs_id"][0]
            full_robust = handle["robust_mcs_id"][0]
        ids = full_ids[grid.lat_slice, grid.lon_slice]
        robust = full_robust[grid.lat_slice, grid.lon_slice]
        ids_in_domain = np.unique(ids[ids > 0])
        full_areas = {
            int(mcs_id): float(full_area_map[full_ids == mcs_id].sum())
            for mcs_id in ids_in_domain
        }
        rows.extend(
            _object_rows(
                timestamp, ids, robust, pilot_lat, pilot_lon, full_areas
            )
        )

    objects = pd.DataFrame(rows)
    if objects.empty:
        lifecycle = pd.DataFrame()
        return objects, lifecycle, grid

    lifecycle = (
        objects.groupby(["track_uid", "year", "mcs_id"], as_index=False)
        .agg(
            lifecycle_start=("time", "min"),
            lifecycle_end=("time", "max"),
            observed_hours_in_domain=("time", "nunique"),
            official_robust_hours_in_domain=("official_robust_flag", "sum"),
            mature_hours_reconstructed=("is_mature_phase_reconstructed", "sum"),
            max_emma_grid_cells=("n_emma_grid_cells", "max"),
            max_official_robust_grid_cells=("official_robust_grid_cells", "max"),
            max_full_track_area_km2=("full_track_area_km2", "max"),
            track_lat_min=("lat_min", "min"),
            track_lat_max=("lat_max", "max"),
            track_lon_min=("lon_min", "min"),
            track_lon_max=("lon_max", "max"),
        )
        .sort_values("track_uid")
        .reset_index(drop=True)
    )
    lifecycle["has_official_robust_mask_in_domain"] = (
        lifecycle["official_robust_hours_in_domain"] > 0
    )
    lifecycle["has_mature_phase_reconstructed_while_in_domain"] = (
        lifecycle["mature_hours_reconstructed"] > 0
    )
    return objects, lifecycle, grid


def load_emma_mask(
    path: str | Path, grid: EmmaGrid
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path) as handle:
        return (
            handle["mcs_id"][0, grid.lat_slice, grid.lon_slice],
            handle["robust_mcs_id"][0, grid.lat_slice, grid.lon_slice],
        )
