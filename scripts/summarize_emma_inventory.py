#!/usr/bin/env python3
"""Count EMMA object-hours and tracks by year on the full 20W-40E, 30-70N domain."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def summarize_year(year_dir: str) -> dict[str, int]:
    root = Path(year_dir)
    paths = sorted(root.rglob("tracking_*.nc"))
    if not paths:
        return {"year": int(root.name), "files": 0, "positive_scenes": 0,
                "object_hours": 0, "robust_object_hours": 0, "tracks": 0}
    with h5py.File(paths[0]) as handle:
        lat = np.asarray(handle["lat"])
        lon = np.asarray(handle["lon"])
    lat_idx = np.flatnonzero((lat >= 30.0) & (lat <= 70.0))
    lon_idx = np.flatnonzero((lon >= -20.0) & (lon <= 40.0))
    ys = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
    xs = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
    tracks: set[int] = set()
    object_hours = 0
    robust_object_hours = 0
    positive_scenes = 0
    for path in paths:
        with h5py.File(path) as handle:
            ids = np.asarray(handle["mcs_id"][0, ys, xs])
            robust = np.asarray(handle["robust_mcs_id"][0, ys, xs])
        current = np.unique(ids[ids > 0]).astype(int)
        robust_current = np.unique(robust[robust > 0]).astype(int)
        if current.size:
            positive_scenes += 1
            object_hours += int(current.size)
            tracks.update(current.tolist())
        robust_object_hours += int(robust_current.size)
    return {
        "year": int(root.name),
        "files": len(paths),
        "positive_scenes": positive_scenes,
        "object_hours": object_hours,
        "robust_object_hours": robust_object_hours,
        "tracks": len(tracks),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emma-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    years = sorted(path for path in args.emma_root.iterdir() if path.is_dir())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(summarize_year, map(str, years)))
    frame = pd.DataFrame(rows).sort_values("year")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    print("\nTotals")
    print(frame[["files", "positive_scenes", "object_hours", "robust_object_hours", "tracks"]].sum())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
