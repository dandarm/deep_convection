#!/usr/bin/env python3
"""Select 30 diverse EMMA object-hours for visual inspection."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd


def iso_z(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seviri-lag-minutes", type=int, default=-30)
    args = parser.parse_args()

    objects = pd.read_parquet(args.objects).copy()
    objects["time"] = pd.to_datetime(objects.time, utc=True)
    objects["month"] = objects.time.dt.month
    # Global size bins make examples comparable between months.  Two distinct
    # tracks per (month, size) cell gives 5 × 3 × 2 = 30 visual cases.
    objects["size_bin"] = pd.qcut(
        objects.n_emma_grid_cells,
        q=3,
        labels=["small", "medium", "large"],
        duplicates="drop",
    )
    selected_rows: list[pd.Series] = []
    used_tracks: set[str] = set()
    for month in sorted(objects.month.unique()):
        for size_bin in ("small", "medium", "large"):
            candidates = objects[
                (objects.month == month) & (objects.size_bin.astype(str) == size_bin)
            ].sort_values(
                ["official_robust_flag", "n_emma_grid_cells"], ascending=[False, False]
            )
            chosen = 0
            for _, row in candidates.iterrows():
                if row.track_uid in used_tracks:
                    continue
                selected_rows.append(row)
                used_tracks.add(str(row.track_uid))
                chosen += 1
                if chosen == 2:
                    break
            if chosen != 2:
                raise RuntimeError(f"Could not select two distinct tracks for month={month} size={size_bin}")

    selected = pd.DataFrame(selected_rows).sort_values("time").reset_index(drop=True)
    selected["target_time"] = selected.time.map(iso_z)
    selected["seviri_time"] = selected.time.map(
        lambda value: iso_z(pd.Timestamp(value).to_pydatetime() + timedelta(minutes=args.seviri_lag_minutes))
    )
    selected["slot_start"] = selected.seviri_time
    selected["slot_end"] = selected.seviri_time.map(
        lambda value: iso_z(pd.Timestamp(value).to_pydatetime() + timedelta(minutes=5))
    )
    columns = [
        "target_time", "seviri_time", "slot_start", "slot_end", "track_uid", "mcs_id",
        "month", "size_bin", "n_emma_grid_cells", "official_robust_flag",
        "centroid_lat_gridmean", "centroid_lon_gridmean", "lat_min", "lat_max", "lon_min", "lon_max",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    selected[columns].to_csv(args.out, index=False)
    print(f"examples={len(selected)} distinct_tracks={selected.track_uid.nunique()} path={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
