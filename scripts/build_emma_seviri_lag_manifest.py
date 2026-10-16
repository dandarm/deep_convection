#!/usr/bin/env python3
"""Select strong EMMA hours for a reproducible SEVIRI temporal-lag audit."""

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd


def iso_z(value: pd.Timestamp) -> str:
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
    parser.add_argument("--per-month", type=int, default=2)
    parser.add_argument("--lag-minutes", type=int, default=-30)
    parser.add_argument("--lag-max-minutes", type=int, default=60)
    args = parser.parse_args()
    if args.per_month < 1:
        raise SystemExit("--per-month must be positive")
    if args.lag_minutes > args.lag_max_minutes:
        raise SystemExit("--lag-minutes must not exceed --lag-max-minutes")

    objects = pd.read_parquet(args.objects)
    objects["time"] = pd.to_datetime(objects.time, utc=True)
    robust = objects[objects.official_robust_flag].copy()
    robust["month"] = robust.time.dt.month
    # One high-signal hour per track, then distinct tracks within each month.
    peaks = robust.loc[
        robust.groupby("track_uid").n_emma_grid_cells.idxmax()
    ].copy()
    selected = (
        peaks.sort_values(["month", "n_emma_grid_cells"], ascending=[True, False])
        .groupby("month", group_keys=False)
        .head(args.per_month)
        .sort_values("time")
    )
    result = selected[
        ["time", "track_uid", "mcs_id", "n_emma_grid_cells", "centroid_lat_gridmean"]
    ].rename(columns={"time": "target_time"})
    result["slot_start"] = result.target_time.map(
        lambda value: iso_z(pd.Timestamp(value).to_pydatetime() + timedelta(minutes=args.lag_minutes))
    )
    # Downloader end is exclusive; include the maximum lag itself.
    result["slot_end"] = result.target_time.map(
        lambda value: iso_z(
            pd.Timestamp(value).to_pydatetime() + timedelta(minutes=args.lag_max_minutes + 5)
        )
    )
    result["target_time"] = result.target_time.map(iso_z)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"selected_hours={len(result)} tracks={result.track_uid.nunique()} path={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
