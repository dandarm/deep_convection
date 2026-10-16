#!/usr/bin/env python3
"""Create a 15-minute MSG/SEVIRI request manifest for DeepStorm footprints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/deepstorm_probe.json")
    args = parser.parse_args()
    config = load_config(args.config)
    catalog = resolve_project_path(ROOT, config["outputs"]["catalog"])
    output = catalog.parent / "msg_seviri_request_manifest.csv"
    table = pd.read_parquet(catalog)
    summary = (
        table.groupby("source_file", as_index=False)
        .agg(
            observation_time=("observation_time", "median"),
            lat_min=("latitude", "min"), lat_max=("latitude", "max"),
            lon_min=("longitude", "min"), lon_max=("longitude", "max"),
            deep_convection_points=("target_dc_any", "sum"),
            iwp_positive_points=("target_iwp_gt_0p5", "sum"),
        )
        .sort_values("observation_time")
    )
    slot_start = pd.to_datetime(summary["observation_time"], utc=True).dt.floor("15min")
    summary["slot_start"] = slot_start.dt.strftime("%Y-%m-%dT%H:%M:%S")
    summary["slot_end"] = (slot_start + pd.DateOffset(minutes=15)).dt.strftime("%Y-%m-%dT%H:%M:%S")
    summary.to_csv(output, index=False)
    print(f"output={output}")
    print(f"source_files={len(summary)} unique_msg_slots={summary[['slot_start', 'slot_end']].drop_duplicates().shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
