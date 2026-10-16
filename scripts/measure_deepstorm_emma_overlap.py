#!/usr/bin/env python3
"""Measure point-sampled agreement between DeepStorm-like targets and EMMA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path
from emma_gpm.emma import load_emma_mask, read_grid


def emma_path(root: Path, time: pd.Timestamp) -> Path:
    hour = time.round("h")
    return root / f"{hour:%m}" / f"tracking_{hour:%Y%m%dT%H}.nc"


def nearest_indices(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.abs(grid[:, None] - values[None, :]).argmin(axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/deepstorm_probe.json")
    parser.add_argument("--pilot-config", default="config/pilot_2020.json")
    args = parser.parse_args()
    data_config = load_config(args.config)
    pilot_config = load_config(args.pilot_config)
    catalog = resolve_project_path(ROOT, data_config["outputs"]["catalog"])
    targets = pd.read_parquet(catalog)
    emma_root = resolve_project_path(ROOT, pilot_config["emma"]["extracted_root"])
    sample = next(emma_root.rglob("tracking_*.nc"))
    grid = read_grid(sample, pilot_config["domain"])
    rows: list[dict[str, object]] = []
    for source_file, frame in targets.groupby("source_file", sort=True):
        time = pd.Timestamp(frame["observation_time"].median())
        path = emma_path(emma_root, time)
        if not path.exists():
            rows.append({"source_file": source_file, "status": "missing_emma"})
            continue
        ids, _ = load_emma_mask(path, grid)
        lat_index = nearest_indices(grid.pilot_lat, frame["latitude"].to_numpy())
        lon_index = nearest_indices(grid.pilot_lon, frame["longitude"].to_numpy())
        in_emma = ids[lat_index, lon_index] > 0
        dc = frame["target_dc_any"].fillna(0).astype(bool).to_numpy()
        iwp = frame["target_iwp_gt_0p5"].fillna(0).astype(bool).to_numpy()
        emma_time = pd.Timestamp(path.stem.removeprefix("tracking_"), tz="UTC")
        rows.append(
            {
                "source_file": source_file,
                "status": "ok",
                "observation_time": time,
                "emma_time": emma_time,
                "time_offset_minutes": abs((time - emma_time).total_seconds()) / 60,
                "target_rows": len(frame),
                "dc_positive": int(dc.sum()),
                "iwp_positive": int(iwp.sum()),
                "target_points_in_emma": int(in_emma.sum()),
                "dc_points_in_emma": int((dc & in_emma).sum()),
                "iwp_points_in_emma": int((iwp & in_emma).sum()),
                "target_fraction_in_emma": float(in_emma.mean()),
                "dc_fraction_in_emma": float((dc & in_emma).sum() / max(1, dc.sum())),
                "iwp_fraction_in_emma": float((iwp & in_emma).sum() / max(1, iwp.sum())),
            }
        )
    output = ROOT / "results" / "deepstorm_emma_point_overlap.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"output={output}")
    print(pd.DataFrame(rows)["status"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
