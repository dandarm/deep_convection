#!/usr/bin/env python3
"""Measure radar quietness across the full 224x224 crop for local N1 objects."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from plot_emma_heatmap_examples import load_ir108, utc_naive  # noqa: E402
from validate_emma_negatives_opera import (  # noqa: E402
    CROP,
    OFFSETS_MINUTES,
    OPERA_CRS,
    download_opera,
    opera_indices_for_crop,
    read_opera,
)
from emma_gpm.seviri import RSS_CRS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--opera-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qind-threshold", type=float, default=0.2)
    args = parser.parse_args()

    frame = pd.read_csv(args.catalog)
    frame = frame[frame["class"].eq("N1_candidate")].copy()
    frame["central_time"] = pd.to_datetime(frame["central_time"])
    transformer = Transformer.from_crs(RSS_CRS, OPERA_CRS, always_xy=True)
    datasets = {}
    records = []
    for central_time, group in frame.groupby("central_time", sort=True):
        center = utc_naive(central_time)
        _, x, y = load_ir108(args.seviri_root, center, datasets)
        opera = [
            read_opera(download_opera(center + pd.Timedelta(offset, unit="min"), args.opera_cache))
            for offset in OFFSETS_MINUTES
        ]
        for item in group.itertuples(index=False):
            row, col, inside = opera_indices_for_crop(
                x, y, int(item.origin_x), int(item.origin_y), transformer
            )
            inside_fraction = float(inside.mean())
            valid_fractions = []
            usable_fractions = []
            maxima = []
            area35 = []
            area40 = []
            for dbzh, qind in opera:
                unique = np.unique(np.column_stack((row[inside], col[inside])), axis=0)
                z = dbzh[unique[:, 0], unique[:, 1]]
                q = qind[unique[:, 0], unique[:, 1]]
                valid = np.isfinite(z)
                no_echo = valid & np.isclose(z, -32.0)
                quality_echo = valid & np.isfinite(q) & (q >= args.qind_threshold)
                usable = no_echo | quality_echo
                valid_fractions.append(float(valid.mean()))
                usable_fractions.append(float(usable.mean()))
                maxima.append(float(np.nanmax(z[usable])) if usable.any() else np.nan)
                area35.append(float(np.count_nonzero(usable & (z >= 35.0)) * 4.0))
                area40.append(float(np.count_nonzero(usable & (z >= 40.0)) * 4.0))
            max_dbzh = float(np.nanmax(maxima)) if np.isfinite(maxima).any() else np.nan
            record = {
                "central_time": center,
                "origin_y": int(item.origin_y),
                "origin_x": int(item.origin_x),
                "shield_id": int(item.shield_id),
                "crop_inside_fraction": inside_fraction,
                "crop_valid_fraction_median": float(np.median(valid_fractions)),
                "crop_usable_fraction_median": float(np.median(usable_fractions)),
                "crop_max_dbzh_5slots": max_dbzh,
                "crop_max_area35_km2_5slots": float(max(area35)),
                "crop_max_area40_km2_5slots": float(max(area40)),
            }
            record["strict_cropwide_n1"] = bool(
                inside_fraction >= 0.80
                and record["crop_valid_fraction_median"] >= 0.70
                and record["crop_usable_fraction_median"] >= 0.05
                and max_dbzh < 30.0
                and record["crop_max_area35_km2_5slots"] == 0.0
            )
            records.append(record)
        print(f"{center:%Y-%m-%d %H:%M} rows={len(group)}", flush=True)
    for dataset in datasets.values():
        dataset.close()
    output = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.strict_cropwide_n1.value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
