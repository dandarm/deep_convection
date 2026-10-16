#!/usr/bin/env python3
"""Compute continuous EMMA–PF overlap metrics for GMI and DPR equally."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path
from emma_gpm.emma import emma_files, load_emma_mask, parse_emma_time, read_grid
from emma_gpm.matching import _ellipse_mask_indices


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    matches = pd.read_parquet(ROOT / "catalogs/gpm_to_emma_pf_2020.parquet")
    objects = pd.read_parquet(interim / "emma_hourly_objects_2020.parquet")
    paths = emma_files(resolve_project_path(ROOT, config["emma"]["extracted_root"]))
    path_by_time = {parse_emma_time(path): path for path in paths}
    grid = read_grid(paths[0], config["domain"])
    object_cells = objects.set_index(["time", "mcs_id"])["n_emma_grid_cells"]

    # A percentage against EMMA is meaningful only where the feature centre is
    # inside the effective EMMA grid. Rows outside that coverage remain in the
    # main GPM catalog but are excluded from this geometric percentage table.
    frame = matches[matches.emma_coverage].copy()
    ellipse_counts: list[int] = []
    target_intersections: list[float] = []
    target_cell_counts: list[float] = []
    mask_cache: dict[pd.Timestamp, np.ndarray] = {}

    for row in frame.itertuples():
        yy, xx = _ellipse_mask_indices(
            float(row.r_lat),
            float(row.r_lon),
            float(row.r_major),
            float(row.r_minor),
            float(row.r_orientation),
            grid.pilot_lat,
            grid.pilot_lon,
        )
        ellipse_counts.append(len(yy))
        if pd.notna(row.mcs_id) and row.mcs_id > 0 and row.emma_time in path_by_time:
            if row.emma_time not in mask_cache:
                mask_cache[row.emma_time] = load_emma_mask(path_by_time[row.emma_time], grid)[0]
            mcs_id = int(row.mcs_id)
            target_intersections.append(int((mask_cache[row.emma_time][yy, xx] == mcs_id).sum()))
            target_cell_counts.append(int(object_cells.loc[(row.emma_time, mcs_id)]))
        else:
            target_intersections.append(np.nan)
            target_cell_counts.append(np.nan)

    frame["pf_ellipse_grid_centres"] = ellipse_counts
    frame["target_emma_intersection_grid_centres"] = target_intersections
    frame["target_emma_grid_centres"] = target_cell_counts
    ellipse_denominator = frame.pf_ellipse_grid_centres.replace(0, np.nan)
    frame["any_emma_fraction_of_pf_ellipse"] = (
        frame.n_overlapped_emma_grid_centres / ellipse_denominator
    )
    frame["target_emma_fraction_of_pf_ellipse"] = (
        frame.target_emma_intersection_grid_centres / ellipse_denominator
    )
    frame["target_emma_fraction_covered_by_pf"] = (
        frame.target_emma_intersection_grid_centres / frame.target_emma_grid_centres
    )
    union = (
        frame.pf_ellipse_grid_centres
        + frame.target_emma_grid_centres
        - frame.target_emma_intersection_grid_centres
    )
    frame["target_emma_pf_iou"] = frame.target_emma_intersection_grid_centres / union
    frame["overlap_metric_note"] = (
        "PF ellipse and EMMA mask rasterized on EMMA grid-cell centres; fitted PF ellipse is approximate"
    )

    keep = [
        "feature_uid", "sensor", "orbit", "observation_time", "emma_time",
        "time_offset_minutes", "mcs_id", "track_uid", "emma_class",
        "r_lon", "r_lat", "r_major", "r_minor", "r_orientation",
        "pf_ellipse_grid_centres", "n_overlapped_emma_grid_centres",
        "target_emma_intersection_grid_centres", "target_emma_grid_centres",
        "any_emma_fraction_of_pf_ellipse", "target_emma_fraction_of_pf_ellipse",
        "target_emma_fraction_covered_by_pf", "target_emma_pf_iou",
        "overlap_metric_note",
    ]
    output = frame[keep]
    output.to_parquet(ROOT / "catalogs/gpm_emma_continuous_overlap_2020.parquet", index=False)

    summary_rows = []
    positive = output[output.target_emma_pf_iou.notna()]
    for sensor, subset in positive.groupby("sensor"):
        for metric in (
            "target_emma_fraction_of_pf_ellipse",
            "target_emma_fraction_covered_by_pf",
            "target_emma_pf_iou",
        ):
            values = subset[metric]
            for quantile in (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0):
                summary_rows.append({
                    "sensor": sensor,
                    "metric": metric,
                    "statistic": f"quantile_{quantile:.2f}",
                    "value": float(values.quantile(quantile)),
                    "feature_count": len(values),
                })
            for label, mask in {
                "0_lt_iou_le_0.05": (values > 0) & (values <= 0.05),
                "iou_ge_0.50": values >= 0.50,
                "iou_ge_0.90": values >= 0.90,
            }.items():
                if metric == "target_emma_pf_iou":
                    summary_rows.append({
                        "sensor": sensor,
                        "metric": metric,
                        "statistic": label,
                        "value": int(mask.sum()),
                        "feature_count": len(values),
                    })
    pd.DataFrame(summary_rows).to_csv(
        ROOT / "results/continuous_overlap_distribution.csv", index=False
    )
    print(f"features within EMMA coverage: {len(output)}")
    print(
        positive.groupby("sensor").target_emma_pf_iou.agg(["count", "min", "median", "max"])
    )


if __name__ == "__main__":
    main()
