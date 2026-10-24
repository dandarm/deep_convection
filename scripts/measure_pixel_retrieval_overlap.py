#!/usr/bin/env python3
"""Measure EMMA overlap for the selected pixel-level GMI/DPR scenes."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from emma_gpm.config import load_config, resolve_project_path
from emma_gpm.emma import emma_files, load_emma_mask, parse_emma_time, read_grid
from plot_orbital_pixel_feature_maps import TIME_TOLERANCE, nearest_index


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    pixels = pd.read_parquet(interim / "orbital_pixel_feature_masks_2020.parquet")
    pixels["observation_time"] = pd.to_datetime(pixels.observation_time, utc=True)
    scenes = pd.read_csv(ROOT / "catalogs/presentation_scene_manifest_2020.csv")
    scenes["emma_time"] = pd.to_datetime(scenes.emma_time, utc=True)

    paths = emma_files(resolve_project_path(ROOT, config["emma"]["extracted_root"]))
    path_by_time = {parse_emma_time(path): path for path in paths}
    grid = read_grid(paths[0], config["domain"])
    rows: list[dict] = []

    for scene in scenes.itertuples(index=False):
        emma, _ = load_emma_mask(path_by_time[scene.emma_time], grid)
        emma_union = emma > 0
        for sensor in ("GMI", "DPR"):
            frame = pixels[
                pixels.orbit.eq(int(scene.orbit))
                & pixels.sensor.eq(sensor)
                & (pixels.observation_time.sub(scene.emma_time).abs() <= TIME_TOLERANCE)
            ].copy()
            iy = nearest_index(frame.lat.to_numpy(), grid.pilot_lat)
            ix = nearest_index(frame.lon.to_numpy(), grid.pilot_lon)
            frame["overlaps_emma"] = emma_union[iy, ix]
            cell_codes = iy.astype(np.int64) * len(grid.pilot_lon) + ix
            precip_cells = np.unique(cell_codes)
            overlap_cells = np.unique(cell_codes[frame.overlaps_emma.to_numpy()])
            component_overlap = frame.groupby("component_uid").overlaps_emma.any()
            touched_ids = np.unique(emma[iy[frame.overlaps_emma], ix[frame.overlaps_emma]])
            touched_ids = touched_ids[touched_ids > 0]
            rows.append(
                {
                    "scene_id": scene.scene_id,
                    "orbit": int(scene.orbit),
                    "emma_time": scene.emma_time.isoformat(),
                    "sensor": sensor,
                    "precipitating_native_pixels": len(frame),
                    "native_pixels_inside_emma": int(frame.overlaps_emma.sum()),
                    "native_pixel_overlap_fraction": float(frame.overlaps_emma.mean()),
                    "precipitating_emma_grid_cells": len(precip_cells),
                    "intersection_emma_grid_cells": len(overlap_cells),
                    "precip_cell_overlap_fraction": len(overlap_cells) / len(precip_cells),
                    "emma_grid_cells": int(emma_union.sum()),
                    "emma_fraction_covered_by_precip_cells": len(overlap_cells) / int(emma_union.sum()),
                    "precip_components": int(component_overlap.size),
                    "components_overlapping_emma": int(component_overlap.sum()),
                    "emma_objects_touched": len(touched_ids),
                }
            )

    detail = pd.DataFrame(rows)
    detail.to_csv(ROOT / "results/pixel_retrieval_overlap_selected_scenes.csv", index=False)
    summary = (
        detail.groupby("sensor", as_index=False)
        .agg(
            scenes=("scene_id", "size"),
            scenes_with_overlap=("intersection_emma_grid_cells", lambda s: int((s > 0).sum())),
            precipitating_native_pixels=("precipitating_native_pixels", "sum"),
            native_pixels_inside_emma=("native_pixels_inside_emma", "sum"),
            precipitating_emma_grid_cells=("precipitating_emma_grid_cells", "sum"),
            intersection_emma_grid_cells=("intersection_emma_grid_cells", "sum"),
            emma_grid_cells=("emma_grid_cells", "sum"),
            precip_components=("precip_components", "sum"),
            components_overlapping_emma=("components_overlapping_emma", "sum"),
        )
    )
    summary["native_pixel_overlap_fraction"] = (
        summary.native_pixels_inside_emma / summary.precipitating_native_pixels
    )
    summary["precip_cell_overlap_fraction"] = (
        summary.intersection_emma_grid_cells / summary.precipitating_emma_grid_cells
    )
    summary["emma_fraction_covered_by_precip_cells"] = (
        summary.intersection_emma_grid_cells / summary.emma_grid_cells
    )
    summary["component_overlap_fraction"] = (
        summary.components_overlapping_emma / summary.precip_components
    )
    summary.to_csv(ROOT / "results/pixel_retrieval_overlap_selected_summary.csv", index=False)
    print(summary.to_string(index=False))

    # Linear warm-season rate extrapolation.  This is an order-of-magnitude
    # projection, not a climatological annual estimate, because MJJAS is not a
    # seasonally representative sample of the full year.
    objects = pd.read_parquet(interim / "emma_hourly_objects_2020.parquet")
    matches = pd.read_parquet(ROOT / "catalogs/gpm_to_emma_pf_2020.parquet")
    lifecycle = pd.read_parquet(ROOT / "catalogs/emma_to_gpm_pf_2020.parquet")
    annual_factor = 366 / 153
    estimates = [
        ("EMMA", "distinct_mcs_tracks", lifecycle.track_uid.nunique()),
        ("EMMA", "hourly_object_masks", len(objects)),
        ("EMMA", "hourly_scenes_with_at_least_one_object", objects.time.nunique()),
    ]
    for sensor in ("GMI", "DPR"):
        sensor_rows = matches[matches.sensor.eq(sensor)]
        estimates.extend(
            [
                (sensor, "pf_object_samples_all_in_domain", len(sensor_rows)),
                (
                    sensor,
                    "pf_object_samples_overlapping_emma",
                    int(sensor_rows.n_overlapped_emma_grid_centres.gt(0).sum()),
                ),
                (sensor, "orbits_with_pf_in_domain", sensor_rows.orbit.nunique()),
            ]
        )
    annual = pd.DataFrame(estimates, columns=["source", "sample_unit", "observed_mjjas_2020"])
    annual["linear_annual_order_of_magnitude"] = (
        annual.observed_mjjas_2020 * annual_factor
    ).round().astype(int)
    annual["annualization_factor"] = annual_factor
    annual["caveat"] = (
        "Linear 366/153 projection of MJJAS rate; warm-season selection makes this non-climatological"
    )
    annual.to_csv(ROOT / "results/training_sample_order_of_magnitude.csv", index=False)


if __name__ == "__main__":
    main()
