#!/usr/bin/env python3
"""Select thirty reproducible, month-balanced EMMA/GPM presentation scenes."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from emma_gpm.config import load_config, resolve_project_path
from extract_raw_dpr_cases import pf_url_for_orbit


def evenly_spaced(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if len(frame) <= count:
        return frame.copy()
    positions = np.unique(np.rint(np.linspace(0, len(frame) - 1, count)).astype(int))
    return frame.iloc[positions].copy()


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    matches = pd.read_parquet(ROOT / "catalogs/gpm_to_emma_pf_2020.parquet")
    inventory = pd.read_parquet(interim / "cmr_gpm_v07_mjjas2020.parquet")
    objects = pd.read_parquet(interim / "emma_hourly_objects_2020.parquet")
    matches["observation_time"] = pd.to_datetime(matches.observation_time, utc=True)
    matches["emma_time"] = pd.to_datetime(matches.emma_time, utc=True)
    object_times = set(pd.to_datetime(objects.time, utc=True))
    dpr = matches[
        matches.sensor.eq("DPR")
        & matches.within_time_tolerance
        & matches.emma_coverage
        & matches.emma_time.isin(object_times)
    ].copy()
    scenes = (
        dpr.groupby(["orbit", "emma_time"], as_index=False)
        .agg(
            observation_time=("observation_time", "min"),
            max_emma_overlap_cells=("n_overlapped_emma_grid_centres", "max"),
            largest_dpr_pf_pixels=("npixels", "max"),
        )
    )
    scenes["month"] = scenes.emma_time.dt.month
    selected: list[pd.DataFrame] = []
    for month in range(5, 10):
        monthly = scenes[scenes.month.eq(month)].copy()
        positive = monthly[monthly.max_emma_overlap_cells.gt(0)].sort_values(
            ["max_emma_overlap_cells", "emma_time"]
        )
        chosen = evenly_spaced(positive, min(6, len(positive)))
        needed = 6 - len(chosen)
        if needed:
            contextual = monthly[
                monthly.max_emma_overlap_cells.eq(0) & ~monthly.orbit.isin(chosen.orbit)
            ].sort_values(["largest_dpr_pf_pixels", "emma_time"], ascending=[False, True])
            chosen = pd.concat([chosen, evenly_spaced(contextual, needed)], ignore_index=True)
        chosen["selection_kind"] = np.where(
            chosen.max_emma_overlap_cells.gt(0),
            "positive_pixel_overlap",
            "context_scene_with_emma_elsewhere_in_domain",
        )
        selected.append(chosen)
    result = pd.concat(selected, ignore_index=True).sort_values("emma_time").reset_index(drop=True)
    if len(result) != 30 or result[["orbit", "emma_time"]].duplicated().any():
        raise RuntimeError("Presentation selection must contain exactly 30 unique scenes")
    urls = {
        int(orbit): pf_url_for_orbit(int(orbit), inventory)[0]
        for orbit in result.orbit.unique()
    }
    result["raw_pf_url"] = result.orbit.astype(int).map(urls)
    result["scene_id"] = [f"scene_{index:02d}" for index in range(1, len(result) + 1)]
    result.to_csv(ROOT / "catalogs/presentation_scene_manifest_2020.csv", index=False)
    print(
        result.groupby(["month", "selection_kind"]).size().rename("scenes").to_string()
    )
    print("\n", result[["scene_id", "emma_time", "orbit", "selection_kind"]].to_string(index=False))


if __name__ == "__main__":
    main()
