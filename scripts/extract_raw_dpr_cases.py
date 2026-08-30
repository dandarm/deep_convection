#!/usr/bin/env python3
"""Extract actual DPR pixel centres for ten reproducible visual-QA cases.

The source is the public orbital Level-1 PF archive (V07A). Only the compact DPR
compound dataset is read via HTTP byte ranges; full ~300 MB orbit files are not
downloaded. Pixel centres are retained as the explicit footprint approximation.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import sys

import fsspec
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path


def nearest_object(row: pd.Series, objects: pd.DataFrame) -> tuple[str, float]:
    same_time = objects[objects.time.eq(row.emma_time)]
    if same_time.empty:
        return "", np.inf
    dlat = np.maximum.reduce([
        same_time.lat_min.to_numpy() - row.lat,
        row.lat - same_time.lat_max.to_numpy(),
        np.zeros(len(same_time)),
    ])
    dlon = np.maximum.reduce([
        same_time.lon_min.to_numpy() - row.lon,
        row.lon - same_time.lon_max.to_numpy(),
        np.zeros(len(same_time)),
    ])
    distance = np.hypot(110.574 * dlat, 111.320 * np.cos(np.deg2rad(row.lat)) * dlon)
    index = int(np.argmin(distance))
    return str(same_time.iloc[index].track_uid), float(distance[index])


def choose_cases(matches: pd.DataFrame, objects: pd.DataFrame) -> pd.DataFrame:
    dpr = matches[matches.sensor.eq("DPR")].copy()
    dpr["month_number"] = dpr.observation_time.dt.month
    hits = (
        dpr[dpr.mcs_id.fillna(0).gt(0)]
        .sort_values("n_overlapped_emma_grid_centres", ascending=False)
        .drop_duplicates(["month_number"])
        .head(5)
        .copy()
    )
    hits["case_kind"] = "hit"
    hits["target_track_uid"] = hits.track_uid
    hits["feature_to_target_bbox_km"] = 0.0

    miss_candidates = dpr[
        dpr.mcs_id.fillna(0).eq(0)
        & dpr.emma_coverage
        & dpr.emma_time.isin(objects.time.unique())
    ].copy()
    nearest = miss_candidates.apply(lambda row: nearest_object(row, objects), axis=1)
    miss_candidates["target_track_uid"] = [item[0] for item in nearest]
    miss_candidates["feature_to_target_bbox_km"] = [item[1] for item in nearest]
    misses = (
        miss_candidates.sort_values(
            ["month_number", "feature_to_target_bbox_km", "npixels"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["month_number"])
        .head(5)
        .copy()
    )
    misses["case_kind"] = "miss"
    cases = pd.concat([hits, misses], ignore_index=True)
    cases["case_id"] = [f"hit_{i+1:02d}" for i in range(len(hits))] + [
        f"miss_{i+1:02d}" for i in range(len(misses))
    ]
    return cases


def pf_url_for_orbit(orbit: int, inventory: pd.DataFrame) -> tuple[str, str]:
    pattern = f".{orbit:06d}.V07A.HDF5"
    selected = inventory[
        inventory.collection.eq("DPR_2A")
        & inventory.producer_granule_id.str.contains(pattern, regex=False)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one CMR DPR granule for orbit {orbit}, found {len(selected)}")
    producer = selected.iloc[0].producer_granule_id
    match = re.search(r"(\d{8}-S\d{6}-E\d{6}\.\d{6}\.V07A\.HDF5)$", producer)
    if not match:
        raise ValueError(f"Cannot translate CMR filename to PF filename: {producer}")
    suffix = match.group(1)
    year_month = suffix[:6]
    name = f"1Z.GPM.PF.{suffix}"
    return f"https://atmos.tamucc.edu/trmm/data/gpm/level_1/{year_month}/{name}", name


def extract_orbit(orbit: int, url: str, domain: dict[str, float]) -> pd.DataFrame:
    fields = [
        "LAT", "LON", "PRECIPTYPE", "STORMHT", "ESPRECIP", "NSPRECIP",
        "NSZ", "DAY", "HOUR", "MINUTE", "MONTH", "SECOND", "YEAR",
    ]
    with fsspec.open(url, "rb", block_size=8 * 1024 * 1024, cache_type="readahead") as stream:
        with h5py.File(stream, "r") as handle:
            record = handle["DPR"].fields(fields)[0]
    lat = record["LAT"]
    lon = record["LON"]
    keep = (
        (lat >= domain["lat_min"]) & (lat <= domain["lat_max"])
        & (lon >= domain["lon_min"]) & (lon <= domain["lon_max"])
    )
    scan, ray = np.where(keep)
    scan_times = pd.to_datetime(
        {
            "year": record["YEAR"].astype(int),
            "month": record["MONTH"].astype(int),
            "day": record["DAY"].astype(int),
            "hour": record["HOUR"].astype(int),
            "minute": record["MINUTE"].astype(int),
            "second": record["SECOND"].astype(int),
        },
        utc=True,
        errors="coerce",
    )
    precip_type_raw = record["PRECIPTYPE"][scan, ray].astype(np.int64)
    major = np.where(precip_type_raw > 0, precip_type_raw // 10_000_000, 0)
    label = np.select(
        [major == 1, major == 2, major == 3],
        ["stratiform", "convective", "other"],
        default="no_precip_or_missing",
    )
    return pd.DataFrame(
        {
            "orbit": orbit,
            "scan": scan,
            "ray": ray,
            "observation_time": scan_times[scan].to_numpy(),
            "lat": lat[scan, ray],
            "lon": lon[scan, ray],
            "type_precip_raw": precip_type_raw,
            "type_precip_major": major,
            "type_precip_label": label,
            "storm_height": record["STORMHT"][scan, ray],
            "estimated_surface_precip": record["ESPRECIP"][scan, ray],
            "near_surface_precip": record["NSPRECIP"][scan, ray],
            # The public PF V07A orbital container stores two NSZ components in
            # its DPR compound field.  The database manual does not define the
            # order explicitly, so retain both without assigning band names.
            "near_surface_reflectivity_component_0": record["NSZ"][scan, ray, 0],
            "near_surface_reflectivity_component_1": record["NSZ"][scan, ray, 1],
            "source_url": url,
            "footprint_representation": "pixel_center",
        }
    )


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    matches = pd.read_parquet(ROOT / "catalogs/gpm_to_emma_pf_2020.parquet")
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    objects = pd.read_parquet(interim / "emma_hourly_objects_2020.parquet")
    inventory = pd.read_parquet(interim / "cmr_gpm_v07_mjjas2020.parquet")
    cases = choose_cases(matches, objects)
    orbit_jobs: dict[int, tuple[str, str]] = {}
    for orbit in cases.orbit.astype(int).unique():
        orbit_jobs[orbit] = pf_url_for_orbit(orbit, inventory)
    checkpoint_dir = interim / "raw_dpr_case_orbits"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    pending: dict[int, tuple[str, str]] = {}
    for orbit, job in orbit_jobs.items():
        checkpoint = checkpoint_dir / f"orbit_{orbit:06d}.parquet"
        if checkpoint.exists():
            frames.append(pd.read_parquet(checkpoint))
            print(f"orbit {orbit}: restored checkpoint", flush=True)
        else:
            pending[orbit] = job
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(extract_orbit, orbit, url, config["domain"]): orbit
            for orbit, (url, _) in pending.items()
        }
        for future in as_completed(futures):
            orbit = futures[future]
            frame = future.result()
            frame.to_parquet(checkpoint_dir / f"orbit_{orbit:06d}.parquet", index=False)
            frames.append(frame)
            print(f"orbit {orbit}: {len(frame)} DPR pixel centres in pilot domain", flush=True)
    pixels = pd.concat(frames, ignore_index=True)
    pixels.to_parquet(interim / "raw_dpr_visual_case_pixels.parquet", index=False)
    cases["raw_pf_url"] = cases.orbit.astype(int).map(lambda value: orbit_jobs[value][0])
    cases.to_csv(ROOT / "catalogs/visual_case_manifest_2020.csv", index=False)
    print(cases[["case_id", "case_kind", "orbit", "observation_time", "target_track_uid"]].to_string(index=False))


if __name__ == "__main__":
    main()
