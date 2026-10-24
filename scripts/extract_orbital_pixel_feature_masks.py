#!/usr/bin/env python3
"""Build resumable pixel-level GMI and DPR precipitation-feature masks.

The compact public Level-1 PF orbit containers expose full sensor arrays.  A
feature is reconstructed as an eight-connected component of pixels whose
retrieved precipitation rate is greater than 0.1 mm h-1.  Components are
labelled on the complete orbital array before clipping to the pilot domain, so
the domain boundary does not artificially split a feature.

Only precipitating pixel centres inside the pilot domain are checkpointed.  No
native NASA GPM granule is downloaded and no model training is performed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import fsspec
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path


PRECIPITATION_THRESHOLD = 0.1
SENSOR_FIELDS = {"GMI": "PRECIP", "DPR": "NSPRECIP"}


def scan_times(record: np.void) -> pd.DatetimeIndex:
    return pd.to_datetime(
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


def extract_sensor(
    record: np.void,
    sensor: str,
    orbit: int,
    url: str,
    domain: dict[str, float],
) -> pd.DataFrame:
    lat = np.asarray(record["LAT"], dtype=np.float32)
    lon = np.asarray(record["LON"], dtype=np.float32)
    rate = np.asarray(record[SENSOR_FIELDS[sensor]], dtype=np.float32)
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(rate) & (rate > PRECIPITATION_THRESHOLD)

    # Eight-neighbour connectivity is an explicit reconstruction choice.  The
    # upstream PF tables do not retain pixel membership for each monthly row.
    labels, n_components = ndimage.label(valid, structure=np.ones((3, 3), dtype=np.uint8))
    inside = valid & (
        (lat >= domain["lat_min"])
        & (lat <= domain["lat_max"])
        & (lon >= domain["lon_min"])
        & (lon <= domain["lon_max"])
    )
    scan, pixel = np.where(inside)
    times = scan_times(record)
    component = labels[scan, pixel].astype(np.int32)
    frame = pd.DataFrame(
        {
            "orbit": np.int32(orbit),
            "sensor": sensor,
            "scan": scan.astype(np.int32),
            "pixel": pixel.astype(np.int16),
            "observation_time": times[scan].to_numpy(),
            "lat": lat[scan, pixel],
            "lon": lon[scan, pixel],
            "precipitation_rate_mm_h": rate[scan, pixel],
            "component_id": component,
        }
    )
    frame["component_uid"] = (
        sensor + ":" + f"{orbit:06d}" + ":" + frame.component_id.astype(str)
    )
    frame["source_url"] = url
    frame["mask_definition"] = (
        f"{SENSOR_FIELDS[sensor]}>{PRECIPITATION_THRESHOLD:g} mm h-1; "
        "8-connected on native swath; pixel centres clipped to pilot domain"
    )
    frame["n_components_full_orbit"] = np.int32(n_components)
    return frame


def extract_orbit(
    orbit: int,
    url: str,
    domain: dict[str, float],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    time_fields = ["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"]
    with fsspec.open(url, "rb", block_size=8 * 1024 * 1024, cache_type="readahead") as stream:
        with h5py.File(stream, "r") as handle:
            for sensor, rate_field in SENSOR_FIELDS.items():
                fields = ["LAT", "LON", rate_field, *time_fields]
                record = handle[sensor].fields(fields)[0]
                frames.append(extract_sensor(record, sensor, orbit, url, domain))
    return pd.concat(frames, ignore_index=True)


def save_parquet_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    frame.to_parquet(partial, index=False)
    partial.replace(destination)


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    manifest_path = ROOT / "catalogs" / "presentation_scene_manifest_2020.csv"
    if not manifest_path.exists():
        manifest_path = ROOT / "catalogs" / "visual_case_manifest_2020.csv"
    manifest = pd.read_csv(manifest_path)
    jobs = (
        manifest[["orbit", "raw_pf_url"]]
        .dropna()
        .drop_duplicates()
        .assign(orbit=lambda table: table.orbit.astype(int))
        .sort_values("orbit")
    )
    checkpoint_dir = interim / "orbital_pixel_feature_masks"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    pending: list[tuple[int, str, Path]] = []
    for row in jobs.itertuples(index=False):
        checkpoint = checkpoint_dir / f"orbit_{row.orbit:06d}.parquet"
        if checkpoint.exists():
            frames.append(pd.read_parquet(checkpoint))
            print(f"orbit {row.orbit}: restored {checkpoint}", flush=True)
        else:
            pending.append((row.orbit, row.raw_pf_url, checkpoint))

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(extract_orbit, orbit, url, config["domain"]): (orbit, checkpoint)
            for orbit, url, checkpoint in pending
        }
        for future in as_completed(futures):
            orbit, checkpoint = futures[future]
            frame = future.result()
            save_parquet_atomic(frame, checkpoint)
            frames.append(frame)
            counts = frame.groupby("sensor").size().to_dict()
            print(f"orbit {orbit}: checkpointed precipitating pixels {counts}", flush=True)

    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["orbit", "sensor", "scan", "pixel"]
    )
    destination = interim / "orbital_pixel_feature_masks_2020.parquet"
    save_parquet_atomic(combined, destination)
    summary = (
        combined.groupby(["orbit", "sensor"], as_index=False)
        .agg(
            precipitating_pixels=("component_uid", "size"),
            components_in_domain=("component_uid", "nunique"),
            start_time=("observation_time", "min"),
            end_time=("observation_time", "max"),
        )
        .sort_values(["orbit", "sensor"])
    )
    summary.to_csv(ROOT / "results/orbital_pixel_feature_mask_inventory.csv", index=False)
    print(f"wrote {len(combined):,} rows to {destination}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
