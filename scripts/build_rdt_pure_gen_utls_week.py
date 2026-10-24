#!/usr/bin/env python3
"""Build improved pure-GEN/UTLS labels for the complete September RSS week.

The source ``RDT_original`` tree is imported read-only.  Products are written
under ``results/`` and can be resumed at day boundaries from an atomic tracker
checkpoint.  Actual Zarr timestamps are sorted and validated; no slot is
invented, interpolated or duplicated.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
from numcodecs import Blosc
from pyproj import CRS, Transformer
import xarray as xr
import zarr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.rdt.pure_gen_utls import (  # noqa: E402
    RDT_ORIGINAL_CLASS_NAMES,
    PureGenUtlsTracker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"E:\Datasets\MSG-SEVIRI\2020"),
    )
    parser.add_argument("--first-day", default="20200914")
    parser.add_argument("--last-day", default="20200920")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "rdt_pure_gen_utls_week_20200914_20200920",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional smoke-test limit. A limited run is never marked complete.",
    )
    return parser.parse_args()


def _orient(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = values
    x_out = x.copy()
    y_out = y.copy()
    if y_out[0] < y_out[-1]:
        output = output[..., ::-1, :]
        y_out = y_out[::-1]
    if x_out[0] > x_out[-1]:
        output = output[..., :, ::-1]
        x_out = x_out[::-1]
    return np.ascontiguousarray(output), x_out, y_out


def _geometry(zarr_path: Path) -> dict[str, np.ndarray | str]:
    with xr.open_zarr(zarr_path, consolidated=True, chunks=None) as dataset:
        x_raw = dataset.x.values.copy()
        y_raw = dataset.y.values.copy()
        projection_wkt = str(dataset.attrs["projection_wkt"])
    dummy = np.zeros((1, len(y_raw), len(x_raw)), dtype=np.uint8)
    _, x, y = _orient(dummy, x_raw, y_raw)
    xx, yy = np.meshgrid(x, y)
    transformer = Transformer.from_crs(
        CRS.from_wkt(projection_wkt), CRS.from_epsg(4326), always_xy=True
    )
    longitude, latitude = transformer.transform(xx, yy)
    longitude = np.asarray(longitude, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)
    valid = np.isfinite(longitude) & np.isfinite(latitude)

    with np.errstate(invalid="ignore"):
        lat_rad = np.radians(latitude)
        lon_rad = np.radians(longitude)
        dlat_dr, dlat_dc = np.gradient(lat_rad)
        dlon_dr, dlon_dc = np.gradient(lon_rad)
        semi_major_km = 6378.137
        eccentricity_sq = 0.00669437999014
        denominator = np.sqrt(
            np.maximum(1.0 - eccentricity_sq * np.sin(lat_rad) ** 2, 1e-6)
        )
        meridional = semi_major_km * (1.0 - eccentricity_sq) / denominator**3
        prime_vertical = semi_major_km / denominator
        jacobian = np.abs(dlat_dr * dlon_dc - dlat_dc * dlon_dr)
        area = meridional * prime_vertical * np.cos(lat_rad) * jacobian
    area = np.where(valid & np.isfinite(area) & (area > 0.0) & (area < 250.0), area, np.nan)
    return {
        "source_x": x_raw.astype(np.float64),
        "source_y": y_raw.astype(np.float64),
        "x": x.astype(np.float64),
        "y": y.astype(np.float64),
        "longitude": longitude.astype(np.float32),
        "latitude": latitude.astype(np.float32),
        "area_km2": area.astype(np.float32),
        "valid_geolocation": valid,
        "projection_wkt": projection_wkt,
    }


def _day_inventory(paths: list[Path]) -> list[dict]:
    inventory = []
    previous_last: pd.Timestamp | None = None
    for path in paths:
        with xr.open_zarr(path, consolidated=True, chunks=None) as dataset:
            times = pd.DatetimeIndex(dataset.time.values)
        if times.hasnans or times.has_duplicates:
            raise ValueError(f"Invalid or duplicate timestamps in {path}")
        order = np.argsort(times.asi8)
        ordered = times[order]
        if len(ordered) != 288:
            raise ValueError(f"Expected 288 RSS observations in {path}, found {len(ordered)}")
        expected_delta = 5 * 60 * 1_000_000_000
        if not np.all(np.diff(ordered.asi8) == expected_delta):
            raise ValueError(f"Non-5-minute cadence in {path}")
        if previous_last is not None and ordered[0] - previous_last != pd.to_timedelta(5, unit="m"):
            raise ValueError(f"Gap between {previous_last} and {ordered[0]}")
        previous_last = ordered[-1]
        inventory.append(
            {
                "path": path,
                "times": ordered,
                "indices": order.astype(np.int64),
            }
        )
    return inventory


def _atomic_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_pickle(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with gzip.open(temporary, "wb", compresslevel=1) as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _open_output(
    output_root: Path,
    total_frames: int,
    height: int,
    width: int,
    metadata: dict,
) -> zarr.Group:
    output_root.mkdir(parents=True, exist_ok=True)
    store_path = output_root / "labels.zarr"
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    if store_path.exists():
        group = zarr.open_group(str(store_path), mode="r+")
        if tuple(group["semantic_label"].shape) != (total_frames, height, width):
            raise ValueError("Existing output shape does not match the requested week")
        if group.attrs.get("algorithm") != metadata["algorithm"]:
            raise ValueError("Existing output belongs to a different algorithm")
        return group
    group = zarr.open_group(str(store_path), mode="w", zarr_format=2)
    group.create_array(
        "semantic_label",
        shape=(total_frames, height, width),
        dtype="u1",
        chunks=(1, 256, 256),
        compressor=compressor,
        fill_value=0,
    )
    group.create_array(
        "time_ns",
        shape=(total_frames,),
        dtype="i8",
        chunks=(288,),
        compressor=compressor,
        fill_value=0,
    )
    group.create_array(
        "valid_observation",
        shape=(height, width),
        dtype="bool",
        chunks=(256, 256),
        compressor=compressor,
        fill_value=False,
    )
    group.attrs.update(metadata)
    return group


def main() -> int:
    args = parse_args()
    paths = sorted(args.source_root.glob("*.zarr"))
    paths = [path for path in paths if args.first_day <= path.stem <= args.last_day]
    if not paths:
        raise FileNotFoundError("No daily Zarr stores match the requested interval")
    expected_names = [stamp.strftime("%Y%m%d") for stamp in pd.date_range(args.first_day, args.last_day, freq="D")]
    if [path.stem for path in paths] != expected_names:
        raise ValueError(f"Expected daily stores {expected_names}, found {[path.stem for path in paths]}")

    inventory = _day_inventory(paths)
    geometry = _geometry(paths[0])
    for path in paths[1:]:
        with xr.open_zarr(path, consolidated=True, chunks=None) as dataset:
            if not np.array_equal(dataset.x.values, geometry["source_x"]):
                raise ValueError(f"Source x axis changes in {path}")
            if not np.array_equal(dataset.y.values, geometry["source_y"]):
                raise ValueError(f"Source y axis changes in {path}")
    height, width = np.asarray(geometry["area_km2"]).shape
    total_frames = len(paths) * 288
    tracker = PureGenUtlsTracker()
    metadata = {
        "algorithm": tracker.version,
        "config_hash": tracker.config_hash,
        "source_root": str(args.source_root.resolve()),
        "first_day": args.first_day,
        "last_day": args.last_day,
        "total_frames": total_frames,
        "cadence_minutes": 5,
        "orientation": "north-up west-left",
        "geometry": "native SEVIRI RSS grid; no parallax",
        "channels": ["WV_062", "WV_073", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134"],
        "semantic_class_names": {str(k): v for k, v in RDT_ORIGINAL_CLASS_NAMES.items()},
        "external_inputs": [],
        "uses_cal": False,
        "uses_opera": False,
        "uses_era5": False,
        "uses_object_local_utls_proxy": True,
        "uses_seviri_ir_cloud_type": True,
    }
    group = _open_output(args.output_root, total_frames, height, width, metadata)
    progress_path = args.output_root / "progress.json"
    metrics_path = args.output_root / "frame_metrics.json"
    checkpoint_path = args.output_root / "tracker_checkpoint.pkl.gz"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        committed_frames = int(progress.get("committed_frames", 0))
        if committed_frames:
            if not checkpoint_path.exists():
                raise FileNotFoundError("Progress exists but the tracker checkpoint is missing")
            with gzip.open(checkpoint_path, "rb") as stream:
                tracker = pickle.load(stream)
            if tracker.config_hash != metadata["config_hash"]:
                raise ValueError("Checkpoint configuration does not match this run")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else []
        metrics = metrics[:committed_frames]
    else:
        committed_frames = 0
        metrics = []

    run_start_frame = committed_frames
    frame_limit = total_frames if args.max_frames is None else min(total_frames, args.max_frames)
    start_wall = time.perf_counter()
    global_index = 0
    for day_number, day in enumerate(inventory):
        day_start = global_index
        day_stop = day_start + 288
        global_index = day_stop
        if day_stop <= committed_frames:
            continue
        if committed_frames not in (0, day_start):
            raise ValueError("Resume checkpoints are supported only at complete day boundaries")
        if day_start >= frame_limit:
            break

        zgroup = zarr.open_group(str(day["path"]), mode="r")
        data = zgroup["seviri"]
        times: pd.DatetimeIndex = day["times"]
        indices: np.ndarray = day["indices"]
        day_metrics = []
        for local_index in range(288):
            output_index = day_start + local_index
            if output_index >= frame_limit:
                break
            started = time.perf_counter()
            raw_index = int(indices[local_index])
            frame = np.asarray(data[raw_index], dtype=np.float32)
            frame, _, _ = _orient(
                frame,
                np.asarray(geometry["source_x"]),
                np.asarray(geometry["source_y"]),
            )
            result = tracker.step_frame(
                frame,
                times[local_index],
                np.asarray(geometry["area_km2"]),
                lat_deg=np.asarray(geometry["latitude"]),
                lon_deg=np.asarray(geometry["longitude"]),
            )
            semantic = result["semantic_label"]
            group["semantic_label"][output_index] = semantic
            group["time_ns"][output_index] = int(times[local_index].value)
            if output_index == 0:
                group["valid_observation"][:] = result["valid_observation"]
            class_counts = np.bincount(semantic.ravel(), minlength=7)
            row = {
                "frame": output_index,
                "time": times[local_index].isoformat(),
                "cells": len(result["cells"]),
                "proxy_confident_cells": int(
                    sum(obj["proxy_confidence"] >= tracker.proxy_config.min_proxy_confidence for obj in result["objects"])
                ),
                "ot_pixels": int((semantic == 6).sum()),
                "class_counts_0_to_6": class_counts.tolist(),
                "elapsed_s": round(time.perf_counter() - started, 3),
            }
            day_metrics.append(row)
            if local_index == 0 or (local_index + 1) % 12 == 0:
                elapsed = time.perf_counter() - start_wall
                completed_this_run = output_index - run_start_frame + 1
                rate = completed_this_run / max(elapsed, 1e-6)
                remaining = max(frame_limit - output_index - 1, 0) / max(rate, 1e-6)
                print(
                    f"[{times[local_index]:%Y-%m-%d %H:%M}] frame {output_index + 1}/{frame_limit} "
                    f"cells={row['cells']} proxy={row['proxy_confident_cells']} OTpx={row['ot_pixels']} "
                    f"{row['elapsed_s']:.2f}s ETA={remaining / 60:.1f}min",
                    flush=True,
                )
            del result, semantic, frame
            if (local_index + 1) % 12 == 0:
                gc.collect()

        if day_stop <= frame_limit:
            metrics.extend(day_metrics)
            _atomic_pickle(checkpoint_path, tracker)
            _atomic_json(metrics_path, metrics)
            committed_frames = day_stop
            _atomic_json(
                progress_path,
                {
                    "status": "complete" if committed_frames == total_frames else "running",
                    "committed_frames": committed_frames,
                    "total_frames": total_frames,
                    "last_time": metrics[-1]["time"],
                    "config_hash": tracker.config_hash,
                },
            )
        else:
            break

    print(
        json.dumps(
            {
                "output": str(args.output_root.resolve()),
                "committed_frames": committed_frames,
                "requested_frames": frame_limit,
                "complete": committed_frames == total_frames,
                "wall_time_s": round(time.perf_counter() - start_wall, 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
