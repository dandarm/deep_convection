#!/usr/bin/env python3
"""Compare pure-GEN/UTLS lifecycle labels with public NWC SAF RDT-CW GIFs.

The comparison is made only for triggering, growing, mature and decay.  Cold
cloud, overshooting-top and triggering-from-split pixels are excluded.  The
teacher always runs causally at the native five-minute RSS cadence; evaluation
occurs only at co-temporal 15-minute archive slots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.nwcsaf_rdt_archive import (  # noqa: E402
    GifGeoReference,
    TARGET_CLASSES,
    TARGET_CLASS_NAMES,
    binary_spatial_metrics,
    extract_reference_phases,
    match_objects,
    rss_grid_to_gif_pixels,
    sample_reference_to_grid,
    sha256_file,
)
from emma_gpm.rdt.pure_gen_utls import PureGenUtlsTracker  # noqa: E402
from emma_gpm.rdt.storage import DomainFrameReader  # noqa: E402
from emma_gpm.rdt_teacher import projected_pixel_areas  # noqa: E402


def _orient_geometry(
    x: np.ndarray,
    y: np.ndarray,
    area: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_out, y_out, area_out = x.copy(), y.copy(), area
    if y_out[0] < y_out[-1]:
        y_out = y_out[::-1]
        area_out = area_out[::-1]
    if x_out[0] > x_out[-1]:
        x_out = x_out[::-1]
        area_out = area_out[:, ::-1]
    return x_out, y_out, np.ascontiguousarray(area_out)


def _geometry(zarr_root: Path, first_time: pd.Timestamp) -> dict:
    path = zarr_root / f"{first_time:%Y}" / f"{first_time:%Y%m%d}.zarr"
    with xr.open_zarr(path, consolidated=True, chunks=None) as dataset:
        x_raw = np.asarray(dataset.x.values, dtype=np.float64)
        y_raw = np.asarray(dataset.y.values, dtype=np.float64)
        projection = str(dataset.attrs["projection_wkt"])
    area = projected_pixel_areas(x_raw, y_raw, projection, allow_invalid=True)
    x, y, area = _orient_geometry(x_raw, y_raw, area)
    xx, yy = np.meshgrid(x, y)
    to_lonlat = Transformer.from_crs(CRS.from_user_input(projection), "EPSG:4326", always_xy=True)
    longitude, latitude = to_lonlat.transform(xx, yy)
    return {
        "x": x,
        "y": y,
        "area_km2": area.astype(np.float32),
        "longitude": np.asarray(longitude, dtype=np.float32),
        "latitude": np.asarray(latitude, dtype=np.float32),
        "projection": projection,
    }


def _available_times(zarr_root: Path, times: pd.DatetimeIndex) -> set[pd.Timestamp]:
    available: set[pd.Timestamp] = set()
    for day in times.normalize().unique():
        path = zarr_root / f"{day:%Y}" / f"{day:%Y%m%d}.zarr"
        if not path.exists():
            continue
        with xr.open_zarr(path, consolidated=True, chunks=None) as dataset:
            available.update(pd.DatetimeIndex(dataset.time.values))
    return available


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest_index(path: Path) -> dict[pd.Timestamp, Path]:
    frame = pd.read_csv(path)
    required = {"time", "status", "path", "sha256"}
    if not required.issubset(frame):
        raise ValueError(f"Archive manifest lacks columns {sorted(required - set(frame))}")
    result: dict[pd.Timestamp, Path] = {}
    for row in frame.itertuples(index=False):
        if row.status == "missing" or not isinstance(row.path, str):
            continue
        gif = Path(row.path)
        if not gif.exists() or sha256_file(gif) != str(row.sha256):
            raise ValueError(f"Missing or changed GIF declared by manifest: {gif}")
        result[pd.Timestamp(row.time)] = gif
    return result


def _aggregate(frame: pd.DataFrame, fields: list[str]) -> dict:
    result: dict[str, dict[str, float | int]] = {}
    if frame.empty:
        return result
    for phase, group in frame.groupby("phase"):
        entry: dict[str, float | int] = {"frames": len(group)}
        for field in fields:
            values = pd.to_numeric(group[field], errors="coerce")
            finite = values[np.isfinite(values)]
            entry[field + "_mean"] = float(finite.mean()) if len(finite) else np.nan
            entry[field + "_median"] = float(finite.median()) if len(finite) else np.nan
        result[str(phase)] = entry
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-root", type=Path, required=True)
    parser.add_argument("--gif-manifest", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "configs" / "nwcsaf_rdt_gif_v2025_msg.json",
    )
    parser.add_argument("--start", required=True, help="Teacher sequence start, UTC")
    parser.add_argument("--end", required=True, help="Exclusive UTC end")
    parser.add_argument(
        "--evaluation-start",
        help="First evaluated slot after causal warm-up; defaults to --start",
    )
    parser.add_argument("--cloud-height-km", type=float, default=12.0)
    parser.add_argument(
        "--height-sensitivity-km",
        type=float,
        nargs="*",
        default=[8.0, 12.0, 15.0],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-centroid-km", type=float, default=150.0)
    args = parser.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    evaluation_start = pd.Timestamp(args.evaluation_start or args.start)
    for name, stamp in (("start", start), ("end", end), ("evaluation-start", evaluation_start)):
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("UTC").tz_localize(None)
        if name == "start":
            start = stamp
        elif name == "end":
            end = stamp
        else:
            evaluation_start = stamp
    if start >= evaluation_start or evaluation_start >= end:
        parser.error("Require start < evaluation-start < end to preserve an explicit warm-up")
    if any(value < 0 for value in args.height_sensitivity_km) or args.cloud_height_km < 0:
        parser.error("Cloud heights must be nonnegative")
    heights = sorted(set([*args.height_sensitivity_km, args.cloud_height_km]))
    times = pd.date_range(start, end, freq="5min", inclusive="left")
    available = _available_times(args.zarr_root, times)
    missing = [stamp for stamp in times if stamp not in available]
    if missing:
        raise ValueError(
            "Teacher sequence is incomplete; missing "
            + ", ".join(stamp.isoformat() for stamp in missing[:10])
        )

    gif_by_time = _manifest_index(args.gif_manifest)
    expected_evaluation_times = [
        stamp for stamp in times if stamp >= evaluation_start and stamp.minute % 15 == 0
    ]
    missing_gifs = [stamp for stamp in expected_evaluation_times if stamp not in gif_by_time]
    if missing_gifs:
        raise ValueError(
            "NWC SAF reference interval is incomplete; missing "
            + ", ".join(stamp.isoformat() for stamp in missing_gifs[:10])
        )
    evaluation_times = expected_evaluation_times
    if not evaluation_times:
        raise ValueError("No quarter-hour evaluation slots in the requested interval")

    georef = GifGeoReference.from_json(args.calibration)
    geometry = _geometry(args.zarr_root, start)
    x, y = geometry["x"], geometry["y"]
    spacing_km = (float(np.median(np.abs(np.diff(y)))) / 1000.0,
                  float(np.median(np.abs(np.diff(x)))) / 1000.0)
    mappings = {
        height: rss_grid_to_gif_pixels(
            x,
            y,
            geometry["projection"],
            georef,
            cloud_height_km=height,
        )
        for height in heights
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "teacher": "rdt_cw_inspired_pure_gen_utls_v2",
        "teacher_sequence_start": start.isoformat(),
        "evaluation_start": evaluation_start.isoformat(),
        "end_exclusive": end.isoformat(),
        "classes": {str(key): value for key, value in TARGET_CLASS_NAMES.items()},
        "excluded": ["non_convective_cold_cloud", "overshooting_top", "triggering_from_split"],
        "gif_manifest": str(args.gif_manifest.resolve()),
        "gif_manifest_sha256": sha256_file(args.gif_manifest),
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": sha256_file(args.calibration),
        "zarr_root": str(args.zarr_root.resolve()),
        "height_sensitivity_km": heights,
        "object_metrics_height_km": args.cloud_height_km,
        "rss_projection": geometry["projection"],
        "reference_projection": georef.geos_crs.to_string(),
        "comparison_grid": "native north-up/west-left RSS grid; categorical nearest-neighbour reference sampling",
    }
    _atomic_json(args.output_dir / "protocol.json", protocol)

    tracker = PureGenUtlsTracker()
    reader = DomainFrameReader(args.zarr_root)
    metric_rows: list[dict] = []
    object_rows: list[dict] = []
    confusion_rows: list[dict] = []
    frame_rows: list[dict] = []
    started = time.monotonic()
    try:
        for frame_index, stamp in enumerate(times):
            frame = reader.load(stamp)
            output = tracker.step_frame(
                frame,
                stamp,
                geometry["area_km2"],
                lat_deg=geometry["latitude"],
                lon_deg=geometry["longitude"],
            )
            if stamp not in evaluation_times:
                continue
            reference = extract_reference_phases(gif_by_time[stamp], georef)
            prediction_labels = np.asarray(output["semantic_label"], dtype=np.uint8)
            observation_valid = np.asarray(output["valid_observation"], dtype=bool)
            for height, (row, column, map_valid) in mappings.items():
                reference_labels, reference_excluded = sample_reference_to_grid(
                    reference, row, column, map_valid
                )
                valid = observation_valid & map_valid & ~reference_excluded
                phases = [("all_target", np.isin(reference_labels, TARGET_CLASSES),
                           np.isin(prediction_labels, TARGET_CLASSES))]
                phases.extend(
                    (
                        TARGET_CLASS_NAMES[class_id],
                        reference_labels == class_id,
                        prediction_labels == class_id,
                    )
                    for class_id in TARGET_CLASSES
                )
                for phase, truth, prediction in phases:
                    values = binary_spatial_metrics(
                        truth,
                        prediction,
                        valid,
                        geometry["area_km2"],
                        spacing_km=spacing_km,
                    )
                    metric_rows.append(
                        {"time": stamp.isoformat(), "height_km": height, "phase": phase, **values}
                    )
                for truth_class in (0, *TARGET_CLASSES):
                    for predicted_class in (0, *TARGET_CLASSES):
                        selected = valid & (reference_labels == truth_class) & (prediction_labels == predicted_class)
                        confusion_rows.append(
                            {
                                "time": stamp.isoformat(),
                                "height_km": height,
                                "reference_class": truth_class,
                                "prediction_class": predicted_class,
                                "pixels": int(selected.sum()),
                                "area_km2": float(
                                    np.sum(np.where(selected, geometry["area_km2"], 0.0), dtype=np.float64)
                                ),
                            }
                        )
                if height == args.cloud_height_km:
                    for phase, truth, prediction in phases:
                        matches, counts = match_objects(
                            truth,
                            prediction,
                            valid,
                            geometry["area_km2"],
                            x,
                            y,
                            spacing_km=spacing_km,
                            maximum_centroid_km=args.maximum_centroid_km,
                        )
                        for match in matches:
                            object_rows.append(
                                {"time": stamp.isoformat(), "height_km": height, "phase": phase, **match}
                            )
                        frame_rows.append(
                            {"time": stamp.isoformat(), "height_km": height, "phase": phase, **counts}
                        )
            elapsed = time.monotonic() - started
            print(
                f"evaluated={stamp.isoformat()} teacher_frame={frame_index + 1}/{len(times)} "
                f"objects={len(output['objects'])} elapsed_seconds={elapsed:.1f}",
                flush=True,
            )
    finally:
        reader.close()

    metrics = pd.DataFrame(metric_rows)
    objects = pd.DataFrame(object_rows)
    confusion = pd.DataFrame(confusion_rows)
    frames = pd.DataFrame(frame_rows)
    for name, table in (
        ("spatial_metrics.csv", metrics),
        ("object_matches.csv", objects),
        ("phase_confusion.csv", confusion),
        ("object_counts.csv", frames),
    ):
        temporary = args.output_dir / (name + ".part")
        table.to_csv(temporary, index=False)
        temporary.replace(args.output_dir / name)

    default_metrics = metrics[metrics.height_km == args.cloud_height_km]
    summary = {
        "status": "complete",
        "evaluated_frames": len(evaluation_times),
        "teacher_frames": len(times),
        "warmup_frames": int((evaluation_start - start).total_seconds() / 300.0),
        "elapsed_seconds": time.monotonic() - started,
        "default_height_km": args.cloud_height_km,
        "spatial": _aggregate(
            default_metrics,
            ["iou", "precision", "recall", "area_ratio", "buffered_iou_5km",
             "buffered_iou_10km", "buffered_iou_20km", "f1_10km"],
        ),
        "matched_objects": int(len(objects)),
        "object_centroid_distance_km_median": (
            float(objects.centroid_distance_km.median()) if not objects.empty else None
        ),
        "object_hausdorff95_km_median": (
            float(objects.hausdorff95_km.median()) if not objects.empty else None
        ),
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
