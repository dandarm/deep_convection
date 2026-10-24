#!/usr/bin/env python3
"""Find independent important-convection clips in the 2022 SEVIRI archive.

Days are evaluated in chronological batches, using one process per day.  Each
daily tracker is warmed with the final 90 minutes of the preceding contiguous
day so that the first valid samples do not start from an empty lifecycle.  The
search stops after the earliest processed prefix that contains the requested
number of distinct, high-value convective tracks.

The complete pixel labels and per-frame metrics are persisted for every day
that is processed.  Rendering therefore never needs to run the teacher again.
``RDT_original`` is imported as a read-only algorithmic reference and is not
written by this script.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import pandas as pd
from numcodecs import Blosc
import zarr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.rdt.pure_gen_utls import PureGenUtlsTracker  # noqa: E402
from scripts.build_rdt_pure_gen_utls_week import _geometry, _orient  # noqa: E402


CHANNELS = ["WV_062", "WV_073", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134"]
PHASE_WEIGHT = {2: 0.80, 3: 1.30, 4: 1.50, 5: 0.90}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(r"E:\Datasets\MSG-SEVIRI\2022"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "rdt_pure_gen_utls_events_2022",
    )
    parser.add_argument("--first-day", default="20220101")
    parser.add_argument("--last-day", default="20221231")
    parser.add_argument("--target-events", type=int, default=30)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--batch-days", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=18)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--min-area-km2", type=float, default=1500.0)
    parser.add_argument("--min-proxy-confidence", type=float, default=0.35)
    parser.add_argument("--max-t-min-k", type=float, default=235.0)
    parser.add_argument("--min-thermal-contrast-k", type=float, default=12.0)
    parser.add_argument("--min-bbox-fill-fraction", type=float, default=0.05)
    parser.add_argument(
        "--max-event-area-km2",
        type=float,
        default=25_000.0,
        help="Selection-only upper bound for non-OT objects; avoids synoptic frontal shields.",
    )
    parser.add_argument("--min-event-separation-minutes", type=float, default=90.0)
    parser.add_argument("--max-days", type=int, default=None, help="Optional integration-test limit.")
    return parser.parse_args()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ordered_inventory(group: zarr.Group) -> tuple[pd.DatetimeIndex, np.ndarray]:
    values = np.asarray(group["time"][:], dtype=np.int64)
    raw_times = pd.DatetimeIndex(pd.to_datetime(values, unit="s"))
    if len(raw_times) != 288 or raw_times.hasnans or raw_times.has_duplicates:
        raise ValueError(f"Expected 288 unique timestamps, received {len(raw_times)}")
    order = np.argsort(raw_times.asi8).astype(np.int64)
    times = raw_times[order]
    if not np.all(np.diff(times.asi8) == 5 * 60 * 1_000_000_000):
        raise ValueError("Daily input does not have an exact five-minute cadence")
    return times, order


def _candidate_from_object(
    row: dict,
    cell: dict,
    *,
    day: str,
    local_frame: int,
    timestamp: pd.Timestamp,
    min_area_km2: float,
) -> dict | None:
    class_id = int(row["class_id"])
    if class_id not in PHASE_WEIGHT:
        return None
    area_km2 = float(row["area_km2"])
    ot_pixels = int(row["ot_pixels"])
    # Persistent OT is independently important.  Without OT, require a large
    # growing/mature system; triggering/decaying objects are retained only if
    # they are substantially larger than the base threshold.
    if ot_pixels <= 0:
        required_area = min_area_km2 if class_id in (3, 4) else 1.75 * min_area_km2
        if area_km2 < required_area:
            return None
    bbox = cell["bbox"]
    score = area_km2 * PHASE_WEIGHT[class_id] + 300.0 * ot_pixels
    if ot_pixels:
        score += 5000.0
    return {
        "day": day,
        "local_frame": local_frame,
        "time": timestamp.isoformat(),
        "track_id": int(row["track_id"]),
        "cell_id": int(row["cell_id"]),
        "class_id": class_id,
        "score": round(float(score), 3),
        "area_km2": round(area_km2, 3),
        "ot_pixels": ot_pixels,
        "proxy_confidence": round(float(row["proxy_confidence"]), 5),
        "t_min_k": round(float(row["t_min_k"]), 3),
        "t_edge_k": round(float(row["t_edge_k"]), 3),
        "t_utls_proxy_k": (
            round(float(row["t_utls_proxy_k"]), 3)
            if np.isfinite(row["t_utls_proxy_k"])
            else None
        ),
        "centre_lat": round(float(cell["lat_deg"]), 4),
        "centre_lon": round(float(cell["lon_deg"]), 4),
        "bbox_yx": [
            int(bbox[0].start),
            int(bbox[0].stop),
            int(bbox[1].start),
            int(bbox[1].stop),
        ],
    }


def _process_day(task: dict) -> dict:
    day_path = Path(task["day_path"])
    previous_path = Path(task["previous_path"]) if task.get("previous_path") else None
    output_root = Path(task["output_root"])
    day = day_path.stem
    final_root = output_root / "daily" / day
    summary_path = final_root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") == "complete"
            and summary.get("algorithm") == task["algorithm"]
            and summary.get("config_hash") == task["config_hash"]
        ):
            summary["reused"] = True
            return summary
        raise ValueError(
            f"Existing output for {day} belongs to a different teacher/configuration; "
            "choose a new output root instead of mixing labels"
        )

    geometry = _geometry(day_path)
    area = np.asarray(geometry["area_km2"])
    latitude = np.asarray(geometry["latitude"])
    longitude = np.asarray(geometry["longitude"])
    source_x = np.asarray(geometry["source_x"])
    source_y = np.asarray(geometry["source_y"])
    height, width = area.shape
    tracker = PureGenUtlsTracker()
    started = time.perf_counter()

    warmup_used = 0
    if previous_path is not None:
        previous_group = zarr.open_group(str(previous_path), mode="r")
        previous_times, previous_indices = _ordered_inventory(previous_group)
        current_group_for_time = zarr.open_group(str(day_path), mode="r")
        current_times_for_time, _ = _ordered_inventory(current_group_for_time)
        if current_times_for_time[0] - previous_times[-1] == pd.Timedelta(minutes=5):
            previous_data = previous_group["seviri"]
            first = max(0, 288 - int(task["warmup_frames"]))
            for index in range(first, 288):
                frame = np.asarray(previous_data[int(previous_indices[index])], dtype=np.float32)
                frame, _, _ = _orient(frame, source_x, source_y)
                tracker.step_frame(
                    frame,
                    previous_times[index],
                    area,
                    lat_deg=latitude,
                    lon_deg=longitude,
                )
                warmup_used += 1

    temporary_root = output_root / "daily" / f".{day}.{uuid.uuid4().hex}.part"
    temporary_root.mkdir(parents=True, exist_ok=False)
    label_group = zarr.open_group(str(temporary_root / "labels.zarr"), mode="w", zarr_format=2)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    labels_array = label_group.create_array(
        "semantic_label",
        shape=(288, height, width),
        chunks=(1, height, width),
        dtype="u1",
        compressor=compressor,
        fill_value=0,
    )
    time_array = label_group.create_array(
        "time_ns",
        shape=(288,),
        chunks=(288,),
        dtype="i8",
        compressor=compressor,
        fill_value=0,
    )
    label_group.attrs.update(
        {
            "algorithm": tracker.version,
            "config_hash": tracker.config_hash,
            "source": str(day_path.resolve()),
            "warmup_frames": warmup_used,
            "channels": CHANNELS,
            "orientation": "north-up west-left",
            "uses_seviri_ir_cloud_type": True,
            "ot_rendering": "exact semantic label pixels; no dilation or rings",
        }
    )

    group = zarr.open_group(str(day_path), mode="r")
    times, indices = _ordered_inventory(group)
    data = group["seviri"]
    metrics: list[dict] = []
    best_by_track: dict[int, dict] = {}
    class_totals = np.zeros(7, dtype=np.int64)
    try:
        for local_frame in range(288):
            frame_started = time.perf_counter()
            frame = np.asarray(data[int(indices[local_frame])], dtype=np.float32)
            frame, _, _ = _orient(frame, source_x, source_y)
            result = tracker.step_frame(
                frame,
                times[local_frame],
                area,
                lat_deg=latitude,
                lon_deg=longitude,
            )
            semantic = np.asarray(result["semantic_label"], dtype=np.uint8)
            labels_array[local_frame] = semantic
            time_array[local_frame] = int(times[local_frame].value)
            counts = np.bincount(semantic.ravel(), minlength=7).astype(np.int64)
            class_totals += counts
            metrics.append(
                {
                    "frame": local_frame,
                    "time": times[local_frame].isoformat(),
                    "cells": len(result["cells"]),
                    "proxy_confident_cells": int(
                        sum(
                            obj["proxy_confidence"] >= tracker.proxy_config.min_proxy_confidence
                            for obj in result["objects"]
                        )
                    ),
                    "ot_pixels": int(counts[6]),
                    "class_counts_0_to_6": counts.tolist(),
                    "elapsed_s": round(time.perf_counter() - frame_started, 3),
                }
            )
            for row in result["objects"]:
                candidate = _candidate_from_object(
                    row,
                    result["cells"][int(row["cell_id"])],
                    day=day,
                    local_frame=local_frame,
                    timestamp=times[local_frame],
                    min_area_km2=float(task["min_area_km2"]),
                )
                if candidate is None:
                    continue
                track_id = int(candidate["track_id"])
                if candidate["score"] > best_by_track.get(track_id, {"score": -np.inf})["score"]:
                    best_by_track[track_id] = candidate
    except Exception:
        # Keep the unique partial directory for forensic inspection/resume
        # rather than silently replacing a potentially useful product.
        raise

    half_before = int(task["clip_frames"]) // 2
    half_after = int(task["clip_frames"]) - half_before - 1
    candidates = [
        value
        for value in best_by_track.values()
        if value["local_frame"] >= half_before
        and value["local_frame"] + half_after < 288
    ]
    candidates.sort(key=lambda value: value["score"], reverse=True)
    summary = {
        "status": "complete",
        "day": day,
        "source": str(day_path.resolve()),
        "algorithm": tracker.version,
        "config_hash": tracker.config_hash,
        "warmup_frames": warmup_used,
        "frames": 288,
        "wall_time_s": round(time.perf_counter() - started, 3),
        "class_pixel_totals_0_to_6": class_totals.tolist(),
        "candidate_tracks": candidates,
    }
    _atomic_json(temporary_root / "frame_metrics.json", metrics)
    _atomic_json(temporary_root / "summary.json", summary)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists():
        raise FileExistsError(f"Refusing to replace existing incomplete output: {final_root}")
    temporary_root.replace(final_root)
    return summary


def _select_events(
    summaries: list[dict],
    *,
    target_events: int,
    clip_frames: int,
    min_separation_minutes: float,
    area_km2: np.ndarray | None = None,
    min_proxy_confidence: float = 0.35,
    max_t_min_k: float = 235.0,
    min_thermal_contrast_k: float = 12.0,
    min_bbox_fill_fraction: float = 0.05,
    max_event_area_km2: float = 25_000.0,
) -> list[dict]:
    candidates = [candidate for summary in summaries for candidate in summary["candidate_tracks"]]
    ranked: list[dict] = []
    for candidate in candidates:
        # Synthetic/unit-test candidates and old manifests without diagnostic
        # fields retain their original score.  Scientific archive candidates
        # use the stricter convective-quality ranking below.
        required = {"proxy_confidence", "t_min_k", "t_edge_k", "area_km2", "ot_pixels", "bbox_yx"}
        if not required.issubset(candidate):
            value = dict(candidate)
            value["selection_score"] = float(candidate["score"])
            ranked.append(value)
            continue
        ot_pixels = int(candidate["ot_pixels"])
        contrast_k = float(candidate["t_edge_k"] - candidate["t_min_k"])
        bbox = [int(value) for value in candidate["bbox_yx"]]
        if area_km2 is None:
            fill_fraction = 1.0
        else:
            footprint = np.asarray(area_km2[bbox[0] : bbox[1], bbox[2] : bbox[3]])
            bbox_area = float(np.nansum(footprint))
            fill_fraction = float(candidate["area_km2"]) / max(bbox_area, 1.0)
            fill_fraction = float(np.clip(fill_fraction, 0.0, 1.0))
        if ot_pixels <= 0 and (
            float(candidate["proxy_confidence"]) < min_proxy_confidence
            or float(candidate["t_min_k"]) > max_t_min_k
            or contrast_k < min_thermal_contrast_k
            or fill_fraction < min_bbox_fill_fraction
            or float(candidate["area_km2"]) > max_event_area_km2
        ):
            continue
        # Area saturates at 15,000 km2: beyond mesoscale size, radiometric
        # depth and compactness are more informative than a larger anvil/front.
        capped_area = min(float(candidate["area_km2"]), 15_000.0)
        spectral_strength = contrast_k + 50.0 * float(candidate["proxy_confidence"])
        selection_score = spectral_strength * np.sqrt(capped_area) * np.sqrt(fill_fraction)
        if ot_pixels:
            selection_score += 10_000.0 + 500.0 * ot_pixels
        value = dict(candidate)
        value["thermal_contrast_k"] = round(contrast_k, 3)
        value["bbox_fill_fraction"] = round(fill_fraction, 5)
        value["selection_score"] = round(float(selection_score), 3)
        ranked.append(value)
    candidates = sorted(ranked, key=lambda value: value["selection_score"], reverse=True)
    selected: list[dict] = []
    separation = timedelta(seconds=int(round(60.0 * min_separation_minutes)))
    for candidate in candidates:
        stamp = pd.Timestamp(candidate["time"])
        if any(abs(stamp - pd.Timestamp(existing["peak_time"])) < separation for existing in selected):
            continue
        before = clip_frames // 2
        after = clip_frames - before - 1
        event = dict(candidate)
        event["event_id"] = ""
        event["peak_time"] = event.pop("time")
        event["clip_start_frame"] = int(event["local_frame"]) - before
        event["clip_stop_frame_exclusive"] = int(event["local_frame"]) + after + 1
        event["clip_start_time"] = (stamp - timedelta(seconds=300 * before)).isoformat()
        event["clip_end_time"] = (stamp + timedelta(seconds=300 * after)).isoformat()
        selected.append(event)
        if len(selected) >= target_events:
            break
    selected.sort(key=lambda value: value["peak_time"])
    for index, event in enumerate(selected, start=1):
        event["event_id"] = f"event_{index:03d}"
    return selected


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    expected_tracker = PureGenUtlsTracker()
    all_paths = sorted(
        path
        for path in args.source_root.glob("*.zarr")
        if args.first_day <= path.stem <= args.last_day
    )
    if args.max_days is not None:
        all_paths = all_paths[: args.max_days]
    if not all_paths:
        raise FileNotFoundError("No daily Zarr stores match the requested interval")
    workers = max(1, min(int(args.workers), os.cpu_count() or 1))
    batch_days = args.batch_days or workers
    summaries_by_day: dict[str, dict] = {}
    selection_area_km2 = np.asarray(_geometry(all_paths[0])["area_km2"])
    wall_started = time.perf_counter()
    for batch_start in range(0, len(all_paths), batch_days):
        batch = all_paths[batch_start : batch_start + batch_days]
        tasks = []
        for path in batch:
            path_index = all_paths.index(path)
            previous = all_paths[path_index - 1] if path_index > 0 else None
            if previous is not None:
                expected = pd.Timestamp(path.stem) - timedelta(days=1)
                if previous.stem != expected.strftime("%Y%m%d"):
                    previous = None
            tasks.append(
                {
                    "day_path": str(path),
                    "previous_path": str(previous) if previous else None,
                    "output_root": str(args.output_root),
                    "warmup_frames": args.warmup_frames,
                    "clip_frames": args.clip_frames,
                    "min_area_km2": args.min_area_km2,
                    "algorithm": expected_tracker.version,
                    "config_hash": expected_tracker.config_hash,
                }
            )
        print(
            f"Processing {batch[0].stem}..{batch[-1].stem} with {min(workers, len(tasks))} workers",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            futures = {pool.submit(_process_day, task): task for task in tasks}
            for future in as_completed(futures):
                summary = future.result()
                summaries_by_day[summary["day"]] = summary
                print(
                    f"  {summary['day']}: {len(summary['candidate_tracks'])} candidate tracks, "
                    f"{summary['wall_time_s'] / 60.0:.1f} min"
                    + (" (reused)" if summary.get("reused") else ""),
                    flush=True,
                )
        ordered = [summaries_by_day[key] for key in sorted(summaries_by_day)]
        selected = _select_events(
            ordered,
            target_events=args.target_events,
            clip_frames=args.clip_frames,
            min_separation_minutes=args.min_event_separation_minutes,
            area_km2=selection_area_km2,
            min_proxy_confidence=args.min_proxy_confidence,
            max_t_min_k=args.max_t_min_k,
            min_thermal_contrast_k=args.min_thermal_contrast_k,
            min_bbox_fill_fraction=args.min_bbox_fill_fraction,
            max_event_area_km2=args.max_event_area_km2,
        )
        manifest = {
            "status": "complete" if len(selected) >= args.target_events else "searching",
            "source_root": str(args.source_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "processed_days": sorted(summaries_by_day),
            "workers": workers,
            "target_events": args.target_events,
            "selected_events": len(selected),
            "clip_shape_tchw": [args.clip_frames, 7, 763, 1762],
            "cadence_minutes": 5,
            "selection": {
                "one_peak_per_daily_track": True,
                "min_area_km2_growing_mature": args.min_area_km2,
                "persistent_ot_always_eligible": True,
                "min_proxy_confidence_without_ot": args.min_proxy_confidence,
                "max_t_min_k_without_ot": args.max_t_min_k,
                "min_thermal_contrast_k_without_ot": args.min_thermal_contrast_k,
                "min_bbox_fill_fraction_without_ot": args.min_bbox_fill_fraction,
                "max_object_area_km2_without_ot": args.max_event_area_km2,
                "min_temporal_separation_minutes": args.min_event_separation_minutes,
                "ranking": (
                    "UTLS confidence and thermal contrast times square-root compact capped area; "
                    "persistent-OT bonus"
                ),
            },
            "events": selected,
            "wall_time_s": round(time.perf_counter() - wall_started, 3),
        }
        _atomic_json(args.output_root / "selected_events.json", manifest)
        _atomic_json(
            args.output_root / "progress.json",
            {
                "processed_days": sorted(summaries_by_day),
                "selected_events": len(selected),
                "target_events": args.target_events,
                "status": manifest["status"],
            },
        )
        print(f"Selected {len(selected)}/{args.target_events} events", flush=True)
        if len(selected) >= args.target_events:
            print(json.dumps(manifest, indent=2))
            return 0
    print(
        f"Archive exhausted with {len(selected)}/{args.target_events} qualifying events",
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
