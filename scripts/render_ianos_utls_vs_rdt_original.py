#!/usr/bin/env python3
"""Render a controlled one-day Ianos comparison of two SEVIRI-only teachers.

The comparison uses the same 288 five-minute observations, the same native
IR10.8 background, palette, alpha and geographic overlay for both videos.
The project teacher is recomputed from midnight with its object-local UTLS
proxy.  The historical ``RDT_original`` pure-GEN labels are read-only inputs.

The 17 September Zarr contains a known payload-order defect between 13:30 and
16:00 UTC.  The permutation already documented by the historical workflow is
applied before either background or project-teacher processing.  A continuity
diagnostic is persisted with the outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import xarray as xr
import zarr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.rdt.pure_gen_utls import PureGenUtlsTracker  # noqa: E402
from scripts.build_rdt_pure_gen_utls_week import _geometry, _orient  # noqa: E402
from scripts.render_rdt_pure_gen_utls_week import (  # noqa: E402
    _font,
    _radiance_composite,
    _static_overlay,
    _video_metadata,
)


DAY = "20200917"
FRAMES_PER_DAY = 288
KNOWN_CORRECTION = {
    "description": (
        "After sorting nominal timestamps, payload positions 162:181 and "
        "181:193 are swapped. Reordering them restores five-minute radiance continuity."
    ),
    "sorted_slice_a": [162, 181],
    "sorted_slice_b": [181, 193],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(r"E:\Datasets\MSG-SEVIRI\2020\20200917.zarr"),
    )
    parser.add_argument(
        "--original-cache",
        type=Path,
        default=(
            ROOT
            / "RDT_original"
            / "results"
            / "comparison_seviri_2020"
            / "_cache_20200917.npz"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "results" / "ianos_20200917_utls_vs_rdt_original",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "figures"
            / "videomae_pretraining"
            / "ianos_20200917_utls_vs_rdt_original"
        ),
    )
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--reuse-project-cache", action="store_true")
    return parser.parse_args()


def corrected_inventory(source: Path) -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, Any]]:
    with xr.open_zarr(source, consolidated=True, chunks=None) as dataset:
        times = pd.DatetimeIndex(dataset.time.values)
    if len(times) != FRAMES_PER_DAY or times.hasnans or times.has_duplicates:
        raise ValueError("Expected 288 finite, unique nominal timestamps")
    sorted_order = np.argsort(times.asi8)
    sorted_times = times[sorted_order]
    expected = pd.date_range("2020-09-17T00:00:00", periods=FRAMES_PER_DAY, freq="5min")
    if not np.array_equal(sorted_times.asi8, expected.asi8):
        raise ValueError("Nominal timestamps do not form the expected complete UTC day")
    corrected_order = np.concatenate(
        [
            sorted_order[:162],
            sorted_order[181:193],
            sorted_order[162:181],
            sorted_order[193:],
        ]
    ).astype(np.int64)
    diagnostic = radiance_continuity_diagnostic(source, sorted_order, corrected_order)
    if diagnostic["corrected_max_mean_absolute_jump_k"] >= diagnostic["sorted_max_mean_absolute_jump_k"]:
        raise RuntimeError("The documented payload correction did not improve radiance continuity")
    return expected, corrected_order, diagnostic


def radiance_continuity_diagnostic(
    source: Path,
    sorted_order: np.ndarray,
    corrected_order: np.ndarray,
) -> dict[str, Any]:
    array = zarr.open_group(str(source), mode="r")["seviri"]

    def jumps(order: np.ndarray) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        previous = np.asarray(array[int(order[149]), 4], dtype=np.float32)
        for position in range(150, 205):
            current = np.asarray(array[int(order[position]), 4], dtype=np.float32)
            absolute = np.abs(current - previous)
            rows.append(
                {
                    "position": position,
                    "mean_absolute_k": float(np.nanmean(absolute)),
                    "p95_absolute_k": float(np.nanpercentile(absolute, 95.0)),
                }
            )
            previous = current
        return rows

    sorted_jumps = jumps(sorted_order)
    corrected_jumps = jumps(corrected_order)
    return {
        **KNOWN_CORRECTION,
        "diagnostic_interval_utc": ["2020-09-17T12:30:00", "2020-09-17T17:00:00"],
        "sorted_max_mean_absolute_jump_k": max(row["mean_absolute_k"] for row in sorted_jumps),
        "corrected_max_mean_absolute_jump_k": max(
            row["mean_absolute_k"] for row in corrected_jumps
        ),
        "sorted_jumps_above_2k": [
            row for row in sorted_jumps if row["mean_absolute_k"] > 2.0
        ],
        "corrected_jumps_above_2k": [
            row for row in corrected_jumps if row["mean_absolute_k"] > 2.0
        ],
    }


def scene_tropopause_proxy(frame: np.ndarray) -> float:
    """Return the scalar pure-RDT scene proxy used by the historical run."""
    cold = np.isfinite(frame[4]) & (frame[4] < 235.0)
    if int(np.count_nonzero(cold)) < 50:
        return 215.0
    value = 0.6 * np.nanpercentile(frame[0][cold], 10.0) + 0.4 * np.nanpercentile(
        frame[4][cold], 5.0
    )
    return float(np.clip(value, 205.0, 226.0))


def build_project_labels(
    source: Path,
    times: pd.DatetimeIndex,
    order: np.ndarray,
    geometry: dict[str, Any],
    cache_dir: Path,
    *,
    reuse: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels_path = cache_dir / "project_utls_labels.npz"
    metrics_path = cache_dir / "frame_metrics.csv"
    if reuse and labels_path.exists() and metrics_path.exists():
        with np.load(labels_path, allow_pickle=False) as archive:
            labels = np.asarray(archive["semantic_label"], dtype=np.uint8)
        with metrics_path.open("r", encoding="utf-8", newline="") as stream:
            metrics = list(csv.DictReader(stream))
        if labels.shape != (FRAMES_PER_DAY, 763, 1762) or len(metrics) != FRAMES_PER_DAY:
            raise ValueError("Existing project cache is incomplete")
        numeric_fields = {
            "frame",
            "cells_project",
            "proxy_valid_cells",
            "project_ot_pixels",
            "original_ot_pixels",
        }
        float_fields = {"project_median_t_utls_k", "original_scene_tropopause_proxy_k"}
        for row in metrics:
            for key in numeric_fields:
                row[key] = int(row[key])
            for key in float_fields:
                row[key] = float(row[key])
        return labels, metrics

    source_array = zarr.open_group(str(source), mode="r")["seviri"]
    tracker = PureGenUtlsTracker()
    area = np.asarray(geometry["area_km2"], dtype=np.float32)
    latitude = np.asarray(geometry["latitude"], dtype=np.float32)
    longitude = np.asarray(geometry["longitude"], dtype=np.float32)
    x_raw = np.asarray(geometry["source_x"])
    y_raw = np.asarray(geometry["source_y"])
    labels = np.zeros((FRAMES_PER_DAY, area.shape[0], area.shape[1]), dtype=np.uint8)
    metrics: list[dict[str, Any]] = []
    for index, timestamp in enumerate(times):
        raw = np.asarray(source_array[int(order[index])], dtype=np.float32)
        frame, _, _ = _orient(raw, x_raw, y_raw)
        result = tracker.step_frame(
            frame,
            timestamp,
            area,
            lat_deg=latitude,
            lon_deg=longitude,
        )
        labels[index] = result["semantic_label"]
        valid_proxy = [
            row
            for row in result["objects"]
            if row["proxy_reason"] == "valid" and np.isfinite(row["t_utls_proxy_k"])
        ]
        metrics.append(
            {
                "frame": index,
                "time": timestamp.isoformat(),
                "cells_project": len(result["objects"]),
                "proxy_valid_cells": len(valid_proxy),
                "project_median_t_utls_k": (
                    float(np.median([row["t_utls_proxy_k"] for row in valid_proxy]))
                    if valid_proxy
                    else np.nan
                ),
                "project_ot_pixels": int(np.count_nonzero(labels[index] == 6)),
                "original_scene_tropopause_proxy_k": scene_tropopause_proxy(frame),
            }
        )
        if index == 0 or (index + 1) % 24 == 0:
            print(
                f"[project UTLS] {index + 1:03d}/{FRAMES_PER_DAY} "
                f"{timestamp:%H:%M} UTC cells={len(result['objects'])} "
                f"proxy_valid={len(valid_proxy)} OTpx={metrics[-1]['project_ot_pixels']}",
                flush=True,
            )
    temporary = labels_path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, semantic_label=labels)
    temporary.replace(labels_path)
    return labels, metrics


def enrich_original_metrics(metrics: list[dict[str, Any]], original: np.ndarray) -> None:
    for index, row in enumerate(metrics):
        row["original_ot_pixels"] = int(np.count_nonzero(original[index] == 6))


def write_metrics(cache_dir: Path, metrics: list[dict[str, Any]]) -> None:
    path = cache_dir / "frame_metrics.csv"
    temporary = path.with_suffix(".csv.part")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    temporary.replace(path)


def render_video(
    *,
    source: Path,
    times: pd.DatetimeIndex,
    order: np.ndarray,
    labels: np.ndarray,
    metrics: list[dict[str, Any]],
    geometry: dict[str, Any],
    output: Path,
    title: str,
    model_key: str,
    fps: float,
) -> Path:
    map_height, map_width = labels.shape[1:]
    canvas_width = map_width + 24
    canvas_height = map_height + 185
    if canvas_height % 2:
        canvas_height += 1
    static = _static_overlay(geometry, (canvas_width, canvas_height))
    header_font = _font(25, bold=True)
    sub_font = _font(15)
    diag_font = _font(14, bold=True)
    valid = np.asarray(geometry["valid_geolocation"], dtype=bool)
    x_raw = np.asarray(geometry["source_x"])
    y_raw = np.asarray(geometry["source_y"])
    source_array = zarr.open_group(str(source), mode="r")["seviri"]
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{canvas_width}x{canvas_height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-metadata",
        f"title={title}",
        "-metadata",
        "comment=Same corrected 2020-09-17 SEVIRI observations; exact label pixels; no visual dilation",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("Unable to open ffmpeg input pipe")
    preview_index = int(np.argmax([row[f"{model_key}_ot_pixels"] for row in metrics]))
    preview_path = output.with_name(output.stem + "_preview.png")
    try:
        for index, timestamp in enumerate(times):
            raw = np.asarray(source_array[int(order[index])], dtype=np.float32)
            frame, _, _ = _orient(raw, x_raw, y_raw)
            map_rgb = _radiance_composite(frame[4], labels[index], valid)
            canvas = Image.new("RGB", (canvas_width, canvas_height), (4, 11, 22))
            canvas.paste(Image.fromarray(map_rgb, mode="RGB"), (12, 80))
            canvas = Image.alpha_composite(canvas.convert("RGBA"), static)
            draw = ImageDraw.Draw(canvas, "RGBA")
            draw.text(
                (14, 13),
                f"{timestamp:%Y-%m-%d %H:%M} UTC  ·  frame {index + 1}/{FRAMES_PER_DAY}",
                font=header_font,
                fill=(248, 250, 252, 255),
            )
            draw.text(
                (canvas_width - 14, 19),
                title,
                font=sub_font,
                anchor="ra",
                fill=(94, 234, 212, 255),
            )
            if model_key == "project":
                t_value = metrics[index]["project_median_t_utls_k"]
                t_text = f"mediana T_UTLS valida={t_value:.2f} K" if np.isfinite(t_value) else "T_UTLS valida=assente"
                diagnostic = (
                    f"{t_text} · celle proxy valide={metrics[index]['proxy_valid_cells']} · "
                    f"OT={metrics[index]['project_ot_pixels']} px"
                )
            else:
                diagnostic = (
                    f"T_trop scena={metrics[index]['original_scene_tropopause_proxy_k']:.2f} K · "
                    f"OT={metrics[index]['original_ot_pixels']} px · cache originale con parallasse"
                )
            draw.text(
                (14, 51),
                diagnostic,
                font=diag_font,
                fill=(226, 232, 240, 255),
            )
            rgb = canvas.convert("RGB")
            if index == preview_index:
                rgb.save(preview_path)
            process.stdin.write(np.asarray(rgb, dtype=np.uint8).tobytes())
            if index == 0 or (index + 1) % 48 == 0:
                print(f"[render {model_key}] {index + 1:03d}/{FRAMES_PER_DAY}", flush=True)
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return preview_path


def main() -> int:
    args = parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffmpeg and ffprobe are required")
    if not args.source.exists():
        raise FileNotFoundError(args.source)
    if not args.original_cache.exists():
        raise FileNotFoundError(args.original_cache)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    times, order, continuity = corrected_inventory(args.source)
    geometry = _geometry(args.source)
    with np.load(args.original_cache, allow_pickle=False) as archive:
        original = np.asarray(archive["sem_pure"], dtype=np.uint8)
    if original.shape != (FRAMES_PER_DAY, 763, 1762):
        raise ValueError(f"Unexpected original cache shape {original.shape}")

    project, metrics = build_project_labels(
        args.source,
        times,
        order,
        geometry,
        args.cache_dir,
        reuse=args.reuse_project_cache,
    )
    enrich_original_metrics(metrics, original)
    write_metrics(args.cache_dir, metrics)

    project_video = args.output_dir / "ianos_20200917_our_object_local_utls.mp4"
    original_video = args.output_dir / "ianos_20200917_rdt_original_pure_scene_tropopause.mp4"
    project_preview = render_video(
        source=args.source,
        times=times,
        order=order,
        labels=project,
        metrics=metrics,
        geometry=geometry,
        output=project_video,
        title="NOSTRO TEACHER · T_UTLS locale multispettrale · no parallasse",
        model_key="project",
        fps=args.fps,
    )
    original_preview = render_video(
        source=args.source,
        times=times,
        order=order,
        labels=original,
        metrics=metrics,
        geometry=geometry,
        output=original_video,
        title="RDT_original PURO · T_trop proxy globale di scena · parallasse",
        model_key="original",
        fps=args.fps,
    )

    summary = {
        "date": "2020-09-17",
        "phase": "Ianos intense/mature day",
        "frames": FRAMES_PER_DAY,
        "cadence_minutes": 5,
        "same_source_observations": True,
        "same_rendering": True,
        "ot_rendering": "exact label pixels; no dilation, ring or halo",
        "project_teacher": {
            "name": "rdt_cw_inspired_pure_gen_utls_v2",
            "state_initialization": "reset at 00:00 UTC",
            "tropopause_reference": "object-local causal multispectral UTLS/anvil proxy",
            "parallax": False,
            "video": str(project_video.resolve()),
            "preview": str(project_preview.resolve()),
            "total_ot_pixel_times": int(sum(row["project_ot_pixels"] for row in metrics)),
            "frames_with_ot": int(sum(row["project_ot_pixels"] > 0 for row in metrics)),
            "video_metadata": _video_metadata(project_video),
        },
        "rdt_original_pure": {
            "name": "RDT_original pure GEN cached run",
            "state_initialization": "reset at 00:00 UTC",
            "tropopause_reference": "global SEVIRI scene scalar proxy clipped to 205-226 K",
            "parallax": True,
            "video": str(original_video.resolve()),
            "preview": str(original_preview.resolve()),
            "total_ot_pixel_times": int(sum(row["original_ot_pixels"] for row in metrics)),
            "frames_with_ot": int(sum(row["original_ot_pixels"] > 0 for row in metrics)),
            "video_metadata": _video_metadata(original_video),
        },
        "important_non_tropopause_differences": [
            "project cloud type is the later causal IR-only implementation",
            "project OT intersects radial OTD with relative-depth candidates and requires persistence",
            "RDT_original cached labels include its parallax remapping and hole filling",
            "therefore this is an end-to-end comparison, not a one-factor causal ablation of T_UTLS",
        ],
        "payload_order_continuity_check": continuity,
        "source": str(args.source.resolve()),
        "original_cache_read_only": str(args.original_cache.resolve()),
        "metrics_csv": str((args.cache_dir / "frame_metrics.csv").resolve()),
    }
    summary_path = args.output_dir / "ianos_20200917_utls_vs_rdt_original_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
