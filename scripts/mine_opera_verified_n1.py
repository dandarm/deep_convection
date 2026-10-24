#!/usr/bin/env python3
"""Mine cold, EMMA-empty clouds and verify them against OPERA.

The OPERA seasonal screen only chooses promising times and places.  This
second-stage miner works on the actual SEVIRI cold-cloud component, requires
the component to fit inside a 224x224 crop, rejects any crop close to an EMMA
MCS, and computes radar coverage/echo metrics on the component plus a small
spatial buffer.  It also stores simple 5-minute temporal descriptors for
later human review; those descriptors do not define N1 by themselves.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import binary_dilation, label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_emma_heatmap_examples import load_ir108, utc_naive  # noqa: E402
from plot_emma_negative_examples import all_mcs_mask  # noqa: E402
from validate_emma_negatives_opera import (  # noqa: E402
    COLD_THRESHOLD_K,
    OFFSETS_MINUTES,
    OPERA_CRS,
    component_metrics,
    download_opera,
    opera_indices_for_crop,
    read_opera,
)
from emma_gpm.seviri import RSS_CRS  # noqa: E402


CROP = 224


def slope(offsets: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return np.nan
    return float(np.polyfit(offsets[finite], values[finite], 1)[0])


def temporal_metrics(
    frames: list[np.ndarray],
    offsets: np.ndarray,
    origin_y: int,
    origin_x: int,
    reference: np.ndarray,
    threshold_k: float,
) -> dict[str, float | int]:
    search = binary_dilation(reference, iterations=16)
    areas: list[float] = []
    minima: list[float] = []
    p10s: list[float] = []
    matched: list[int] = []
    for frame in frames:
        crop = frame[origin_y : origin_y + CROP, origin_x : origin_x + CROP]
        components, count = label(
            np.isfinite(crop) & (crop < threshold_k),
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        best_id = 0
        best_overlap = 0
        for component_id in range(1, count + 1):
            overlap = int(np.count_nonzero((components == component_id) & search))
            if overlap > best_overlap:
                best_id, best_overlap = component_id, overlap
        if best_id == 0:
            areas.append(np.nan)
            minima.append(np.nan)
            p10s.append(np.nan)
            matched.append(0)
            continue
        mask = components == best_id
        values = crop[mask]
        areas.append(float(mask.sum()))
        minima.append(float(np.nanmin(values)))
        p10s.append(float(np.nanquantile(values, 0.10)))
        matched.append(1)

    area_values = np.asarray(areas, dtype=float)
    minimum_values = np.asarray(minima, dtype=float)
    p10_values = np.asarray(p10s, dtype=float)
    pre = offsets <= 0
    return {
        "video_matched_frames": int(sum(matched)),
        "video_area_min_pixels": float(np.nanmin(area_values)),
        "video_area_max_pixels": float(np.nanmax(area_values)),
        "video_area_slope_px_per_min": slope(offsets, area_values),
        "video_precentral_area_slope_px_per_min": slope(offsets[pre], area_values[pre]),
        "video_min_bt_min_k": float(np.nanmin(minimum_values)),
        "video_min_bt_slope_k_per_min": slope(offsets, minimum_values),
        "video_precentral_min_bt_slope_k_per_min": slope(
            offsets[pre], minimum_values[pre]
        ),
        "video_p10_bt_slope_k_per_min": slope(offsets, p10_values),
    }


def radar_class(metrics: dict[str, float | int]) -> str:
    observable = (
        metrics["opera_inside_fraction"] >= 0.80
        and metrics["opera_valid_fraction_median"] >= 0.70
        and metrics["opera_good_fraction_median"] >= 0.05
    )
    if not observable:
        return "NU"
    convective = (
        metrics["opera_max_area40_km2_5slots"] >= 16.0
        and metrics["opera_slots_area35_ge_16km2"] >= 2
    )
    if convective:
        return "N2_candidate"
    quiet = (
        metrics["opera_max_dbzh_5slots"] < 30.0
        and metrics["opera_max_area35_km2_5slots"] == 0.0
    )
    return "N1_candidate" if quiet else "NU"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--emma-root", type=Path, required=True)
    parser.add_argument("--opera-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold-k", type=float, default=COLD_THRESHOLD_K)
    parser.add_argument("--shield-threshold-k", type=float, default=253.0)
    parser.add_argument("--emma-buffer-pixels", type=int, default=16)
    parser.add_argument("--radar-buffer-pixels", type=int, default=4)
    parser.add_argument("--component-margin-pixels", type=int, default=16)
    parser.add_argument("--min-component-pixels", type=int, default=24)
    parser.add_argument("--min-shield-pixels", type=int, default=512)
    parser.add_argument("--per-scene", type=int, default=10)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.limit is not None:
        manifest = manifest.head(args.limit)
    datasets: dict[Path, object] = {}
    emma_lookup: dict[object, object] = {}
    rss_to_opera = Transformer.from_crs(RSS_CRS, OPERA_CRS, always_xy=True)
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for number, source in enumerate(manifest.itertuples(index=False), start=1):
        center = utc_naive(source.central_time)
        target = utc_naive(source.emma_target_time)
        offsets = np.arange(-30, 31, 5, dtype=int)
        timestamps = [center + timedelta(minutes=int(value)) for value in offsets]
        try:
            loaded = [load_ir108(args.seviri_root, time, datasets) for time in timestamps]
            frames = [item[0] for item in loaded]
            x, y = loaded[0][1], loaded[0][2]
            central = frames[len(frames) // 2]
            mcs = all_mcs_mask(args.emma_root, target, x, y, emma_lookup)
            blocked = binary_dilation(mcs, iterations=args.emma_buffer_pixels)
            opera_frames = [
                read_opera(download_opera(center + timedelta(minutes=int(value)), args.opera_cache))
                for value in OFFSETS_MINUTES
            ]
        except Exception as exc:
            skipped.append(
                {"central_time": center.isoformat(), "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue

        shield_labels, count = label(
            np.isfinite(central) & (central < args.shield_threshold_k),
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        shield_sizes = np.bincount(shield_labels.ravel())
        eligible_shield_ids = np.flatnonzero(
            shield_sizes >= args.min_shield_pixels
        )
        eligible_shield_ids = eligible_shield_ids[eligible_shield_ids != 0]
        scene_candidates: list[dict[str, object]] = []
        for shield_id in eligible_shield_ids:
            shield = shield_labels == shield_id
            shield_pixels = int(shield_sizes[shield_id])
            cold_core = shield & (central < args.threshold_k)
            cold_pixels = int(cold_core.sum())
            if cold_pixels < args.min_component_pixels:
                continue
            yy, xx = np.where(shield)
            bbox_h = int(yy.max() - yy.min() + 1)
            bbox_w = int(xx.max() - xx.min() + 1)
            maximum_extent = CROP - 2 * args.component_margin_pixels
            if bbox_h > maximum_extent or bbox_w > maximum_extent:
                continue
            center_y = int(round((int(yy.min()) + int(yy.max())) / 2))
            center_x = int(round((int(xx.min()) + int(xx.max())) / 2))
            y0 = int(np.clip(center_y - CROP // 2, 0, central.shape[0] - CROP))
            x0 = int(np.clip(center_x - CROP // 2, 0, central.shape[1] - CROP))
            crop = central[y0 : y0 + CROP, x0 : x0 + CROP]
            if np.isfinite(crop).mean() < 0.98:
                continue
            # A negative crop must not contain an EMMA MCS or its 48-km buffer.
            if blocked[y0 : y0 + CROP, x0 : x0 + CROP].any():
                continue
            shield_crop = shield[y0 : y0 + CROP, x0 : x0 + CROP]
            cold_crop = cold_core[y0 : y0 + CROP, x0 : x0 + CROP]
            local_y, local_x = np.where(shield_crop)
            if (
                local_y.min() < args.component_margin_pixels
                or local_y.max() >= CROP - args.component_margin_pixels
                or local_x.min() < args.component_margin_pixels
                or local_x.max() >= CROP - args.component_margin_pixels
            ):
                continue
            radar_footprint = binary_dilation(
                shield_crop, iterations=args.radar_buffer_pixels
            )
            opera_row, opera_col, inside = opera_indices_for_crop(
                x, y, x0, y0, rss_to_opera
            )
            metrics = component_metrics(
                radar_footprint,
                opera_row,
                opera_col,
                inside,
                opera_frames,
                qind_threshold=0.20,
            )
            classification = radar_class(metrics)
            video = temporal_metrics(
                frames, offsets.astype(float), y0, x0, cold_crop, args.threshold_k
            )
            scene_candidates.append(
                {
                    "central_time": center,
                    "emma_target_time": target,
                    "origin_y": y0,
                    "origin_x": x0,
                    "shield_id": shield_id,
                    "class": classification,
                    "central_cold_component_pixels": cold_pixels,
                    "central_shield_pixels": shield_pixels,
                    "cold_fraction_lt241k": float(np.mean(crop < args.threshold_k)),
                    "ir108_min_k": float(np.nanmin(crop[cold_crop])),
                    "ir108_p10_k": float(np.nanquantile(crop[cold_crop], 0.10)),
                    "shield_bbox_height": bbox_h,
                    "shield_bbox_width": bbox_w,
                    "crop_valid_fraction": float(np.isfinite(crop).mean()),
                    **metrics,
                    **video,
                }
            )
        scene_candidates.sort(
            key=lambda item: (
                item["class"] == "N1_candidate",
                item["central_shield_pixels"],
            ),
            reverse=True,
        )
        rows.extend(scene_candidates[: args.per_scene])
        print(
            f"scene={number}/{len(manifest)} time={center.isoformat()} "
            f"cold_components={count} retained={min(len(scene_candidates), args.per_scene)}",
            flush=True,
        )

    for dataset in datasets.values():
        dataset.close()
    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(
            args.out.with_name(args.out.stem + "_skipped.csv"), index=False
        )
    print(
        f"candidates={len(result)} classes={result['class'].value_counts().to_dict() if len(result) else {}} "
        f"skipped_scenes={len(skipped)} out={args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
