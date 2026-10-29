#!/usr/bin/env python3
"""Benchmark and evaluation of NWC SAF Pure-SEVIRI Convection Initiation (GEO-CI)
on real 7-channel SEVIRI satellite data and comparison with RDT convective onset.

Demonstrates:
1. Autonomous execution of Pure-SEVIRI GEO-CI on real SEVIRI Zarr observations.
2. Verification of forecast lead times (ci_prob30, ci_prob60, ci_prob90) and monotonicity.
3. Temporal lead-time analysis: verifying that CI provides early warning BEFORE RDT matures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xarray as xr

# Add project root and local directories to path
REPO_ROOT = Path(__file__).resolve().parents[2]
CI_ORIG_ROOT = Path(__file__).resolve().parents[1]
RDT_ORIG_ROOT = REPO_ROOT / "RDT_original"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(CI_ORIG_ROOT))
if RDT_ORIG_ROOT.exists():
    sys.path.insert(0, str(RDT_ORIG_ROOT))

from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.seviri_clips import SeviriZarrClipReader
from emma_gpm.rdt import RDTTeacherConfig, build_rdt_teacher_labels

from ci_nwcsaf import (
    NwcSafCiConfig,
    NwcSafCiTracker,
    CiProbabilityClass,
    run_nwcsaf_ci_sequence,
    SEVIRI_CHANNELS,
)


def load_real_seviri_window(
    zarr_root: Path,
    start_time: str,
    origin_y: int,
    origin_x: int,
    num_frames: int = 16,
    crop_size: int = 224,
) -> Dict[str, Any]:
    """Loads a spatio-temporal cube directly from the SEVIRI Zarr store."""
    reader = SeviriZarrClipReader(
        zarr_root,
        num_frames=num_frames,
        cadence_minutes=5,
        crop_size=crop_size,
        require_finite=False,
    )
    start_ts = pd.Timestamp(start_time)
    cube_raw = reader.load(
        start_ts,
        origin_y=origin_y,
        origin_x=origin_x,
        global_mean=None,
        global_std=None,
    )
    times = pd.date_range(start_ts, periods=num_frames, freq="5min")

    return {
        "cube": cube_raw,  # shape (T, 7, H, W) in Kelvin
        "times": times,
        "origin_y": origin_y,
        "origin_x": origin_x,
    }


def evaluate_ci_and_rdt_on_window(
    seviri_cube: np.ndarray,
    times: pd.DatetimeIndex,
    ci_config: Optional[NwcSafCiConfig] = None,
) -> Dict[str, Any]:
    """Runs Pure-SEVIRI CI and RDT on the same window and evaluates lead-time statistics."""
    if ci_config is None:
        ci_config = NwcSafCiConfig(mode="night_or_pure_ir")

    t_len, c_len, h, w = seviri_cube.shape
    tracker = NwcSafCiTracker(config=ci_config)

    ci_prob30_stack = np.zeros((t_len, h, w), dtype=np.uint8)
    ci_prob60_stack = np.zeros((t_len, h, w), dtype=np.uint8)
    ci_prob90_stack = np.zeros((t_len, h, w), dtype=np.uint8)
    eligible_stack = np.zeros((t_len, h, w), dtype=bool)
    pre_ci_stack = np.zeros((t_len, h, w), dtype=bool)

    t_start = time.perf_counter()
    step_results = []
    for t_idx in range(t_len):
        frame_cube = seviri_cube[t_idx]
        stamp = times[t_idx]
        res = tracker.step(frame_cube, timestamp=stamp)
        step_results.append(res)
        ci_prob30_stack[t_idx] = res.ci_prob30
        ci_prob60_stack[t_idx] = res.ci_prob60
        ci_prob90_stack[t_idx] = res.ci_prob90
        eligible_stack[t_idx] = res.eligible_mask
        pre_ci_stack[t_idx] = res.pre_ci_mask
    ci_runtime_s = time.perf_counter() - t_start

    # Run RDT teacher to obtain ground-truth convective cell development
    t_start = time.perf_counter()
    rdt_cfg = RDTTeacherConfig()
    # pixel area nominal 9 km²
    pixel_area_km2 = np.full((h, w), ci_config.pixel_size_km ** 2, dtype=np.float32)
    rdt_out = build_rdt_teacher_labels(
        seviri_cube, times=times, config=rdt_cfg, pixel_area_km2=pixel_area_km2
    )
    rdt_runtime_s = time.perf_counter() - t_start

    rdt_semantic = rdt_out["semantic_label"]  # 0: bg, 1: non-conv, 2: dev, 3: mature, 4: OT
    rdt_convective = (rdt_semantic >= 2)

    # Lead-time verification: check monotonicity P30 <= P60 <= P90
    monotonicity_violations = int(np.sum((ci_prob30_stack > ci_prob60_stack) | (ci_prob60_stack > ci_prob90_stack)))

    # Compute lead-time relationship:
    # For every pixel where RDT first becomes convective at time t_rdt (t_rdt >= 2),
    # did CI show positive initiation probability at t_rdt - 15m or t_rdt - 30m?
    rdt_first_onset = np.zeros((h, w), dtype=int)
    ci_lead_hits_30m = 0
    ci_lead_hits_60m = 0
    total_convective_onsets = 0

    for r in range(h):
        for c in range(w):
            conv_frames = np.where(rdt_convective[:, r, c])[0]
            if len(conv_frames) > 0:
                first_f = int(conv_frames[0])
                # Only evaluate if cell initiated inside the observation window (first_f >= 2)
                if first_f >= 2:
                    total_convective_onsets += 1
                    # Check 15m prior (first_f - 3 frames at 5m RSS)
                    prior_f_15m = max(0, first_f - 3)
                    prior_f_30m = max(0, first_f - 6)

                    # Did CI flag initiation in prior frames?
                    if np.any(ci_prob30_stack[prior_f_15m:first_f, r, c] > 0) or np.any(ci_prob60_stack[prior_f_15m:first_f, r, c] > 0):
                        ci_lead_hits_30m += 1
                    if np.any(ci_prob60_stack[prior_f_30m:first_f, r, c] > 0) or np.any(ci_prob90_stack[prior_f_30m:first_f, r, c] > 0):
                        ci_lead_hits_60m += 1

    total_voxels = t_len * h * w
    ci_pos_30 = int(np.sum(ci_prob30_stack > 0))
    ci_pos_60 = int(np.sum(ci_prob60_stack > 0))
    ci_pos_90 = int(np.sum(ci_prob90_stack > 0))
    rdt_conv_voxels = int(np.sum(rdt_convective))

    return {
        "total_voxels": total_voxels,
        "ci_runtime_s": ci_runtime_s,
        "rdt_runtime_s": rdt_runtime_s,
        "monotonicity_violations": monotonicity_violations,
        "ci_prob30_positive_voxels": ci_pos_30,
        "ci_prob30_pct": float(ci_pos_30 / total_voxels * 100.0),
        "ci_prob60_positive_voxels": ci_pos_60,
        "ci_prob60_pct": float(ci_pos_60 / total_voxels * 100.0),
        "ci_prob90_positive_voxels": ci_pos_90,
        "ci_prob90_pct": float(ci_pos_90 / total_voxels * 100.0),
        "rdt_convective_voxels": rdt_conv_voxels,
        "rdt_convective_pct": float(rdt_conv_voxels / total_voxels * 100.0),
        "total_convective_onsets": total_convective_onsets,
        "ci_lead_hits_30m": ci_lead_hits_30m,
        "ci_lead_pod_30m": float(ci_lead_hits_30m / max(1, total_convective_onsets)),
        "ci_lead_hits_60m": ci_lead_hits_60m,
        "ci_lead_pod_60m": float(ci_lead_hits_60m / max(1, total_convective_onsets)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run NWC SAF Pure-SEVIRI GEO-CI Benchmark on Real SEVIRI Data."
    )
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path(r"E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N"),
        help="Path to SEVIRI 7-channel Zarr dataset",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "catalogs" / "seviri_pretraining_random_100.csv",
        help="Manifest with sample windows",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=3,
        help="Number of windows to evaluate",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=CI_ORIG_ROOT / "results" / "ci_seviri_benchmark_report.json",
        help="Output path for JSON summary report",
    )
    args = parser.parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  NWC SAF PURE-SEVIRI GEO-CI CONVECTION INITIATION BENCHMARK")
    print("=" * 78)
    print(f"SEVIRI Zarr Root: {args.zarr_root}")
    print(f"Manifest:         {args.manifest}")
    print(f"Num Windows:      {args.num_windows}\n")

    if not args.zarr_root.exists():
        print(f"Warning: Zarr root {args.zarr_root} not found. Running on synthetic validation scenes.")
        # Synthetic run
        from CI_original.tests.test_ci_nwcsaf import create_synthetic_seviri_cube
        cube = np.stack([create_synthetic_seviri_cube(h=128, w=128) for _ in range(8)], axis=0)
        times = pd.date_range("2020-09-15 12:00", periods=8, freq="5min")
        res = evaluate_ci_and_rdt_on_window(cube, times)
        results = [res]
    else:
        df = pd.read_csv(args.manifest)
        results = []
        for idx in range(min(args.num_windows, len(df))):
            row = df.iloc[idx]
            clip_id = row.get("clip_id", f"clip_{idx}")
            start_time = row["start_time"]
            origin_y = int(row["origin_y"])
            origin_x = int(row["origin_x"])

            print(f"[{idx+1}/{args.num_windows}] Processing {clip_id} ({start_time}) origin=({origin_y}, {origin_x}) ...")
            try:
                win = load_real_seviri_window(
                    args.zarr_root,
                    start_time=start_time,
                    origin_y=origin_y,
                    origin_x=origin_x,
                    num_frames=16,
                    crop_size=224,
                )
                res = evaluate_ci_and_rdt_on_window(win["cube"], win["times"])
                res["clip_id"] = clip_id
                res["start_time"] = str(start_time)
                results.append(res)

                print(
                    f"   Done in {res['ci_runtime_s']:.2f}s | "
                    f"CI prob30: {res['ci_prob30_pct']:.2f}% | "
                    f"prob60: {res['ci_prob60_pct']:.2f}% | "
                    f"prob90: {res['ci_prob90_pct']:.2f}% | "
                    f"Monotonicity violations: {res['monotonicity_violations']}"
                )
                if res['total_convective_onsets'] > 0:
                    print(
                        f"   Convective Onsets: {res['total_convective_onsets']} | "
                        f"CI Lead POD 30m: {res['ci_lead_pod_30m']*100.0:.1f}% | "
                        f"CI Lead POD 60m: {res['ci_lead_pod_60m']*100.0:.1f}%"
                    )
            except Exception as e:
                print(f"   Error processing window {idx}: {e}")

    # Summary report
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Benchmark completed. Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
