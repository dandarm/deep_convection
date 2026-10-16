#!/usr/bin/env python3
"""Quantitative comparison between the original NWC SAF GEO-RDT-CW algorithm
and the repository's proxy RDT teachers on identical SEVIRI spatial-temporal domains.

Compares:
1. nwcsaf_rdt_cw_v5.0 (Original EUMETSAT NWC SAF GEO-RDT-CW in RDT_original/rdt_nwcsaf)
2. multispectral_utls_v0 (7-channel UTLS proxy teacher in src/emma_gpm/rdt)
3. refined_v2 (2-channel adaptive contour-tree proxy teacher in src/emma_gpm/rdt_refined.py)
4. legacy_fixed_2ch (2-channel fixed-threshold proxy teacher in src/emma_gpm/rdt_teacher.py)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
RDT_ORIG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(RDT_ORIG_ROOT))

from emma_gpm.rdt import RDTTeacherConfig, build_rdt_teacher_labels
from emma_gpm.rdt_context import context_window
from emma_gpm.rdt_refined import RefinedRDTConfig, label_refined_clip
from emma_gpm.rdt_teacher import RDTConfig, label_rdt_clip, projected_pixel_areas
from emma_gpm.seviri_clips import SeviriZarrClipReader

from rdt_nwcsaf import NwcSafRdtConfig, SEMANTIC_CLASS_NAMES, run_nwcsaf_rdt_sequence


def _binary_metrics(ref: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute POD/Recall, Precision, FAR, CSI/IoU, Dice/F1, and Frequency Bias."""
    ref_b = np.asarray(ref, dtype=bool)
    pred_b = np.asarray(pred, dtype=bool)
    tp = int(np.logical_and(ref_b, pred_b).sum())
    fp = int(np.logical_and(~ref_b, pred_b).sum())
    fn = int(np.logical_and(ref_b, ~pred_b).sum())
    tn = int(np.logical_and(~ref_b, ~pred_b).sum())

    union = tp + fp + fn
    ref_pos = tp + fn
    pred_pos = tp + fp

    iou_csi = float(tp / union) if union > 0 else (1.0 if ref_pos == 0 and pred_pos == 0 else 0.0)
    dice_f1 = (
        float(2 * tp / (2 * tp + fp + fn))
        if (2 * tp + fp + fn) > 0
        else (1.0 if ref_pos == 0 and pred_pos == 0 else 0.0)
    )
    pod_recall = float(tp / ref_pos) if ref_pos > 0 else np.nan
    precision = float(tp / pred_pos) if pred_pos > 0 else np.nan
    far = float(fp / pred_pos) if pred_pos > 0 else np.nan
    freq_bias = float(pred_pos / ref_pos) if ref_pos > 0 else np.nan

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "ref_pixels": ref_pos,
        "pred_pixels": pred_pos,
        "iou_csi": iou_csi,
        "dice_f1": dice_f1,
        "pod_recall": pod_recall,
        "precision": precision,
        "far": far,
        "freq_bias": freq_bias,
    }


def compute_comparison_metrics(
    ref_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> dict[str, Any]:
    """Compute 5-class confusion matrix, per-class IoU/Dice, and physical regime metrics."""
    ref_flat = np.asarray(ref_labels, dtype=np.int64).ravel()
    pred_flat = np.asarray(pred_labels, dtype=np.int64).ravel()
    if ref_flat.shape != pred_flat.shape:
        raise ValueError(f"Shape mismatch: {ref_labels.shape} vs {pred_labels.shape}")

    conf = np.zeros((5, 5), dtype=np.int64)
    np.add.at(conf, (ref_flat, pred_flat), 1)
    overall_acc = float(np.trace(conf) / max(int(conf.sum()), 1))

    per_class: dict[str, dict[str, float]] = {}
    for cls_id, cls_name in SEMANTIC_CLASS_NAMES.items():
        per_class[f"class_{cls_id}_{cls_name}"] = _binary_metrics(
            ref_labels == cls_id, pred_labels == cls_id
        )

    regimes = {
        "cloud_tower_mask_ge1": _binary_metrics(ref_labels >= 1, pred_labels >= 1),
        "convective_mask_234": _binary_metrics(
            np.isin(ref_labels, [2, 3, 4]), np.isin(pred_labels, [2, 3, 4])
        ),
        "developing_convective_eq2": _binary_metrics(ref_labels == 2, pred_labels == 2),
        "mature_decaying_convective_eq3": _binary_metrics(ref_labels == 3, pred_labels == 3),
        "overshooting_top_eq4": _binary_metrics(ref_labels == 4, pred_labels == 4),
    }

    return {
        "overall_pixel_accuracy": overall_acc,
        "confusion_matrix_5x5": conf.tolist(),
        "per_class": per_class,
        "regimes": regimes,
    }


def load_window_with_halo(
    zarr_root: Path,
    start_time: str,
    origin_y: int,
    origin_x: int,
    *,
    warmup_frames: int = 3,
    eval_frames: int = 16,
    crop_size: int = 224,
    halo: int = 64,
) -> dict[str, Any]:
    """Load a (T_total, 7, H_halo, W_halo) SEVIRI window in Kelvin with north-up/west-left orientation."""
    start = pd.Timestamp(start_time)
    context_start = start - pd.Timedelta(minutes=5 * warmup_frames)
    day_path = zarr_root / f"{context_start:%Y}" / f"{context_start:%Y%m%d}.zarr"
    with xr.open_zarr(day_path, consolidated=True, chunks=None) as ds:
        gx = ds.x.values.copy()
        gy = ds.y.values.copy()
        projection = ds.attrs["projection_wkt"]

    yy, xx, size, roi = context_window(
        int(origin_y),
        int(origin_x),
        len(gy),
        len(gx),
        crop=crop_size,
        halo=halo,
        north_reversed=gy[0] < gy[-1],
        west_reversed=gx[0] > gx[-1],
    )
    total_frames = warmup_frames + eval_frames
    with SeviriZarrClipReader(zarr_root, num_frames=total_frames, crop_size=size) as reader:
        raw = reader.load(context_start, origin_y=yy, origin_x=xx)

    x = gx[xx : xx + size]
    y = gy[yy : yy + size]
    x = x if x[0] < x[-1] else x[::-1]
    y = y if y[0] > y[-1] else y[::-1]
    area_km2 = projected_pixel_areas(x, y, projection, allow_invalid=True)

    xx_grid, yy_grid = np.meshgrid(x, y)
    lon_grid, lat_grid = Transformer.from_crs(
        CRS.from_user_input(projection), "EPSG:4326", always_xy=True
    ).transform(xx_grid, yy_grid)

    times = pd.date_range(context_start, periods=total_frames, freq="5min")
    return {
        "raw": raw,
        "times": times,
        "area_km2": area_km2,
        "lat_deg": np.asarray(lat_grid, dtype=np.float32),
        "lon_deg": np.asarray(lon_grid, dtype=np.float32),
        "roi": roi,
        "warmup_frames": warmup_frames,
        "eval_frames": eval_frames,
        "start_time": start.isoformat(),
        "origin_y": int(origin_y),
        "origin_x": int(origin_x),
    }


def evaluate_all_four_algorithms_on_window(win: dict[str, Any]) -> dict[str, Any]:
    """Run Original NWC SAF RDT-CW and the 3 repository proxies on the exact same window."""
    raw = win["raw"]
    times = win["times"]
    area_km2 = win["area_km2"]
    lat_deg = win["lat_deg"]
    lon_deg = win["lon_deg"]
    roi = win["roi"]
    w_frames = win["warmup_frames"]

    # 1. Original NWC SAF GEO-RDT-CW (v5.0)
    t0 = time.perf_counter()
    orig_cfg = NwcSafRdtConfig(enforce_stretched_mask=False)
    orig_out = run_nwcsaf_rdt_sequence(
        raw, times.to_numpy(), area_km2, orig_cfg, lat_deg=lat_deg, lon_deg=lon_deg
    )
    orig_time_s = time.perf_counter() - t0
    orig_sem = orig_out["semantic_label"][w_frames:, roi[0], roi[1]]
    orig_tracks = orig_out["track_id_map"][w_frames:, roi[0], roi[1]]
    orig_base_k = orig_out["base_threshold_map_k"][w_frames:, roi[0], roi[1]]

    # 2. Proxy 1: multispectral_utls_v0 (Main 7-channel teacher in src/emma_gpm/rdt)
    t0 = time.perf_counter()
    utls_cfg = RDTTeacherConfig()
    utls_out = build_rdt_teacher_labels(
        raw, times=times, config=utls_cfg, pixel_area_km2=area_km2
    )
    utls_time_s = time.perf_counter() - t0
    utls_sem = utls_out["semantic_label"][w_frames:, roi[0], roi[1]]
    utls_tracks = utls_out["track_id"][w_frames:, roi[0], roi[1]]
    utls_base_k = utls_out["t_edge_k"][w_frames:, roi[0], roi[1]]

    # 3. Proxy 2: refined_v2 (2-channel adaptive contour-tree teacher in src/emma_gpm/rdt_refined.py)
    t0 = time.perf_counter()
    ref2_cfg = RefinedRDTConfig()
    ref2_out = label_refined_clip(raw, times, area_km2, ref2_cfg)
    ref2_time_s = time.perf_counter() - t0
    ref2_sem = ref2_out["classes"][w_frames:, roi[0], roi[1]]
    ref2_tracks = ref2_out["tracks"][w_frames:, roi[0], roi[1]]
    ref2_base_full = np.where(
        ref2_out["classes"] >= 1, float(ref2_cfg.thresholds_k[0]), np.nan
    ).astype(np.float32)
    ref2_objs = ref2_out["objects"]
    if len(ref2_objs) > 0 and "threshold_k" in ref2_objs.columns:
        for _, orow in ref2_objs.iterrows():
            f_idx = int(orow["frame"])
            t_id = int(orow["track_id"])
            ref2_base_full[f_idx][ref2_out["tracks"][f_idx] == t_id] = float(orow["threshold_k"])
    ref2_base_k = ref2_base_full[w_frames:, roi[0], roi[1]]

    # 4. Proxy 3: legacy_fixed_2ch (2-channel fixed-threshold teacher in src/emma_gpm/rdt_teacher.py)
    t0 = time.perf_counter()
    leg_cfg = RDTConfig()
    leg_out = label_rdt_clip(raw, times, area_km2, leg_cfg)
    leg_time_s = time.perf_counter() - t0
    leg_sem = leg_out["classes"][w_frames:, roi[0], roi[1]]
    leg_tracks = leg_out["tracks"][w_frames:, roi[0], roi[1]]
    leg_base_k = np.where(leg_sem >= 1, float(leg_cfg.thresholds_k[0]), np.nan).astype(np.float32)

    def _summarize_algo(
        sem: np.ndarray,
        tracks: np.ndarray,
        base_k: np.ndarray,
        elapsed_s: float,
    ) -> dict[str, Any]:
        unique_tracks = set(np.unique(tracks)) - {0}
        track_durations = [
            int(np.sum(np.any(tracks == tid, axis=(1, 2)))) for tid in unique_tracks
        ]
        valid_base = base_k[np.isfinite(base_k) & (sem >= 1)]
        return {
            "elapsed_seconds": float(elapsed_s),
            "sec_per_frame": float(elapsed_s / len(raw)),
            "cloud_pixels": int((sem >= 1).sum()),
            "convective_pixels": int(np.isin(sem, [2, 3, 4]).sum()),
            "developing_pixels": int((sem == 2).sum()),
            "mature_decaying_pixels": int((sem == 3).sum()),
            "ot_pixels": int((sem == 4).sum()),
            "unique_tracks_in_roi": len(unique_tracks),
            "mean_track_duration_frames": (
                float(np.mean(track_durations)) if track_durations else 0.0
            ),
            "mean_base_temp_k": float(np.mean(valid_base)) if len(valid_base) else np.nan,
        }

    return {
        "labels": {
            "nwcsaf_rdt_cw_v5.0": orig_sem,
            "multispectral_utls_v0": utls_sem,
            "refined_v2": ref2_sem,
            "legacy_fixed_2ch": leg_sem,
        },
        "base_temps_k": {
            "nwcsaf_rdt_cw_v5.0": orig_base_k,
            "multispectral_utls_v0": utls_base_k,
            "refined_v2": ref2_base_k,
            "legacy_fixed_2ch": leg_base_k,
        },
        "summaries": {
            "nwcsaf_rdt_cw_v5.0": _summarize_algo(orig_sem, orig_tracks, orig_base_k, orig_time_s),
            "multispectral_utls_v0": _summarize_algo(utls_sem, utls_tracks, utls_base_k, utls_time_s),
            "refined_v2": _summarize_algo(ref2_sem, ref2_tracks, ref2_base_k, ref2_time_s),
            "legacy_fixed_2ch": _summarize_algo(leg_sem, leg_tracks, leg_base_k, leg_time_s),
        },
    }


def select_convective_benchmark_windows(
    zarr_root: Path,
    catalog_csv: Path,
    smoke_csv: Path,
    max_windows: int = 8,
) -> list[dict[str, Any]]:
    """Select real SEVIRI benchmark windows spanning active deep convection and mixed cloud scenes."""
    candidates: list[dict[str, Any]] = []
    if smoke_csv.exists():
        df_smoke = pd.read_csv(smoke_csv)
        for _, row in df_smoke.iterrows():
            candidates.append(
                {
                    "clip_id": str(row["clip_id"]),
                    "start_time": str(row["start_time"]),
                    "origin_y": int(row["origin_y"]),
                    "origin_x": int(row["origin_x"]),
                }
            )
    if catalog_csv.exists():
        df_cat = pd.read_csv(catalog_csv)
        for _, row in df_cat.iterrows():
            candidates.append(
                {
                    "clip_id": str(row["clip_id"]),
                    "start_time": str(row["start_time"]),
                    "origin_y": int(row["origin_y"]),
                    "origin_x": int(row["origin_x"]),
                }
            )

    scored: list[tuple[float, dict[str, Any]]] = []
    for cand in candidates:
        try:
            win = load_window_with_halo(
                zarr_root,
                cand["start_time"],
                cand["origin_y"],
                cand["origin_x"],
                warmup_frames=3,
                eval_frames=16,
                crop_size=224,
                halo=64,
            )
            roi = win["roi"]
            ir_roi = win["raw"][3:, 4, roi[0], roi[1]]
            btd_roi = win["raw"][3:, 0, roi[0], roi[1]] - ir_roi
            cold_235_frac = float((ir_roi <= 235.0).mean())
            cold_220_pix = int((ir_roi <= 220.0).sum())
            pos_btd_pix = int((btd_roi >= 0.0).sum())
            min_ir = float(np.nanmin(ir_roi))
            score = cold_220_pix + 5.0 * pos_btd_pix + max(0.0, 250.0 - min_ir) * 50.0
            cand_meta = {
                **cand,
                "min_ir108_k": min_ir,
                "cold_le235k_fraction": cold_235_frac,
                "cold_le220k_pixels": cold_220_pix,
                "pos_btd_pixels": pos_btd_pix,
            }
            scored.append((score, cand_meta))
        except Exception:
            continue

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:max_windows]]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Original NWC SAF GEO-RDT-CW vs Proxy RDT Teachers on real SEVIRI data."
    )
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path(r"E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N"),
    )
    parser.add_argument(
        "--catalog-csv",
        type=Path,
        default=ROOT / "catalogs" / "seviri_pretraining_random_100.csv",
    )
    parser.add_argument(
        "--smoke-csv",
        type=Path,
        default=ROOT / "catalogs" / "seviri_pretraining_smoke.csv",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=6,
        help="Number of 19-frame (3 warmup + 16 eval) SEVIRI windows to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RDT_ORIG_ROOT / "results" / "comparison_seviri_2020",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] Selecting top {args.num_windows} convective SEVIRI windows from {args.zarr_root} ...")
    selected_windows = select_convective_benchmark_windows(
        args.zarr_root, args.catalog_csv, args.smoke_csv, max_windows=args.num_windows
    )
    if not selected_windows:
        raise RuntimeError("No valid SEVIRI windows found in archive.")

    for idx, wmeta in enumerate(selected_windows, 1):
        print(
            f"  Window {idx}: {wmeta['clip_id']} ({wmeta['start_time']}) "
            f"origin=({wmeta['origin_y']},{wmeta['origin_x']}) "
            f"min_IR108={wmeta['min_ir108_k']:.2f}K cold<=220K={wmeta['cold_le220k_pixels']}px"
        )

    all_labels: dict[str, list[np.ndarray]] = {
        "nwcsaf_rdt_cw_v5.0": [],
        "multispectral_utls_v0": [],
        "refined_v2": [],
        "legacy_fixed_2ch": [],
    }
    all_base_k: dict[str, list[np.ndarray]] = {k: [] for k in all_labels}
    per_window_reports: list[dict[str, Any]] = []

    print(f"\n[2/3] Running all 4 algorithms across {len(selected_windows)} windows (19 frames x 352x352 halo -> 16 frames x 224x224 ROI)...")
    for idx, wmeta in enumerate(selected_windows, 1):
        win = load_window_with_halo(
            args.zarr_root,
            wmeta["start_time"],
            wmeta["origin_y"],
            wmeta["origin_x"],
            warmup_frames=3,
            eval_frames=16,
            crop_size=224,
            halo=64,
        )
        res = evaluate_all_four_algorithms_on_window(win)
        for algo_name in all_labels:
            all_labels[algo_name].append(res["labels"][algo_name])
            all_base_k[algo_name].append(res["base_temps_k"][algo_name])

        win_metrics = {
            proxy: compute_comparison_metrics(
                res["labels"]["nwcsaf_rdt_cw_v5.0"], res["labels"][proxy]
            )
            for proxy in ("multispectral_utls_v0", "refined_v2", "legacy_fixed_2ch")
        }
        per_window_reports.append(
            {
                "window": wmeta,
                "summaries": res["summaries"],
                "vs_original_metrics": win_metrics,
            }
        )
        s_orig = res["summaries"]["nwcsaf_rdt_cw_v5.0"]
        s_utls = res["summaries"]["multispectral_utls_v0"]
        s_ref2 = res["summaries"]["refined_v2"]
        print(
            f"  [{idx}/{len(selected_windows)}] {wmeta['clip_id']}: "
            f"Orig(conv={s_orig['convective_pixels']}, OT={s_orig['ot_pixels']}) | "
            f"UTLS_v0(conv={s_utls['convective_pixels']}, OT={s_utls['ot_pixels']}, "
            f"conv_IoU={win_metrics['multispectral_utls_v0']['regimes']['convective_mask_234']['iou_csi']:.3f}) | "
            f"Refined_v2(conv={s_ref2['convective_pixels']}, OT={s_ref2['ot_pixels']}, "
            f"conv_IoU={win_metrics['refined_v2']['regimes']['convective_mask_234']['iou_csi']:.3f})"
        )

    print("\n[3/3] Aggregating pooled voxel-level & object-level fidelity metrics...")
    stacked_labels = {k: np.concatenate(v, axis=0) for k, v in all_labels.items()}
    stacked_base_k = {k: np.concatenate(v, axis=0) for k, v in all_base_k.items()}

    ref_sem = stacked_labels["nwcsaf_rdt_cw_v5.0"]
    ref_base = stacked_base_k["nwcsaf_rdt_cw_v5.0"]

    pooled_vs_original: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for proxy in ("multispectral_utls_v0", "refined_v2", "legacy_fixed_2ch"):
        m = compute_comparison_metrics(ref_sem, stacked_labels[proxy])
        # Base temperature comparison on jointly detected cloud pixels
        joint_cloud = (
            (ref_sem >= 1)
            & (stacked_labels[proxy] >= 1)
            & np.isfinite(ref_base)
            & np.isfinite(stacked_base_k[proxy])
        )
        if joint_cloud.any():
            diff_base = stacked_base_k[proxy][joint_cloud] - ref_base[joint_cloud]
            base_bias_k = float(np.mean(diff_base))
            base_mae_k = float(np.mean(np.abs(diff_base)))
            base_corr = (
                float(np.corrcoef(ref_base[joint_cloud], stacked_base_k[proxy][joint_cloud])[0, 1])
                if int(joint_cloud.sum()) > 2
                else np.nan
            )
        else:
            base_bias_k = np.nan
            base_mae_k = np.nan
            base_corr = np.nan

        m["base_temperature_comparison"] = {
            "joint_cloud_pixels": int(joint_cloud.sum()),
            "bias_proxy_minus_orig_k": base_bias_k,
            "mae_k": base_mae_k,
            "pearson_r": base_corr,
        }
        pooled_vs_original[proxy] = m

        reg = m["regimes"]
        pc = m["per_class"]
        summary_rows.append(
            {
                "proxy_algorithm": proxy,
                "overall_5class_acc": m["overall_pixel_accuracy"],
                "cloud_tower_iou": reg["cloud_tower_mask_ge1"]["iou_csi"],
                "cloud_tower_dice": reg["cloud_tower_mask_ge1"]["dice_f1"],
                "cloud_tower_pod": reg["cloud_tower_mask_ge1"]["pod_recall"],
                "cloud_tower_precision": reg["cloud_tower_mask_ge1"]["precision"],
                "cloud_tower_bias": reg["cloud_tower_mask_ge1"]["freq_bias"],
                "convective_234_iou": reg["convective_mask_234"]["iou_csi"],
                "convective_234_dice": reg["convective_mask_234"]["dice_f1"],
                "convective_234_pod": reg["convective_mask_234"]["pod_recall"],
                "convective_234_precision": reg["convective_mask_234"]["precision"],
                "convective_234_far": reg["convective_mask_234"]["far"],
                "convective_234_bias": reg["convective_mask_234"]["freq_bias"],
                "class1_nonconv_iou": pc["class_1_non_convective_cloud"]["iou_csi"],
                "class2_developing_iou": pc["class_2_convective_triggering_growing"]["iou_csi"],
                "class3_mature_iou": pc["class_3_convective_mature_decaying"]["iou_csi"],
                "class4_ot_iou": pc["class_4_overshooting_top"]["iou_csi"],
                "class4_ot_dice": pc["class_4_overshooting_top"]["dice_f1"],
                "base_temp_bias_k": base_bias_k,
                "base_temp_mae_k": base_mae_k,
                "base_temp_corr": base_corr,
            }
        )

    algo_totals: dict[str, Any] = {}
    for algo_name, sem_arr in stacked_labels.items():
        total_sec = sum(
            w["summaries"][algo_name]["elapsed_seconds"] for w in per_window_reports
        )
        total_frames = len(selected_windows) * 19
        algo_totals[algo_name] = {
            "total_evaluated_voxels": int(sem_arr.size),
            "background_voxels_c0": int((sem_arr == 0).sum()),
            "non_convective_voxels_c1": int((sem_arr == 1).sum()),
            "developing_voxels_c2": int((sem_arr == 2).sum()),
            "mature_decaying_voxels_c3": int((sem_arr == 3).sum()),
            "overshooting_top_voxels_c4": int((sem_arr == 4).sum()),
            "total_cloud_voxels_ge1": int((sem_arr >= 1).sum()),
            "total_convective_voxels_234": int(np.isin(sem_arr, [2, 3, 4]).sum()),
            "mean_sec_per_frame": float(total_sec / max(total_frames, 1)),
            "mean_track_duration_frames": float(
                np.mean([w["summaries"][algo_name]["mean_track_duration_frames"] for w in per_window_reports])
            ),
        }

    report_payload = {
        "zarr_root": str(args.zarr_root),
        "num_windows": len(selected_windows),
        "eval_frames_per_window": 16,
        "warmup_frames_per_window": 3,
        "roi_shape": [224, 224],
        "halo_pixels": 64,
        "algorithm_totals": algo_totals,
        "pooled_vs_original": pooled_vs_original,
        "per_window_reports": per_window_reports,
    }

    json_path = args.output_dir / "comparison_report.json"
    csv_path = args.output_dir / "comparison_summary.csv"
    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    print(f"\nSaved JSON report to: {json_path}")
    print(f"Saved CSV summary to: {csv_path}\n")
    print("=== POOLED COMPARISON SUMMARY (Reference = Original NWC SAF GEO-RDT-CW v5.0) ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
