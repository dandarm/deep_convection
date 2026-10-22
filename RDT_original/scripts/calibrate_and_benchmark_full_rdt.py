#!/usr/bin/env python3
"""End-to-End ERA5 + OPERA Statistical Calibration (CAL) and 5-Way Benchmark for NWC SAF RDT-CW.

Pipeline steps:
1. Selects calibration windows (with cold clouds and OPERA radar coverage) and held-out
   evaluation windows from ``catalogs/seviri_100clips_2020_random_manifest.csv``.
2. Fetches & caches API-free ERA5 reanalysis fields (WMO T_tropo, LI, SHW, KI, CAPE, NWPCONV,
   U700, V700, T_surf) from Google Cloud ARCO-ERA5 Zarr for each window.
3. Matches detected RDT cloud cells against co-located EUMETNET OPERA ``DBZH_QIND`` radar
   composites to fit the Météo-France ``CAL`` logistic regression weights and distill the
   7-channel SEVIRI-only NWP surrogates (saved to ``RDT_original/results/calibrated_rdt_model.json``).
4. Benchmarks all 5 algorithms on the held-out evaluation windows:
   - ``nwcsaf_rdt_full_era5_cal`` (Reference: ERA5 NWP + GEO-CT + GEO-HRW + OPERA-calibrated CAL)
   - ``nwcsaf_rdt_seviri_only_cal`` (Distilled SEVIRI-only RDT: 7-ch SEVIRI surrogates + GEO-CT + HRW + CAL)
   - ``multispectral_utls_v0`` (Repository 7-channel proxy teacher)
   - ``refined_v2`` (Repository 2-channel adaptive proxy teacher)
   - ``legacy_fixed_2ch`` (Repository 2-channel fixed-threshold proxy teacher)
5. Evaluates both:
   - Voxel-level agreement against the full multi-source ``nwcsaf_rdt_full_era5_cal`` reference;
   - Independent ground-truth verification against OPERA radar convective cores (DBZH >= 35 dBZ).
"""

from __future__ import annotations

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
sys.path.insert(0, str(RDT_ORIG_ROOT / "scripts"))

from emma_gpm.rdt import RDTTeacherConfig, build_rdt_teacher_labels
from emma_gpm.rdt_context import context_window
from emma_gpm.rdt_refined import RefinedRDTConfig, label_refined_clip
from emma_gpm.rdt_teacher import RDTConfig, label_rdt_clip, projected_pixel_areas
from emma_gpm.seviri_clips import SeviriZarrClipReader

from rdt_nwcsaf import (
    NwcSafRdtConfig,
    SEMANTIC_CLASS_NAMES,
    build_feature_vector,
    build_seviri_to_opera_lookup,
    classify_vertical_category,
    compute_temporal_predictors,
    fetch_era5_europe_hour,
    fit_calibrated_rdt_model,
    interpolate_era5_to_seviri_grid,
    opera_file_path,
    predict_seviri_only_nwp_surrogates,
    read_opera_h5,
    run_nwcsaf_rdt_sequence,
    select_history_depth_regime,
    verify_cell_with_opera,
)


from compare_original_vs_proxies import select_convective_benchmark_windows


def _binary_metrics(ref: np.ndarray, pred: np.ndarray) -> dict[str, float]:
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
    ref_flat = np.asarray(ref_labels, dtype=np.int64).ravel()
    pred_flat = np.asarray(pred_labels, dtype=np.int64).ravel()
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
            ref_labels >= 2, pred_labels >= 2
        ),
        "developing_convective_eq2": _binary_metrics(ref_labels == 2, pred_labels == 2),
        "mature_decaying_convective_eq3": _binary_metrics(ref_labels == 3, pred_labels == 3),
        "triggering_eq2": _binary_metrics(ref_labels == 2, pred_labels == 2),
        "growing_eq3": _binary_metrics(ref_labels == 3, pred_labels == 3),
        "mature_eq4": _binary_metrics(ref_labels == 4, pred_labels == 4),
        "decaying_eq5": _binary_metrics(ref_labels == 5, pred_labels == 5),
        "overshooting_top_eq6": _binary_metrics(ref_labels == 6, pred_labels == 6),
        "overshooting_top_eq4": _binary_metrics(ref_labels == 4, pred_labels == 4),
    }

    return {
        "overall_pixel_accuracy": overall_acc,
        "confusion_matrix_5x5": conf.tolist(),
        "per_class": per_class,
        "regimes": regimes,
    }


def load_window_with_halo_and_coords(
    zarr_root: Path,
    start_time: str,
    origin_y: int,
    origin_x: int,
    *,
    warmup_frames: int = 3,
    eval_frames: int = 16,
    crop_size: int = 224,
    halo: int = 48,
) -> dict[str, Any]:
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
        "x_m": x,
        "y_m": y,
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


def collect_calibration_samples_for_window(
    win: dict[str, Any],
    era5_cache_dir: Path,
    opera_root: Path,
) -> tuple[
    list[dict[str, object]],
    list[tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray]],
]:
    """Extract cell-level (features, OPERA label) records and pixel-level ERA5 surrogate samples."""
    raw = win["raw"]
    times = win["times"]
    x_m, y_m = win["x_m"], win["y_m"]
    area_km2 = win["area_km2"]
    lat_deg = win["lat_deg"]
    lon_deg = win["lon_deg"]

    # Fetch ERA5 for the window midpoint and interpolate to SEVIRI grid
    mid_ts = times[len(times) // 2]
    era5_eu = fetch_era5_europe_hour(mid_ts, era5_cache_dir)
    era5_grid = interpolate_era5_to_seviri_grid(era5_eu, x_m, y_m)

    # Collect pixel-level surrogate samples (ERA5 T_tropo and NWPCONV from 7-ch SEVIRI)
    mid_frame = raw[len(times) // 2]
    wv062 = mid_frame[0].astype(np.float64)
    wv073 = mid_frame[1].astype(np.float64)
    ir097 = mid_frame[3].astype(np.float64)
    ir108 = mid_frame[4].astype(np.float64)
    ir120 = mid_frame[5].astype(np.float64)
    ir134 = mid_frame[6].astype(np.float64)

    wv62_amb = float(np.nanpercentile(wv062, 25))
    wv73_amb = float(np.nanpercentile(wv073, 25))
    ir97_amb = float(np.nanpercentile(ir097, 25))
    ir134_amb = float(np.nanpercentile(ir134, 25))
    ir108_amb = float(np.nanpercentile(ir108, 75))

    # Subsample every 8th pixel for fast regression
    sub_s = (slice(None, None, 8), slice(None, None, 8))
    n_sub = wv062[sub_s].size
    X_tropo = np.column_stack(
        [
            np.ones(n_sub),
            0.5 * wv062[sub_s].ravel() + 0.5 * wv62_amb,
            0.5 * wv073[sub_s].ravel() + 0.5 * wv73_amb,
            0.5 * ir097[sub_s].ravel() + 0.5 * ir97_amb,
            0.5 * ir134[sub_s].ravel() + 0.5 * ir134_amb,
            np.full(n_sub, ir108_amb),
            np.abs(lat_deg[sub_s].ravel()) / 45.0,
        ]
    )
    y_tropo = era5_grid.t_tropo_k[sub_s].ravel().astype(np.float64)

    X_nwp = np.column_stack(
        [
            np.ones(n_sub),
            (np.maximum(ir108[sub_s].ravel(), ir108_amb) - 273.15) / 20.0,
            (ir108[sub_s].ravel() - ir134[sub_s].ravel()) / 25.0,
            np.clip(ir108[sub_s].ravel() - ir120[sub_s].ravel(), -1.0, 4.0) / 2.0,
            (wv073[sub_s].ravel() - wv062[sub_s].ravel()) / 15.0,
        ]
    )
    y_nwp = era5_grid.nwpconv_mask[sub_s].ravel().astype(np.float64)

    # Run detection & tracking with ERA5 inputs
    cfg = NwcSafRdtConfig(enforce_stretched_mask=False)
    seq_out = run_nwcsaf_rdt_sequence(
        raw,
        times.to_numpy(),
        area_km2,
        cfg,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        tropopause_temp_k=era5_grid.t_tropo_k,
        nwp_instability_mask=era5_grid.nwpconv_mask,
        use_ancillary_ct=True,
        use_hrw_amv=True,
        u700_ms=era5_grid.u700_ms,
        v700_ms=era5_grid.v700_ms,
        t_surf_k=era5_grid.t_surf_k,
    )

    opera_row, opera_col, opera_inside = build_seviri_to_opera_lookup(x_m, y_m)
    opera_cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    records: list[dict[str, object]] = []
    for t_idx, ts in enumerate(times):
        # Match to nearest 15-min OPERA composite
        ts_15 = pd.Timestamp(ts).round("15min")
        if abs((pd.Timestamp(ts) - ts_15).total_seconds()) > 300.0:
            continue
        op_key = ts_15.isoformat()
        if op_key not in opera_cache:
            op_path = opera_file_path(opera_root, ts_15)
            if op_path.exists():
                try:
                    opera_cache[op_key] = read_opera_h5(op_path)
                except Exception:
                    opera_cache[op_key] = None
            else:
                opera_cache[op_key] = None
        op_data = opera_cache[op_key]
        if op_data is None:
            continue
        dbzh, qind = op_data

        f_rec = seq_out["frame_records"][t_idx]
        cell_id_map = f_rec["cell_id_map"]
        for cid, cell in f_rec["cells"].items():
            full_mask = cell_id_map == cid
            ver = verify_cell_with_opera(
                full_mask, opera_row, opera_col, opera_inside, dbzh, qind
            )
            if ver.label not in (0, 1):
                continue
            vcat = int(cell["vertical_cat"])
            d_regime = str(cell["depth_regime"])
            has_hist = 1 if d_regime != "0" else 0
            t_tropo_cell = float(cell.get("tropopause_temp_k", 215.0))
            x_vec = build_feature_vector(cell, cell, d_regime, t_tropo_k=t_tropo_cell)
            records.append(
                {
                    "vertical_cat": vcat,
                    "depth_regime": d_regime,
                    "has_history": has_hist,
                    "features": x_vec.tolist(),
                    "label": int(ver.label),
                    "max_dbzh": ver.max_dbzh,
                    "area35_km2": ver.area35_km2,
                }
            )

    return records, [(X_tropo, y_tropo)], [(X_nwp, y_nwp)]


def verify_predictions_against_opera_on_window(
    win: dict[str, Any],
    algo_sem_maps: dict[str, np.ndarray],
    opera_root: Path,
) -> dict[str, dict[str, int]]:
    """Verify each algorithm's convective predictions against OPERA radar cores (DBZH >= 35 dBZ)."""
    times = win["times"][win["warmup_frames"] :]
    x_m, y_m = win["x_m"], win["y_m"]
    roi = win["roi"]
    opera_row, opera_col, opera_inside = build_seviri_to_opera_lookup(x_m, y_m)
    row_roi = opera_row[roi[0], roi[1]]
    col_roi = opera_col[roi[0], roi[1]]
    in_roi = opera_inside[roi[0], roi[1]]

    # Dilate radar convective core (>= 35 dBZ) by 3 pixels (~9 km) to account for anvil overhang / parallax
    from scipy.ndimage import binary_dilation

    counts = {
        name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "eval_Cloud_objects_tp": 0, "eval_Cloud_objects_fp": 0, "eval_Cloud_objects_fn": 0}
        for name in algo_sem_maps
    }

    for t_idx, ts in enumerate(times):
        ts_15 = pd.Timestamp(ts).round("15min")
        if abs((pd.Timestamp(ts) - ts_15).total_seconds()) > 120.0:
            continue  # Evaluate at exact 15-min OPERA timestamps
        op_path = opera_file_path(opera_root, ts_15)
        if not op_path.exists():
            continue
        dbzh, qind = read_opera_h5(op_path)

        z_map = np.full(row_roi.shape, np.nan, dtype=np.float32)
        q_map = np.full(row_roi.shape, np.nan, dtype=np.float32)
        z_map[in_roi] = dbzh[row_roi[in_roi], col_roi[in_roi]]
        q_map[in_roi] = qind[row_roi[in_roi], col_roi[in_roi]]

        valid_obs = in_roi & (np.isclose(z_map, -32.0) | (np.isfinite(z_map) & (q_map >= 0.20)))
        if int(np.count_nonzero(valid_obs)) < 500:
            continue

        radar_core_35 = valid_obs & (z_map >= 35.0)
        radar_quiet = valid_obs & (z_map < 28.0)

        # Evaluate object-level (cloud tower) and pixel-level convective detection
        from scipy.ndimage import label as nd_label

        for name, sem_3d in algo_sem_maps.items():
            sem = sem_3d[t_idx]
            cloud_mask = (sem >= 1) & valid_obs
            conv_mask = (sem >= 2) & valid_obs

            # Object-level evaluation on detected cloud towers
            lbl, n_obj = nd_label(sem >= 1)
            for oid in range(1, n_obj + 1):
                omask = lbl == oid
                ver = verify_cell_with_opera(omask, row_roi, col_roi, in_roi, dbzh, qind)
                if ver.label not in (0, 1):
                    continue
                pred_conv = bool(np.any(sem[omask] >= 2))
                if ver.label == 1 and pred_conv:
                    counts[name]["tp"] += int(np.count_nonzero(omask & valid_obs))
                    counts[name]["eval_Cloud_objects_tp"] += 1
                elif ver.label == 0 and pred_conv:
                    counts[name]["fp"] += int(np.count_nonzero(omask & valid_obs))
                    counts[name]["eval_Cloud_objects_fp"] += 1
                elif ver.label == 1 and not pred_conv:
                    counts[name]["fn"] += int(np.count_nonzero(omask & valid_obs))
                    counts[name]["eval_Cloud_objects_fn"] += 1
                else:
                    counts[name]["tn"] += int(np.count_nonzero(omask & valid_obs))

    return counts


def main() -> None:
    catalog_csv = ROOT / "catalogs" / "seviri_pretraining_random_100.csv"
    smoke_csv = ROOT / "catalogs" / "seviri_pretraining_smoke.csv"
    zarr_root = Path(r"E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N")
    opera_root = Path(r"E:\Datasets\Deep_convection\raw\opera")
    era5_cache_dir = RDT_ORIG_ROOT / "results" / "era5_cache"
    out_dir = RDT_ORIG_ROOT / "results" / "comparison_seviri_2020"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "benchmark_progress.log"
    log_file.write_text("", encoding="utf-8")

    def log_msg(msg: str) -> None:
        ts_str = pd.Timestamp.now().strftime("%H:%M:%S")
        line = f"[{ts_str}] {msg}"
        print(line, flush=True)
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    win_cache_path = out_dir / "selected_12_windows.json"
    if win_cache_path.exists():
        all_windows = json.loads(win_cache_path.read_text(encoding="utf-8"))
        log_msg(f"Loaded {len(all_windows)} cached convective windows from {win_cache_path.name}")
    else:
        log_msg("=== Selecting Calibration & Held-Out Evaluation Windows ===")
        all_windows = select_convective_benchmark_windows(
            zarr_root, catalog_csv, smoke_csv, max_windows=12
        )
        win_cache_path.write_text(json.dumps(all_windows, indent=2), encoding="utf-8")
        log_msg(f"Saved {len(all_windows)} selected windows to {win_cache_path.name}")

    eval_windows = all_windows[:6]
    train_windows = all_windows[6:12]

    log_msg("=== STEP 1: Collecting ERA5 + OPERA Calibration Dataset ===")
    all_records: list[dict[str, object]] = []
    all_tropo_samples: list[tuple[np.ndarray, np.ndarray]] = []
    all_nwp_samples: list[tuple[np.ndarray, np.ndarray]] = []

    for idx_w, row in enumerate(train_windows, 1):
        cid = str(row["clip_id"])
        log_msg(f"  [Train {idx_w}/{len(train_windows)}] Loading & matching {cid} ({row['start_time']})...")
        win = load_window_with_halo_and_coords(
            zarr_root, str(row["start_time"]), int(row["origin_y"]), int(row["origin_x"])
        )
        recs, t_samp, n_samp = collect_calibration_samples_for_window(
            win, era5_cache_dir, opera_root
        )
        all_records.extend(recs)
        all_tropo_samples.extend(t_samp)
        all_nwp_samples.extend(n_samp)
        log_msg(f"    -> Collected {len(recs)} OPERA-verified cloud cells (cumulative={len(all_records)})")

    log_msg("=== STEP 2: Fitting NWC SAF 'CAL' Logistic Weights & SEVIRI-Only Surrogates ===")
    cal_model = fit_calibrated_rdt_model(all_records, all_tropo_samples, all_nwp_samples)
    model_path = out_dir / "calibrated_rdt_model.json"
    cal_model.to_json(model_path)
    log_msg(f"  Saved calibrated model to {model_path}")
    log_msg(f"  Training summary: {json.dumps(cal_model.training_summary)}")

    log_msg("=== STEP 3: Running 5-Way Benchmark on Held-Out Evaluation Windows ===")
    algo_names = [
        "nwcsaf_rdt_seviri_only_cal",
        "multispectral_utls_v0",
        "refined_v2",
        "legacy_fixed_2ch",
    ]
    ref_all_list: list[np.ndarray] = []
    preds_all_dict: dict[str, list[np.ndarray]] = {k: [] for k in algo_names}
    tropo_errors: list[np.ndarray] = []

    opera_totals = {
        k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "eval_Cloud_objects_tp": 0, "eval_Cloud_objects_fp": 0, "eval_Cloud_objects_fn": 0}
        for k in ["nwcsaf_rdt_full_era5_cal", *algo_names]
    }
    per_window_reports: list[dict[str, Any]] = []

    for idx_e, row in enumerate(eval_windows, 1):
        cid = str(row["clip_id"])
        log_msg(f"  [Eval {idx_e}/{len(eval_windows)}] Window {cid} ({row['start_time']})...")
        win = load_window_with_halo_and_coords(
            zarr_root, str(row["start_time"]), int(row["origin_y"]), int(row["origin_x"])
        )
        raw = win["raw"]
        times = win["times"]
        x_m, y_m = win["x_m"], win["y_m"]
        area_km2 = win["area_km2"]
        lat_deg = win["lat_deg"]
        lon_deg = win["lon_deg"]
        roi = win["roi"]
        w_frames = win["warmup_frames"]

        # Fetch ERA5 for evaluation window
        mid_ts = times[len(times) // 2]
        era5_eu = fetch_era5_europe_hour(mid_ts, era5_cache_dir)
        era5_grid = interpolate_era5_to_seviri_grid(era5_eu, x_m, y_m)

        # 1. Reference: Full Multi-Source NWC SAF RDT-CW (ERA5 NWP + GEO-CT + GEO-HRW + CAL)
        t0 = time.perf_counter()
        full_cfg = NwcSafRdtConfig(
            enforce_stretched_mask=False,
            nwp_conv_filter=True,
            discri_scheme="CAL",
        )
        full_out = run_nwcsaf_rdt_sequence(
            raw,
            times.to_numpy(),
            area_km2,
            full_cfg,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            tropopause_temp_k=era5_grid.t_tropo_k,
            nwp_instability_mask=era5_grid.nwpconv_mask,
            calibrated_model=cal_model,
            use_ancillary_ct=True,
            use_hrw_amv=True,
            u700_ms=era5_grid.u700_ms,
            v700_ms=era5_grid.v700_ms,
            t_surf_k=era5_grid.t_surf_k,
        )
        full_time_s = time.perf_counter() - t0
        full_sem = full_out["semantic_label"][w_frames:, roi[0], roi[1]]

        # 2. Distilled SEVIRI-Only Calibrated RDT-CW (Zero external NWP or radar at runtime!)
        t0 = time.perf_counter()
        mid_f = raw[len(times) // 2]
        ch_dict = {
            "WV_062": mid_f[0],
            "WV_073": mid_f[1],
            "IR_087": mid_f[2],
            "IR_097": mid_f[3],
            "IR_108": mid_f[4],
            "IR_120": mid_f[5],
            "IR_134": mid_f[6],
        }
        sev_tropo_k, sev_nwpconv = predict_seviri_only_nwp_surrogates(
            cal_model, ch_dict, lat_deg=lat_deg
        )
        tropo_errors.append((sev_tropo_k - era5_grid.t_tropo_k)[roi[0], roi[1]].ravel())

        sev_cfg = NwcSafRdtConfig(
            enforce_stretched_mask=False,
            nwp_conv_filter=True,
            discri_scheme="CAL",
        )
        sev_out = run_nwcsaf_rdt_sequence(
            raw,
            times.to_numpy(),
            area_km2,
            sev_cfg,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            tropopause_temp_k=sev_tropo_k,
            nwp_instability_mask=sev_nwpconv,
            calibrated_model=cal_model,
            use_ancillary_ct=True,
            use_hrw_amv=True,
        )
        sev_time_s = time.perf_counter() - t0
        sev_sem = sev_out["semantic_label"][w_frames:, roi[0], roi[1]]

        # 3. Proxy 1: multispectral_utls_v0
        t0 = time.perf_counter()
        utls_out = build_rdt_teacher_labels(
            raw, times=times, config=RDTTeacherConfig(), pixel_area_km2=area_km2
        )
        utls_time_s = time.perf_counter() - t0
        utls_sem = utls_out["semantic_label"][w_frames:, roi[0], roi[1]]

        # 4. Proxy 2: refined_v2
        t0 = time.perf_counter()
        ref2_out = label_refined_clip(raw, times, area_km2, RefinedRDTConfig())
        ref2_time_s = time.perf_counter() - t0
        ref2_sem = ref2_out["classes"][w_frames:, roi[0], roi[1]]

        # 5. Proxy 3: legacy_fixed_2ch
        t0 = time.perf_counter()
        leg_out = label_rdt_clip(raw, times, area_km2, RDTConfig())
        leg_time_s = time.perf_counter() - t0
        leg_sem = leg_out["classes"][w_frames:, roi[0], roi[1]]

        ref_all_list.append(full_sem)
        preds_all_dict["nwcsaf_rdt_seviri_only_cal"].append(sev_sem)
        preds_all_dict["multispectral_utls_v0"].append(utls_sem)
        preds_all_dict["refined_v2"].append(ref2_sem)
        preds_all_dict["legacy_fixed_2ch"].append(leg_sem)

        # Verify all 5 against OPERA radar on this window
        op_win = verify_predictions_against_opera_on_window(
            win,
            {
                "nwcsaf_rdt_full_era5_cal": full_sem,
                "nwcsaf_rdt_seviri_only_cal": sev_sem,
                "multispectral_utls_v0": utls_sem,
                "refined_v2": ref2_sem,
                "legacy_fixed_2ch": leg_sem,
            },
            opera_root,
        )
        for k, v in op_win.items():
            for m_key, m_val in v.items():
                opera_totals[k][m_key] += m_val

        win_metrics = {
            k: compute_comparison_metrics(full_sem, p_list[-1])
            for k, p_list in preds_all_dict.items()
        }
        per_window_reports.append(
            {
                "clip_id": cid,
                "start_time": str(row["start_time"]),
                "era5_tropo_median_k": float(np.nanmedian(era5_grid.t_tropo_k)),
                "seviri_tropo_median_k": float(np.nanmedian(sev_tropo_k)),
                "era5_nwpconv_fraction": float(np.mean(era5_grid.nwpconv_mask)),
                "elapsed_s": {
                    "nwcsaf_rdt_full_era5_cal": round(full_time_s, 3),
                    "nwcsaf_rdt_seviri_only_cal": round(sev_time_s, 3),
                    "multispectral_utls_v0": round(utls_time_s, 3),
                    "refined_v2": round(ref2_time_s, 3),
                    "legacy_fixed_2ch": round(leg_time_s, 3),
                },
                "vs_full_era5_reference": {
                    k: {
                        "overall_acc": m["overall_pixel_accuracy"],
                        "cloud_dice": m["regimes"]["cloud_tower_mask_ge1"]["dice_f1"],
                        "conv_dice": m["regimes"]["convective_mask_234"]["dice_f1"],
                        "conv_iou": m["regimes"]["convective_mask_234"]["iou_csi"],
                        "conv_pod": m["regimes"]["convective_mask_234"]["pod_recall"],
                        "conv_prec": m["regimes"]["convective_mask_234"]["precision"],
                    }
                    for k, m in win_metrics.items()
                },
            }
        )
        log_msg(
            f"    -> SEVIRI-only vs Full-ERA5 RDT: Cloud Dice={win_metrics['nwcsaf_rdt_seviri_only_cal']['regimes']['cloud_tower_mask_ge1']['dice_f1']:.4f}, "
            f"Convective Dice={win_metrics['nwcsaf_rdt_seviri_only_cal']['regimes']['convective_mask_234']['dice_f1']:.4f} | "
            f"UTLS_v0 Conv Dice={win_metrics['multispectral_utls_v0']['regimes']['convective_mask_234']['dice_f1']:.4f}"
        )

    ref_stack = np.concatenate(ref_all_list, axis=0)
    agg_vs_full = {
        k: compute_comparison_metrics(ref_stack, np.concatenate(p_list, axis=0))
        for k, p_list in preds_all_dict.items()
    }

    # Summarize OPERA ground-truth verification metrics
    opera_summary: dict[str, dict[str, float]] = {}
    for k, c in opera_totals.items():
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        otp, ofp, ofn = c["eval_Cloud_objects_tp"], c["eval_Cloud_objects_fp"], c["eval_Cloud_objects_fn"]
        opera_summary[k] = {
            "pixel_pod_recall": float(tp / max(tp + fn, 1)),
            "pixel_precision": float(tp / max(tp + fp, 1)),
            "pixel_far": float(fp / max(tp + fp, 1)),
            "pixel_csi_iou": float(tp / max(tp + fp + fn, 1)),
            "pixel_dice_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "cell_pod_recall": float(otp / max(otp + ofn, 1)),
            "cell_precision": float(otp / max(otp + ofp, 1)),
            "cell_csi_iou": float(otp / max(otp + ofp + ofn, 1)),
            "cell_dice_f1": float(2 * otp / max(2 * otp + ofp + ofn, 1)),
            "verified_convective_cells_tp": otp,
            "verified_false_alarm_cells_fp": ofp,
            "verified_missed_cells_fn": ofn,
        }

    all_tropo_err = np.concatenate(tropo_errors)
    tropo_mae_k = float(np.nanmean(np.abs(all_tropo_err)))
    tropo_rmse_k = float(np.sqrt(np.nanmean(all_tropo_err ** 2)))

    final_report = {
        "calibration_summary": cal_model.training_summary,
        "seviri_tropopause_surrogate_accuracy": {
            "mae_vs_era5_wmo_k": round(tropo_mae_k, 3),
            "rmse_vs_era5_wmo_k": round(tropo_rmse_k, 3),
        },
        "aggregated_vs_full_era5_rdt_reference": agg_vs_full,
        "independent_opera_radar_ground_truth_verification": opera_summary,
        "per_window_reports": per_window_reports,
    }

    report_path = out_dir / "full_calibrated_benchmark_report.json"
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    log_msg(f"Saved full benchmark report to {report_path}")

    log_msg("================ SUMMARY: FIDELITY TO FULL MULTI-SOURCE RDT (ERA5+CT+HRW+CAL) ================")
    for k, m in agg_vs_full.items():
        c_dice = m["regimes"]["cloud_tower_mask_ge1"]["dice_f1"]
        conv = m["regimes"]["convective_mask_234"]
        log_msg(
            f"  {k:28s} | Overall Acc: {m['overall_pixel_accuracy']*100:5.2f}% | "
            f"Cloud Dice: {c_dice:.4f} | Conv Dice: {conv['dice_f1']:.4f} "
            f"(POD={conv['pod_recall']*100:5.1f}%, Prec={conv['precision']*100:5.1f}%)"
        )

    log_msg("================ SUMMARY: INDEPENDENT OPERA RADAR GROUND-TRUTH VERIFICATION ================")
    for k, om in opera_summary.items():
        log_msg(
            f"  {k:28s} | Cell F1/Dice: {om['cell_dice_f1']:.4f} (POD={om['cell_pod_recall']*100:5.1f}%, Prec={om['cell_precision']*100:5.1f}%) | "
            f"Pixel F1/Dice: {om['pixel_dice_f1']:.4f} (CSI={om['pixel_csi_iou']:.4f})"
        )


if __name__ == "__main__":
    main()
