#!/usr/bin/env python3
"""Evaluate Pure Uncalibrated SEVIRI RDT (GEN) and diagnose OPERA Radar Truth across thresholds & scales.

Answers two key scientific questions:
1. How does a truly *Pure* SEVIRI-only RDT (``nwcsaf_rdt_seviri_pure_gen``) — using ZERO
   OPERA-calibrated weights (``discri_scheme="GEN"``), ZERO ERA5 NWP fields or ERA5-fitted
   surrogates, ZERO external NWC SAF ancillary products (``use_ancillary_ct=False``,
   ``use_hrw_amv=False``), and ZERO lightning data — perform compared to:
   - ``nwcsaf_rdt_full_era5_cal`` (Full ERA5 + GEO-CT/HRW + OPERA CAL weights)
   - ``nwcsaf_rdt_seviri_only_cal`` (7-ch SEVIRI + OPERA CAL weights + ERA5-distilled surrogates)
   - Repository proxy teachers (``multispectral_utls_v0``, ``refined_v2``, ``legacy_fixed_2ch``)

2. Why are object-level Precision & Recall against instantaneous 35 dBZ OPERA radar moderate
   (~32-45%), and how does verification change when:
   - Varying the OPERA reflectivity threshold: 35 dBZ vs 40 dBZ vs 45 dBZ vs 50 dBZ;
   - Accounting for physical cloud-top vs precipitation time lag (+/- 15 min window);
   - Stratifying by cell size (small single-cell < 500 km2 vs mesoscale >= 500 km2 vs >= 2000 km2);
   - Stratifying by RDT lifecycle phase (Growing/Triggering Class 2 vs Mature/Decaying Anvil Class 3).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import label as nd_label

ROOT = Path(__file__).resolve().parents[2]
RDT_ORIG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(RDT_ORIG_ROOT))
sys.path.insert(0, str(RDT_ORIG_ROOT / "scripts"))

from emma_gpm.rdt import RDTTeacherConfig, build_rdt_teacher_labels
from emma_gpm.rdt_refined import RefinedRDTConfig, label_refined_clip
from emma_gpm.rdt_teacher import RDTConfig, label_rdt_clip

from rdt_nwcsaf import (
    CalibratedRdtModel,
    NwcSafRdtConfig,
    build_seviri_to_opera_lookup,
    fetch_era5_europe_hour,
    interpolate_era5_to_seviri_grid,
    opera_file_path,
    predict_seviri_only_nwp_surrogates,
    read_opera_h5,
    run_nwcsaf_rdt_sequence,
)
from calibrate_and_benchmark_full_rdt import (
    _binary_metrics,
    compute_comparison_metrics,
)
from emma_gpm.rdt_context import context_window
from emma_gpm.rdt_teacher import projected_pixel_areas
from pyproj import CRS, Transformer
import xarray as xr


def fast_load_window_with_halo(
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
    with xr.open_zarr(day_path, consolidated=True) as ds:
        gx = ds.x.values.copy()
        gy = ds.y.values.copy()
        projection = ds.attrs["projection_wkt"]
        t_all = pd.to_datetime(ds.time.values, utc=True)

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
        times = pd.date_range(context_start, periods=total_frames, freq="5min")
        t_map = {ts: idx for idx, ts in enumerate(t_all)}
        t_indices = [t_map[ts] for ts in times]

        flip_y = bool(gy[0] < gy[-1])
        flip_x = bool(gx[0] > gx[-1])
        y_raw = slice(yy, yy + size)
        x_raw = slice(xx, xx + size)

        raw = np.asarray(ds["seviri"].isel(time=t_indices, y=y_raw, x=x_raw).values, dtype=np.float32)
        x = gx[xx : xx + size]
        y = gy[yy : yy + size]
        if flip_y:
            raw = raw[:, :, ::-1, :]
            y = y[::-1]
        if flip_x:
            raw = raw[:, :, :, ::-1]
            x = x[::-1]

    area_km2 = projected_pixel_areas(x, y, projection, allow_invalid=True)
    xx_grid, yy_grid = np.meshgrid(x, y)
    lon_grid, lat_grid = Transformer.from_crs(
        CRS.from_user_input(projection), "EPSG:4326", always_xy=True
    ).transform(xx_grid, yy_grid)
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
    }


def run_pure_uncalibrated_seviri_rdt(
    raw: np.ndarray,
    times: np.ndarray,
    area_km2: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
) -> dict[str, Any]:
    """Run 100% Pure Uncalibrated SEVIRI RDT (no CAL weights, no ERA5, no CT/HRW, no lightning).

    Uses only the uncalibrated NWC SAF ATBD Generic ('GEN') physical discrimination rules
    and a purely physical radiative tropopause estimate from the 5th percentile of opaque cold
    cloud tops in WV_062 and IR_108.
    """
    mid_f = raw[len(times) // 2]
    wv062 = mid_f[0].astype(np.float64)
    ir108 = mid_f[4].astype(np.float64)
    ir134 = mid_f[6].astype(np.float64)

    # Purely physical radiative tropopause estimate (no regression weights):
    # Coldest 5th percentile of upper-tropospheric WV_062 / IR_108, bounded to [205, 228] K
    cold_mask = np.isfinite(ir108) & (ir108 < 235.0)
    if int(np.count_nonzero(cold_mask)) >= 50:
        t_tropo_val = float(
            0.6 * np.nanpercentile(wv062[cold_mask], 10)
            + 0.4 * np.nanpercentile(ir108[cold_mask], 5)
        )
    else:
        t_tropo_val = float(np.nanpercentile(wv062, 5))
    t_tropo_val = float(np.clip(t_tropo_val, 205.0, 226.0))
    pure_tropo_k = np.full(ir108.shape, t_tropo_val, dtype=np.float32)

    pure_cfg = NwcSafRdtConfig(
        enforce_stretched_mask=False,
        nwp_conv_filter=False,  # No ERA5 NWPCONV mask!
        discri_scheme="GEN",    # Uncalibrated physical ATBD rules (no OPERA beta weights!)
    )
    return run_nwcsaf_rdt_sequence(
        raw,
        times,
        area_km2,
        pure_cfg,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        tropopause_temp_k=pure_tropo_k,
        nwp_instability_mask=None,
        calibrated_model=None,
        use_ancillary_ct=False,
        use_hrw_amv=False,
    )


def analyze_opera_thresholds_and_scales(
    win: dict[str, Any],
    algo_sem_maps: dict[str, np.ndarray],
    opera_root: Path,
) -> dict[str, Any]:
    """Evaluate cells against OPERA radar across thresholds (35, 40, 45, 50 dBZ), lag (+/-15m), & scale."""
    times = win["times"][win["warmup_frames"] :]
    x_m, y_m = win["x_m"], win["y_m"]
    roi = win["roi"]
    area_roi = win["area_km2"][roi[0], roi[1]]
    opera_row, opera_col, opera_inside = build_seviri_to_opera_lookup(x_m, y_m)
    row_roi = opera_row[roi[0], roi[1]]
    col_roi = opera_col[roi[0], roi[1]]
    in_roi = opera_inside[roi[0], roi[1]]

    thresholds_dbz = [35.0, 40.0, 45.0, 50.0]
    results: dict[str, Any] = {
        name: {
            f"dbz_{int(th)}": {
                "inst_cell_tp": 0,
                "inst_cell_fp": 0,
                "inst_cell_fn": 0,
                "lag15_cell_tp": 0,
                "lag15_cell_fp": 0,
                "lag15_cell_fn": 0,
                "meso500_tp": 0,
                "meso500_fp": 0,
                "meso500_fn": 0,
                "small_tp": 0,
                "small_fp": 0,
                "small_fn": 0,
                "growing_cls2_tp": 0,
                "growing_cls2_fp": 0,
                "mature_cls3_tp": 0,
                "mature_cls3_fp": 0,
                "pixel_tp": 0,
                "pixel_fp": 0,
                "pixel_fn": 0,
            }
            for th in thresholds_dbz
        }
        for name in algo_sem_maps
    }

    opera_cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    def _get_opera(ts_15: pd.Timestamp) -> tuple[np.ndarray, np.ndarray] | None:
        k = ts_15.isoformat()
        if k not in opera_cache:
            p = opera_file_path(opera_root, ts_15)
            opera_cache[k] = read_opera_h5(p) if p.exists() else None
        return opera_cache[k]

    for t_idx, ts in enumerate(times):
        ts_15 = pd.Timestamp(ts).round("15min")
        if abs((pd.Timestamp(ts) - ts_15).total_seconds()) > 120.0:
            continue
        op0 = _get_opera(ts_15)
        if op0 is None:
            continue
        dbzh0, qind0 = op0

        # Also load -15 min and +15 min OPERA slots to account for cloud-top vs surface-rain phase lag
        lag_slots = [op0]
        for delta_m in (-15, 15):
            op_d = _get_opera(ts_15 + pd.Timedelta(minutes=delta_m))
            if op_d is not None:
                lag_slots.append(op_d)

        rc = np.clip(row_roi, 0, dbzh0.shape[0] - 1)
        cc = np.clip(col_roi, 0, dbzh0.shape[1] - 1)
        z0 = np.where(in_roi, dbzh0[rc, cc], np.nan)
        q0 = np.where(in_roi, qind0[rc, cc], np.nan)
        valid_obs = in_roi & (np.isclose(z0, -32.0) | (np.isfinite(z0) & (q0 >= 0.40)))
        if int(np.count_nonzero(valid_obs)) < 500:
            continue

        # Max reflectivity across [-15, 0, +15] min window
        z_lag_max = z0.copy()
        for dbzh_l, qind_l in lag_slots[1:]:
            zl = np.where(in_roi, dbzh_l[rc, cc], np.nan)
            ql = np.where(in_roi, qind_l[rc, cc], np.nan)
            ok_l = in_roi & np.isfinite(zl) & (ql >= 0.40)
            z_lag_max = np.where(ok_l, np.fmax(z_lag_max, zl), z_lag_max)

        for name, sem_3d in algo_sem_maps.items():
            sem = sem_3d[t_idx]
            lbl, n_obj = nd_label(sem >= 1)
            for oid in range(1, n_obj + 1):
                omask = lbl == oid
                obs_m = omask & valid_obs
                if int(np.count_nonzero(obs_m)) < 6:
                    continue
                if float(np.count_nonzero(obs_m) / max(int(np.count_nonzero(omask)), 1)) < 0.70:
                    continue

                cell_area_km2 = float(np.nansum(area_roi[omask]))
                pred_conv = bool(np.any(sem[omask] >= 2))
                has_cls2 = bool(np.any(sem[omask] == 2))
                n_pix = int(np.count_nonzero(obs_m))

                # Unique OPERA 2x2 km pixels inside cell
                u_rc = np.unique(np.column_stack((rc[obs_m], cc[obs_m])), axis=0)
                vals_0 = dbzh0[u_rc[:, 0], u_rc[:, 1]]
                q_vals_0 = qind0[u_rc[:, 0], u_rc[:, 1]]
                good_0 = vals_0[np.isfinite(vals_0) & (q_vals_0 >= 0.40)]
                max_z0 = float(np.max(good_0)) if good_0.size > 0 else -32.0

                max_z_lag = max_z0
                for dbzh_l, qind_l in lag_slots[1:]:
                    vl = dbzh_l[u_rc[:, 0], u_rc[:, 1]]
                    ql = qind_l[u_rc[:, 0], u_rc[:, 1]]
                    gl = vl[np.isfinite(vl) & (ql >= 0.40)]
                    if gl.size > 0:
                        max_z_lag = max(max_z_lag, float(np.max(gl)))

                for th in thresholds_dbz:
                    rdict = results[name][f"dbz_{int(th)}"]
                    area_th_0 = float(np.count_nonzero(good_0 >= th) * 4.0) if good_0.size > 0 else 0.0
                    min_core_km2 = 16.0 if th <= 40.0 else 8.0

                    # Ambiguous intermediate echo (between 28 dBZ and th) is excluded from strict binary truth
                    truth_inst = 1 if area_th_0 >= min_core_km2 else (0 if max_z0 < 28.0 else -1)
                    truth_lag = 1 if max_z_lag >= th else (0 if max_z_lag < 28.0 else -1)

                    if truth_inst == 1 and pred_conv:
                        rdict["inst_cell_tp"] += 1
                        rdict["pixel_tp"] += n_pix
                        if cell_area_km2 >= 500.0:
                            rdict["meso500_tp"] += 1
                        else:
                            rdict["small_tp"] += 1
                        if has_cls2:
                            rdict["growing_cls2_tp"] += 1
                        else:
                            rdict["mature_cls3_tp"] += 1
                    elif truth_inst == 0 and pred_conv:
                        rdict["inst_cell_fp"] += 1
                        rdict["pixel_fp"] += n_pix
                        if cell_area_km2 >= 500.0:
                            rdict["meso500_fp"] += 1
                        else:
                            rdict["small_fp"] += 1
                        if has_cls2:
                            rdict["growing_cls2_fp"] += 1
                        else:
                            rdict["mature_cls3_fp"] += 1
                    elif truth_inst == 1 and not pred_conv:
                        rdict["inst_cell_fn"] += 1
                        rdict["pixel_fn"] += n_pix
                        if cell_area_km2 >= 500.0:
                            rdict["meso500_fn"] += 1
                        else:
                            rdict["small_fn"] += 1

                    if truth_lag == 1 and pred_conv:
                        rdict["lag15_cell_tp"] += 1
                    elif truth_lag == 0 and pred_conv:
                        rdict["lag15_cell_fp"] += 1
                    elif truth_lag == 1 and not pred_conv:
                        rdict["lag15_cell_fn"] += 1

    return results


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    pod = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    csi = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "pod": round(pod, 4), "precision": round(prec, 4), "f1": round(f1, 4), "csi": round(csi, 4)}


def main() -> None:
    zarr_root = Path(r"E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N")
    opera_root = Path(r"E:\Datasets\Deep_convection\raw\opera")
    era5_cache_dir = RDT_ORIG_ROOT / "results" / "era5_cache"
    out_dir = RDT_ORIG_ROOT / "results" / "comparison_seviri_2020"

    all_windows = json.loads((out_dir / "selected_12_windows.json").read_text(encoding="utf-8"))
    eval_windows = all_windows[:6]
    cal_model = CalibratedRdtModel.from_json(RDT_ORIG_ROOT / "rdt_nwcsaf" / "default_calibrated_model.json")

    algo_names = [
        "nwcsaf_rdt_seviri_only_cal",
        "nwcsaf_rdt_seviri_pure_gen",
        "multispectral_utls_v0",
        "refined_v2",
        "legacy_fixed_2ch",
    ]
    ref_all: list[np.ndarray] = []
    preds_all: dict[str, list[np.ndarray]] = {k: [] for k in algo_names}

    all_algos = ["nwcsaf_rdt_full_era5_cal", *algo_names]
    agg_opera: dict[str, Any] = None

    for idx, row in enumerate(eval_windows, 1):
        cid = str(row["clip_id"])
        print(f"[{idx}/6] Evaluating {cid} ({row['start_time']})...", flush=True)
        win = fast_load_window_with_halo(
            zarr_root, str(row["start_time"]), int(row["origin_y"]), int(row["origin_x"])
        )
        raw = win["raw"]
        times = win["times"]
        x_m, y_m = win["x_m"], win["y_m"]
        area_km2 = win["area_km2"]
        lat_deg, lon_deg = win["lat_deg"], win["lon_deg"]
        roi = win["roi"]
        w_f = win["warmup_frames"]

        mid_ts = times[len(times) // 2]
        era5_grid = interpolate_era5_to_seviri_grid(
            fetch_era5_europe_hour(mid_ts, era5_cache_dir), x_m, y_m
        )

        # 1. Full ERA5 + GEO-CT/HRW + CAL
        full_cfg = NwcSafRdtConfig(enforce_stretched_mask=False, nwp_conv_filter=True, discri_scheme="CAL")
        full_out = run_nwcsaf_rdt_sequence(
            raw, times.to_numpy(), area_km2, full_cfg,
            lat_deg=lat_deg, lon_deg=lon_deg,
            tropopause_temp_k=era5_grid.t_tropo_k,
            nwp_instability_mask=era5_grid.nwpconv_mask,
            calibrated_model=cal_model,
            use_ancillary_ct=True, use_hrw_amv=True,
            u700_ms=era5_grid.u700_ms, v700_ms=era5_grid.v700_ms, t_surf_k=era5_grid.t_surf_k,
        )
        full_sem = full_out["semantic_label"][w_f:, roi[0], roi[1]]

        # 2. Calibrated 7-ch SEVIRI
        mid_f = raw[len(times) // 2]
        ch_dict = {
            "WV_062": mid_f[0], "WV_073": mid_f[1], "IR_087": mid_f[2],
            "IR_097": mid_f[3], "IR_108": mid_f[4], "IR_120": mid_f[5], "IR_134": mid_f[6],
        }
        sev_tropo_k, sev_nwpconv = predict_seviri_only_nwp_surrogates(cal_model, ch_dict, lat_deg=lat_deg)
        sev_cal_out = run_nwcsaf_rdt_sequence(
            raw, times.to_numpy(), area_km2, full_cfg,
            lat_deg=lat_deg, lon_deg=lon_deg,
            tropopause_temp_k=sev_tropo_k, nwp_instability_mask=sev_nwpconv,
            calibrated_model=cal_model, use_ancillary_ct=True, use_hrw_amv=True,
        )
        sev_cal_sem = sev_cal_out["semantic_label"][w_f:, roi[0], roi[1]]

        # 3. Pure Uncalibrated 7-ch SEVIRI (GEN, no CAL weights, no ERA5, no CT/HRW)
        pure_out = run_pure_uncalibrated_seviri_rdt(raw, times.to_numpy(), area_km2, lat_deg, lon_deg)
        pure_sem = pure_out["semantic_label"][w_f:, roi[0], roi[1]]

        # 4-6. Repository proxies
        utls_sem = build_rdt_teacher_labels(
            raw, times=times, config=RDTTeacherConfig(), pixel_area_km2=area_km2
        )["semantic_label"][w_f:, roi[0], roi[1]]
        ref2_sem = label_refined_clip(raw, times, area_km2, RefinedRDTConfig())["classes"][w_f:, roi[0], roi[1]]
        leg_sem = label_rdt_clip(raw, times, area_km2, RDTConfig())["classes"][w_f:, roi[0], roi[1]]

        ref_all.append(full_sem)
        preds_all["nwcsaf_rdt_seviri_only_cal"].append(sev_cal_sem)
        preds_all["nwcsaf_rdt_seviri_pure_gen"].append(pure_sem)
        preds_all["multispectral_utls_v0"].append(utls_sem)
        preds_all["refined_v2"].append(ref2_sem)
        preds_all["legacy_fixed_2ch"].append(leg_sem)

        win_op = analyze_opera_thresholds_and_scales(
            win,
            {
                "nwcsaf_rdt_full_era5_cal": full_sem,
                "nwcsaf_rdt_seviri_only_cal": sev_cal_sem,
                "nwcsaf_rdt_seviri_pure_gen": pure_sem,
                "multispectral_utls_v0": utls_sem,
                "refined_v2": ref2_sem,
                "legacy_fixed_2ch": leg_sem,
            },
            opera_root,
        )
        if agg_opera is None:
            agg_opera = win_op
        else:
            for a_name in all_algos:
                for th_k in agg_opera[a_name]:
                    for m_k in agg_opera[a_name][th_k]:
                        agg_opera[a_name][th_k][m_k] += win_op[a_name][th_k][m_k]

    ref_concat = np.concatenate(ref_all, axis=0)
    vs_ref_summary = {}
    for k in algo_names:
        m = compute_comparison_metrics(ref_concat, np.concatenate(preds_all[k], axis=0))
        vs_ref_summary[k] = {
            "overall_acc": round(m["overall_pixel_accuracy"], 4),
            "cloud_iou": round(m["regimes"]["cloud_tower_mask_ge1"]["iou_csi"], 4),
            "cloud_dice": round(m["regimes"]["cloud_tower_mask_ge1"]["dice_f1"], 4),
            "conv_iou": round(m["regimes"]["convective_mask_234"]["iou_csi"], 4),
            "conv_dice": round(m["regimes"]["convective_mask_234"]["dice_f1"], 4),
            "conv_pod": round(m["regimes"]["convective_mask_234"]["pod_recall"], 4),
            "conv_prec": round(m["regimes"]["convective_mask_234"]["precision"], 4),
        }

    opera_summary: dict[str, Any] = {}
    assert agg_opera is not None
    for a_name in all_algos:
        opera_summary[a_name] = {}
        for th_k, d in agg_opera[a_name].items():
            opera_summary[a_name][th_k] = {
                "instantaneous_all_cells": _prf(d["inst_cell_tp"], d["inst_cell_fp"], d["inst_cell_fn"]),
                "lag15min_all_cells": _prf(d["lag15_cell_tp"], d["lag15_cell_fp"], d["lag15_cell_fn"]),
                "mesoscale_ge500km2_cells": _prf(d["meso500_tp"], d["meso500_fp"], d["meso500_fn"]),
                "small_lt500km2_cells": _prf(d["small_tp"], d["small_fp"], d["small_fn"]),
                "pixel_level": _prf(d["pixel_tp"], d["pixel_fp"], d["pixel_fn"]),
                "growing_cls2_precision": round(
                    float(d["growing_cls2_tp"] / max(d["growing_cls2_tp"] + d["growing_cls2_fp"], 1)), 4
                ),
                "mature_cls3_precision": round(
                    float(d["mature_cls3_tp"] / max(d["mature_cls3_tp"] + d["mature_cls3_fp"], 1)), 4
                ),
            }

    out_json = out_dir / "pure_vs_calibrated_and_opera_analysis.json"
    out_json.write_text(
        json.dumps({"vs_full_era5_reference": vs_ref_summary, "opera_MultiThreshold_analysis": opera_summary}, indent=2),
        encoding="utf-8",
    )
    print("=== SUMMARY VS FULL ERA5 REFERENCE ===")
    print(json.dumps(vs_ref_summary, indent=2))
    print("=== SUMMARY VS OPERA RADAR (35, 40, 45, 50 dBZ & MESOSCALE vs SMALL) ===")
    for a_name in ["nwcsaf_rdt_full_era5_cal", "nwcsaf_rdt_seviri_only_cal", "nwcsaf_rdt_seviri_pure_gen", "multispectral_utls_v0"]:
        print(f"\n--- {a_name} ---")
        for th_k in ["dbz_35", "dbz_40", "dbz_45", "dbz_50"]:
            print(th_k, json.dumps(opera_summary[a_name][th_k]))


if __name__ == "__main__":
    main()
