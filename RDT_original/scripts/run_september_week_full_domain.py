#!/usr/bin/env python3
"""Full-Domain (763 x 1762) September 2020 Week Verification & 3-Video Generator.

Evaluates and renders over the full MSG-SEVIRI Rapid Scan domain (763 x 1762 pixels)
stored in ``E:\\Datasets\\MSG-SEVIRI\\2020`` (2020-09-14 through 2020-09-20, 7 full days
including Medicane Ianos and continental European deep convection):
1. Quantitative ablation on the 6 held-out evaluation windows:
   - Pure IR-derived Ancillary Products (GEO-CT + GEO-HRW from 7 thermal SEVIRI channels)
     on ``nwcsaf_rdt_seviri_pure_gen``
   - 3D Geostationary Parallax Correction (``apply_parallax_correction_2d``) vs OPERA radar
2. Full-Domain (763 x 1762) September 2020 Week Verification & 3 Separate Full-Domain MP4 Videos
   in ``RDT_original/results/comparison_seviri_2020/``:
   - ``sept2020_full_domain_1_OPERA_radar.mp4``
   - ``sept2020_full_domain_2_RDT_completo.mp4``
   - ``sept2020_full_domain_3_RDT_puro.mp4``
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer
from scipy import ndimage
import xarray as xr
import zarr

ROOT = Path(__file__).resolve().parents[2]
RDT_ORIG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(RDT_ORIG_ROOT))
sys.path.insert(0, str(RDT_ORIG_ROOT / "scripts"))

from emma_gpm.rdt_teacher import projected_pixel_areas

from rdt_nwcsaf import (
    CalibratedRdtModel,
    NwcSafRdtConfig,
    NwcSafRdtTracker,
    build_seviri_to_opera_lookup,
    fetch_era5_europe_hour,
    interpolate_era5_to_seviri_grid,
    opera_file_path,
    read_opera_h5,
    run_nwcsaf_rdt_sequence,
)
from calibrate_and_benchmark_full_rdt import compute_comparison_metrics
from analyze_pure_and_opera_truth import (
    _prf,
    analyze_opera_thresholds_and_scales,
    fast_load_window_with_halo,
    run_pure_uncalibrated_seviri_rdt,
)

SEPT_ZARR_ROOT = Path(r"E:\Datasets\MSG-SEVIRI\2020")
OPERA_ROOT = Path(r"E:\Datasets\Deep_convection\raw\opera")
ERA5_CACHE_DIR = RDT_ORIG_ROOT / "results" / "era5_cache"
OUT_DIR = RDT_ORIG_ROOT / "results" / "comparison_seviri_2020"

# Pure 4 semantic class RGBA colors (no artificial borders, no overlapping yellow):
# 1: Non-convective cold cloud -> Steel Blue
# 2: Developing / Growing convective cell -> Vibrant Orange
# 3: Mature / Decaying convective cell -> Crimson Red
# 4: Overshooting Top (OT) -> Bright Magenta / Violet
CLASS_COLORS: dict[int, tuple[int, int, int, float]] = {
    1: (59, 130, 246, 0.58),
    2: (249, 115, 22, 0.80),
    3: (220, 38, 38, 0.80),
    4: (232, 121, 249, 0.95),
}

TURBO_CMAP = matplotlib.colormaps["turbo"]


def _orient_north_up_west_left(
    data: np.ndarray, y_raw: np.ndarray, x_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure north-up (y descending) and west-left (x ascending)."""
    out = data
    y_out = y_raw.copy()
    x_out = x_raw.copy()
    if y_out[0] < y_out[-1]:
        out = out[..., ::-1, :]
        y_out = y_out[::-1]
    if x_out[0] > x_out[-1]:
        out = out[..., :, ::-1]
        x_out = x_out[::-1]
    return np.ascontiguousarray(out), y_out, x_out


def _load_full_domain_geometry(zarr_path: Path) -> dict[str, Any]:
    """Load and orient (north-up, west-left) the 763x1762 geostationary coordinates & pixel areas."""
    with xr.open_zarr(zarr_path, consolidated=True) as ds:
        x_raw = ds.x.values.copy()
        y_raw = ds.y.values.copy()
        proj_wkt = ds.attrs["projection_wkt"]

    dummy = np.zeros((1, len(y_raw), len(x_raw)), dtype=np.float32)
    _, y_m, x_m = _orient_north_up_west_left(dummy, y_raw, x_raw)

    crs = CRS.from_wkt(proj_wkt)
    xx, yy = np.meshgrid(x_m, y_m)
    to_ll = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon_deg, lat_deg = to_ll.transform(xx, yy)
    lon_deg = np.where(np.isfinite(lon_deg), lon_deg, np.nan)
    lat_deg = np.where(np.isfinite(lat_deg), lat_deg, np.nan)

    # Vectorized WGS84 ellipsoidal pixel area (km^2) via Jacobian d(lat, lon)/d(row, col)
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    dlat_dr, dlat_dc = np.gradient(lat_rad)
    dlon_dr, dlon_dc = np.gradient(lon_rad)
    a_wgs = 6378.137
    e2_wgs = 0.00669437999014
    sin_lat = np.sin(lat_rad)
    denom = np.sqrt(np.maximum(1.0 - e2_wgs * sin_lat * sin_lat, 1e-6))
    r_merid = a_wgs * (1.0 - e2_wgs) / (denom ** 3)
    r_prime = a_wgs / denom
    jac_rad2 = np.abs(dlat_dr * dlon_dc - dlat_dc * dlon_dr)
    area_km2 = (r_merid * r_prime * np.cos(lat_rad) * jac_rad2).astype(np.float32)
    area_km2 = np.where(np.isfinite(area_km2) & (area_km2 > 0.0) & (area_km2 < 250.0), area_km2, np.nan)

    opera_row, opera_col, opera_inside = build_seviri_to_opera_lookup(x_m, y_m)

    return {
        "x_raw": x_raw,
        "y_raw": y_raw,
        "x_m": x_m,
        "y_m": y_m,
        "area_km2": area_km2,
        "lat_deg": np.asarray(lat_deg, dtype=np.float32),
        "lon_deg": np.asarray(lon_deg, dtype=np.float32),
        "opera_row": opera_row,
        "opera_col": opera_col,
        "opera_inside": opera_inside,
    }


def _ir108_to_rgb_base(ir108_k: np.ndarray) -> np.ndarray:
    """Convert IR 10.8 brightness temperature (K) to an enhanced grayscale RGB image."""
    norm = np.clip((300.0 - np.nan_to_num(ir108_k, nan=295.0)) / (300.0 - 202.0), 0.0, 1.0)
    gray = (30.0 + 205.0 * norm).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _overlay_semantic_mask(base_rgb: np.ndarray, sem_label: np.ndarray) -> np.ndarray:
    """Alpha-blend strictly the 4 semantic classes onto IR10.8 RGB."""
    out = base_rgb.astype(np.float32).copy()
    for cls_id, (r, g, b, alpha) in CLASS_COLORS.items():
        mask = sem_label == cls_id
        if np.any(mask):
            out[mask, 0] = (1.0 - alpha) * out[mask, 0] + alpha * r
            out[mask, 1] = (1.0 - alpha) * out[mask, 1] + alpha * g
            out[mask, 2] = (1.0 - alpha) * out[mask, 2] + alpha * b
    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_opera_gradient_panel(
    base_rgb: np.ndarray,
    dbzh_grid: np.ndarray,
    qind_grid: np.ndarray,
    inside_mask: np.ndarray,
    qind_min: float = 0.20,
    dbzh_min: float = -5.0,
) -> np.ndarray:
    """Render OPERA DBZH continuous gradient ('turbo', -5..65 dBZ)."""
    out = base_rgb.astype(np.float32).copy()
    outside = ~inside_mask | ~np.isfinite(dbzh_grid)
    if np.any(outside):
        out[outside] = 0.68 * out[outside] + 0.32 * np.array([70.0, 75.0, 86.0], dtype=np.float32)

    echo = (
        inside_mask
        & np.isfinite(dbzh_grid)
        & ~np.isclose(dbzh_grid, -32.0)
        & np.isfinite(qind_grid)
        & (qind_grid >= qind_min)
        & (dbzh_grid >= dbzh_min)
    )
    if np.any(echo):
        norm_val = np.clip((dbzh_grid[echo] - (-5.0)) / (65.0 - (-5.0)), 0.0, 1.0)
        rgba = (TURBO_CMAP(norm_val)[:, :3] * 255.0).astype(np.float32)
        alpha = 0.78
        out[echo] = (1.0 - alpha) * out[echo] + alpha * rgba

    return np.clip(out, 0, 255).astype(np.uint8)


def _compose_single_full_domain_frame(
    rgb_map: np.ndarray,
    title: str,
    subtitle: str,
    is_opera_video: bool = False,
) -> np.ndarray:
    """Compose a single-panel full-domain (763 x 1762) video frame with header and legend bar."""
    h, w, _ = rgb_map.shape
    header_h = 64
    footer_h = 52
    pad = 12
    total_w = w + 2 * pad
    total_h = header_h + h + footer_h + 2 * pad
    if total_w % 2 == 1:
        total_w += 1
    if total_h % 2 == 1:
        total_h += 1

    canvas = Image.new("RGB", (total_w, total_h), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    try:
        font_h1 = ImageFont.truetype("arialbd.ttf", 21)
        font_sub = ImageFont.truetype("arial.ttf", 15)
        font_leg = ImageFont.truetype("arialbd.ttf", 14)
    except Exception:
        font_h1 = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_leg = ImageFont.load_default()

    draw.text((pad + 4, 10), title, fill=(248, 250, 252), font=font_h1)
    draw.text((pad + 4, 38), subtitle, fill=(148, 163, 184), font=font_sub)

    panel_img = Image.fromarray(rgb_map)
    canvas.paste(panel_img, (pad, header_h))
    draw.rectangle([pad - 1, header_h - 1, pad + w, header_h + h], outline=(71, 85, 105), width=1)

    fy = header_h + h + 16
    lx = pad + 6
    if is_opera_video:
        cb_w = 280
        for dx in range(cb_w):
            val = dx / max(cb_w - 1, 1)
            r, g, b, _ = TURBO_CMAP(val)
            col = (int(r * 255), int(g * 255), int(b * 255))
            draw.line([(lx + dx, fy), (lx + dx, fy + 18)], fill=col)
        draw.rectangle([lx, fy, lx + cb_w, fy + 18], outline=(210, 215, 225))
        draw.text(
            (lx + cb_w + 12, fy + 1),
            "OPERA Radar DBZH [-5 .. 20 .. 35 .. 50 .. 65 dBZ] (QIND >= 0.20) su SEVIRI IR 10.8 [K]",
            fill=(226, 232, 240),
            font=font_leg,
        )
    else:
        legend_items = [
            ((59, 130, 246), "Classe 1: Nube fredda non convettiva"),
            ((249, 115, 22), "Classe 2: Convezione in sviluppo (Growing/Triggering)"),
            ((220, 38, 38), "Classe 3: Convezione matura / in decadimento"),
            ((232, 121, 249), "Classe 4: Overshooting Top (OT)"),
        ]
        for color, label in legend_items:
            draw.rectangle([lx, fy, lx + 20, fy + 18], fill=color, outline=(255, 255, 255))
            draw.text((lx + 26, fy + 1), label, fill=(226, 232, 240), font=font_leg)
            lx += 26 + int(len(label) * 7.8) + 28

    return np.asarray(canvas, dtype=np.uint8)


def _encode_mp4(frames: list[np.ndarray], out_mp4: Path, fps: float = 1.2) -> None:
    """Encode list of RGB frames into a smooth 12-fps H.264 MP4 via ffmpeg at input rate ``fps``."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_mp4.parent / f"_tmp_{out_mp4.stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(tmp_dir / f"frame_{idx:04d}.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", f"{fps:.4f}",
        "-i", str(tmp_dir / "frame_%04d.png"),
        "-vf", "fps=12",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "19",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def evaluate_heldout_ancillary_and_parallax_ablations(
    cal_model: CalibratedRdtModel,
) -> dict[str, Any]:
    """Run controlled ablations on the 6 held-out evaluation windows to quantify:
    1. Pure IR-derived Ancillary Products (GEO-CT + GEO-HRW) on pure_gen
    2. 3D Geostationary Parallax Correction vs OPERA radar
    """
    zarr_root = Path(r"E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N")
    all_windows = json.loads((OUT_DIR / "selected_12_windows.json").read_text(encoding="utf-8"))
    eval_windows = all_windows[:6]

    ref_all: list[np.ndarray] = []
    pure_base_all: list[np.ndarray] = []
    pure_ir_anc_all: list[np.ndarray] = []

    agg_opera: dict[str, Any] | None = None
    ablation_algos = [
        "full_no_parallax",
        "full_with_parallax",
        "pure_no_parallax",
        "pure_ir_anc_with_parallax",
    ]

    for idx, row in enumerate(eval_windows, 1):
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
            fetch_era5_europe_hour(mid_ts, ERA5_CACHE_DIR), x_m, y_m
        )

        full_cfg = NwcSafRdtConfig(enforce_stretched_mask=False, nwp_conv_filter=True, discri_scheme="CAL")
        pure_cfg = NwcSafRdtConfig(enforce_stretched_mask=False, nwp_conv_filter=False, discri_scheme="GEN")

        # Radiative tropopause from 7ch SEVIRI for pure models
        mid_f = raw[len(times) // 2]
        cold_m = np.isfinite(mid_f[4]) & (mid_f[4] < 235.0)
        if int(np.count_nonzero(cold_m)) >= 50:
            t_tropo_val = float(0.6 * np.nanpercentile(mid_f[0][cold_m], 10) + 0.4 * np.nanpercentile(mid_f[4][cold_m], 5))
        else:
            t_tropo_val = float(np.nanpercentile(mid_f[0], 5))
        pure_tropo_k = np.full(mid_f[4].shape, float(np.clip(t_tropo_val, 205.0, 226.0)), dtype=np.float32)

        full_no_par = run_nwcsaf_rdt_sequence(
            raw, times.to_numpy(), area_km2, full_cfg,
            lat_deg=lat_deg, lon_deg=lon_deg,
            tropopause_temp_k=era5_grid.t_tropo_k, nwp_instability_mask=era5_grid.nwpconv_mask,
            calibrated_model=cal_model, use_ancillary_ct=True, use_hrw_amv=True,
            use_parallax_correction=False,
            u700_ms=era5_grid.u700_ms, v700_ms=era5_grid.v700_ms, t_surf_k=era5_grid.t_surf_k,
        )["semantic_label"][w_f:, roi[0], roi[1]]

        full_with_par = run_nwcsaf_rdt_sequence(
            raw, times.to_numpy(), area_km2, full_cfg,
            lat_deg=lat_deg, lon_deg=lon_deg,
            tropopause_temp_k=era5_grid.t_tropo_k, nwp_instability_mask=era5_grid.nwpconv_mask,
            calibrated_model=cal_model, use_ancillary_ct=True, use_hrw_amv=True,
            use_parallax_correction=True,
            u700_ms=era5_grid.u700_ms, v700_ms=era5_grid.v700_ms, t_surf_k=era5_grid.t_surf_k,
        )["semantic_label"][w_f:, roi[0], roi[1]]

        pure_no_par = run_pure_uncalibrated_seviri_rdt(
            raw, times.to_numpy(), area_km2, lat_deg, lon_deg
        )["semantic_label"][w_f:, roi[0], roi[1]]

        pure_ir_anc = run_nwcsaf_rdt_sequence(
            raw, times.to_numpy(), area_km2, pure_cfg,
            lat_deg=lat_deg, lon_deg=lon_deg,
            tropopause_temp_k=pure_tropo_k, nwp_instability_mask=None,
            calibrated_model=None, use_ancillary_ct=True, use_hrw_amv=True,
            use_parallax_correction=False,
        )["semantic_label"][w_f:, roi[0], roi[1]]

        pure_ir_anc_par = run_nwcsaf_rdt_sequence(
            raw, times.to_numpy(), area_km2, pure_cfg,
            lat_deg=lat_deg, lon_deg=lon_deg,
            tropopause_temp_k=pure_tropo_k, nwp_instability_mask=None,
            calibrated_model=None, use_ancillary_ct=True, use_hrw_amv=True,
            use_parallax_correction=True,
        )["semantic_label"][w_f:, roi[0], roi[1]]

        ref_all.append(full_no_par)
        pure_base_all.append(pure_no_par)
        pure_ir_anc_all.append(pure_ir_anc)

        win_op = analyze_opera_thresholds_and_scales(
            win,
            {
                "full_no_parallax": full_no_par,
                "full_with_parallax": full_with_par,
                "pure_no_parallax": pure_no_par,
                "pure_ir_anc_with_parallax": pure_ir_anc_par,
            },
            OPERA_ROOT,
        )
        if agg_opera is None:
            agg_opera = win_op
        else:
            for a_name in ablation_algos:
                for th_k in agg_opera[a_name]:
                    for m_k in agg_opera[a_name][th_k]:
                        agg_opera[a_name][th_k][m_k] += win_op[a_name][th_k][m_k]

    ref_concat = np.concatenate(ref_all, axis=0)
    m_base = compute_comparison_metrics(ref_concat, np.concatenate(pure_base_all, axis=0))
    m_anc = compute_comparison_metrics(ref_concat, np.concatenate(pure_ir_anc_all, axis=0))

    assert agg_opera is not None
    parallax_opera_summary: dict[str, Any] = {}
    for a_name in ablation_algos:
        parallax_opera_summary[a_name] = {}
        for th_k, d in agg_opera[a_name].items():
            parallax_opera_summary[a_name][th_k] = {
                "instantaneous_all_cells": _prf(d["inst_cell_tp"], d["inst_cell_fp"], d["inst_cell_fn"]),
                "lag15min_all_cells": _prf(d["lag15_cell_tp"], d["lag15_cell_fp"], d["lag15_cell_fn"]),
                "mesoscale_ge500km2_cells": _prf(d["meso500_tp"], d["meso500_fp"], d["meso500_fn"]),
                "pixel_level": _prf(d["pixel_tp"], d["pixel_fp"], d["pixel_fn"]),
            }

    return {
        "pure_gen_without_ancillaries_vs_full": {
            "overall_acc": round(m_base["overall_pixel_accuracy"], 4),
            "cloud_dice": round(m_base["regimes"]["cloud_tower_mask_ge1"]["dice_f1"], 4),
            "conv_dice": round(m_base["regimes"]["convective_mask_234"]["dice_f1"], 4),
            "conv_pod": round(m_base["regimes"]["convective_mask_234"]["pod_recall"], 4),
            "conv_prec": round(m_base["regimes"]["convective_mask_234"]["precision"], 4),
        },
        "pure_gen_with_ir_ancillaries_vs_full": {
            "overall_acc": round(m_anc["overall_pixel_accuracy"], 4),
            "cloud_dice": round(m_anc["regimes"]["cloud_tower_mask_ge1"]["dice_f1"], 4),
            "conv_dice": round(m_anc["regimes"]["convective_mask_234"]["dice_f1"], 4),
            "conv_pod": round(m_anc["regimes"]["convective_mask_234"]["pod_recall"], 4),
            "conv_prec": round(m_anc["regimes"]["convective_mask_234"]["precision"], 4),
        },
        "parallax_and_ancillary_vs_opera": parallax_opera_summary,
    }


def _make_snapshot(cell: dict[str, Any], cid: int, event: str, frame_idx: int) -> dict[str, Any]:
    return {
        "frame_index": frame_idx,
        "timestamp": None,
        "cell_id": cid,
        "track_id": cid,
        "event": event,
        "border": cell["border"],
        "seed_yx": cell["seed_yx"],
        "t_bt_k": cell["t_bt_k"],
        "t_st_k": cell["t_st_k"],
        "t_min_k": cell["t_min_k"],
        "t_mean_k": cell["t_mean_k"],
        "contrast_k": cell["contrast_k"],
        "area_km2": cell["area_km2"],
        "vertical_profile": cell["vertical_profile"],
        "btd_wv62_ir108_max_k": cell["btd_wv62_ir108_max_k"],
        "btd_wv62_ir108_p90_k": cell["btd_wv62_ir108_p90_k"],
        "cooling_rate_k_per_h": cell["cooling_rate_k_per_h"],
        "is_convective": cell["is_convective"],
        "vertical_cat": cell["vertical_cat"],
        "phase_code": cell["phase_code"],
        "severity": cell["severity"],
    }


def _evaluate_pair_both_models(
    f_prev: np.ndarray,
    f_curr: np.ndarray,
    area_km2: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    era5_grid: Any,
    pure_tropo_p: np.ndarray,
    pure_tropo_c: np.ndarray,
    cal_model: CalibratedRdtModel,
) -> dict[str, Any]:
    """Run authentic 15-min pair (f_prev -> f_curr) sharing Stage 0-4 detection/tracking/OTD
    and executing Stage 5-6 (discrimination + parallax correction) separately for Full and Pure.
    """
    from rdt_nwcsaf.ancillary import (
        apply_parallax_correction_2d,
        compute_hrw_amv_guess_field,
        compute_nwcsaf_cloud_type,
    )
    from rdt_nwcsaf.detection import detect_rdt_cells
    from rdt_nwcsaf.discrimination import discriminate_and_classify_cell
    from rdt_nwcsaf.otd import detect_cell_overshooting_tops
    from rdt_nwcsaf.tracking import track_rdt_cells

    full_cfg = NwcSafRdtConfig(enforce_stretched_mask=False, nwp_conv_filter=True, discri_scheme="CAL")
    pure_cfg = NwcSafRdtConfig(enforce_stretched_mask=False, nwp_conv_filter=False, discri_scheme="GEN")
    h, w = area_km2.shape
    pure_tropo_p_scalar = float(pure_tropo_p[0, 0])
    pure_tropo_c_scalar = float(pure_tropo_c[0, 0])

    # --- Frame t - 15 min (f_prev) ---
    ct_prev = compute_nwcsaf_cloud_type(
        f_prev[4], ir120=f_prev[5], ir087=f_prev[2], wv062=f_prev[0], wv073=f_prev[1],
        t_tropo_k=pure_tropo_p, t_surf_k=292.0,
    )
    prev_labels, prev_cells_base, prev_hierarchy, _ = detect_rdt_cells(
        f_prev, area_km2, pure_cfg, ct_cloudy_mask=ct_prev.opaque_cloud_mask, tropopause_temp_k=pure_tropo_p
    )

    prev_cells_full: dict[int, dict[str, Any]] = {}
    prev_cells_pure: dict[int, dict[str, Any]] = {}
    hist_full: dict[int, list[dict[str, Any]]] = {}
    hist_pure: dict[int, list[dict[str, Any]]] = {}

    for cid, base_c in prev_cells_base.items():
        base_c["motion_yx"] = np.zeros(2, dtype=float)
        base_c["event"] = "new"
        base_c["track_id"] = cid
        cy_i = int(np.clip(round(float(base_c["weight_centroid_yx"][0])), 0, h - 1))
        cx_i = int(np.clip(round(float(base_c["weight_centroid_yx"][1])), 0, w - 1))
        c_lat = float(lat_deg[cy_i, cx_i])
        c_lon = float(lon_deg[cy_i, cx_i])
        base_c["lat_deg"] = c_lat
        base_c["lon_deg"] = c_lon
        ot_p = detect_cell_overshooting_tops(base_c, f_prev, area_km2, pure_cfg, cell_lat_deg=c_lat)
        base_c["ot_confirmed_count"] = int(ot_p["confirmed_count"])
        base_c["ot_confirmed_pixels"] = int(ot_p["confirmed_pixels"])

        bbox = base_c["bbox"]
        mask = base_c["mask"]
        loc_nwp = era5_grid.nwpconv_mask[bbox][mask]
        nwp_val = int(np.max(loc_nwp)) if len(loc_nwp) > 0 else 2
        loc_tr_f = era5_grid.t_tropo_k[bbox][mask]
        tropo_f = float(np.nanmedian(loc_tr_f)) if len(loc_tr_f) > 0 else 215.0

        cf = dict(base_c)
        df = discriminate_and_classify_cell(
            cf, None, None, [], None, prev_hierarchy, ot_p, 15.0, full_cfg,
            nwp_mask_value=nwp_val, lightning_strokes=0, cell_lat_deg=c_lat,
            calibrated_model=cal_model, cell_tropo_temp_k=tropo_f,
        )
        cf.update(df)
        prev_cells_full[cid] = cf
        hist_full[cid] = [_make_snapshot(cf, cid, "new", 0)]

        cp = dict(base_c)
        dp = discriminate_and_classify_cell(
            cp, None, None, [], None, prev_hierarchy, ot_p, 15.0, pure_cfg,
            nwp_mask_value=2, lightning_strokes=0, cell_lat_deg=c_lat,
            calibrated_model=None, cell_tropo_temp_k=pure_tropo_p_scalar,
        )
        cp.update(dp)
        prev_cells_pure[cid] = cp
        hist_pure[cid] = [_make_snapshot(cp, cid, "new", 0)]

    # --- Frame t (f_curr) ---
    ct_curr = compute_nwcsaf_cloud_type(
        f_curr[4], ir120=f_curr[5], ir087=f_curr[2], wv062=f_curr[0], wv073=f_curr[1],
        t_tropo_k=pure_tropo_c, t_surf_k=292.0,
    )
    dy_px, dx_px = compute_hrw_amv_guess_field(
        f_prev[4], f_curr[4], prev_wv062=f_prev[0], curr_wv062=f_curr[0], dt_seconds=900.0
    )
    guess_motion_yx = np.stack([dy_px, dx_px], axis=0)

    curr_labels, curr_cells_base, curr_hierarchy, _ = detect_rdt_cells(
        f_curr, area_km2, pure_cfg, ct_cloudy_mask=ct_curr.opaque_cloud_mask, tropopause_temp_k=pure_tropo_c
    )

    parents, children, edges, primary_parent, events, _ = track_rdt_cells(
        prev_cells_pure, prev_labels, prev_hierarchy, f_prev[4],
        curr_cells_base, curr_labels, curr_hierarchy, f_curr[4],
        area_km2, pure_cfg, guess_motion_yx=guess_motion_yx,
    )

    sem_full = np.zeros((h, w), dtype=np.uint8)
    sem_pure = np.zeros((h, w), dtype=np.uint8)
    ot_mask_full = np.zeros((h, w), dtype=bool)
    ot_mask_pure = np.zeros((h, w), dtype=bool)
    n_conv_full = 0
    n_conv_pure = 0

    for cid, base_c in curr_cells_base.items():
        ev = events.get(cid, "new")
        p_main = primary_parent.get(cid)
        p_list = parents.get(cid, [])

        cy_i = int(np.clip(round(float(base_c["weight_centroid_yx"][0])), 0, h - 1))
        cx_i = int(np.clip(round(float(base_c["weight_centroid_yx"][1])), 0, w - 1))
        c_lat = float(lat_deg[cy_i, cx_i])
        c_lon = float(lon_deg[cy_i, cx_i])
        base_c["lat_deg"] = c_lat
        base_c["lon_deg"] = c_lon
        base_c["event"] = ev

        ot_c = detect_cell_overshooting_tops(base_c, f_curr, area_km2, pure_cfg, cell_lat_deg=c_lat)
        base_c["ot_confirmed_count"] = int(ot_c["confirmed_count"])
        base_c["ot_confirmed_pixels"] = int(ot_c["confirmed_pixels"])
        bbox = base_c["bbox"]
        mask = base_c["mask"]
        ot_conf_loc = ot_c["confirmed_mask"] & mask

        loc_nwp = era5_grid.nwpconv_mask[bbox][mask]
        nwp_val = int(np.max(loc_nwp)) if len(loc_nwp) > 0 else 2
        loc_tr_f = era5_grid.t_tropo_k[bbox][mask]
        tropo_f = float(np.nanmedian(loc_tr_f)) if len(loc_tr_f) > 0 else 215.0

        # Full model discrimination
        cf = dict(base_c)
        prev_f = prev_cells_full.get(p_main) if p_main is not None else None
        conv_p_f = prev_f if (prev_f is not None and prev_f.get("is_convective")) else None
        if conv_p_f is None:
            for pid in sorted(p_list):
                if prev_cells_full[pid].get("is_convective"):
                    conv_p_f = prev_cells_full[pid]
                    break
        hf = hist_full.get(p_main, []) if (p_main is not None and ev in ("continuation", "merge_main", "split_main")) else []
        df = discriminate_and_classify_cell(
            cf, prev_f, conv_p_f, hf, prev_hierarchy, curr_hierarchy, ot_c, 15.0, full_cfg,
            nwp_mask_value=nwp_val, lightning_strokes=0, cell_lat_deg=c_lat,
            calibrated_model=cal_model, cell_tropo_temp_k=tropo_f,
        )
        sem_full[bbox][mask] = int(df["class_id"])
        if df["is_convective"]:
            n_conv_full += 1
            if np.any(ot_conf_loc):
                ot_mask_full[bbox][ot_conf_loc] = True
                sem_full[bbox][ot_conf_loc] = 4

        # Pure model discrimination
        cp = dict(base_c)
        prev_p = prev_cells_pure.get(p_main) if p_main is not None else None
        conv_p_p = prev_p if (prev_p is not None and prev_p.get("is_convective")) else None
        if conv_p_p is None:
            for pid in sorted(p_list):
                if prev_cells_pure[pid].get("is_convective"):
                    conv_p_p = prev_cells_pure[pid]
                    break
        hp = hist_pure.get(p_main, []) if (p_main is not None and ev in ("continuation", "merge_main", "split_main")) else []
        dp = discriminate_and_classify_cell(
            cp, prev_p, conv_p_p, hp, prev_hierarchy, curr_hierarchy, ot_c, 15.0, pure_cfg,
            nwp_mask_value=2, lightning_strokes=0, cell_lat_deg=c_lat,
            calibrated_model=None, cell_tropo_temp_k=pure_tropo_c_scalar,
        )
        sem_pure[bbox][mask] = int(dp["class_id"])
        if dp["is_convective"]:
            n_conv_pure += 1
            if np.any(ot_conf_loc):
                ot_mask_pure[bbox][ot_conf_loc] = True
                sem_pure[bbox][ot_conf_loc] = 4

    # Stage 6: 3D Geostationary Parallax Correction
    sem_full = apply_parallax_correction_2d(
        sem_full, f_curr[4], lat_deg, lon_deg, t_surf_k=era5_grid.t_surf_k, t_tropo_k=era5_grid.t_tropo_k, fill_value=0
    )
    ot_mask_full = apply_parallax_correction_2d(
        ot_mask_full.astype(np.uint8), f_curr[4], lat_deg, lon_deg, t_surf_k=era5_grid.t_surf_k, t_tropo_k=era5_grid.t_tropo_k, fill_value=0
    ).astype(bool)
    sem_pure = apply_parallax_correction_2d(
        sem_pure, f_curr[4], lat_deg, lon_deg, t_surf_k=292.0, t_tropo_k=pure_tropo_c, fill_value=0
    )
    ot_mask_pure = apply_parallax_correction_2d(
        ot_mask_pure.astype(np.uint8), f_curr[4], lat_deg, lon_deg, t_surf_k=292.0, t_tropo_k=pure_tropo_c, fill_value=0
    ).astype(bool)

    return {
        "sem_full": sem_full,
        "sem_pure": sem_pure,
        "n_conv_full": n_conv_full,
        "n_conv_pure": n_conv_pure,
        "n_ot_full": int(ot_mask_full.sum()),
        "n_ot_pure": int(ot_mask_pure.sum()),
    }


def _process_single_day_worker(args: tuple[str, list[int], dict[str, Any]]) -> dict[str, Any]:
    """Worker process that evaluates all uniform slots for a single day Zarr."""
    zarr_path_str, slot_indices, geom = args
    zp = Path(zarr_path_str)
    cal_model = CalibratedRdtModel.from_json(RDT_ORIG_ROOT / "rdt_nwcsaf" / "default_calibrated_model.json")

    area_km2 = geom["area_km2"]
    lat_deg = geom["lat_deg"]
    lon_deg = geom["lon_deg"]
    x_m, y_m = geom["x_m"], geom["y_m"]
    opera_row, opera_col, opera_inside = geom["opera_row"], geom["opera_col"], geom["opera_inside"]
    h, w = area_km2.shape

    g = zarr.open(str(zp), mode="r")
    times_all = pd.date_range(zp.stem, periods=288, freq="5min", tz="UTC")

    # Preload the 3 synoptic ERA5 grids for this day (06:00, 13:00, 19:00 UTC)
    synoptic_hours = [6, 13, 19]
    era5_day_grids: dict[int, Any] = {}
    for sh in synoptic_hours:
        ts_syn = pd.Timestamp(f"{zp.stem[:4]}-{zp.stem[4:6]}-{zp.stem[6:8]}T{sh:02d}:00:00Z")
        era5_day_grids[sh] = interpolate_era5_to_seviri_grid(
            fetch_era5_europe_hour(ts_syn, ERA5_CACHE_DIR), x_m, y_m
        )

    opera_cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    def _get_opera(ts_15: pd.Timestamp) -> tuple[np.ndarray, np.ndarray] | None:
        k = ts_15.isoformat()
        if k not in opera_cache:
            if len(opera_cache) >= 4:
                opera_cache.clear()
            p = opera_file_path(OPERA_ROOT, ts_15)
            opera_cache[k] = read_opera_h5(p) if p.exists() else None
        return opera_cache[k]

    import gc

    day_frames_opera: list[np.ndarray] = []
    day_frames_full: list[np.ndarray] = []
    day_frames_pure: list[np.ndarray] = []
    day_ref_full: list[np.ndarray] = []
    day_pred_pure: list[np.ndarray] = []

    day_opera_stats: dict[str, dict[str, dict[str, int]]] = {
        m_name: {
            f"dbz_{th}": {
                "inst_cell_tp": 0, "inst_cell_fp": 0, "inst_cell_fn": 0,
                "lag15_cell_tp": 0, "lag15_cell_fp": 0, "lag15_cell_fn": 0,
                "meso500_tp": 0, "meso500_fp": 0, "meso500_fn": 0,
                "small_tp": 0, "small_fp": 0, "small_fn": 0,
                "pixel_tp": 0, "pixel_fp": 0, "pixel_fn": 0,
            }
            for th in (35, 40, 45, 50)
        }
        for m_name in ("nwcsaf_rdt_full_era5_cal", "nwcsaf_rdt_seviri_pure_gen")
    }

    for s_idx, idx_curr in enumerate(slot_indices, start=1):
        t0 = time.perf_counter()
        idx_prev = idx_curr - 3
        ts_curr = times_all[idx_curr]
        nearest_sh = min(synoptic_hours, key=lambda sh: abs(ts_curr.hour - sh))
        era5_grid = era5_day_grids[nearest_sh]

        raw_prev = np.asarray(g["seviri"][idx_prev], dtype=np.float32)
        f_prev, _, _ = _orient_north_up_west_left(raw_prev, geom["y_raw"], geom["x_raw"])
        cold_p = np.isfinite(f_prev[4]) & (f_prev[4] < 235.0)
        t_tr_p = float(0.6 * np.nanpercentile(f_prev[0][cold_p], 10) + 0.4 * np.nanpercentile(f_prev[4][cold_p], 5)) if int(np.count_nonzero(cold_p)) >= 50 else 215.0
        pure_tropo_p = np.full(f_prev[4].shape, float(np.clip(t_tr_p, 205.0, 226.0)), dtype=np.float32)

        raw_curr = np.asarray(g["seviri"][idx_curr], dtype=np.float32)
        f_curr, _, _ = _orient_north_up_west_left(raw_curr, geom["y_raw"], geom["x_raw"])
        cold_c = np.isfinite(f_curr[4]) & (f_curr[4] < 235.0)
        t_tr_c = float(0.6 * np.nanpercentile(f_curr[0][cold_c], 10) + 0.4 * np.nanpercentile(f_curr[4][cold_c], 5)) if int(np.count_nonzero(cold_c)) >= 50 else 215.0
        pure_tropo_c = np.full(f_curr[4].shape, float(np.clip(t_tr_c, 205.0, 226.0)), dtype=np.float32)

        res = _evaluate_pair_both_models(
            f_prev, f_curr, area_km2, lat_deg, lon_deg, era5_grid, pure_tropo_p, pure_tropo_c, cal_model
        )
        gc.collect()
        sem_full = res["sem_full"]
        sem_pure = res["sem_pure"]
        n_conv_full = res["n_conv_full"]
        n_conv_pure = res["n_conv_pure"]
        n_ot_full = res["n_ot_full"]
        n_ot_pure = res["n_ot_pure"]

        day_ref_full.append(sem_full[None, ...])
        day_pred_pure.append(sem_pure[None, ...])

        # OPERA verification on full domain
        ts_15 = ts_curr.round("15min")
        op0 = _get_opera(ts_15)
        dbzh_vis = np.full((h, w), np.nan, dtype=np.float32)
        qind_vis = np.zeros((h, w), dtype=np.float32)

        if op0 is not None:
            dbzh0, qind0 = op0
            rc = np.clip(opera_row, 0, dbzh0.shape[0] - 1)
            cc = np.clip(opera_col, 0, dbzh0.shape[1] - 1)
            dbzh_vis = np.where(opera_inside, dbzh0[rc, cc], np.nan)
            qind_vis = np.where(opera_inside, qind0[rc, cc], np.nan)

            valid_obs = opera_inside & (np.isclose(dbzh_vis, -32.0) | (np.isfinite(dbzh_vis) & (qind_vis >= 0.40)))
            lag_slots = [op0]
            for dm in (-15, 15):
                op_d = _get_opera(ts_15 + pd.Timedelta(minutes=dm))
                if op_d is not None:
                    lag_slots.append(op_d)

            for m_name, sem_map in [
                ("nwcsaf_rdt_full_era5_cal", sem_full),
                ("nwcsaf_rdt_seviri_pure_gen", sem_pure),
            ]:
                lbl, _ = ndimage.label(sem_map >= 1, structure=np.ones((3, 3), dtype=bool))
                bboxes = ndimage.find_objects(lbl)
                for oid, bbox in enumerate(bboxes, start=1):
                    if bbox is None:
                        continue
                    omask_loc = lbl[bbox] == oid
                    obs_loc = omask_loc & valid_obs[bbox]
                    n_obs = int(np.count_nonzero(obs_loc))
                    if n_obs < 6:
                        continue
                    if float(n_obs / max(int(np.count_nonzero(omask_loc)), 1)) < 0.70:
                        continue

                    cell_area_km2 = float(np.nansum(area_km2[bbox][omask_loc]))
                    pred_conv = bool(np.any(np.isin(sem_map[bbox][omask_loc], [2, 3, 4])))

                    u_rc = np.unique(np.column_stack((rc[bbox][obs_loc], cc[bbox][obs_loc])), axis=0)
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

                    for th in (35.0, 40.0, 45.0, 50.0):
                        rdict = day_opera_stats[m_name][f"dbz_{int(th)}"]
                        area_th_0 = float(np.count_nonzero(good_0 >= th) * 4.0) if good_0.size > 0 else 0.0
                        min_core_km2 = 16.0 if th <= 40.0 else 8.0
                        truth_inst = 1 if area_th_0 >= min_core_km2 else (0 if max_z0 < 28.0 else -1)
                        truth_lag = 1 if max_z_lag >= th else (0 if max_z_lag < 28.0 else -1)

                        if truth_inst == 1 and pred_conv:
                            rdict["inst_cell_tp"] += 1
                            rdict["pixel_tp"] += n_obs
                            if cell_area_km2 >= 500.0:
                                rdict["meso500_tp"] += 1
                            else:
                                rdict["small_tp"] += 1
                        elif truth_inst == 0 and pred_conv:
                            rdict["inst_cell_fp"] += 1
                            rdict["pixel_fp"] += n_obs
                            if cell_area_km2 >= 500.0:
                                rdict["meso500_fp"] += 1
                            else:
                                rdict["small_fp"] += 1
                        elif truth_inst == 1 and not pred_conv:
                            rdict["inst_cell_fn"] += 1
                            rdict["pixel_fn"] += n_obs
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

        ir_frame = f_curr[4]
        base_rgb = _ir108_to_rgb_base(ir_frame)
        ts_str = ts_curr.strftime("%Y-%m-%d %H:%M UTC")

        rgb_opera = _overlay_opera_gradient_panel(
            base_rgb, dbzh_vis, qind_vis, opera_inside, qind_min=0.20, dbzh_min=-5.0
        )
        day_frames_opera.append(
            _compose_single_full_domain_frame(
                rgb_opera,
                f"1. OPERA Radar Reflectivity [dBZ] su Dominio Completo SEVIRI (763x1762) — {ts_str}",
                "Settimana 14-20 Settembre 2020 (Cadenza Uniforme 2h, 7 Giorni) | Gradiente continuo 'turbo' [-5..65 dBZ]",
                is_opera_video=True,
            )
        )

        rgb_full = _overlay_semantic_mask(base_rgb, sem_full)
        day_frames_full.append(
            _compose_single_full_domain_frame(
                rgb_full,
                f"2. RDT Nostro Completo (ERA5 + GEO-CT/HRW + OPERA CAL + Parallasse) — {ts_str}",
                f"Dominio Completo SEVIRI (763x1762) | Cadenza Uniforme 2h | Celle convettive attive: {n_conv_full} | Pixel OT: {n_ot_full}",
                is_opera_video=False,
            )
        )

        rgb_pure = _overlay_semantic_mask(base_rgb, sem_pure)
        day_frames_pure.append(
            _compose_single_full_domain_frame(
                rgb_pure,
                f"3. RDT Puro 7 Canali SEVIRI (Senza CAL, Senza ERA5, CT/HRW da IR + Parallasse) — {ts_str}",
                f"Dominio Completo SEVIRI (763x1762) | Cadenza Uniforme 2h | Celle convettive attive: {n_conv_pure} | Pixel OT: {n_ot_pure}",
                is_opera_video=False,
            )
        )

        dt_s = time.perf_counter() - t0
        print(
            f"  [Day {zp.stem} {s_idx:02d}/{len(slot_indices):02d}] {ts_str} | "
            f"Full conv={n_conv_full}, Pure conv={n_conv_pure} | {dt_s:.1f}s",
            flush=True,
        )

    opera_cache.clear()
    gc.collect()
    return {
        "day": zp.stem,
        "frames_opera": day_frames_opera,
        "frames_full": day_frames_full,
        "frames_pure": day_frames_pure,
        "ref_full": np.concatenate(day_ref_full, axis=0),
        "pred_pure": np.concatenate(day_pred_pure, axis=0),
        "opera_stats": day_opera_stats,
    }


def run_september_week_full_domain_and_videos(
    cal_model: CalibratedRdtModel,
) -> dict[str, Any]:
    """Run full-domain (763 x 1762) evaluation and generate 3 full-domain MP4 videos
    with 100% uniform 2-hour cadence across all 7 days (2020-09-14 to 2020-09-20,
    84 uniform slots -> 70.0 seconds video duration at 1.2 fps, zero speed changes).
    """
    from concurrent.futures import ProcessPoolExecutor

    zarr_days = sorted(SEPT_ZARR_ROOT.glob("202009*.zarr"))
    week_zarrs = [p for p in zarr_days if "20200914" <= p.stem <= "20200920"]
    if not week_zarrs:
        week_zarrs = zarr_days

    print(f"[Sept 2020 Full Domain] Loading geometry from {week_zarrs[0].name}...", flush=True)
    geom = _load_full_domain_geometry(week_zarrs[0])
    h, w = geom["area_km2"].shape
    print(f"[Sept 2020 Full Domain] Grid shape: ({h}, {w}), pixel area mean={float(np.nanmean(geom['area_km2'])):.2f} km^2", flush=True)

    # Pre-warm the 3 synoptic ERA5 files (06:00, 13:00, 19:00 UTC) for all 7 days
    for zp in week_zarrs:
        for sh in (6, 13, 19):
            ts_syn = pd.Timestamp(f"{zp.stem[:4]}-{zp.stem[4:6]}-{zp.stem[6:8]}T{sh:02d}:00:00Z")
            fetch_era5_europe_hour(ts_syn, ERA5_CACHE_DIR)

    # Strictly uniform 2-hour schedule across every day of the 7-day week:
    # 00:15, 02:15, 04:15, 06:15, 08:15, 10:15, 12:15, 14:15, 16:15, 18:15, 20:15, 22:15 UTC
    # (12 uniform slots/day * 7 days = 84 uniform slots; at 1.2 fps -> 70.0 seconds video duration)
    uniform_daily_slots = list(range(3, 288, 24))
    total_slots = len(uniform_daily_slots) * len(week_zarrs)
    print(
        f"[Sept 2020 Full Domain] Launching 7-day parallel evaluation: {len(uniform_daily_slots)} uniform slots/day "
        f"x {len(week_zarrs)} days = {total_slots} uniform slots (duration={total_slots / 1.2:.1f}s at 1.2 fps).",
        flush=True,
    )

    t_start_all = time.perf_counter()
    worker_args = [(str(zp), uniform_daily_slots, geom) for zp in week_zarrs]
    with ProcessPoolExecutor(max_workers=4) as pool:
        day_results = list(pool.map(_process_single_day_worker, worker_args))

    # Sort strictly in chronological order (20200914 .. 20200920)
    day_results.sort(key=lambda r: r["day"])

    frames_opera: list[np.ndarray] = []
    frames_full: list[np.ndarray] = []
    frames_pure: list[np.ndarray] = []
    ref_full_list: list[np.ndarray] = []
    pred_pure_list: list[np.ndarray] = []

    opera_stats: dict[str, dict[str, dict[str, int]]] = {
        m_name: {
            f"dbz_{th}": {
                "inst_cell_tp": 0, "inst_cell_fp": 0, "inst_cell_fn": 0,
                "lag15_cell_tp": 0, "lag15_cell_fp": 0, "lag15_cell_fn": 0,
                "meso500_tp": 0, "meso500_fp": 0, "meso500_fn": 0,
                "small_tp": 0, "small_fp": 0, "small_fn": 0,
                "pixel_tp": 0, "pixel_fp": 0, "pixel_fn": 0,
            }
            for th in (35, 40, 45, 50)
        }
        for m_name in ("nwcsaf_rdt_full_era5_cal", "nwcsaf_rdt_seviri_pure_gen")
    }

    for dres in day_results:
        frames_opera.extend(dres["frames_opera"])
        frames_full.extend(dres["frames_full"])
        frames_pure.extend(dres["frames_pure"])
        ref_full_list.append(dres["ref_full"])
        pred_pure_list.append(dres["pred_pure"])
        for m_name in opera_stats:
            for th_k in opera_stats[m_name]:
                for stat_k, val in dres["opera_stats"][m_name][th_k].items():
                    opera_stats[m_name][th_k][stat_k] += int(val)

    mp4_opera = OUT_DIR / "sept2020_full_domain_1_OPERA_radar.mp4"
    mp4_full = OUT_DIR / "sept2020_full_domain_2_RDT_completo.mp4"
    mp4_pure = OUT_DIR / "sept2020_full_domain_3_RDT_puro.mp4"

    video_fps = 1.2  # 84 uniform frames / 1.2 fps = 70.0 seconds (1 min 10 s)
    print(
        f"[Sept 2020 Full Domain] Encoding 3 uniform-cadence full-domain MP4 videos "
        f"({len(frames_opera)} frames at {video_fps} fps -> {len(frames_opera)/video_fps:.1f}s duration) in {time.perf_counter()-t_start_all:.1f}s...",
        flush=True,
    )
    _encode_mp4(frames_opera, mp4_opera, fps=video_fps)
    _encode_mp4(frames_full, mp4_full, fps=video_fps)
    _encode_mp4(frames_pure, mp4_pure, fps=video_fps)

    m_full_dom = compute_comparison_metrics(
        np.concatenate(ref_full_list, axis=0),
        np.concatenate(pred_pure_list, axis=0),
    )

    opera_summary: dict[str, Any] = {}
    for m_name, mdict in opera_stats.items():
        opera_summary[m_name] = {}
        for th_k, d in mdict.items():
            opera_summary[m_name][th_k] = {
                "instantaneous_all_cells": _prf(d["inst_cell_tp"], d["inst_cell_fp"], d["inst_cell_fn"]),
                "lag15min_all_cells": _prf(d["lag15_cell_tp"], d["lag15_cell_fp"], d["lag15_cell_fn"]),
                "mesoscale_ge500km2_cells": _prf(d["meso500_tp"], d["meso500_fp"], d["meso500_fn"]),
                "small_lt500km2_cells": _prf(d["small_tp"], d["small_fp"], d["small_fn"]),
                "pixel_level": _prf(d["pixel_tp"], d["pixel_fp"], d["pixel_fn"]),
            }

    return {
        "september_week_days": [p.stem for p in week_zarrs],
        "num_full_domain_slots": len(frames_opera),
        "uniform_cadence_hours": 2.0,
        "video_fps": video_fps,
        "video_duration_seconds": round(len(frames_opera) / video_fps, 2),
        "domain_shape_hw": [int(h), int(w)],
        "pure_vs_full_agreement_full_domain": {
            "overall_acc": round(m_full_dom["overall_pixel_accuracy"], 4),
            "cloud_iou": round(m_full_dom["regimes"]["cloud_tower_mask_ge1"]["iou_csi"], 4),
            "cloud_dice": round(m_full_dom["regimes"]["cloud_tower_mask_ge1"]["dice_f1"], 4),
            "conv_iou": round(m_full_dom["regimes"]["convective_mask_234"]["iou_csi"], 4),
            "conv_dice": round(m_full_dom["regimes"]["convective_mask_234"]["dice_f1"], 4),
            "conv_pod": round(m_full_dom["regimes"]["convective_mask_234"]["pod_recall"], 4),
            "conv_prec": round(m_full_dom["regimes"]["convective_mask_234"]["precision"], 4),
        },
        "opera_verification_full_domain": opera_summary,
        "videos": {
            "video_1_opera": str(mp4_opera),
            "video_2_rdt_completo": str(mp4_full),
            "video_3_rdt_puro": str(mp4_pure),
        },
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cal_model = CalibratedRdtModel.from_json(RDT_ORIG_ROOT / "rdt_nwcsaf" / "default_calibrated_model.json")
    out_json = OUT_DIR / "september_2020_full_domain_verification.json"

    print("=== Stage 1: Held-Out Ablations (IR-only Ancillaries & Parallax Correction) ===", flush=True)
    if out_json.exists():
        existing = json.loads(out_json.read_text(encoding="utf-8"))
        ablation_res = existing.get("heldout_ablations") or evaluate_heldout_ancillary_and_parallax_ablations(cal_model)
    else:
        ablation_res = evaluate_heldout_ancillary_and_parallax_ablations(cal_model)

    print(
        "  Pure GEN (no ancillaries) Convective Dice vs Full:",
        ablation_res["pure_gen_without_ancillaries_vs_full"],
        flush=True,
    )
    print(
        "  Pure GEN (+ IR-only CT/HRW) Convective Dice vs Full:",
        ablation_res["pure_gen_with_ir_ancillaries_vs_full"],
        flush=True,
    )

    print("=== Stage 2: Full-Domain (763x1762) September 2020 Week Verification & 3 Videos ===", flush=True)
    sept_res = run_september_week_full_domain_and_videos(cal_model)

    full_report = {
        "heldout_ablations": ablation_res,
        "september_2020_full_domain": sept_res,
    }
    out_json.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"Saved complete report -> {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
