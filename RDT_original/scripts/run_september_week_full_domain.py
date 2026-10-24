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
import os
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

# Full 4-phase official RDT convective lifecycle schema:
# 1: Non-convective cold cloud -> Steel Blue
# 2: Triggering convective cell -> Bright Yellow
# 3: Growing convective cell -> Vibrant Orange
# 4: Mature convective cell -> Vivid Crimson Red
# 5: Decaying convective cell -> Dark Brick / Bordeaux Red
# 6: Overshooting Top (OT) -> Bright Magenta / Violet
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    1: (59, 130, 246),
    2: (250, 204, 21),
    3: (249, 115, 22),
    4: (220, 38, 38),
    5: (153, 27, 27),
    6: (245, 105, 255),
}


class SmoothSemanticRenderer:
    """Renders semantic masks with temporal color smoothing (EMA) to prevent 1-frame flickering."""

    def __init__(self, shape: tuple[int, int], gamma: float = 0.38):
        h, w = shape
        self.gamma = gamma
        self.conv_color_buffer = np.zeros((h, w, 3), dtype=np.float32)
        self.conv_active = np.zeros((h, w), dtype=bool)

    def render(self, base_rgb: np.ndarray, sem_label: np.ndarray) -> np.ndarray:
        out = base_rgb.astype(np.float32).copy()

        # 1. Non-convective cold cloud (Class 1 -> Steel Blue)
        cirrus = (sem_label == 1)
        if np.any(cirrus):
            alpha_c = 0.55
            out[cirrus, 0] = (1.0 - alpha_c) * out[cirrus, 0] + alpha_c * 59.0
            out[cirrus, 1] = (1.0 - alpha_c) * out[cirrus, 1] + alpha_c * 130.0
            out[cirrus, 2] = (1.0 - alpha_c) * out[cirrus, 2] + alpha_c * 246.0

        # 2. Convective lifecycle classes (2: Triggering, 3: Growing, 4: Mature, 5: Decaying)
        conv_mask = (sem_label >= 2) & (sem_label <= 5)
        if np.any(conv_mask):
            target_rgb = np.zeros_like(self.conv_color_buffer)
            for cls_id, (r, g, b) in [
                (2, (250.0, 204.0, 21.0)),  # Triggering -> Bright Yellow
                (3, (249.0, 115.0, 22.0)),  # Growing -> Vibrant Orange
                (4, (220.0, 38.0, 38.0)),   # Mature -> Vivid Crimson Red
                (5, (153.0, 27.0, 27.0)),   # Decaying -> Dark Brick Red
            ]:
                m = (sem_label == cls_id)
                if np.any(m):
                    target_rgb[m] = [r, g, b]

            # Smoothly blend for persistent convective pixels
            smooth_pixels = conv_mask & self.conv_active
            self.conv_color_buffer[smooth_pixels] = (
                (1.0 - self.gamma) * self.conv_color_buffer[smooth_pixels] + self.gamma * target_rgb[smooth_pixels]
            )
            # Newly convective pixels take target color immediately
            new_pixels = conv_mask & (~self.conv_active)
            self.conv_color_buffer[new_pixels] = target_rgb[new_pixels]

            alpha = 0.82
            out[conv_mask] = (1.0 - alpha) * out[conv_mask] + alpha * self.conv_color_buffer[conv_mask]

        self.conv_active = conv_mask

        # 3. Overshooting Tops (Class 6 -> Bright Vivid Magenta with 1-px dilation for high visibility)
        ot_mask = (sem_label == 6)
        if np.any(ot_mask):
            ot_dil = ndimage.binary_dilation(ot_mask, iterations=1)
            alpha_ot = 0.96
            out[ot_dil, 0] = (1.0 - alpha_ot) * out[ot_dil, 0] + alpha_ot * 245.0
            out[ot_dil, 1] = (1.0 - alpha_ot) * out[ot_dil, 1] + alpha_ot * 105.0
            out[ot_dil, 2] = (1.0 - alpha_ot) * out[ot_dil, 2] + alpha_ot * 255.0

        return np.clip(out, 0, 255).astype(np.uint8)

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


def fill_internal_mask_holes(mask_2d: np.ndarray, max_hole_size: int = 16) -> np.ndarray:
    """Fill small discrete coordinate-splatting holes (<= 16 px) to eliminate white necklaces."""
    if not np.any(mask_2d > 0):
        return mask_2d
    is_cloud = mask_2d > 0
    filled = ndimage.binary_fill_holes(is_cloud)
    holes = filled & (~is_cloud)
    if not np.any(holes):
        return mask_2d
    lbl, num_holes = ndimage.label(holes, structure=np.ones((3, 3), dtype=bool))
    counts = np.bincount(lbl.ravel())
    small_holes = np.isin(lbl, np.where((counts > 0) & (counts <= max_hole_size))[0]) & (lbl > 0)
    out = mask_2d.copy()
    out[small_holes] = ndimage.maximum_filter(mask_2d, size=3)[small_holes]
    return out


def _ir108_to_rgb_base(ir108_k: np.ndarray) -> np.ndarray:
    """Convert IR 10.8 brightness temperature (K) to an enhanced grayscale RGB image."""
    norm = np.clip((300.0 - np.nan_to_num(ir108_k, nan=295.0)) / (300.0 - 202.0), 0.0, 1.0)
    gray = (30.0 + 205.0 * norm).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _overlay_semantic_mask(base_rgb: np.ndarray, sem_label: np.ndarray) -> np.ndarray:
    """Alpha-blend strictly the semantic classes onto IR10.8 RGB."""
    out = base_rgb.astype(np.float32).copy()
    for cls_id, (r, g, b) in CLASS_COLORS.items():
        mask = sem_label == cls_id
        if np.any(mask):
            alpha = 0.55 if cls_id == 1 else (0.96 if cls_id == 6 else 0.82)
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
    is_opera_video: bool = False,
) -> np.ndarray:
    """Compose a single-panel full-domain (763 x 1762) video frame with single title and clean legend."""
    h, w, _ = rgb_map.shape
    header_h = 48
    footer_h = 44
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
        font_h1 = ImageFont.truetype("arialbd.ttf", 20)
        font_leg = ImageFont.truetype("arialbd.ttf", 13)
    except Exception:
        font_h1 = ImageFont.load_default()
        font_leg = ImageFont.load_default()

    # Single-line title without subtitle
    draw.text((pad + 4, 14), title, fill=(248, 250, 252), font=font_h1)

    panel_img = Image.fromarray(rgb_map)
    canvas.paste(panel_img, (pad, header_h))
    draw.rectangle([pad - 1, header_h - 1, pad + w, header_h + h], outline=(71, 85, 105), width=1)

    fy = header_h + h + 13
    lx = pad + 6
    if is_opera_video:
        cb_w = 260
        for dx in range(cb_w):
            val = dx / max(cb_w - 1, 1)
            r, g, b, _ = TURBO_CMAP(val)
            col = (int(r * 255), int(g * 255), int(b * 255))
            draw.line([(lx + dx, fy), (lx + dx, fy + 16)], fill=col)
        draw.rectangle([lx, fy, lx + cb_w, fy + 16], outline=(210, 215, 225))
        draw.text(
            (lx + cb_w + 12, fy + 1),
            "Riflettivita Radar OPERA [-5 .. 20 .. 35 .. 50 .. 65 dBZ] (QIND >= 0.20) su SEVIRI IR 10.8 [K]",
            fill=(226, 232, 240),
            font=font_leg,
        )
    else:
        legend_items = [
            ((59, 130, 246), "Nube fredda"),
            ((250, 204, 21), "Triggering"),
            ((249, 115, 22), "Growing"),
            ((220, 38, 38), "Mature"),
            ((153, 27, 27), "Decaying"),
            ((245, 105, 255), "Overshooting Top (OT)"),
        ]
        for color, label in legend_items:
            draw.rectangle([lx, fy, lx + 18, fy + 16], fill=color, outline=(255, 255, 255))
            draw.text((lx + 24, fy + 1), label, fill=(226, 232, 240), font=font_leg)
            lx += 24 + int(len(label) * 7.5) + 24

    return np.asarray(canvas, dtype=np.uint8)


class FfmpegStreamWriter:
    """Streams raw RGB frames directly to ffmpeg stdin to encode MP4 with zero disk churn."""

    def __init__(self, out_path: Path, width: int, height: int, fps: float = 33.6):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path = out_path
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", f"{fps:.4f}",
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "19",
            str(out_path),
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def write_frame(self, frame_rgb: np.ndarray) -> None:
        self.proc.stdin.write(frame_rgb.tobytes())

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()


def _encode_mp4(frames: list[np.ndarray], out_mp4: Path, fps: float = 1.2) -> None:
    """Encode list of RGB frames into a smooth H.264 MP4 via ffmpeg."""
    if not frames:
        return
    h, w, _ = frames[0].shape
    writer = FfmpegStreamWriter(out_mp4, w, h, fps=fps)
    for frame in frames:
        writer.write_frame(frame)
    writer.close()


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
    # Extract only the key for cell["t_bt_k"] into a plain lightweight dict to break the reference cycle to hierarchy
    t_bt = cell.get("t_bt_k", 240.0)
    vp: dict[int, float] = {}
    raw_vp = cell.get("vertical_profile")
    if raw_vp is not None:
        try:
            k = int(round((t_bt - 198.15) / 1.0))
            if k in raw_vp:
                vp[k] = float(raw_vp[k])
        except Exception:
            pass
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
        "vertical_profile": vp,
        "btd_wv62_ir108_max_k": cell["btd_wv62_ir108_max_k"],
        "btd_wv62_ir108_p90_k": cell["btd_wv62_ir108_p90_k"],
        "cooling_rate_k_per_h": cell["cooling_rate_k_per_h"],
        "is_convective": cell["is_convective"],
        "vertical_cat": cell["vertical_cat"],
        "phase_code": cell["phase_code"],
        "severity": cell["severity"],
    }


def _process_single_day_worker(args: tuple[str, dict[str, Any], str, float]) -> dict[str, Any]:
    """Worker process that continuously processes all 288 5-minute frames for a single day Zarr,
    streaming the 3 video outputs directly to ffmpeg without accumulating frames in RAM.
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

    zarr_path_str, geom, out_dir_str, fps = args
    zp = Path(zarr_path_str)
    out_dir = Path(out_dir_str)
    cal_model = CalibratedRdtModel.from_json(RDT_ORIG_ROOT / "rdt_nwcsaf" / "default_calibrated_model.json")

    area_km2 = geom["area_km2"]
    lat_deg = geom["lat_deg"]
    lon_deg = geom["lon_deg"]
    x_m, y_m = geom["x_m"], geom["y_m"]
    opera_row, opera_col, opera_inside = geom["opera_row"], geom["opera_col"], geom["opera_inside"]
    h, w = area_km2.shape

    # Exact canvas dimensions matching _compose_single_full_domain_frame
    pad = 12
    total_w = w + 2 * pad
    total_h = 48 + h + 44 + 2 * pad
    if total_w % 2 == 1:
        total_w += 1
    if total_h % 2 == 1:
        total_h += 1

    day_opera_mp4 = out_dir / f"_day_{zp.stem}_opera.mp4"
    day_full_mp4 = out_dir / f"_day_{zp.stem}_full.mp4"
    day_pure_mp4 = out_dir / f"_day_{zp.stem}_pure.mp4"
    stats_cache = out_dir / f"_stats_{zp.stem}.json"
    prog_file = out_dir / f"_progress_{zp.stem}.txt"
    cache_file = out_dir / f"_cache_{zp.stem}.npz"

    # If already computed with cache and 100% complete (288/288 frames), resume without re-computing!
    if (
        cache_file.exists()
        and prog_file.exists()
        and "DONE (288/288 frames)" in prog_file.read_text(encoding="utf-8")
        and day_opera_mp4.exists() and day_full_mp4.exists() and day_pure_mp4.exists()
    ):
        print(f"[Day {zp.stem}] Found completed 288-frame video segments. Resuming without re-computing!", flush=True)
        stats = json.loads(stats_cache.read_text(encoding="utf-8")) if stats_cache.exists() else {}
        return {
            "day": zp.stem,
            "day_opera_mp4": str(day_opera_mp4),
            "day_full_mp4": str(day_full_mp4),
            "day_pure_mp4": str(day_pure_mp4),
            "opera_stats": stats,
        }

    writer_opera = FfmpegStreamWriter(day_opera_mp4, total_w, total_h, fps=fps)
    writer_full = FfmpegStreamWriter(day_full_mp4, total_w, total_h, fps=fps)
    writer_pure = FfmpegStreamWriter(day_pure_mp4, total_w, total_h, fps=fps)

    g = zarr.open(str(zp), mode="r")
    # Read actual unix timestamps from the Zarr archive and sort strictly chronologically
    if "time" in g:
        raw_unix = np.asarray(g["time"][:], dtype=np.int64)
        chronological_indices = np.argsort(raw_unix)
        if zp.stem == "20200917":
            # Specific fix for 20200917 where chunks 19..30 (13:30-14:25 UTC) and 0..18 (14:30-16:00 UTC) were swapped in raw Zarr timestamps
            chronological_indices = np.concatenate([
                chronological_indices[:162],
                chronological_indices[181:193],
                chronological_indices[162:181],
                chronological_indices[193:],
            ])
        times_all = pd.date_range(f"{zp.stem[:4]}-{zp.stem[4:6]}-{zp.stem[6:8]} 00:00:00", periods=288, freq="5min", tz="UTC")
    else:
        chronological_indices = np.arange(288)
        times_all = pd.date_range(zp.stem, periods=288, freq="5min", tz="UTC")

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

    pure_cfg = NwcSafRdtConfig(enforce_stretched_mask=False, nwp_conv_filter=False, discri_scheme="GEN")
    full_cfg = NwcSafRdtConfig(enforce_stretched_mask=False, nwp_conv_filter=True, discri_scheme="CAL", limittropo=True)

    prev_f: np.ndarray | None = None
    prev_labels: np.ndarray | None = None
    prev_hierarchy: ThresholdSliceHierarchy | None = None
    prev_cells_full: dict[int, dict[str, Any]] = {}
    prev_cells_pure: dict[int, dict[str, Any]] = {}
    hist_full: dict[int, list[dict[str, Any]]] = {}
    hist_pure: dict[int, list[dict[str, Any]]] = {}

    # Sequential continuous 5-minute stepping in true chronological order
    num_frames = min(int(os.environ.get("RDT_MAX_FRAMES_PER_DAY", "288")), len(chronological_indices))
    renderer_full = SmoothSemanticRenderer((h, w), gamma=0.38)
    renderer_pure = SmoothSemanticRenderer((h, w), gamma=0.38)
    daily_sem_full = np.zeros((num_frames, h, w), dtype=np.uint8)
    daily_sem_pure = np.zeros((num_frames, h, w), dtype=np.uint8)
    for step_idx in range(num_frames):
        t0 = time.perf_counter()
        raw_idx = int(chronological_indices[step_idx])
        ts_curr = times_all[step_idx]
        nearest_sh = min(synoptic_hours, key=lambda sh: abs(ts_curr.hour - sh))
        era5_grid = era5_day_grids[nearest_sh]

        raw_curr = np.asarray(g["seviri"][raw_idx], dtype=np.float32)
        f_curr, _, _ = _orient_north_up_west_left(raw_curr, geom["y_raw"], geom["x_raw"])
        cold_c = np.isfinite(f_curr[4]) & (f_curr[4] < 235.0)
        t_tr_c = float(0.6 * np.nanpercentile(f_curr[0][cold_c], 10) + 0.4 * np.nanpercentile(f_curr[4][cold_c], 5)) if int(np.count_nonzero(cold_c)) >= 50 else 215.0
        pure_tropo_c = np.full(f_curr[4].shape, float(np.clip(t_tr_c, 205.0, 226.0)), dtype=np.float32)
        pure_tropo_c_scalar = float(np.clip(t_tr_c, 205.0, 226.0))

        ct_curr = compute_nwcsaf_cloud_type(
            f_curr[4], ir120=f_curr[5], ir087=f_curr[2], wv062=f_curr[0], wv073=f_curr[1],
            t_tropo_k=pure_tropo_c, t_surf_k=292.0,
        )

        if prev_f is not None:
            dy_px, dx_px = compute_hrw_amv_guess_field(
                prev_f[4], f_curr[4], prev_wv062=prev_f[0], curr_wv062=f_curr[0], dt_seconds=300.0
            )
            guess_motion_yx = np.stack([dy_px, dx_px], axis=0)
        else:
            guess_motion_yx = None

        curr_labels, curr_cells_base, curr_hierarchy, _ = detect_rdt_cells(
            f_curr, area_km2, pure_cfg, ct_cloudy_mask=ct_curr.opaque_cloud_mask, tropopause_temp_k=pure_tropo_c
        )

        if prev_labels is not None and prev_hierarchy is not None and prev_f is not None:
            parents, children, edges, primary_parent, events, _ = track_rdt_cells(
                prev_cells_pure, prev_labels, prev_hierarchy, prev_f[4],
                curr_cells_base, curr_labels, curr_hierarchy, f_curr[4],
                area_km2, pure_cfg, guess_motion_yx=guess_motion_yx,
            )
        else:
            parents, children, edges, primary_parent = {}, {}, {}, {}
            events = {cid: "new" for cid in curr_cells_base}

        curr_cells_full: dict[int, dict[str, Any]] = {}
        curr_cells_pure: dict[int, dict[str, Any]] = {}
        new_hist_full: dict[int, list[dict[str, Any]]] = {}
        new_hist_pure: dict[int, list[dict[str, Any]]] = {}

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
            prev_f_c = prev_cells_full.get(p_main) if p_main is not None else None
            conv_p_f = prev_f_c if (prev_f_c is not None and prev_f_c.get("is_convective")) else None
            if conv_p_f is None:
                for pid in sorted(p_list):
                    if prev_cells_full.get(pid, {}).get("is_convective"):
                        conv_p_f = prev_cells_full[pid]
                        break
            hf = hist_full.get(p_main, []) if (p_main is not None and ev in ("continuation", "merge_main", "split_main")) else []
            df = discriminate_and_classify_cell(
                cf, prev_f_c, conv_p_f, hf, prev_hierarchy, curr_hierarchy, ot_c, 5.0, full_cfg,
                nwp_mask_value=nwp_val, lightning_strokes=0, cell_lat_deg=c_lat,
                calibrated_model=cal_model, cell_tropo_temp_k=tropo_f,
            )
            cf.update(df)
            curr_cells_full[cid] = cf
            new_hist_full[cid] = (hf + [_make_snapshot(cf, cid, ev, step_idx)])[-12:]
            sem_full[bbox][mask] = int(df["class_id"])
            if df["is_convective"]:
                n_conv_full += 1
                if np.any(ot_conf_loc):
                    ot_mask_full[bbox][ot_conf_loc] = True
                    sem_full[bbox][ot_conf_loc] = 6

            # Pure model discrimination
            cp = dict(base_c)
            prev_p_c = prev_cells_pure.get(p_main) if p_main is not None else None
            conv_p_p = prev_p_c if (prev_p_c is not None and prev_p_c.get("is_convective")) else None
            if conv_p_p is None:
                for pid in sorted(p_list):
                    if prev_cells_pure.get(pid, {}).get("is_convective"):
                        conv_p_p = prev_cells_pure[pid]
                        break
            hp = hist_pure.get(p_main, []) if (p_main is not None and ev in ("continuation", "merge_main", "split_main")) else []
            dp = discriminate_and_classify_cell(
                cp, prev_p_c, conv_p_p, hp, prev_hierarchy, curr_hierarchy, ot_c, 5.0, pure_cfg,
                nwp_mask_value=2, lightning_strokes=0, cell_lat_deg=c_lat,
                calibrated_model=None, cell_tropo_temp_k=pure_tropo_c_scalar,
            )
            cp.update(dp)
            curr_cells_pure[cid] = cp
            new_hist_pure[cid] = (hp + [_make_snapshot(cp, cid, ev, step_idx)])[-12:]
            sem_pure[bbox][mask] = int(dp["class_id"])
            if dp["is_convective"]:
                n_conv_pure += 1
                if np.any(ot_conf_loc):
                    ot_mask_pure[bbox][ot_conf_loc] = True
                    sem_pure[bbox][ot_conf_loc] = 6

        # Stage 6: 3D Geostationary Parallax Correction
        sem_full = apply_parallax_correction_2d(
            sem_full, f_curr[4], lat_deg, lon_deg, t_surf_k=era5_grid.t_surf_k, t_tropo_k=era5_grid.t_tropo_k, fill_value=0
        )
        sem_pure = apply_parallax_correction_2d(
            sem_pure, f_curr[4], lat_deg, lon_deg, t_surf_k=292.0, t_tropo_k=pure_tropo_c, fill_value=0
        )
        # Fill discrete forward-splatting coordinate holes (eliminates white necklaces)
        sem_full = fill_internal_mask_holes(sem_full, max_hole_size=16)
        sem_pure = fill_internal_mask_holes(sem_pure, max_hole_size=16)

        # Update tracking state for next step
        prev_f = f_curr
        prev_labels = curr_labels
        prev_hierarchy = curr_hierarchy
        prev_cells_full = curr_cells_full
        prev_cells_pure = curr_cells_pure
        hist_full = new_hist_full
        hist_pure = new_hist_pure

        # OPERA radar matching (every 15 min or closest)
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

            if step_idx % 3 == 0:  # Evaluate contingency table on 15-min intervals
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
                        pred_conv = bool(np.any(sem_map[bbox][omask_loc] >= 2))

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

        # Render 3 full-domain frames and write to stdin stream
        ir_frame = f_curr[4]
        base_rgb = _ir108_to_rgb_base(ir_frame)
        ts_str = ts_curr.strftime("%Y-%m-%d %H:%M UTC")

        # Save masks to daily arrays for caching
        daily_sem_full[step_idx] = sem_full.astype(np.uint8)
        daily_sem_pure[step_idx] = sem_pure.astype(np.uint8)

        # 1. OPERA radar
        rgb_opera = _overlay_opera_gradient_panel(
            base_rgb, dbzh_vis, qind_vis, opera_inside, qind_min=0.20, dbzh_min=-5.0
        )
        f_op = _compose_single_full_domain_frame(
            rgb_opera,
            f"1. Radar OPERA (Verità a terra) — {ts_str}",
            is_opera_video=True,
        )
        writer_opera.write_frame(f_op)

        # Parallax-shifted IR background ensures 100% pixel-perfect alignment without exposed rims
        ir_full_ground = apply_parallax_correction_2d(
            ir_frame, ir_frame, lat_deg, lon_deg, t_surf_k=era5_grid.t_surf_k, t_tropo_k=era5_grid.t_tropo_k, fill_value=290.0
        )
        base_rgb_full = _ir108_to_rgb_base(ir_full_ground)
        rgb_full = renderer_full.render(base_rgb_full, sem_full)
        f_fu = _compose_single_full_domain_frame(
            rgb_full,
            f"2. RDT Completo 7 Canali SEVIRI (CAL, ERA5, CT/HRW, Parallasse) — {ts_str}",
            is_opera_video=False,
        )
        writer_full.write_frame(f_fu)

        ir_pure_ground = apply_parallax_correction_2d(
            ir_frame, ir_frame, lat_deg, lon_deg, t_surf_k=292.0, t_tropo_k=pure_tropo_c, fill_value=290.0
        )
        base_rgb_pure = _ir108_to_rgb_base(ir_pure_ground)
        rgb_pure = renderer_pure.render(base_rgb_pure, sem_pure)
        f_pu = _compose_single_full_domain_frame(
            rgb_pure,
            f"3. RDT Puro 7 Canali SEVIRI (Senza CAL, Senza ERA5, CT/HRW da IR, Parallasse) — {ts_str}",
            is_opera_video=False,
        )
        writer_pure.write_frame(f_pu)

        dt_s = time.perf_counter() - t0
        if (step_idx + 1) % 4 == 0 or step_idx == 0 or step_idx == num_frames - 1:
            import gc
            gc.collect()
            try:
                (out_dir / f"_progress_{zp.stem}.txt").write_text(
                    f"Frame {step_idx+1:03d}/{num_frames:03d} ({ts_str}) | Full={n_conv_full} Pure={n_conv_pure} | {dt_s:.2f}s",
                    encoding="utf-8",
                )
            except Exception:
                pass
            print(
                f"  [Day {zp.stem} Frame {step_idx+1:03d}/{num_frames:03d}] {ts_str} | "
                f"Full conv={n_conv_full}, Pure conv={n_conv_pure} | {dt_s:.2f}s/frame",
                flush=True,
            )

    writer_opera.close()
    writer_full.close()
    writer_pure.close()
    opera_cache.clear()
    try:
        stats_cache.write_text(json.dumps(day_opera_stats, indent=2), encoding="utf-8")
        np.savez_compressed(
            out_dir / f"_cache_{zp.stem}.npz",
            sem_full=daily_sem_full,
            sem_pure=daily_sem_pure,
        )
        (out_dir / f"_progress_{zp.stem}.txt").write_text(f"DONE ({num_frames}/{num_frames} frames)", encoding="utf-8")
        print(f"[Day {zp.stem}] Saved intermediate cache _cache_{zp.stem}.npz successfully!", flush=True)
    except Exception as e:
        print(f"[Day {zp.stem}] Warning saving cache: {e}", flush=True)

    return {
        "day": zp.stem,
        "day_opera_mp4": str(day_opera_mp4),
        "day_full_mp4": str(day_full_mp4),
        "day_pure_mp4": str(day_pure_mp4),
        "opera_stats": day_opera_stats,
    }


def run_september_week_full_domain_and_videos(
    cal_model: CalibratedRdtModel,
) -> dict[str, Any]:
    """Run full-domain (763 x 1762) continuous 5-minute evaluation and generate 3 fluid MP4 videos
    covering all 7 days of the September week (2020-09-14 to 2020-09-20, 2016 frames total at 33.6 fps -> 60.0s).
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

    # Pre-warm synoptic ERA5 files (06:00, 13:00, 19:00 UTC) for all 7 days
    for zp in week_zarrs:
        for sh in (6, 13, 19):
            ts_syn = pd.Timestamp(f"{zp.stem[:4]}-{zp.stem[4:6]}-{zp.stem[6:8]}T{sh:02d}:00:00Z")
            fetch_era5_europe_hour(ts_syn, ERA5_CACHE_DIR)

    video_fps = 33.6  # 2016 continuous frames / 33.6 fps = 60.0 seconds total duration
    total_frames = len(week_zarrs) * 288
    print(
        f"[Sept 2020 Full Domain] Launching 7-day parallel continuous 5-minute processing: "
        f"288 frames/day x {len(week_zarrs)} days = {total_frames} frames "
        f"(video duration = {total_frames / video_fps:.1f}s at {video_fps:.1f} fps).",
        flush=True,
    )

    t_start_all = time.perf_counter()
    worker_args = [(str(zp), geom, str(OUT_DIR), video_fps) for zp in week_zarrs]
    with ProcessPoolExecutor(max_workers=2) as pool:
        day_results = list(pool.map(_process_single_day_worker, worker_args))

    day_results.sort(key=lambda r: r["day"])

    # Concatenate the 7 daily segments into final 60s weekly MP4 videos
    print("[Sept 2020 Full Domain] Concatenating 7 daily segments into 3 final full-week 60s MP4 videos...", flush=True)
    mp4_opera = OUT_DIR / "sept2020_full_domain_1_OPERA_radar.mp4"
    mp4_full = OUT_DIR / "sept2020_full_domain_2_RDT_completo.mp4"
    mp4_pure = OUT_DIR / "sept2020_full_domain_3_RDT_puro.mp4"

    for tag, final_mp4 in [
        ("opera", mp4_opera),
        ("full", mp4_full),
        ("pure", mp4_pure),
    ]:
        list_file = OUT_DIR / f"_concat_list_{tag}.txt"
        lines = [f"file '{Path(r[f'day_{tag}_mp4']).name}'\n" for r in day_results]
        list_file.write_text("".join(lines), encoding="utf-8")
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file.name), "-c", "copy", str(final_mp4.name),
        ]
        subprocess.run(concat_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(OUT_DIR))
        list_file.unlink(missing_ok=True)
        for r in day_results:
            Path(r[f"day_{tag}_mp4"]).unlink(missing_ok=True)

    # Mirror videos to brain artifacts directory so user can view them in the chat UI
    artifact_dir = Path(r"C:\Users\Daniele.LOKI\.gemini\antigravity\brain\6ce9d192-9e7a-46f3-923e-ada02993fbe9")
    if artifact_dir.exists():
        for f in (mp4_opera, mp4_full, mp4_pure):
            shutil.copy2(f, artifact_dir / f.name)
        print(f"[Sept 2020 Full Domain] Successfully mirrored videos to artifacts directory: {artifact_dir}", flush=True)

    # Aggregate OPERA stats across the 7 days
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
        op_d = dres.get("opera_stats", {})
        for m_name in opera_stats:
            if m_name not in op_d:
                continue
            for th_k in opera_stats[m_name]:
                if th_k not in op_d[m_name]:
                    continue
                for stat_k, val in op_d[m_name][th_k].items():
                    opera_stats[m_name][th_k][stat_k] += int(val)

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

    total_time_s = time.perf_counter() - t_start_all
    print(f"[Sept 2020 Full Domain] Full run completed in {total_time_s/60.0:.1f} minutes!", flush=True)

    return {
        "september_week_days": [p.stem for p in week_zarrs],
        "num_full_domain_slots": total_frames,
        "cadence_minutes": 5.0,
        "video_fps": video_fps,
        "video_duration_seconds": round(total_frames / video_fps, 2),
        "domain_shape_hw": [int(h), int(w)],
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
