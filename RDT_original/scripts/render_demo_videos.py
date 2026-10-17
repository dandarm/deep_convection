#!/usr/bin/env python3
"""Render side-by-side demonstration videos (MP4 + GIF) of RDT convective cells.

Generates two demonstration videos in ``RDT_original/results/comparison_seviri_2020/``
with the exact same 4-panel layout:
1. Panel 1 (Top-Left): SEVIRI IR 10.8 + OPERA Radar Reflectivity [dBZ] continuous gradient
   (``turbo`` colormap from -5 to 65 dBZ with QIND >= 0.20, matching ``scripts/render_opera_ir108_week.py``).
2. Panel 2 (Top-Right): Full Multi-Source RDT (``nwcsaf_rdt_full_era5_cal``: ERA5 NWP + GEO-CT/HRW + OPERA CAL weights).
3. Panel 3 (Bottom-Left): Calibrated 7-Channel SEVIRI RDT (``nwcsaf_rdt_seviri_only_cal``: 7ch SEVIRI + OPERA CAL weights).
4. Panel 4 (Bottom-Right): Pure Uncalibrated 7-Channel SEVIRI RDT (``nwcsaf_rdt_seviri_pure_gen``:
   ZERO calibration weights, ZERO ERA5 NWP, ZERO ancillary products, ZERO lightning).

Only the 4 semantic class colors are used in Panels 2, 3, and 4 (no artificial yellow borders or radar contours).
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
RDT_ORIG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(RDT_ORIG_ROOT))
sys.path.insert(0, str(RDT_ORIG_ROOT / "scripts"))

from emma_gpm.rdt_teacher import projected_pixel_areas

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
from analyze_pure_and_opera_truth import run_pure_uncalibrated_seviri_rdt


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


def _ir108_to_rgb_base(ir108_k: np.ndarray) -> np.ndarray:
    """Convert IR 10.8 brightness temperature (K) to an enhanced grayscale RGB image."""
    norm = np.clip((300.0 - np.nan_to_num(ir108_k, nan=295.0)) / (300.0 - 202.0), 0.0, 1.0)
    gray = (30.0 + 205.0 * norm).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _overlay_semantic_mask(
    base_rgb: np.ndarray,
    sem_label: np.ndarray,
) -> np.ndarray:
    """Alpha-blend strictly the 4 semantic classes onto IR10.8 RGB (no extra borders or radar lines)."""
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
    """Render OPERA DBZH continuous gradient ('turbo', -5..65 dBZ) exactly like render_opera_ir108_week.py."""
    out = base_rgb.astype(np.float32).copy()
    # Shade outside OPERA coverage slightly darker
    outside = ~inside_mask | ~np.isfinite(dbzh_grid)
    if np.any(outside):
        out[outside] = 0.65 * out[outside] + 0.35 * np.array([75.0, 80.0, 90.0], dtype=np.float32)

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
        alpha = 0.76
        out[echo] = (1.0 - alpha) * out[echo] + alpha * rgba

    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_turbo_colorbar(draw: ImageDraw.ImageDraw, x0: int, y0: int, width: int, height: int, font: Any) -> None:
    """Draw a continuous -5..65 dBZ 'turbo' colorbar in the footer."""
    for dx in range(width):
        val = dx / max(width - 1, 1)
        r, g, b, _ = TURBO_CMAP(val)
        col = (int(r * 255), int(g * 255), int(b * 255))
        draw.line([(x0 + dx, y0), (x0 + dx, y0 + height)], fill=col)
    draw.rectangle([x0, y0, x0 + width, y0 + height], outline=(210, 215, 225))
    draw.text((x0 + width + 8, y0 + 1), "OPERA DBZH [-5 .. 35 .. 50 .. 65 dBZ]", fill=(230, 235, 245), font=font)


def _compose_4panel_frame(
    panels: list[tuple[str, str, np.ndarray]],
    header_title: str,
    header_subtitle: str,
    panel_scale: int = 2,
) -> np.ndarray:
    """Assemble a 2x2 grid of panels with dark header, titles, and clean 4-class + radar gradient legend."""
    h0, w0, _ = panels[0][2].shape
    hp, wp = h0 * panel_scale, w0 * panel_scale

    pad = 16
    title_h = 46
    header_h = 68
    footer_h = 56

    total_w = pad * 3 + wp * 2
    total_h = header_h + (title_h + hp + pad) * 2 + footer_h
    if total_w % 2 == 1:
        total_w += 1
    if total_h % 2 == 1:
        total_h += 1

    canvas = Image.new("RGB", (total_w, total_h), (18, 22, 30))
    draw = ImageDraw.Draw(canvas)

    try:
        font_h1 = ImageFont.truetype("arialbd.ttf", 20)
        font_h2 = ImageFont.truetype("arial.ttf", 15)
        font_p1 = ImageFont.truetype("arialbd.ttf", 15)
        font_p2 = ImageFont.truetype("arial.ttf", 13)
        font_leg = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_h1 = font_h2 = font_p1 = font_p2 = font_leg = ImageFont.load_default()

    draw.text((pad, 12), header_title, fill=(245, 248, 255), font=font_h1)
    draw.text((pad, 40), header_subtitle, fill=(175, 195, 225), font=font_h2)

    for idx, (p_title, p_stats, rgb) in enumerate(panels):
        row = idx // 2
        col = idx % 2
        x0 = pad + col * (wp + pad)
        y0 = header_h + row * (title_h + hp + pad)

        draw.rectangle([x0, y0, x0 + wp, y0 + title_h - 4], fill=(30, 36, 48))
        draw.text((x0 + 8, y0 + 5), p_title, fill=(255, 255, 255), font=font_p1)
        draw.text((x0 + 8, y0 + 25), p_stats, fill=(185, 215, 250), font=font_p2)

        pil_panel = Image.fromarray(rgb).resize((wp, hp), resample=Image.Resampling.NEAREST)
        canvas.paste(pil_panel, (x0, y0 + title_h))
        draw.rectangle([x0, y0 + title_h, x0 + wp, y0 + title_h + hp], outline=(75, 88, 110), width=1)

    # Footer: continuous OPERA colorbar + strictly the 4 semantic class swatches
    fy = total_h - footer_h + 18
    _draw_turbo_colorbar(draw, pad, fy, width=150, height=18, font=font_leg)

    fx = pad + 430
    class_legend = [
        ((59, 130, 246), "Classe 1: Nube Fredda Non-Conv."),
        ((249, 115, 22), "Classe 2: Convezione in Sviluppo"),
        ((220, 38, 38), "Classe 3: Convezione Matura/Decad."),
        ((232, 121, 249), "Classe 4: Overshooting Top"),
    ]
    for color, label in class_legend:
        draw.rectangle([fx, fy, fx + 18, fy + 18], fill=color, outline=(220, 220, 220))
        draw.text((fx + 24, fy + 1), label, fill=(230, 235, 245), font=font_leg)
        fx += 28 + len(label) * 7 + 20

    return np.asarray(canvas, dtype=np.uint8)


def _encode_mp4_and_gif(
    frames: list[np.ndarray],
    mp4_path: Path,
    gif_path: Path,
    fps: int = 4,
) -> None:
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / max(fps, 1)),
        loop=0,
        optimize=False,
    )

    ffmpeg_exe = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.EXE"
    if Path(ffmpeg_exe).exists():
        tmp_dir = mp4_path.parent / f"_tmp_frames_{mp4_path.stem}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            for i, p_img in enumerate(pil_frames):
                p_img.save(tmp_dir / f"frame_{i:04d}.png")
            cmd = [
                ffmpeg_exe,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(tmp_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(mp4_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _load_zarr_crop(
    ds: xr.Dataset,
    y_slice: slice,
    x_slice: slice,
) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray, str]:
    gx_full = ds["x"].values.copy()
    gy_full = ds["y"].values.copy()
    projection = ds.attrs["projection_wkt"]
    t_all = pd.to_datetime(ds["time"].values, utc=True)
    order = np.argsort(t_all)

    flip_y = bool(gy_full[0] < gy_full[-1])
    flip_x = bool(gx_full[0] > gx_full[-1])

    sub = np.asarray(ds["seviri"][:, :, y_slice, x_slice].values, dtype=np.float32)[order]
    x_sub = gx_full[x_slice]
    y_sub = gy_full[y_slice]
    if flip_y:
        sub = sub[:, :, ::-1, :]
        y_sub = y_sub[::-1]
    if flip_x:
        sub = sub[:, :, :, ::-1]
        x_sub = x_sub[::-1]

    return sub, pd.DatetimeIndex(t_all[order]), x_sub, y_sub, projection


def render_5min_outbreak_video(
    zarr_root: Path,
    opera_root: Path,
    era5_cache_dir: Path,
    cal_model: CalibratedRdtModel,
    out_dir: Path,
) -> dict[str, Path]:
    """Render Video 1: 5-minute RSS outbreak on 2020-06-08 (15:30 -> 18:00 UTC)."""
    print("[Video 1] Loading 2020-06-08 5-min window...", flush=True)
    ds = xr.open_zarr(zarr_root / "2020" / "20200608.zarr")
    ys = slice(298, 650)  # 352 px
    xs = slice(443, 795)  # 352 px
    arr_sub, t_sorted, x_m, y_m, projection = _load_zarr_crop(ds, ys, xs)

    keep = [i for i, ts in enumerate(t_sorted) if ts >= pd.Timestamp("2020-06-08T15:30:00Z")]
    times = pd.DatetimeIndex([t_sorted[i] for i in keep])
    raw = arr_sub[keep]

    area_km2 = projected_pixel_areas(x_m, y_m, projection, allow_invalid=True)
    xx_grid, yy_grid = np.meshgrid(x_m, y_m)
    lon_deg, lat_deg = Transformer.from_crs(
        CRS.from_user_input(projection), "EPSG:4326", always_xy=True
    ).transform(xx_grid, yy_grid)
    lat_deg = np.asarray(lat_deg, dtype=np.float32)
    lon_deg = np.asarray(lon_deg, dtype=np.float32)

    mid_ts = times[len(times) // 2]
    era5_eu = fetch_era5_europe_hour(mid_ts, era5_cache_dir)
    era5_grid = interpolate_era5_to_seviri_grid(era5_eu, x_m, y_m)

    cfg = NwcSafRdtConfig(enforce_stretched_mask=False, nwp_conv_filter=True, discri_scheme="CAL")
    full_out = run_nwcsaf_rdt_sequence(
        raw, times.to_numpy(), area_km2, cfg,
        lat_deg=lat_deg, lon_deg=lon_deg,
        tropopause_temp_k=era5_grid.t_tropo_k,
        nwp_instability_mask=era5_grid.nwpconv_mask,
        calibrated_model=cal_model,
        use_ancillary_ct=True, use_hrw_amv=True,
        u700_ms=era5_grid.u700_ms, v700_ms=era5_grid.v700_ms, t_surf_k=era5_grid.t_surf_k,
    )

    mid_f = raw[len(times) // 2]
    ch_dict = {
        "WV_062": mid_f[0], "WV_073": mid_f[1], "IR_087": mid_f[2],
        "IR_097": mid_f[3], "IR_108": mid_f[4], "IR_120": mid_f[5], "IR_134": mid_f[6],
    }
    sev_tropo_k, sev_nwpconv = predict_seviri_only_nwp_surrogates(cal_model, ch_dict, lat_deg=lat_deg)
    sev_cal_out = run_nwcsaf_rdt_sequence(
        raw, times.to_numpy(), area_km2, cfg,
        lat_deg=lat_deg, lon_deg=lon_deg,
        tropopause_temp_k=sev_tropo_k, nwp_instability_mask=sev_nwpconv,
        calibrated_model=cal_model, use_ancillary_ct=True, use_hrw_amv=True,
    )

    pure_out = run_pure_uncalibrated_seviri_rdt(raw, times.to_numpy(), area_km2, lat_deg, lon_deg)

    opera_row, opera_col, opera_inside = build_seviri_to_opera_lookup(x_m, y_m)
    opera_cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    frames_rgb: list[np.ndarray] = []
    for t_idx in range(6, len(times)):
        ts = pd.Timestamp(times[t_idx])
        ts_15 = ts.round("15min")
        op_key = ts_15.isoformat()
        if op_key not in opera_cache:
            op_path = opera_file_path(opera_root, ts_15)
            opera_cache[op_key] = read_opera_h5(op_path) if op_path.exists() else None
        op_data = opera_cache[op_key]

        ir108 = raw[t_idx, 4]
        base_rgb = _ir108_to_rgb_base(ir108)

        dbzh_grid = np.full_like(ir108, np.nan, dtype=np.float32)
        qind_grid = np.full_like(ir108, np.nan, dtype=np.float32)
        if op_data is not None:
            dbzh, qind = op_data
            rc = np.clip(opera_row, 0, dbzh.shape[0] - 1)
            cc = np.clip(opera_col, 0, dbzh.shape[1] - 1)
            dbzh_grid = np.where(opera_inside, dbzh[rc, cc], np.nan)
            qind_grid = np.where(opera_inside, qind[rc, cc], np.nan)

        f_rec = full_out["frame_records"][t_idx]
        s_rec = sev_cal_out["frame_records"][t_idx]
        p_rec = pure_out["frame_records"][t_idx]

        full_sem = full_out["semantic_label"][t_idx]
        sev_sem = sev_cal_out["semantic_label"][t_idx]
        pure_sem = pure_out["semantic_label"][t_idx]

        p1 = _overlay_opera_gradient_panel(base_rgb, dbzh_grid, qind_grid, opera_inside)
        p2 = _overlay_semantic_mask(base_rgb, full_sem)
        p3 = _overlay_semantic_mask(base_rgb, sev_sem)
        p4 = _overlay_semantic_mask(base_rgb, pure_sem)

        n_full_conv = int(sum(1 for c in f_rec["cells"].values() if c.get("is_convective", False)))
        n_sev_conv = int(sum(1 for c in s_rec["cells"].values() if c.get("is_convective", False)))
        n_pure_conv = int(sum(1 for c in p_rec["cells"].values() if c.get("is_convective", False)))
        op_35 = int(np.count_nonzero(np.isfinite(dbzh_grid) & (qind_grid >= 0.2) & (dbzh_grid >= 35.0)))
        op_45 = int(np.count_nonzero(np.isfinite(dbzh_grid) & (qind_grid >= 0.2) & (dbzh_grid >= 45.0)))

        panels = [
            (
                "1. OSSERVAZIONE RADAR: IR 10.8 + OPERA DBZH [Gradiente]",
                f"Pixel >= 35 dBZ: {op_35} | Nuclei intensi >= 45 dBZ: {op_45} px",
                p1,
            ),
            (
                "2. RDT COMPLETO (ERA5 NWP + GEO-CT/HRW + Pesi CAL OPERA)",
                f"Celle convettive: {n_full_conv} | Area convettiva: {int(np.isin(full_sem, [2,3,4]).sum())} px",
                p2,
            ),
            (
                "3. RDT 7ch SEVIRI CALIBRATO (Solo 7ch SEVIRI + Pesi CAL)",
                f"Celle convettive: {n_sev_conv} | Area convettiva: {int(np.isin(sev_sem, [2,3,4]).sum())} px",
                p3,
            ),
            (
                "4. RDT 7ch SEVIRI PURO (Zero Pesi CAL, Zero ERA5, Zero Ancillari)",
                f"Celle convettive: {n_pure_conv} | Area convettiva: {int(np.isin(pure_sem, [2,3,4]).sum())} px",
                p4,
            ),
        ]
        frame_img = _compose_4panel_frame(
            panels,
            header_title=f"Confronto Celle RDT-CW vs Radar OPERA | {ts.strftime('%Y-%m-%d %H:%M UTC')} (Cadenza 5 min)",
            header_subtitle=f"Dominio Europeo (Lat {float(lat_deg.min()):.1f}-{float(lat_deg.max()):.1f}N, Lon {float(lon_deg.min()):.1f}-{float(lon_deg.max()):.1f}E, 352x352 px)",
            panel_scale=2,
        )
        frames_rgb.append(frame_img)

    mp4_path = out_dir / "demo_5min_20200608_convective_cells.mp4"
    gif_path = out_dir / "demo_5min_20200608_convective_cells.gif"
    keyframe_path = out_dir / "demo_5min_20200608_keyframe.png"
    _encode_mp4_and_gif(frames_rgb, mp4_path, gif_path, fps=4)
    Image.fromarray(frames_rgb[len(frames_rgb) // 2]).save(keyframe_path)
    print(f"[Video 1] Saved {mp4_path.name} and {gif_path.name} ({len(frames_rgb)} frames)", flush=True)
    return {"mp4": mp4_path, "gif": gif_path, "keyframe": keyframe_path}


def render_48h_multiday_video(
    zarr_root: Path,
    opera_root: Path,
    era5_cache_dir: Path,
    cal_model: CalibratedRdtModel,
    out_dir: Path,
) -> dict[str, Path]:
    """Render Video 2: Continuous 2-day (48-hour, 2020-05-04 -> 2020-05-05) evolution with identical 4 panels."""
    print("[Video 2] Loading 2-day continuous SEVIRI + OPERA sequence (2020-05-04 to 2020-05-05)...", flush=True)
    y_slice = slice(330, 610)  # 280 px North-South
    x_slice = slice(300, 640)  # 340 px West-East

    ds4 = xr.open_zarr(zarr_root / "2020" / "20200504.zarr")
    ds5 = xr.open_zarr(zarr_root / "2020" / "20200505.zarr")

    arr4, t4, x_m, y_m, projection = _load_zarr_crop(ds4, y_slice, x_slice)
    arr5, t5, _, _, _ = _load_zarr_crop(ds5, y_slice, x_slice)

    times_all = t4.append(t5)
    arr_all = np.concatenate([arr4, arr5], axis=0)

    dedup_idx: list[int] = []
    seen_ts: set[pd.Timestamp] = set()
    for i, ts in enumerate(times_all):
        if ts.minute in (0, 15, 30, 45) and ts not in seen_ts:
            seen_ts.add(ts)
            dedup_idx.append(i)

    import gc

    # Use every 2nd 15-min frame (96 frames at 30-min cadence across the full 48 hours)
    dedup_idx = dedup_idx[::2]
    raw_sub = np.asarray(arr_all[dedup_idx], dtype=np.float32)
    del arr4, arr5, arr_all
    gc.collect()

    times_sub = pd.DatetimeIndex([times_all[i] for i in dedup_idx])
    area_km2 = projected_pixel_areas(x_m, y_m, projection, allow_invalid=True)

    xx_grid, yy_grid = np.meshgrid(x_m, y_m)
    lon_deg, lat_deg = Transformer.from_crs(
        CRS.from_user_input(projection), "EPSG:4326", always_xy=True
    ).transform(xx_grid, yy_grid)
    lat_deg = np.asarray(lat_deg, dtype=np.float32)
    lon_deg = np.asarray(lon_deg, dtype=np.float32)

    era5_d1 = interpolate_era5_to_seviri_grid(
        fetch_era5_europe_hour(pd.Timestamp("2020-05-04T15:00:00Z"), era5_cache_dir), x_m, y_m
    )
    era5_d2 = interpolate_era5_to_seviri_grid(
        fetch_era5_europe_hour(pd.Timestamp("2020-05-05T15:00:00Z"), era5_cache_dir), x_m, y_m
    )
    t_tropo_blend = 0.5 * (era5_d1.t_tropo_k + era5_d2.t_tropo_k)
    nwpconv_blend = np.logical_or(era5_d1.nwpconv_mask, era5_d2.nwpconv_mask)
    u700_blend = 0.5 * (era5_d1.u700_ms + era5_d2.u700_ms)
    v700_blend = 0.5 * (era5_d1.v700_ms + era5_d2.v700_ms)
    tsurf_blend = 0.5 * (era5_d1.t_surf_k + era5_d2.t_surf_k)

    cfg = NwcSafRdtConfig(enforce_stretched_mask=False, nwp_conv_filter=True, discri_scheme="CAL")
    print("[Video 2] Running 1/3: Full Multi-Source RDT...", flush=True)
    full_out = run_nwcsaf_rdt_sequence(
        raw_sub, times_sub.to_numpy(), area_km2, cfg,
        lat_deg=lat_deg, lon_deg=lon_deg,
        tropopause_temp_k=t_tropo_blend, nwp_instability_mask=nwpconv_blend,
        calibrated_model=cal_model, use_ancillary_ct=True, use_hrw_amv=True,
        u700_ms=u700_blend, v700_ms=v700_blend, t_surf_k=tsurf_blend,
    )
    full_sem_all = full_out["semantic_label"].copy()
    full_counts = [
        int(sum(1 for c in f["cells"].values() if c.get("is_convective", False)))
        for f in full_out["frame_records"]
    ]
    del full_out
    gc.collect()

    print("[Video 2] Running 2/3: Calibrated 7-Channel SEVIRI RDT...", flush=True)
    mid_f = raw_sub[len(times_sub) // 2]
    ch_dict = {
        "WV_062": mid_f[0], "WV_073": mid_f[1], "IR_087": mid_f[2],
        "IR_097": mid_f[3], "IR_108": mid_f[4], "IR_120": mid_f[5], "IR_134": mid_f[6],
    }
    sev_tropo_k, sev_nwpconv = predict_seviri_only_nwp_surrogates(cal_model, ch_dict, lat_deg=lat_deg)
    sev_cal_out = run_nwcsaf_rdt_sequence(
        raw_sub, times_sub.to_numpy(), area_km2, cfg,
        lat_deg=lat_deg, lon_deg=lon_deg,
        tropopause_temp_k=sev_tropo_k, nwp_instability_mask=sev_nwpconv,
        calibrated_model=cal_model, use_ancillary_ct=True, use_hrw_amv=True,
    )
    sev_sem_all = sev_cal_out["semantic_label"].copy()
    sev_counts = [
        int(sum(1 for c in f["cells"].values() if c.get("is_convective", False)))
        for f in sev_cal_out["frame_records"]
    ]
    del sev_cal_out
    gc.collect()

    print("[Video 2] Running 3/3: Pure Uncalibrated 7-Channel SEVIRI RDT (GEN)...", flush=True)
    pure_out = run_pure_uncalibrated_seviri_rdt(raw_sub, times_sub.to_numpy(), area_km2, lat_deg, lon_deg)
    pure_sem_all = pure_out["semantic_label"].copy()
    pure_counts = [
        int(sum(1 for c in f["cells"].values() if c.get("is_convective", False)))
        for f in pure_out["frame_records"]
    ]
    del pure_out
    gc.collect()

    opera_row, opera_col, opera_inside = build_seviri_to_opera_lookup(x_m, y_m)
    frames_rgb: list[np.ndarray] = []

    print("[Video 2] Compositing 48-hour video frames...", flush=True)
    for t_idx in range(1, len(times_sub)):
        ts = pd.Timestamp(times_sub[t_idx])
        op_path = opera_file_path(opera_root, ts.round("15min"))
        op_data = read_opera_h5(op_path) if op_path.exists() else None

        ir108 = raw_sub[t_idx, 4]
        base_rgb = _ir108_to_rgb_base(ir108)

        dbzh_grid = np.full_like(ir108, np.nan, dtype=np.float32)
        qind_grid = np.full_like(ir108, np.nan, dtype=np.float32)
        if op_data is not None:
            dbzh, qind = op_data
            rc = np.clip(opera_row, 0, dbzh.shape[0] - 1)
            cc = np.clip(opera_col, 0, dbzh.shape[1] - 1)
            dbzh_grid = np.where(opera_inside, dbzh[rc, cc], np.nan)
            qind_grid = np.where(opera_inside, qind[rc, cc], np.nan)

        full_sem = full_sem_all[t_idx]
        sev_sem = sev_sem_all[t_idx]
        pure_sem = pure_sem_all[t_idx]

        p1 = _overlay_opera_gradient_panel(base_rgb, dbzh_grid, qind_grid, opera_inside)
        p2 = _overlay_semantic_mask(base_rgb, full_sem)
        p3 = _overlay_semantic_mask(base_rgb, sev_sem)
        p4 = _overlay_semantic_mask(base_rgb, pure_sem)

        n_full_conv = full_counts[t_idx]
        n_sev_conv = sev_counts[t_idx]
        n_pure_conv = pure_counts[t_idx]
        op_35 = int(np.count_nonzero(np.isfinite(dbzh_grid) & (qind_grid >= 0.2) & (dbzh_grid >= 35.0)))
        op_45 = int(np.count_nonzero(np.isfinite(dbzh_grid) & (qind_grid >= 0.2) & (dbzh_grid >= 45.0)))

        panels = [
            (
                "1. OSSERVAZIONE RADAR: IR 10.8 + OPERA DBZH [Gradiente]",
                f"Pixel >= 35 dBZ: {op_35} | Nuclei intensi >= 45 dBZ: {op_45} px",
                p1,
            ),
            (
                "2. RDT COMPLETO (ERA5 NWP + GEO-CT/HRW + Pesi CAL OPERA)",
                f"Celle convettive: {n_full_conv} | Area convettiva: {int(np.isin(full_sem, [2,3,4]).sum())} px",
                p2,
            ),
            (
                "3. RDT 7ch SEVIRI CALIBRATO (Solo 7ch SEVIRI + Pesi CAL)",
                f"Celle convettive: {n_sev_conv} | Area convettiva: {int(np.isin(sev_sem, [2,3,4]).sum())} px",
                p3,
            ),
            (
                "4. RDT 7ch SEVIRI PURO (Zero Pesi CAL, Zero ERA5, Zero Ancillari)",
                f"Celle convettive: {n_pure_conv} | Area convettiva: {int(np.isin(pure_sem, [2,3,4]).sum())} px",
                p4,
            ),
        ]
        frame_img = _compose_4panel_frame(
            panels,
            header_title=f"Evoluzione Multi-Giornaliera (48 Ore: 4-5 Maggio 2020) | {ts.strftime('%Y-%m-%d %H:%M UTC')}",
            header_subtitle=f"Europa Sud-Occidentale (Lat {float(lat_deg.min()):.1f}-{float(lat_deg.max()):.1f}N, Lon {float(lon_deg.min()):.1f}-{float(lon_deg.max()):.1f}E)",
            panel_scale=2,
        )
        frames_rgb.append(frame_img)

    mp4_path = out_dir / "demo_48h_multiday_20200504_20200505.mp4"
    gif_path = out_dir / "demo_48h_multiday_20200504_20200505.gif"
    keyframe_path = out_dir / "demo_48h_multiday_20200504_20200505_keyframe.png"
    _encode_mp4_and_gif(frames_rgb, mp4_path, gif_path, fps=6)
    Image.fromarray(frames_rgb[int(len(frames_rgb) * 0.78)]).save(keyframe_path)
    print(f"[Video 2] Saved {mp4_path.name} and {gif_path.name} ({len(frames_rgb)} frames)", flush=True)
    return {"mp4": mp4_path, "gif": gif_path, "keyframe": keyframe_path}


def main() -> None:
    zarr_root = Path(r"E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N")
    opera_root = Path(r"E:\Datasets\Deep_convection\raw\opera")
    era5_cache_dir = RDT_ORIG_ROOT / "results" / "era5_cache"
    out_dir = RDT_ORIG_ROOT / "results" / "comparison_seviri_2020"
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_model = CalibratedRdtModel.from_json(RDT_ORIG_ROOT / "rdt_nwcsaf" / "default_calibrated_model.json")

    if "--only-48h" not in sys.argv:
        render_5min_outbreak_video(zarr_root, opera_root, era5_cache_dir, cal_model, out_dir)
    v2 = render_48h_multiday_video(zarr_root, opera_root, era5_cache_dir, cal_model, out_dir)
    print("All updated demonstration videos generated successfully:", v2["mp4"], flush=True)


if __name__ == "__main__":
    main()
