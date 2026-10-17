#!/usr/bin/env python3
"""Render side-by-side demonstration videos (MP4 + GIF) of RDT convective cells.

Generates two demonstration videos in ``RDT_original/results/comparison_seviri_2020/``:
1. ``demo_5min_20200608_convective_cells.mp4`` (and ``.gif``):
   High-cadence 5-minute RSS tracking on 2020-06-08 (15:30 -> 18:00 UTC) comparing:
   - Ground Truth: OPERA European Radar Composite (DBZH >= 35 dBZ) & OPERA-verified RDT cells
   - Full Multi-Source RDT (``nwcsaf_rdt_full_era5_cal``: ERA5 + GEO-CT/HRW + OPERA CAL weights)
   - Pure 7-Channel SEVIRI RDT (``nwcsaf_rdt_seviri_only_cal``: 7ch SEVIRI only + OPERA CAL weights)
   - Repository Proxy Teacher (``multispectral_utls_v0``)

2. ``demo_48h_multiday_20200504_20200505.mp4`` (and ``.gif``):
   Continuous 2-day (48-hour, 2020-05-04 00:00 -> 2020-05-05 23:45 UTC) evolution over Europe
   comparing OPERA Radar Ground Truth, Full ERA5+OPERA RDT, Pure 7-ch SEVIRI RDT, and their
   pixel-by-pixel agreement map.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer
from scipy import ndimage
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
RDT_ORIG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(RDT_ORIG_ROOT))
sys.path.insert(0, str(RDT_ORIG_ROOT / "scripts"))

from emma_gpm.rdt import RDTTeacherConfig, build_rdt_teacher_labels
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
    verify_cell_with_opera,
)
from calibrate_and_benchmark_full_rdt import load_window_with_halo_and_coords


# Semantic class RGBA colors:
# 1: Non-convective cold cloud (Cool Steel Blue)
# 2: Developing / Growing convective cell (Vibrant Orange)
# 3: Mature / Decaying convective cell (Crimson Red)
# 4: Overshooting Top (Bright Yellow-White)
CLASS_COLORS: dict[int, tuple[int, int, int, float]] = {
    1: (65, 145, 235, 0.52),
    2: (255, 155, 20, 0.72),
    3: (230, 35, 45, 0.72),
    4: (255, 250, 90, 0.92),
}


def _ir108_to_rgb_base(ir108_k: np.ndarray) -> np.ndarray:
    """Convert IR 10.8 brightness temperature (K) to an enhanced grayscale RGB image."""
    # Warm surface (300 K) -> dark gray (35), cold cloud tops (205 K) -> bright white (235)
    norm = np.clip((300.0 - np.nan_to_num(ir108_k, nan=295.0)) / (300.0 - 202.0), 0.0, 1.0)
    gray = (30.0 + 205.0 * norm).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _overlay_semantic_mask(
    base_rgb: np.ndarray,
    sem_label: np.ndarray,
    opera_dbzh_35: np.ndarray | None = None,
) -> np.ndarray:
    """Alpha-blend the 5-class semantic mask and optional OPERA >=35 dBZ contour onto IR10.8 RGB."""
    out = base_rgb.astype(np.float32).copy()
    for cls_id, (r, g, b, alpha) in CLASS_COLORS.items():
        mask = sem_label == cls_id
        if np.any(mask):
            out[mask, 0] = (1.0 - alpha) * out[mask, 0] + alpha * r
            out[mask, 1] = (1.0 - alpha) * out[mask, 1] + alpha * g
            out[mask, 2] = (1.0 - alpha) * out[mask, 2] + alpha * b

    # Draw outer contour of convective cells (class in {2, 3, 4}) for crisp cell boundaries
    conv = np.isin(sem_label, [2, 3, 4])
    if np.any(conv):
        eroded = ndimage.binary_erosion(conv, iterations=1)
        border = conv & ~eroded
        out[border] = (255, 235, 180)

    # Overlay cyan contour where OPERA radar reflectivity >= 35 dBZ
    if opera_dbzh_35 is not None and np.any(opera_dbzh_35):
        dil = ndimage.binary_dilation(opera_dbzh_35, iterations=1)
        ero = ndimage.binary_erosion(opera_dbzh_35, iterations=1)
        op_border = dil & ~ero
        out[op_border] = (0, 255, 255)

    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_opera_truth_panel(
    base_rgb: np.ndarray,
    dbzh_grid: np.ndarray,
    verified_conv_mask: np.ndarray,
    nonconv_cloud_mask: np.ndarray,
) -> np.ndarray:
    """Render the Ground Truth panel: OPERA radar echoes + OPERA-verified convective cells."""
    out = base_rgb.astype(np.float32).copy()

    # Non-convective cold cloud towers (verified < 35 dBZ by OPERA) in light blue
    if np.any(nonconv_cloud_mask):
        a = 0.42
        out[nonconv_cloud_mask, 0] = (1.0 - a) * out[nonconv_cloud_mask, 0] + a * 65
        out[nonconv_cloud_mask, 1] = (1.0 - a) * out[nonconv_cloud_mask, 1] + a * 145
        out[nonconv_cloud_mask, 2] = (1.0 - a) * out[nonconv_cloud_mask, 2] + a * 235

    # OPERA-verified convective cells (cell with >= 16 km2 of >= 35 dBZ) in warm red/orange fill
    if np.any(verified_conv_mask):
        a = 0.58
        out[verified_conv_mask, 0] = (1.0 - a) * out[verified_conv_mask, 0] + a * 240
        out[verified_conv_mask, 1] = (1.0 - a) * out[verified_conv_mask, 1] + a * 70
        out[verified_conv_mask, 2] = (1.0 - a) * out[verified_conv_mask, 2] + a * 40
        eroded = ndimage.binary_erosion(verified_conv_mask, iterations=1)
        out[verified_conv_mask & ~eroded] = (255, 220, 120)

    # Direct OPERA radar reflectivity cores: 25-35 dBZ (green), >=35 dBZ (cyan), >=42 dBZ (magenta)
    echo_25 = np.isfinite(dbzh_grid) & (dbzh_grid >= 25.0) & (dbzh_grid < 35.0)
    echo_35 = np.isfinite(dbzh_grid) & (dbzh_grid >= 35.0) & (dbzh_grid < 42.0)
    echo_42 = np.isfinite(dbzh_grid) & (dbzh_grid >= 42.0)
    if np.any(echo_25):
        out[echo_25] = 0.45 * out[echo_25] + 0.55 * np.array([40, 220, 100], dtype=np.float32)
    if np.any(echo_35):
        out[echo_35] = np.array([0, 255, 255], dtype=np.float32)
    if np.any(echo_42):
        out[echo_42] = np.array([255, 60, 255], dtype=np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_agreement_panel(
    base_rgb: np.ndarray,
    full_sem: np.ndarray,
    sev_sem: np.ndarray,
) -> np.ndarray:
    """Render pixel-level agreement between Full ERA5 RDT and Pure 7-ch SEVIRI RDT."""
    out = base_rgb.astype(np.float32).copy()
    full_conv = np.isin(full_sem, [2, 3, 4])
    sev_conv = np.isin(sev_sem, [2, 3, 4])
    full_cld = full_sem == 1
    sev_cld = sev_sem == 1

    both_conv = full_conv & sev_conv
    both_cld = full_cld & sev_cld
    only_sev_conv = sev_conv & ~full_conv
    only_full_conv = full_conv & ~sev_conv

    if np.any(both_cld):
        out[both_cld] = 0.48 * out[both_cld] + 0.52 * np.array([65, 145, 235], dtype=np.float32)
    if np.any(both_conv):
        out[both_conv] = 0.28 * out[both_conv] + 0.72 * np.array([30, 225, 95], dtype=np.float32)
    if np.any(only_sev_conv):
        out[only_sev_conv] = 0.30 * out[only_sev_conv] + 0.70 * np.array([255, 210, 40], dtype=np.float32)
    if np.any(only_full_conv):
        out[only_full_conv] = 0.30 * out[only_full_conv] + 0.70 * np.array([245, 60, 220], dtype=np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def _compose_4panel_frame(
    panels: list[tuple[str, str, np.ndarray]],
    header_title: str,
    header_subtitle: str,
    footer_legend: list[tuple[tuple[int, int, int], str]],
    panel_scale: int = 2,
) -> np.ndarray:
    """Assemble a 2x2 grid of panels with dark header, titles, and color legend."""
    h0, w0, _ = panels[0][2].shape
    hp, wp = h0 * panel_scale, w0 * panel_scale

    pad = 16
    title_h = 46
    header_h = 68
    footer_h = 54

    total_w = pad * 3 + wp * 2
    total_h = header_h + (title_h + hp + pad) * 2 + footer_h
    # Ensure even dimensions for H.264 yuv420p encoding
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

    # Header text
    draw.text((pad, 12), header_title, fill=(245, 248, 255), font=font_h1)
    draw.text((pad, 40), header_subtitle, fill=(175, 195, 225), font=font_h2)

    # 2x2 panels
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

    # Footer legend
    fy = total_h - footer_h + 16
    fx = pad
    for color, label in footer_legend:
        draw.rectangle([fx, fy, fx + 18, fy + 18], fill=color, outline=(220, 220, 220))
        draw.text((fx + 24, fy + 1), label, fill=(230, 235, 245), font=font_leg)
        fx += 28 + len(label) * 7 + 22

    return np.asarray(canvas, dtype=np.uint8)


def _encode_mp4_and_gif(
    frames: list[np.ndarray],
    mp4_path: Path,
    gif_path: Path,
    fps: int = 4,
) -> None:
    """Save frames as an H.264 MP4 video (via ffmpeg) and an animated GIF (via PIL)."""
    # Save GIF directly via PIL
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
    """Load only the requested north-up, west-left spatial crop from a daily Zarr store."""
    gx_full = ds["x"].values.copy()
    gy_full = ds["y"].values.copy()
    projection = ds.attrs["projection_wkt"]
    t_all = pd.to_datetime(ds["time"].values, utc=True)
    order = np.argsort(t_all)

    ny = len(gy_full)
    nx = len(gx_full)
    flip_y = bool(gy_full[0] < gy_full[-1])
    flip_x = bool(gx_full[0] > gx_full[-1])

    y_raw = slice(ny - y_slice.stop, ny - y_slice.start) if flip_y else y_slice
    x_raw = slice(nx - x_slice.stop, nx - x_slice.start) if flip_x else x_slice

    sub = np.asarray(ds["seviri"][:, :, y_raw, x_raw].values, dtype=np.float32)[order]
    if flip_y:
        sub = sub[:, :, ::-1, :]
        gy_full = gy_full[::-1]
    if flip_x:
        sub = sub[:, :, :, ::-1]
        gx_full = gx_full[::-1]

    return sub, pd.DatetimeIndex(t_all[order]), gx_full[x_slice], gy_full[y_slice], projection


def render_5min_outbreak_video(
    zarr_root: Path,
    opera_root: Path,
    era5_cache_dir: Path,
    cal_model: CalibratedRdtModel,
    out_dir: Path,
) -> dict[str, Path]:
    """Render Video 1: 5-minute RSS outbreak on 2020-06-08 (random_092 context window, 31 frames)."""
    print("[Video 1] Loading 2020-06-08 5-min window (random_092 + context halo)...", flush=True)
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
        raw,
        times.to_numpy(),
        area_km2,
        cfg,
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
    sev_tropo_k, sev_nwpconv = predict_seviri_only_nwp_surrogates(cal_model, ch_dict, lat_deg=lat_deg)
    sev_out = run_nwcsaf_rdt_sequence(
        raw,
        times.to_numpy(),
        area_km2,
        cfg,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        tropopause_temp_k=sev_tropo_k,
        nwp_instability_mask=sev_nwpconv,
        calibrated_model=cal_model,
        use_ancillary_ct=True,
        use_hrw_amv=True,
    )

    utls_out = build_rdt_teacher_labels(
        raw, times=times, config=RDTTeacherConfig(), pixel_area_km2=area_km2
    )

    opera_row, opera_col, opera_inside = build_seviri_to_opera_lookup(x_m, y_m)
    opera_cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    frames_rgb: list[np.ndarray] = []
    # Render from frame 6 onwards (after 30-min tracker warmup)
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
        ver_conv_mask = np.zeros_like(ir108, dtype=bool)
        ver_nonconv_mask = np.zeros_like(ir108, dtype=bool)
        n_ver_conv_cells = 0

        f_rec = full_out["frame_records"][t_idx]
        s_rec = sev_out["frame_records"][t_idx]

        if op_data is not None:
            dbzh, qind = op_data
            rc = np.clip(opera_row, 0, dbzh.shape[0] - 1)
            cc = np.clip(opera_col, 0, dbzh.shape[1] - 1)
            dbzh_grid = np.where(opera_inside, dbzh[rc, cc], np.nan)

            obj_map = f_rec["cell_id_map"]
            for cid in f_rec["cells"]:
                cmask = obj_map == cid
                ver = verify_cell_with_opera(
                    cmask, opera_row, opera_col, opera_inside, dbzh, qind
                )
                if ver.label == 1:
                    ver_conv_mask |= cmask
                    n_ver_conv_cells += 1
                elif ver.label == 0:
                    ver_nonconv_mask |= cmask

        op_echo35 = np.isfinite(dbzh_grid) & (dbzh_grid >= 35.0)
        full_sem = full_out["semantic_label"][t_idx]
        sev_sem = sev_out["semantic_label"][t_idx]
        utls_sem = utls_out["semantic_label"][t_idx]

        p1 = _overlay_opera_truth_panel(base_rgb, dbzh_grid, ver_conv_mask, ver_nonconv_mask)
        p2 = _overlay_semantic_mask(base_rgb, full_sem, op_echo35)
        p3 = _overlay_semantic_mask(base_rgb, sev_sem, op_echo35)
        p4 = _overlay_semantic_mask(base_rgb, utls_sem, op_echo35)

        n_full_conv = int(sum(1 for c in f_rec["cells"].values() if c.get("is_convective", False)))
        n_sev_conv = int(sum(1 for c in s_rec["cells"].values() if c.get("is_convective", False)))
        conv_agreement = float(
            2.0
            * np.logical_and(np.isin(full_sem, [2, 3, 4]), np.isin(sev_sem, [2, 3, 4])).sum()
            / max(
                int(np.isin(full_sem, [2, 3, 4]).sum() + np.isin(sev_sem, [2, 3, 4]).sum()),
                1,
            )
        )

        panels = [
            (
                "1. VERITA OSSERVATIVA: Eco Radar OPERA + Celle Verificate",
                f"Celle convettive verificate radar (>=35 dBZ): {n_ver_conv_cells} | Pixel >=35 dBZ: {int(op_echo35.sum())}",
                p1,
            ),
            (
                "2. RDT COMPLETO (ERA5 NWP + GEO-CT/HRW + Pesi CAL OPERA)",
                f"Celle convettive: {n_full_conv} | Area convettiva: {int(np.isin(full_sem, [2,3,4]).sum())} px",
                p2,
            ),
            (
                "3. RDT PURO 7 CANALI SEVIRI (Solo 7ch SEVIRI + Pesi CAL)",
                f"Celle convettive: {n_sev_conv} | Accordo (Dice) col Modello Completo: {conv_agreement:.3f}",
                p3,
            ),
            (
                "4. PROXY ATTUALE REPOSITORY (multispectral_utls_v0)",
                f"Area convettiva predetta: {int(np.isin(utls_sem, [2,3,4]).sum())} px",
                p4,
            ),
        ]
        legend = [
            ((0, 255, 255), "Eco Radar OPERA >=35 dBZ"),
            ((255, 60, 255), "Nucleo Radar >=42 dBZ"),
            ((65, 145, 235), "Classe 1: Nube Fredda Non-Conv."),
            ((255, 155, 20), "Classe 2: Convezione in Sviluppo"),
            ((230, 35, 45), "Classe 3: Convezione Matura"),
            ((255, 250, 90), "Classe 4: Overshooting Top"),
        ]
        frame_img = _compose_4panel_frame(
            panels,
            header_title=f"Confronto Celle Convettive RDT-CW vs Radar OPERA | {ts.strftime('%Y-%m-%d %H:%M UTC')} (Cadenza 5 min)",
            header_subtitle=f"Dominio Europeo (Lat {float(lat_deg.min()):.1f}-{float(lat_deg.max()):.1f}N, Lon {float(lon_deg.min()):.1f}-{float(lon_deg.max()):.1f}E, 352x352 px)",
            footer_legend=legend,
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
    """Render Video 2: Continuous 2-day (48-hour, 2020-05-04 -> 2020-05-05) evolution over Europe."""
    print("[Video 2] Loading 2-day continuous SEVIRI + OPERA sequence (2020-05-04 to 2020-05-05)...", flush=True)
    y_slice = slice(330, 610)  # 280 px North-South
    x_slice = slice(300, 640)  # 340 px West-East

    ds4 = xr.open_zarr(zarr_root / "2020" / "20200504.zarr")
    ds5 = xr.open_zarr(zarr_root / "2020" / "20200505.zarr")

    arr4, t4, x_m, y_m, projection = _load_zarr_crop(ds4, y_slice, x_slice)
    arr5, t5, _, _, _ = _load_zarr_crop(ds5, y_slice, x_slice)

    times_all = t4.append(t5)
    arr_all = np.concatenate([arr4, arr5], axis=0)

    # Filter to regular 15-minute marks so every frame aligns with an OPERA radar composite
    dedup_idx: list[int] = []
    seen_ts: set[pd.Timestamp] = set()
    for i, ts in enumerate(times_all):
        if ts.minute in (0, 15, 30, 45) and ts not in seen_ts:
            seen_ts.add(ts)
            dedup_idx.append(i)

    raw_sub = np.asarray(arr_all[dedup_idx], dtype=np.float32)
    times_sub = pd.DatetimeIndex([times_all[i] for i in dedup_idx])
    area_km2 = projected_pixel_areas(x_m, y_m, projection, allow_invalid=True)

    xx_grid, yy_grid = np.meshgrid(x_m, y_m)
    lon_deg, lat_deg = Transformer.from_crs(
        CRS.from_user_input(projection), "EPSG:4326", always_xy=True
    ).transform(xx_grid, yy_grid)
    lat_deg = np.asarray(lat_deg, dtype=np.float32)
    lon_deg = np.asarray(lon_deg, dtype=np.float32)

    print(
        f"[Video 2] Loaded {len(times_sub)} frames ({times_sub[0]} -> {times_sub[-1]}), shape={raw_sub.shape}",
        flush=True,
    )

    # Fetch ERA5 representative synoptic state for Day 1 (2020-05-04 15:00) and Day 2 (2020-05-05 15:00)
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
    print("[Video 2] Running Full Multi-Source RDT (ERA5 + GEO-CT/HRW + OPERA CAL)...", flush=True)
    full_out = run_nwcsaf_rdt_sequence(
        raw_sub,
        times_sub.to_numpy(),
        area_km2,
        cfg,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        tropopause_temp_k=t_tropo_blend,
        nwp_instability_mask=nwpconv_blend,
        calibrated_model=cal_model,
        use_ancillary_ct=True,
        use_hrw_amv=True,
        u700_ms=u700_blend,
        v700_ms=v700_blend,
        t_surf_k=tsurf_blend,
    )

    print("[Video 2] Running Pure 7-Channel SEVIRI RDT (Distilled SEVIRI-only)...", flush=True)
    mid_f = raw_sub[len(times_sub) // 2]
    ch_dict = {
        "WV_062": mid_f[0],
        "WV_073": mid_f[1],
        "IR_087": mid_f[2],
        "IR_097": mid_f[3],
        "IR_108": mid_f[4],
        "IR_120": mid_f[5],
        "IR_134": mid_f[6],
    }
    sev_tropo_k, sev_nwpconv = predict_seviri_only_nwp_surrogates(cal_model, ch_dict, lat_deg=lat_deg)
    sev_out = run_nwcsaf_rdt_sequence(
        raw_sub,
        times_sub.to_numpy(),
        area_km2,
        cfg,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        tropopause_temp_k=sev_tropo_k,
        nwp_instability_mask=sev_nwpconv,
        calibrated_model=cal_model,
        use_ancillary_ct=True,
        use_hrw_amv=True,
    )

    opera_row, opera_col, opera_inside = build_seviri_to_opera_lookup(x_m, y_m)
    frames_rgb: list[np.ndarray] = []

    print("[Video 2] Compositing 48-hour video frames...", flush=True)
    # Render every 30-min frame across the 48 hours (~96 frames)
    for t_idx in range(2, len(times_sub), 2):
        ts = pd.Timestamp(times_sub[t_idx])
        op_path = opera_file_path(opera_root, ts.round("15min"))
        op_data = read_opera_h5(op_path) if op_path.exists() else None

        ir108 = raw_sub[t_idx, 4]
        base_rgb = _ir108_to_rgb_base(ir108)

        dbzh_grid = np.full_like(ir108, np.nan, dtype=np.float32)
        ver_conv_mask = np.zeros_like(ir108, dtype=bool)
        ver_nonconv_mask = np.zeros_like(ir108, dtype=bool)
        n_ver_conv_cells = 0

        f_rec = full_out["frame_records"][t_idx]
        s_rec = sev_out["frame_records"][t_idx]

        if op_data is not None:
            dbzh, qind = op_data
            rc = np.clip(opera_row, 0, dbzh.shape[0] - 1)
            cc = np.clip(opera_col, 0, dbzh.shape[1] - 1)
            dbzh_grid = np.where(opera_inside, dbzh[rc, cc], np.nan)

            obj_map = f_rec["cell_id_map"]
            for cid in f_rec["cells"]:
                cmask = obj_map == cid
                ver = verify_cell_with_opera(
                    cmask, opera_row, opera_col, opera_inside, dbzh, qind
                )
                if ver.label == 1:
                    ver_conv_mask |= cmask
                    n_ver_conv_cells += 1
                elif ver.label == 0:
                    ver_nonconv_mask |= cmask

        op_echo35 = np.isfinite(dbzh_grid) & (dbzh_grid >= 35.0)
        full_sem = full_out["semantic_label"][t_idx]
        sev_sem = sev_out["semantic_label"][t_idx]

        p1 = _overlay_opera_truth_panel(base_rgb, dbzh_grid, ver_conv_mask, ver_nonconv_mask)
        p2 = _overlay_semantic_mask(base_rgb, full_sem, op_echo35)
        p3 = _overlay_semantic_mask(base_rgb, sev_sem, op_echo35)
        p4 = _overlay_agreement_panel(base_rgb, full_sem, sev_sem)

        n_full_conv = int(sum(1 for c in f_rec["cells"].values() if c.get("is_convective", False)))
        n_sev_conv = int(sum(1 for c in s_rec["cells"].values() if c.get("is_convective", False)))
        acc_pct = float(100.0 * (full_sem == sev_sem).mean())

        panels = [
            (
                "1. VERITA OSSERVATIVA: Eco Radar OPERA + Celle Verificate",
                f"Celle convettive verificate radar (>=35 dBZ): {n_ver_conv_cells} | Eco >=35 dBZ: {int(op_echo35.sum())} px",
                p1,
            ),
            (
                "2. RDT COMPLETO (ERA5 NWP + GEO-CT/HRW + Pesi CAL OPERA)",
                f"Celle convettive: {n_full_conv} | Area convettiva: {int(np.isin(full_sem, [2,3,4]).sum())} px",
                p2,
            ),
            (
                "3. RDT PURO 7 CANALI SEVIRI (Solo 7ch SEVIRI + Pesi CAL)",
                f"Celle convettive: {n_sev_conv} | Area convettiva: {int(np.isin(sev_sem, [2,3,4]).sum())} px",
                p3,
            ),
            (
                "4. MAPPA DI ACCORDO: Completo vs Puro 7ch SEVIRI",
                f"Accordo esatto su tutto il dominio: {acc_pct:.2f}% (Verde = Convezione concorde)",
                p4,
            ),
        ]
        legend = [
            ((0, 255, 255), "Eco Radar OPERA >=35 dBZ"),
            ((65, 145, 235), "Classe 1: Nube Fredda Non-Conv."),
            ((255, 155, 20), "Classe 2: Conv. in Sviluppo"),
            ((230, 35, 45), "Classe 3: Conv. Matura"),
            ((30, 225, 95), "Pannello 4: Accordo Convettivo"),
            ((255, 210, 40), "Pannello 4: Differenza Bordi"),
        ]
        frame_img = _compose_4panel_frame(
            panels,
            header_title=f"Evoluzione Multi-Giornaliera (48 Ore: 4-5 Maggio 2020) | {ts.strftime('%Y-%m-%d %H:%M UTC')}",
            header_subtitle=f"Europa Sud-Occidentale (Lat {float(lat_deg.min()):.1f}-{float(lat_deg.max()):.1f}N, Lon {float(lon_deg.min()):.1f}-{float(lon_deg.max()):.1f}E)",
            footer_legend=legend,
            panel_scale=2,
        )
        frames_rgb.append(frame_img)

    mp4_path = out_dir / "demo_48h_multiday_20200504_20200505.mp4"
    gif_path = out_dir / "demo_48h_multiday_20200504_20200505.gif"
    keyframe_path = out_dir / "demo_48h_multiday_20200504_20200505_keyframe.png"
    _encode_mp4_and_gif(frames_rgb, mp4_path, gif_path, fps=6)
    # Save peak afternoon convective frame as keyframe PNG
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

    v1 = render_5min_outbreak_video(zarr_root, opera_root, era5_cache_dir, cal_model, out_dir)
    v2 = render_48h_multiday_video(zarr_root, opera_root, era5_cache_dir, cal_model, out_dir)
    print("All demonstration videos generated successfully:", v1["mp4"], v2["mp4"], flush=True)


if __name__ == "__main__":
    main()
