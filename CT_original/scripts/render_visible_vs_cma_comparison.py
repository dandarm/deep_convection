"""Extract Visible Channels (VIS006, VIS008, IR_016 & Day Natural RGB) for Daytime Clips and Compare with CMA Pure-IR.

Allows direct naked-eye inspection of daytime cloudiness from solar channels:
- VIS006 (0.6 µm Solar Visible Reflectance %)
- VIS008 (0.8 µm Solar Near-IR Reflectance %)
- Day Natural Color RGB Composite (R=IR_016 1.6µm, G=VIS008 0.8µm, B=VIS006 0.6µm):
    * White = Low/Mid Water Clouds
    * Bright Cyan = Glaciated Ice Clouds / Convective Anvils (ice absorbs 1.6 µm)
    * Green/Brown = Clear Land & Vegetation
    * Dark Navy/Black = Clear Sea
compared side-by-side against IR_108, Testa 1 (CMA Pure-IR), Testa 2 (CT Piano A), and Testa 3 (RDT with Orange Mature Core).
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from satpy import Scene
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CT_original.ct_nwcsaf.ancillary import (
    compute_coastal_mask,
    resolve_geostationary_land_mask,
)
from CT_original.ct_nwcsaf.config import NwcSafCmaCtConfig
from CT_original.ct_nwcsaf.pipeline import NwcSafCmaCtTracker
from CT_original.scripts.render_probing_spec_video import (
    encode_mp4_and_gif,
    LEGEND_5_REGIMES,
    LEGEND_CMA,
    LEGEND_RDT_5,
    map_cma_to_3_classes,
    map_ct_to_5_regimes,
    map_rdt_semantic_to_5_classes,
    PALETTE_5_REGIMES,
    PALETTE_CMA,
    PALETTE_RDT_5,
    render_color_map,
)
from emma_gpm.rdt.pure_gen_utls import PureGenUtlsTracker
from emma_gpm.rdt.storage import DomainFrameReader
from emma_gpm.seviri import geographic_bbox_to_geos_bounds, native_slot_start_from_name
from scripts.build_rdt_pure_gen_utls_week import _geometry


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(font_name, size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def load_visible_channels_from_nat(
    nat_path: Path,
    bounds_xy: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load VIS006, VIS008, and IR_016 reflectance (%) from a native MSG .nat file on the North-Up West-Left grid."""
    scn = Scene(reader="seviri_l1b_native", filenames=[str(nat_path)])
    scn.load(["VIS006", "VIS008", "IR_016"])
    cropped = scn.crop(xy_bbox=bounds_xy)
    # Flip [::-1, ::-1] to match the North-Up, West-Left orientation of DomainFrameReader
    vis06 = np.asarray(cropped["VIS006"].values, dtype=np.float32)[::-1, ::-1]
    vis08 = np.asarray(cropped["VIS008"].values, dtype=np.float32)[::-1, ::-1]
    ir016 = np.asarray(cropped["IR_016"].values, dtype=np.float32)[::-1, ::-1]
    return vis06, vis08, ir016


def render_vis_albedo_thumbnail(
    refl_pct: np.ndarray,
    is_earth_disk: np.ndarray,
    is_coast: np.ndarray,
    target_wh: tuple[int, int],
    r_max: float = 85.0,
    gamma: float = 0.75,
) -> Image.Image:
    """Render a solar reflectance channel (0..r_max %) with gamma correction so clouds and coastlines are crisp."""
    valid = np.where(np.isfinite(refl_pct), np.clip(refl_pct, 0.0, r_max), 0.0)
    norm = np.power(valid / r_max, gamma)
    gray = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[~is_earth_disk] = (35, 42, 54)
    rgb[is_coast] = (255, 220, 100)  # Subtle warm gold coastline for high contrast on dark sea
    return Image.fromarray(rgb).resize(target_wh, resample=Image.Resampling.BILINEAR)


def render_day_natural_rgb(
    vis06: np.ndarray,
    vis08: np.ndarray,
    ir016: np.ndarray,
    is_earth_disk: np.ndarray,
    is_coast: np.ndarray,
    target_wh: tuple[int, int],
) -> Image.Image:
    """Render EUMETSAT standard Day Natural Color RGB (R=IR_016, G=VIS008, B=VIS006).

    In this physical composite:
    - Clear ocean: Dark navy blue
    - Vegetated land: Green
    - Bare soil / desert: Tan / brown
    - Low/mid water clouds: White
    - High ice clouds / convective anvils: Bright Cyan (because ice crystals strongly absorb 1.6 µm!)
    """
    def _scale(ch: np.ndarray, hi: float = 90.0, gamma: float = 0.72) -> np.ndarray:
        v = np.where(np.isfinite(ch), np.clip(ch, 0.0, hi), 0.0)
        return (np.power(v / hi, gamma) * 255.0).astype(np.uint8)

    r = _scale(ir016, 85.0, 0.75)
    g = _scale(vis08, 90.0, 0.72)
    b = _scale(vis06, 90.0, 0.72)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~is_earth_disk] = (35, 42, 54)
    rgb[is_coast] = (245, 245, 220)
    return Image.fromarray(rgb).resize(target_wh, resample=Image.Resampling.BILINEAR)


def render_ir108_thumbnail(
    ir108: np.ndarray,
    is_earth_disk: np.ndarray,
    is_coast: np.ndarray,
    target_wh: tuple[int, int],
) -> Image.Image:
    t_min, t_max = 210.0, 310.0
    valid = np.where(np.isfinite(ir108), ir108, t_min)
    norm = np.clip((t_max - valid) / (t_max - t_min), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[~is_earth_disk] = (35, 42, 54)
    rgb[is_coast] = (140, 200, 250)
    return Image.fromarray(rgb).resize(target_wh, resample=Image.Resampling.BILINEAR)


def compose_visible_vs_cma_frame(
    vis06: np.ndarray,
    vis08: np.ndarray,
    ir016: np.ndarray,
    ir108: np.ndarray,
    cma_3: np.ndarray,
    ct_5: np.ndarray,
    rdt_5: np.ndarray,
    is_earth_disk: np.ndarray,
    is_coast: np.ndarray,
    timestamp_str: str,
    period_label: str,
) -> np.ndarray:
    """Compose a 1920x1080 Full HD frame comparing Visible Channels & Natural RGB (top row) with the 3 Probing Heads (bottom row)."""
    canvas = Image.new("RGB", (1920, 1080), color=(12, 16, 24))
    draw = ImageDraw.Draw(canvas)

    f_title = _load_font(17, bold=True)
    f_sub = _load_font(14, bold=True)
    f_card_hdr = _load_font(13, bold=True)
    f_leg = _load_font(12, bold=True)

    # Compute cloud fraction in VIS006 (>16% reflectance) vs CMA Pure-IR
    valid_disk = is_earth_disk & np.isfinite(vis06) & np.isfinite(ir108)
    vis_cloud_pct = float((vis06[valid_disk] > 16.0).mean() * 100.0)
    cma_cloud_pct = float((cma_3[valid_disk] >= 1).mean() * 100.0)
    cma_opaque_pct = float((np.isin(ct_5[valid_disk], [1, 2, 3])).mean() * 100.0)
    cirrus_pct = float((ct_5[valid_disk] == 4).mean() * 100.0)

    # Header bar
    draw.rectangle([0, 0, 1920, 44], fill=(18, 25, 38))
    draw.line([(0, 44), (1920, 44)], fill=(40, 55, 78), width=1)
    draw.text(
        (40, 12),
        f"CONFRONTO DIURNO: VIS006 / VIS008 / RGB vs 3 TESTE PURE-IR  [{period_label}]",
        fill=(245, 248, 255),
        font=f_title,
    )
    hdr_right = (
        f"UTC: {timestamp_str}  |  "
        f"Nubi Vis: {vis_cloud_pct:.1f}%  |  "
        f"Opache IR: {cma_opaque_pct:.1f}% + Cirri IR: {cirrus_pct:.1f}% (CMA: {cma_cloud_pct:.1f}%)"
    )
    draw.text((1060, 14), hdr_right, fill=(130, 230, 255), font=f_sub)

    # Top Row: 4 Optical / Radiometric Views (VIS006, VIS008, Day Natural RGB, IR_108)
    top_w, top_h = 444, 390
    top_gap = 16
    top_x0 = 48
    top_y0 = 58

    top_panels = [
        (
            "1. VIS006 (0.6 µm Visibile Rosso) — Riflettanza Nubi",
            render_vis_albedo_thumbnail(vis06, is_earth_disk, is_coast, (top_w, top_h)),
            "Bianco = Nubi riflettenti a occhio nudo | Scuro = Mare e terra sereni",
        ),
        (
            "2. VIS008 (0.8 µm Vicino Infrarosso) — Nubi & Vegetazione",
            render_vis_albedo_thumbnail(vis08, is_earth_disk, is_coast, (top_w, top_h)),
            "Contrasto netto coste/vegetazione | Scuro = Mare sereno (albedo < 6%)",
        ),
        (
            "3. RGB Naturale Diurno (IR_016, VIS008, VIS006)",
            render_day_natural_rgb(vis06, vis08, ir016, is_earth_disk, is_coast, (top_w, top_h)),
            "Bianco = Nubi basse acqua | Ciano = Ghiaccio/Incudini | Verde = Terra | Blu scuro = Mare",
        ),
        (
            "4. IR_108 (10.8 µm Finestra Termica) [210–310 K]",
            render_ir108_thumbnail(ir108, is_earth_disk, is_coast, (top_w, top_h)),
            "Rileva anche i cirri d'alta quota semitrasparenti nel visibile",
        ),
    ]

    for i, (p_title, p_img, p_sub) in enumerate(top_panels):
        cx = top_x0 + i * (top_w + top_gap)
        draw.rectangle([cx, top_y0, cx + top_w, top_y0 + 24], fill=(24, 34, 50))
        draw.text((cx + 8, top_y0 + 4), p_title, fill=(230, 242, 255), font=f_card_hdr)
        canvas.paste(p_img, (cx, top_y0 + 24))
        draw.rectangle([cx, top_y0 + 24, cx + top_w, top_y0 + 24 + top_h], outline=(50, 70, 98), width=1)
        draw.rectangle([cx, top_y0 + 24 + top_h + 4, cx + top_w, top_y0 + 24 + top_h + 28], fill=(18, 24, 36))
        draw.text((cx + 8, top_y0 + 24 + top_h + 9), p_sub, fill=(180, 210, 240), font=f_leg)

    # Bottom Row: The 3 Probing Heads (CMA, CT Piano A, RDT with Orange Mature Core)
    bot_w, bot_h = 596, 420
    bot_gap = 18
    bot_x0 = 48
    bot_y0 = top_y0 + 24 + top_h + 42

    probes = [
        {
            "title": "TESTA 1: CLOUD MASK (CMA PURE-IR CON LINEA DI COSTA)",
            "map": render_color_map(cma_3, PALETTE_CMA, is_earth_disk, is_coast, (bot_w, bot_h)),
            "legend": LEGEND_CMA,
        },
        {
            "title": "TESTA 2: CLOUD TYPE PIANO A (4 CLASSI NUBE + CLEAR SKY)",
            "map": render_color_map(ct_5, PALETTE_5_REGIMES, is_earth_disk, is_coast, (bot_w, bot_h)),
            "legend": LEGEND_5_REGIMES,
        },
        {
            "title": "TESTA 3: RDT CONVEZIONE (MATURO IN ARANCIONE, OT IN ROSSO-MAGENTA)",
            "map": render_color_map(rdt_5, PALETTE_RDT_5, is_earth_disk, is_coast, (bot_w, bot_h)),
            "legend": LEGEND_RDT_5,
        },
    ]

    for k, p in enumerate(probes):
        col_x = bot_x0 + k * (bot_w + bot_gap)
        draw.rectangle([col_x, bot_y0, col_x + bot_w, bot_y0 + 26], fill=(22, 32, 48))
        draw.text((col_x + 10, bot_y0 + 5), p["title"], fill=(255, 230, 130), font=f_card_hdr)

        canvas.paste(p["map"], (col_x, bot_y0 + 26))
        draw.rectangle([col_x, bot_y0 + 26, col_x + bot_w, bot_y0 + 26 + bot_h], outline=(50, 70, 98), width=1)

        leg_y0 = bot_y0 + 26 + bot_h + 6
        draw.rectangle([col_x, leg_y0, col_x + bot_w, leg_y0 + 60], fill=(18, 24, 36), outline=(38, 52, 74), width=1)

        pill_x = col_x + 10
        pill_y = leg_y0 + 8
        n_items = len(p["legend"])
        for item_idx, (lbl, rgb) in enumerate(p["legend"]):
            draw.rectangle([pill_x, pill_y, pill_x + 14, pill_y + 14], fill=rgb, outline=(255, 255, 255), width=1)
            draw.text((pill_x + 18, pill_y - 1), lbl, fill=(235, 242, 255), font=f_leg)

            if n_items == 3:
                if item_idx == 1:
                    pill_x = col_x + 10
                    pill_y += 24
                else:
                    pill_x += 286
            else:
                if item_idx == 2:
                    pill_x = col_x + 10
                    pill_y += 24
                elif item_idx == 3:
                    pill_x = col_x + 280
                else:
                    pill_x += 194

    return np.asarray(canvas)


def render_daytime_visible_comparison(
    zarr_path: Path,
    nat_dir: Path,
    out_dir: Path,
    period_tag: str,
    period_label: str,
    max_slots: int = 36,
    render_every_n_slots: int = 3,
) -> dict[str, object]:
    """Render daytime Visible (VIS006/VIS008/RGB) vs 3-Head Pure-IR video and peak preview PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    geom = _geometry(zarr_path)
    area_km2 = geom["area_km2"]
    lat_deg = geom["latitude"]
    lon_deg = geom["longitude"]
    H, W = lat_deg.shape

    land_mask = resolve_geostationary_land_mask(lat_deg, lon_deg, (H, W), default_land=False)
    coast_mask = compute_coastal_mask(land_mask)
    is_earth_disk = np.isfinite(lat_deg) & np.isfinite(lon_deg)
    is_coast = coast_mask & is_earth_disk

    bounds = geographic_bbox_to_geos_bounds(-20.0, 40.0, 30.0, 70.0)

    # Index available .nat files by their 5-min nominal slot start
    nat_map: dict[pd.Timestamp, Path] = {}
    for np_file in sorted(nat_dir.rglob("*.nat")):
        try:
            slot_dt = pd.Timestamp(native_slot_start_from_name(np_file))
            nat_map[slot_dt] = np_file
        except Exception:
            continue

    with xr.open_zarr(zarr_path, consolidated=True, chunks=None) as ds_x:
        all_times = pd.DatetimeIndex(ds_x.time.values)[:max_slots]

    reader = DomainFrameReader(zarr_path.parent.parent)
    ct_tracker = NwcSafCmaCtTracker(NwcSafCmaCtConfig.full_domain_pure_ir())
    rdt_tracker = PureGenUtlsTracker()

    video_frames: list[np.ndarray] = []
    peak_conv = -1
    peak_frame: np.ndarray | None = None
    peak_ts_str = ""

    print(f"\n>>> Rendering Visible vs Pure-IR comparison for {period_tag} ({len(all_times)} slots)...", flush=True)

    for idx, ts in enumerate(all_times):
        cube = reader.load(ts)
        ct_res = ct_tracker.step(cube, lat_deg=lat_deg, lon_deg=lon_deg, land_mask=land_mask)
        rdt_res = rdt_tracker.step_frame(cube, ts, area_km2, lat_deg=lat_deg, lon_deg=lon_deg)

        if idx >= 3 and ((idx - 3) % render_every_n_slots == 0) and (ts in nat_map):
            vis06, vis08, ir016 = load_visible_channels_from_nat(nat_map[ts], bounds.xy_bbox)
            cma_3 = map_cma_to_3_classes(ct_res.cma, ct_res.cma_cloudsnow)
            ct_5 = map_ct_to_5_regimes(ct_res.ct)
            rdt_5 = map_rdt_semantic_to_5_classes(rdt_res["semantic_label"])

            frame = compose_visible_vs_cma_frame(
                vis06=vis06,
                vis08=vis08,
                ir016=ir016,
                ir108=cube[4],
                cma_3=cma_3,
                ct_5=ct_5,
                rdt_5=rdt_5,
                is_earth_disk=is_earth_disk,
                is_coast=is_coast,
                timestamp_str=ts.strftime("%Y-%m-%d %H:%M"),
                period_label=period_label,
            )
            video_frames.append(frame)

            n_conv = int((rdt_5 >= 2).sum())
            if n_conv > peak_conv:
                peak_conv = n_conv
                peak_frame = frame
                peak_ts_str = ts.strftime("%Y%m%dT%H%MZ")

        del ct_res, rdt_res, cube
        if (idx + 1) % 6 == 0:
            gc.collect()

    reader.close()
    del ct_tracker, rdt_tracker, reader
    gc.collect()

    video_path = out_dir / f"visible_vs_cma_probing_{period_tag}.mp4"
    gif_path = out_dir / f"visible_vs_cma_probing_{period_tag}.gif"
    preview_path = out_dir / f"visible_vs_cma_preview_{period_tag}_{peak_ts_str}.png"

    encode_mp4_and_gif(video_frames, video_path, gif_path, fps=2)
    if peak_frame is not None:
        Image.fromarray(peak_frame).save(preview_path)

    print(f"  [OK] Visible comparison {period_tag}: {len(video_frames)} frames -> {preview_path.name}", flush=True)
    return {
        "period_tag": period_tag,
        "period_label": period_label,
        "rendered_frames": len(video_frames),
        "video_mp4": str(video_path),
        "animated_gif": str(gif_path),
        "peak_preview_png": str(preview_path),
    }


def main():
    staging_root = Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_cma_ct_pilot")
    out_dir = staging_root / "visible_vs_cma_daytime_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    daytime_cases = [
        (
            Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_rdt_pilot\rss_zarr\2026\20261027.zarr"),
            Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_rdt_pilot\rss_native_0900_1300"),
            "20261027_med_atlantic",
            "2026-10-27 09:15-12:45 UTC",
            46,
            3,
        ),
        (
            Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_rdt_archive_scan\rss_zarr_video_batch\candidate_015\2025\20250720.zarr"),
            Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_rdt_archive_scan\rss_native_video_batch\candidate_015"),
            "20250720_severe_summer",
            "2025-07-20 12:15-14:55 UTC",
            36,
            3,
        ),
        (
            Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_rdt_archive_scan\rss_zarr_video_batch\candidate_019\2025\20250605.zarr"),
            Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_rdt_archive_scan\rss_native_video_batch\candidate_019"),
            "20250605_continental_outbreak",
            "2025-06-05 12:15-14:55 UTC",
            36,
            3,
        ),
        (
            Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_rdt_archive_scan\rss_zarr_pilot\2025\20250801.zarr"),
            Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_rdt_archive_scan\rss_native"),
            "20250801_alpine_po_valley",
            "2025-08-01 11:15-13:55 UTC",
            36,
            3,
        ),
    ]

    results = []
    for zarr_p, nat_d, tag, lbl, m_slots, step_n in daytime_cases:
        if not zarr_p.exists() or not nat_d.exists():
            print(f"Skipping {tag} (missing path)")
            continue
        info = render_daytime_visible_comparison(
            zarr_path=zarr_p,
            nat_dir=nat_d,
            out_dir=out_dir,
            period_tag=tag,
            period_label=lbl,
            max_slots=m_slots,
            render_every_n_slots=step_n,
        )
        results.append(info)

    summary_path = out_dir / "visible_comparison_summary.json"
    summary_path.write_text(json.dumps({"cases": results}, indent=2), encoding="utf-8")
    print(f"\nCompleted {len(results)} daytime Visible vs Pure-IR comparison videos! Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
