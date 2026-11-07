"""Comprehensive Multi-Temporal Consistency Verification between Pure-IR Cloud Type & Pure-IR RDT Convective Cells.

Scientific Objective:
Verify that convective storm cells detected by the causal multispectral UTLS RDT teacher
are physically nested inside the 'High Opaque' cloud category (Class 3 in 5-Regime CT)
and inside the Cloud Mask (CMA) across wide multi-day and multi-year time intervals.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CT_original.ct_nwcsaf.config import NwcSafCmaCtConfig
from CT_original.ct_nwcsaf.pipeline import NwcSafCmaCtTracker
from CT_original.scripts.render_probing_spec_video import (
    map_ct_to_5_regimes,
    PALETTE_5_REGIMES,
)
from emma_gpm.rdt.pure_gen_utls import PureGenUtlsTracker
from emma_gpm.rdt.storage import DomainFrameReader
from scripts.build_rdt_pure_gen_utls_week import _geometry

# RDT Semantic label names
RDT_CLASSES = {
    0: "Background / Clear",
    1: "Cold Non-Convective Cloud",
    2: "Convective Triggering / Growing",
    3: "Convective Mature Core",
    4: "Decaying Cell",
    5: "Severe Convection",
    6: "Persistent Overshooting Top (OT)",
}

CT_5_NAMES = {
    0: "0: Clear Sky",
    1: "1: Low Cloud (>650 hPa)",
    2: "2: Mid Cloud (450-650 hPa)",
    3: "3: High Opaque Cloud (<450 hPa)",
    4: "4: Cirrus & Multilayer",
}


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(font_name, size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def render_consistency_diagnostic_figure(
    ir108: np.ndarray,
    cma_mask: np.ndarray,
    ct_5: np.ndarray,
    rdt_sem: np.ndarray,
    timestamp_str: str,
    stats: dict[str, float | int],
    out_path: Path,
) -> None:
    """Render a 4-panel diagnostic figure comparing IR108, CMA, CT-5, and RDT Convective Nesting."""
    H, W = ir108.shape
    scale = 0.55
    disp_w, disp_h = int(W * scale), int(H * scale)

    canvas_w, canvas_h = 1920, 1080
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(14, 18, 27))
    draw = ImageDraw.Draw(canvas)

    f_title = _load_font(20, bold=True)
    f_sub = _load_font(14, bold=False)
    f_panel = _load_font(15, bold=True)
    f_leg = _load_font(12, bold=True)
    f_stat = _load_font(13, bold=False)

    # Header Strip
    draw.rectangle([0, 0, canvas_w, 48], fill=(20, 28, 44))
    draw.line([(0, 48), (canvas_w, 48)], fill=(40, 55, 80), width=1)
    draw.text((36, 12), "VERIFICA FISICA DI NIDIFICAZIONE: CLOUD TYPE (HIGH OPAQUE) vs RDT (CELLE CONVETTIVE)", fill=(245, 248, 255), font=f_title)
    draw.text((1350, 14), f"SLOT UTC: {timestamp_str}  |  SEVIRI 7-CH IR", fill=(130, 215, 255), font=f_sub)

    # Compute panel positions (2x2 grid)
    # Target size for each panel to fit in 1920x1080:
    pw = 890
    ph = int(pw * (H / W))
    if ph > 420:
        ph = 420
        pw = int(ph * (W / H))

    p_gap_x = 40
    p_gap_y = 45
    p_x0 = (canvas_w - (2 * pw + p_gap_x)) // 2
    p_y0 = 65

    # Panel 1: IR 10.8 Brightness Temperature (Inverted Greyscale)
    valid_ir = np.where(np.isfinite(ir108), ir108, 300.0)
    norm_ir = np.clip((305.0 - valid_ir) / (305.0 - 210.0), 0.0, 1.0)
    gray = (norm_ir * 255.0).astype(np.uint8)
    p1_rgb = np.stack([gray, gray, gray], axis=-1)
    p1_img = Image.fromarray(p1_rgb).resize((pw, ph), resample=Image.Resampling.BILINEAR)

    # Panel 2: Cloud Mask (CMA)
    p2_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    p2_rgb[~cma_mask] = (30, 41, 59)      # Clear: dark slate
    p2_rgb[cma_mask] = (245, 245, 250)     # Cloud: white
    p2_img = Image.fromarray(p2_rgb).resize((pw, ph), resample=Image.Resampling.NEAREST)

    # Panel 3: Cloud Type (5 Regimes)
    p3_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for c_val, rgb in PALETTE_5_REGIMES.items():
        p3_rgb[ct_5 == c_val] = rgb
    p3_img = Image.fromarray(p3_rgb).resize((pw, ph), resample=Image.Resampling.NEAREST)

    # Panel 4: RDT Convective Nesting Overlay
    # High Opaque is muted slate/blue, Convective Triggering/Growing is Yellow, Mature is Red, OT is Magenta
    p4_rgb = np.full((H, W, 3), 20, dtype=np.uint8)
    p4_rgb[ct_5 == 0] = (25, 33, 44)      # Clear
    p4_rgb[ct_5 == 1] = (45, 55, 72)      # Low
    p4_rgb[ct_5 == 2] = (55, 68, 88)      # Mid
    p4_rgb[ct_5 == 3] = (70, 90, 120)     # High Opaque Shield (Stratiform Anvil background)
    p4_rgb[ct_5 == 4] = (40, 50, 65)      # Cirrus

    # Highlight RDT Convective Cells
    is_trig_grow = np.isin(rdt_sem, [2, 3])
    is_mature = np.isin(rdt_sem, [4, 5])
    is_ot = rdt_sem == 6

    p4_rgb[is_trig_grow] = (250, 204, 21) # Yellow
    p4_rgb[is_mature] = (239, 68, 68)     # Crimson Red
    p4_rgb[is_ot] = (244, 63, 94)         # Bright Rose / Magenta

    p4_img = Image.fromarray(p4_rgb).resize((pw, ph), resample=Image.Resampling.NEAREST)

    panels = [
        ("A. SEVIRI IR 10.8 µm (Radiometria Infrarosso Termico)", p1_img, "Inversione termica standard: Bianco = Vette fredde (<215 K), Scuro = Suolo caldo"),
        ("B. Cloud Mask (CMA Pure-IR)", p2_img, "Copertura nuvolosa binaria: Bianco = Nube (CMA=1), Scuro = Sereno (CMA=0)"),
        ("C. Cloud Type Piano A (5 Macro-Regimi Fisici)", p3_img, "Giallo=Basso, Verde=Medio, Rosso=High Opaque (<450 hPa), Ciano=Cirro, Blu=Sereno"),
        ("D. RDT Convective Nesting (Nidificazione Fisica)", p4_img, "Blu=High Opaque Shield, Giallo=Innesco convettivo, Rosso=Nucleo Maturo, Magenta=OT"),
    ]

    coords = [
        (p_x0, p_y0),
        (p_x0 + pw + p_gap_x, p_y0),
        (p_x0, p_y0 + ph + p_gap_y),
        (p_x0 + pw + p_gap_x, p_y0 + ph + p_gap_y),
    ]

    for (title, img, sub), (cx, cy) in zip(panels, coords):
        # Header banner
        draw.rectangle([cx, cy, cx + pw, cy + 24], fill=(24, 34, 50))
        draw.text((cx + 8, cy + 3), title, fill=(245, 235, 160), font=f_panel)

        # Image paste & outline
        canvas.paste(img, (cx, cy + 24))
        draw.rectangle([cx, cy + 24, cx + pw, cy + 24 + ph], outline=(50, 70, 98), width=1)

        # Subtitle
        draw.text((cx + 6, cy + 24 + ph + 4), sub, fill=(180, 205, 235), font=f_stat)

    # Footer banner with quantitative consistency metrics
    fy0 = canvas_h - 44
    draw.rectangle([0, fy0, canvas_w, canvas_h], fill=(18, 25, 38))
    draw.line([(0, fy0), (canvas_w, fy0)], fill=(40, 55, 80), width=1)
    stat_text = (
        f"METRICHE DI CONSISTENZA FISICA:  "
        f"Convective Pixels in High Opaque: {stats['conv_in_high_opaque_pct']:.2f}%  |  "
        f"Convective Pixels in Cloud Mask: {stats['conv_in_cma_pct']:.2f}%  |  "
        f"False Exclusion (Clear/Low): {stats['conv_in_clear_or_low_pct']:.4f}%  |  "
        f"Convective Density in High Opaque: {stats['convective_density_in_high_opaque_pct']:.2f}%"
    )
    draw.text((36, fy0 + 12), stat_text, fill=(130, 245, 210), font=f_leg)

    canvas.save(out_path)
    print(f"Saved consistency diagnostic preview at {out_path}", flush=True)


def verify_dataset_consistency(
    zarr_path: Path,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run both Pure-IR Cloud Type and Pure-IR RDT across frames and compute nesting metrics."""
    print(f"\n=======================================================", flush=True)
    print(f"Processing archive: {zarr_path.name}", flush=True)
    print(f"=======================================================", flush=True)

    geom = _geometry(zarr_path)
    area_km2 = geom["area_km2"]
    lat_deg = geom["latitude"]
    lon_deg = geom["longitude"]

    reader = DomainFrameReader(zarr_path.parent.parent)

    # Extract available timestamps using xarray to decode CF time units
    import xarray as xr
    with xr.open_zarr(zarr_path, consolidated=True, chunks=None) as ds_x:
        times = pd.DatetimeIndex(ds_x.time.values)
    if max_frames is not None:
        times = times[:max_frames]

    print(f"Total slots to evaluate: {len(times)} (Cadence: 5 min)", flush=True)

    ct_cfg = NwcSafCmaCtConfig.full_domain_pure_ir()
    ct_tracker = NwcSafCmaCtTracker(ct_cfg)
    rdt_tracker = PureGenUtlsTracker()

    total_conv_pixels = 0
    total_conv_in_cma = 0
    total_conv_in_high_opaque = 0
    total_conv_in_cirrus = 0
    total_conv_in_mid = 0
    total_conv_in_low = 0
    total_conv_in_clear = 0
    total_high_opaque_pixels = 0

    # Cross contingency: RDT label (0..6) vs CT 5 classes (0..4)
    contingency = np.zeros((7, 5), dtype=np.int64)

    preview_done = False
    mid_idx = len(times) // 2

    for idx, ts in enumerate(times):
        cube = reader.load(ts)

        # Step CT Pure-IR
        ct_res = ct_tracker.step(cube, lat_deg=lat_deg, lon_deg=lon_deg)
        ct_5 = map_ct_to_5_regimes(ct_res.ct)
        cma_mask = ct_res.cma_cloudsnow == 1

        # Step RDT Pure-IR
        rdt_res = rdt_tracker.step_frame(cube, ts, area_km2, lat_deg=lat_deg, lon_deg=lon_deg)
        rdt_sem = rdt_res["semantic_label"]

        # Convective mask: Triggering (2), Growing (3), Mature (4), Decaying (5), OT (6)
        is_conv = rdt_sem >= 2
        n_conv = int(is_conv.sum())
        total_conv_pixels += n_conv

        is_high_opaque = ct_5 == 3
        total_high_opaque_pixels += int(is_high_opaque.sum())

        total_conv_in_cma += int((is_conv & cma_mask).sum())
        total_conv_in_high_opaque += int((is_conv & is_high_opaque).sum())
        total_conv_in_cirrus += int((is_conv & (ct_5 == 4)).sum())
        total_conv_in_mid += int((is_conv & (ct_5 == 2)).sum())
        total_conv_in_low += int((is_conv & (ct_5 == 1)).sum())
        total_conv_in_clear += int((is_conv & (ct_5 == 0)).sum())

        for r_cls in range(7):
            m_r = rdt_sem == r_cls
            for c_cls in range(5):
                contingency[r_cls, c_cls] += int((m_r & (ct_5 == c_cls)).sum())

        if (idx + 1) % 10 == 0 or idx == len(times) - 1:
            curr_recall = (total_conv_in_high_opaque / max(1, total_conv_pixels)) * 100
            curr_cma_rec = (total_conv_in_cma / max(1, total_conv_pixels)) * 100
            print(
                f"  [{idx+1:02d}/{len(times):02d}] {ts:%Y-%m-%d %H:%M} | "
                f"Conv Pixels: {total_conv_pixels:,} | "
                f"In High Opaque: {curr_recall:.2f}% | "
                f"In CMA: {curr_cma_rec:.2f}%",
                flush=True,
            )

        # Render preview at middle timestamp if convective cells exist
        if not preview_done and (idx >= mid_idx and n_conv > 1000 or idx == len(times) - 1):
            out_preview = Path(__file__).parent / f"convective_consistency_{zarr_path.stem}.png"
            curr_stats = {
                "conv_in_high_opaque_pct": (total_conv_in_high_opaque / max(1, total_conv_pixels)) * 100,
                "conv_in_cma_pct": (total_conv_in_cma / max(1, total_conv_pixels)) * 100,
                "conv_in_clear_or_low_pct": ((total_conv_in_clear + total_conv_in_low) / max(1, total_conv_pixels)) * 100,
                "convective_density_in_high_opaque_pct": (total_conv_pixels / max(1, total_high_opaque_pixels)) * 100,
            }
            render_consistency_diagnostic_figure(
                ir108=cube[4],
                cma_mask=cma_mask,
                ct_5=ct_5,
                rdt_sem=rdt_sem,
                timestamp_str=ts.strftime("%Y-%m-%d %H:%M UTC"),
                stats=curr_stats,
                out_path=out_preview,
            )
            preview_done = True

        del ct_res, rdt_res, cube, ct_5, cma_mask, rdt_sem
        if (idx + 1) % 4 == 0:
            gc.collect()

    reader.close()
    del ct_tracker, rdt_tracker, reader
    gc.collect()

    conv_in_ho_pct = (total_conv_in_high_opaque / max(1, total_conv_pixels)) * 100
    conv_in_cirrus_pct = (total_conv_in_cirrus / max(1, total_conv_pixels)) * 100
    conv_in_mid_pct = (total_conv_in_mid / max(1, total_conv_pixels)) * 100
    conv_in_low_pct = (total_conv_in_low / max(1, total_conv_pixels)) * 100
    conv_in_clear_pct = (total_conv_in_clear / max(1, total_conv_pixels)) * 100
    conv_in_cma_pct = (total_conv_in_cma / max(1, total_conv_pixels)) * 100
    ho_density = (total_conv_pixels / max(1, total_high_opaque_pixels)) * 100

    results = {
        "dataset": zarr_path.name,
        "frames_evaluated": len(times),
        "total_convective_pixels": int(total_conv_pixels),
        "total_high_opaque_pixels": int(total_high_opaque_pixels),
        "convective_recall_in_high_opaque_pct": float(conv_in_ho_pct),
        "convective_recall_in_cirrus_pct": float(conv_in_cirrus_pct),
        "convective_recall_in_mid_cloud_pct": float(conv_in_mid_pct),
        "convective_recall_in_low_cloud_pct": float(conv_in_low_pct),
        "convective_recall_in_clear_sky_pct": float(conv_in_clear_pct),
        "convective_recall_in_cma_cloud_pct": float(conv_in_cma_pct),
        "convective_density_in_high_opaque_shield_pct": float(ho_density),
        "contingency_matrix_rdt_rows_ct_cols": contingency.tolist(),
    }
    return results


def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    archives = [
        Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_rdt_pilot\rss_zarr\2026\20261027.zarr"),
        Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_rdt_archive_scan\rss_zarr_pilot\2024\20240829.zarr"),
        Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_rdt_archive_scan\rss_zarr_pilot\2025\20250801.zarr"),
    ]

    all_results = []
    grand_conv_total = 0
    grand_conv_in_ho = 0
    grand_conv_in_cma = 0
    grand_conv_in_clear_low = 0
    grand_ho_total = 0

    for p in archives:
        if not p.exists():
            print(f"Skipping missing archive: {p}")
            continue
        # Limit to 36 frames per archive for fast multi-day verification
        res = verify_dataset_consistency(p, max_frames=36)
        all_results.append(res)

        grand_conv_total += res["total_convective_pixels"]
        grand_conv_in_ho += int(res["convective_recall_in_high_opaque_pct"] * res["total_convective_pixels"] / 100)
        grand_conv_in_cma += int(res["convective_recall_in_cma_cloud_pct"] * res["total_convective_pixels"] / 100)
        grand_conv_in_clear_low += int(
            (res["convective_recall_in_low_cloud_pct"] + res["convective_recall_in_clear_sky_pct"])
            * res["total_convective_pixels"]
            / 100
        )
        grand_ho_total += res["total_high_opaque_pixels"]

    summary = {
        "title": "Multi-Temporal Physical Consistency Verification: Cloud Type (High Opaque) vs RDT (Convective Cells)",
        "num_datasets": len(all_results),
        "total_frames_evaluated": sum(r["frames_evaluated"] for r in all_results),
        "grand_convective_pixels_analyzed": grand_conv_total,
        "grand_convective_recall_in_high_opaque_pct": float(grand_conv_in_ho / max(1, grand_conv_total) * 100),
        "grand_convective_recall_in_cma_pct": float(grand_conv_in_cma / max(1, grand_conv_total) * 100),
        "grand_false_exclusion_clear_or_low_pct": float(grand_conv_in_clear_low / max(1, grand_conv_total) * 100),
        "grand_convective_density_in_high_opaque_shield_pct": float(grand_conv_total / max(1, grand_ho_total) * 100),
        "datasets": all_results,
    }

    report_path = out_dir / "consistency_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=======================================================", flush=True)
    print("VERIFICATION COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Total Convective Pixels Evaluated: {grand_conv_total:,}", flush=True)
    print(f"Overall Convective Recall in High Opaque: {summary['grand_convective_recall_in_high_opaque_pct']:.2f}%", flush=True)
    print(f"Overall Convective Recall in Cloud Mask:  {summary['grand_convective_recall_in_cma_pct']:.2f}%", flush=True)
    print(f"False Exclusions (Clear Sky / Low Cloud): {summary['grand_false_exclusion_clear_or_low_pct']:.4f}%", flush=True)
    print(f"Report JSON: {report_path}", flush=True)


if __name__ == "__main__":
    main()
