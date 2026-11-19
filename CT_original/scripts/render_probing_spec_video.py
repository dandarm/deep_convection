"""Render Multi-Period 3-Head VideoMAE Probing Videos (7 SEVIRI Channels + CMA, CT Piano A, RDT).

Layout (1920 x 1080 Full HD, clean legends only, no text slides):
- Top slim bar: Title + Slot UTC + Real-time Convective Cell / OT count & Nesting Recall
- Row 1 (4 channels): WV_062, WV_073, IR_087, IR_097
- Row 2 (3 channels): IR_108, IR_120, IR_134 (centered)
- Row 3 (3 Probing Heads):
    1. Testa 1 — Cloud Mask (CMA Pure-IR)
    2. Testa 2 — Cloud Type Piano A (4 Cloud Classes + Clear Sky = 5 Regimes)
    3. Testa 3 — RDT Pure-IR (Convective Cells & Lifecycle Nested in High Opaque)
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
from scipy import ndimage
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
from RDT_original.rdt_nwcsaf.config import NwcSafRdtConfig
from RDT_original.rdt_nwcsaf.pipeline import NwcSafRdtTracker
from emma_gpm.rdt.storage import DomainFrameReader
from scripts.build_rdt_pure_gen_utls_week import _geometry


# -----------------------------------------------------------------------------
# Channel & 3-Head Probing Taxonomy Definitions (Piano B Excluded)
# -----------------------------------------------------------------------------

CHANNEL_SPECS = [
    {"name": "WV_062", "wl": "6.2 µm", "desc": "Upper WV", "t_min": 210.0, "t_max": 250.0},
    {"name": "WV_073", "wl": "7.3 µm", "desc": "Mid WV", "t_min": 215.0, "t_max": 270.0},
    {"name": "IR_087", "wl": "8.7 µm", "desc": "Phase/Ice", "t_min": 215.0, "t_max": 305.0},
    {"name": "IR_097", "wl": "9.7 µm", "desc": "Ozone", "t_min": 210.0, "t_max": 280.0},
    {"name": "IR_108", "wl": "10.8 µm", "desc": "Window", "t_min": 215.0, "t_max": 310.0},
    {"name": "IR_120", "wl": "12.0 µm", "desc": "Split-Win", "t_min": 215.0, "t_max": 305.0},
    {"name": "IR_134", "wl": "13.4 µm", "desc": "CO2 Slice", "t_min": 210.0, "t_max": 275.0},
]

# Testa 1: Cloud Mask (CMA Pure-IR)
PALETTE_CMA = {
    0: (30, 41, 59),     # Clear Sky: dark slate #1E293B
    1: (241, 245, 249),  # Cloud Filled: crisp white #F1F5F9
    2: (148, 163, 184),  # Cloud Contaminated / Edge: cool grey-blue #94A3B8
}

LEGEND_CMA = [
    ("0: Clear Sky", PALETTE_CMA[0]),
    ("1: Cloud Filled", PALETTE_CMA[1]),
    ("2: Cloud Contaminated", PALETTE_CMA[2]),
]

# Testa 2: Cloud Type (5 Macro-Regimi Fisici: Clear Sky + 4 Cloud Classes)
PALETTE_5_REGIMES = {
    0: (30, 41, 59),     # Clear Sky: dark slate #1E293B
    1: (245, 158, 11),   # Low / Boundary-Layer: warm amber #F59E0B
    2: (16, 185, 129),   # Mid-Level: emerald green #10B981
    3: (239, 68, 68),    # Deep / High Opaque Convective: crimson red #EF4444
    4: (6, 182, 212),    # High Semitransparent / Cirrus: bright cyan #06B6D4
}

LEGEND_5_REGIMES = [
    ("0: Clear Sky", PALETTE_5_REGIMES[0]),
    ("1: Low Cloud", PALETTE_5_REGIMES[1]),
    ("2: Mid Cloud", PALETTE_5_REGIMES[2]),
    ("3: High Opaque", PALETTE_5_REGIMES[3]),
    ("4: Cirrus & Multilayer", PALETTE_5_REGIMES[4]),
]

# Testa 3: RDT Pure-IR (5 Classi del Ciclo di Vita Convettivo)
PALETTE_RDT_5 = {
    0: (30, 41, 59),     # Background / Clear or Warm Cloud: dark slate #1E293B
    1: (59, 130, 246),   # Cold Non-Convective Anvil: vivid royal blue #3B82F6
    2: (250, 224, 20),   # Triggering / Growing Updraft: vivid lemon yellow #FAE014
    3: (255, 136, 0),    # Mature / Decaying Convective Core: vivid orange #FF8800
    4: (235, 0, 85),     # Severe / Overshooting Top (OT): vivid crimson-magenta #EB0055
}

LEGEND_RDT_5 = [
    ("0: Clear / Non-Cold", PALETTE_RDT_5[0]),
    ("1: Cold Anvil", PALETTE_RDT_5[1]),
    ("2: Triggering / Growing", PALETTE_RDT_5[2]),
    ("3: Mature / Decaying", PALETTE_RDT_5[3]),
    ("4: Severe / OT", PALETTE_RDT_5[4]),
]


def map_ct_to_5_regimes(ct: np.ndarray) -> np.ndarray:
    """Piano A: 5 macro-regimi fisici (0=Clear Sky + 4 Cloud Classes)."""
    out = np.zeros(ct.shape, dtype=np.uint8)
    out[np.isin(ct, [5, 6, 10])] = 1          # Low
    out[ct == 7] = 2                          # Mid
    out[np.isin(ct, [8, 9])] = 3              # High Opaque Convective
    out[np.isin(ct, [11, 12, 13, 14, 15])] = 4  # Cirrus & Multilayer
    return out


def map_cma_to_3_classes(cma_ext: np.ndarray, cma_bin: np.ndarray) -> np.ndarray:
    """Map CMA extended (1=Clear, 2=Cloud Filled, 3=Cloud Contaminated, 4=Snow) to {0, 1, 2}."""
    out = np.zeros(cma_bin.shape, dtype=np.uint8)
    out[cma_bin == 1] = 1
    out[cma_ext == 3] = 2
    return out


def map_rdt_semantic_to_5_classes(rdt_sem: np.ndarray) -> np.ndarray:
    """Map PureGenUtlsTracker semantic_label (0..6) to the 5 canonical RDT probing classes."""
    out = np.zeros(rdt_sem.shape, dtype=np.uint8)
    out[rdt_sem == 1] = 1                     # Cold non-convective cloud
    out[np.isin(rdt_sem, [2, 3])] = 2         # Triggering / Growing
    out[np.isin(rdt_sem, [4, 5])] = 3         # Mature / Decaying
    out[rdt_sem == 6] = 4                     # Overshooting Top (OT)
    return out


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(font_name, size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _scene_tropopause_temp_k(cube: np.ndarray) -> float:
    """Estimate scene scalar tropopause proxy (K) from WV_062 and IR_108 over cold cloud (< 235 K)."""
    cold_c = np.isfinite(cube[4]) & (cube[4] < 235.0)
    if int(np.count_nonzero(cold_c)) >= 50:
        t_tr_c = float(
            0.6 * np.nanpercentile(cube[0][cold_c], 10)
            + 0.4 * np.nanpercentile(cube[4][cold_c], 5)
        )
    else:
        t_tr_c = 215.0
    return float(np.clip(t_tr_c, 205.0, 226.0))


def render_color_map(
    class_map: np.ndarray,
    palette: dict[int, tuple[int, int, int]],
    is_earth_disk: np.ndarray,
    is_coast_or_grid: np.ndarray,
    target_wh: tuple[int, int],
    *,
    preserve_high_classes: bool = False,
) -> Image.Image:
    """Render a 2D integer class map into an RGB PIL Image with coastlines and pure black space background."""
    H, W = class_map.shape
    if preserve_high_classes:
        work_map = class_map.copy()
        conv_dil = ndimage.maximum_filter(
            np.where(is_earth_disk & (class_map >= 2), class_map, 0), size=2
        )
        work_map = np.where(conv_dil >= 2, np.maximum(work_map, conv_dil), work_map)
        # Ensure 1-4 px Overshooting Tops (class 4) survive the ~3x panel downsample
        ot_dil = ndimage.maximum_filter(is_earth_disk & (class_map == 4), size=4)
        work_map[ot_dil & is_earth_disk] = 4
    else:
        work_map = class_map

    rgb_arr = np.zeros((H, W, 3), dtype=np.uint8)
    for c_val, rgb in palette.items():
        m = is_earth_disk & (work_map == c_val)
        rgb_arr[m] = rgb
    rgb_arr[~is_earth_disk] = (0, 0, 0)
    rgb_arr[is_coast_or_grid] = (235, 240, 245)
    img = Image.fromarray(rgb_arr)
    return img.resize(target_wh, resample=Image.Resampling.NEAREST)


def render_channel_thumbnail(
    tb_map: np.ndarray,
    spec: dict[str, str | float],
    is_earth_disk: np.ndarray,
    is_coast_or_grid: np.ndarray,
    target_wh: tuple[int, int],
) -> Image.Image:
    """Render a single SEVIRI channel with inverted grayscale enhancement, pure black space, and subtle coastlines."""
    t_min, t_max = float(spec["t_min"]), float(spec["t_max"])
    valid_tb = np.where(np.isfinite(tb_map), tb_map, t_min)
    norm = np.clip((t_max - valid_tb) / (t_max - t_min), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    rgb_arr = np.stack([gray, gray, gray], axis=-1)
    rgb_arr[~is_earth_disk] = (0, 0, 0)
    rgb_arr[is_coast_or_grid] = (140, 200, 250)  # Subtle cyan coastlines
    img = Image.fromarray(rgb_arr)
    return img.resize(target_wh, resample=Image.Resampling.BILINEAR)


def compose_three_heads_frame(
    cube: np.ndarray,
    cma_3: np.ndarray,
    ct_5: np.ndarray,
    rdt_5: np.ndarray,
    is_earth_disk: np.ndarray,
    is_coast_or_grid: np.ndarray,
    timestamp_str: str,
    period_label: str = "",
) -> np.ndarray:
    """Compose the Full HD 1920x1080 dashboard frame with 7 SEVIRI channels + 3 Probing Heads.

    Layout:
    - Top slim bar: Title + Timestamp
    - Row 1 (4 channels): WV_062, WV_073, IR_087, IR_097
    - Row 2 (3 channels): IR_108, IR_120, IR_134 (centered)
    - Row 3 (3 Probing Heads): Testa 1 (CMA), Testa 2 (CT), Testa 3 (RDT)
    - Map legends only (no text slides).
    """
    canvas = Image.new("RGB", (1920, 1080), color=(12, 16, 24))
    draw = ImageDraw.Draw(canvas)

    f_title = _load_font(17, bold=True)
    f_sub = _load_font(15, bold=True)
    f_card_hdr = _load_font(14, bold=True)
    f_leg = _load_font(13, bold=True)

    # -------------------------------------------------------------------------
    # 1. Main Header Strip (y: 0..42)
    # -------------------------------------------------------------------------
    draw.rectangle([0, 0, 1920, 42], fill=(18, 25, 38))
    draw.line([(0, 42), (1920, 42)], fill=(40, 55, 78), width=1)
    hdr_left = "SEVIRI 7-CH IR  |  3-HEAD PROBING: CLOUD MASK + CLOUD TYPE + RDT CONVEZIONE"
    draw.text((48, 11), hdr_left, fill=(245, 248, 255), font=f_title)

    hdr_right = f"{timestamp_str} UTC"
    draw.text((1685, 12), hdr_right, fill=(130, 225, 255), font=f_sub)

    # -------------------------------------------------------------------------
    # 2. Section 1: SEVIRI Channels in 2 Rows (Row 1: 4 ch, Row 2: 3 ch)
    # -------------------------------------------------------------------------
    ch_w, ch_h = 444, 232
    ch_gap = 16

    row1_x0 = 48
    row1_y0 = 50

    for i in range(4):
        spec = CHANNEL_SPECS[i]
        col_x = row1_x0 + i * (ch_w + ch_gap)
        tb_map = cube[i]
        thumb = render_channel_thumbnail(
            tb_map, spec, is_earth_disk, is_coast_or_grid, (ch_w, ch_h)
        )

        draw.rectangle([col_x, row1_y0, col_x + ch_w, row1_y0 + 22], fill=(24, 34, 50))
        ch_title = f"{spec['name']} ({spec['wl']})  [{spec['t_min']:.0f}–{spec['t_max']:.0f} K]"
        draw.text((col_x + 8, row1_y0 + 3), ch_title, fill=(225, 238, 255), font=f_card_hdr)

        canvas.paste(thumb, (col_x, row1_y0 + 22))
        draw.rectangle([col_x, row1_y0 + 22, col_x + ch_w, row1_y0 + 22 + ch_h], outline=(50, 70, 98), width=1)

    row2_w_total = 3 * ch_w + 2 * ch_gap
    row2_x0 = (1920 - row2_w_total) // 2
    row2_y0 = row1_y0 + 22 + ch_h + 12

    for j, idx in enumerate([4, 5, 6]):
        spec = CHANNEL_SPECS[idx]
        col_x = row2_x0 + j * (ch_w + ch_gap)
        tb_map = cube[idx]
        thumb = render_channel_thumbnail(
            tb_map, spec, is_earth_disk, is_coast_or_grid, (ch_w, ch_h)
        )

        draw.rectangle([col_x, row2_y0, col_x + ch_w, row2_y0 + 22], fill=(24, 34, 50))
        ch_title = f"{spec['name']} ({spec['wl']})  [{spec['t_min']:.0f}–{spec['t_max']:.0f} K]"
        draw.text((col_x + 8, row2_y0 + 3), ch_title, fill=(225, 238, 255), font=f_card_hdr)

        canvas.paste(thumb, (col_x, row2_y0 + 22))
        draw.rectangle([col_x, row2_y0 + 22, col_x + ch_w, row2_y0 + 22 + ch_h], outline=(50, 70, 98), width=1)

    # -------------------------------------------------------------------------
    # 3. Section 2: The 3 Probing Heads (Row 3: CMA, CT, RDT)
    # -------------------------------------------------------------------------
    bot_w, bot_h = 596, 311
    bot_gap = 18
    bot_x0 = 48
    bot_y0 = row2_y0 + 22 + ch_h + 16

    probes = [
        {
            "title": "TESTA 1: CLOUD MASK",
            "map": render_color_map(cma_3, PALETTE_CMA, is_earth_disk, is_coast_or_grid, (bot_w, bot_h)),
            "legend": LEGEND_CMA,
        },
        {
            "title": "TESTA 2: CLOUD TYPE",
            "map": render_color_map(ct_5, PALETTE_5_REGIMES, is_earth_disk, is_coast_or_grid, (bot_w, bot_h)),
            "legend": LEGEND_5_REGIMES,
        },
        {
            "title": "TESTA 3: RDT CONVEZIONE",
            "map": render_color_map(
                rdt_5,
                PALETTE_RDT_5,
                is_earth_disk,
                is_coast_or_grid,
                (bot_w, bot_h),
                preserve_high_classes=True,
            ),
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
        draw.rectangle([col_x, leg_y0, col_x + bot_w, leg_y0 + 64], fill=(18, 24, 36), outline=(38, 52, 74), width=1)

        pill_x = col_x + 10
        pill_y = leg_y0 + 9
        n_items = len(p["legend"])
        for item_idx, (lbl, rgb) in enumerate(p["legend"]):
            draw.rectangle([pill_x, pill_y, pill_x + 14, pill_y + 14], fill=rgb, outline=(255, 255, 255), width=1)
            draw.text((pill_x + 18, pill_y - 1), lbl, fill=(235, 242, 255), font=f_leg)

            if n_items == 3:
                if item_idx == 1:
                    pill_x = col_x + 10
                    pill_y += 26
                else:
                    pill_x += 286
            else:  # 5 items: 3 on line 1, 2 on line 2
                if item_idx == 2:
                    pill_x = col_x + 10
                    pill_y += 26
                elif item_idx == 3:
                    pill_x = col_x + 280
                else:
                    pill_x += 194

    return np.asarray(canvas)


def encode_mp4_and_gif(
    video_frames: list[np.ndarray],
    video_path: Path,
    gif_out_path: Path,
    fps: int = 2,
) -> None:
    """Encode a list of 1920x1080 RGB frames into an all-intra (-g 1) H.264 MP4 and a 960x540 animated GIF."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        "1920x1080",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-g",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for fr in video_frames:
        proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    proc.stdin.close()
    err_bytes = proc.stderr.read() if proc.stderr is not None else b""
    proc.wait()
    if proc.returncode != 0:
        err = err_bytes.decode("utf-8", errors="replace") if err_bytes else ""
        raise RuntimeError(f"FFmpeg failed ({proc.returncode}): {err}")

    gif_stride = max(1, len(video_frames) // 72)
    gif_frames = [
        Image.fromarray(fr).resize((960, 540), resample=Image.Resampling.BOX)
        for fr in video_frames[::gif_stride]
    ]
    gif_frames[0].save(
        gif_out_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=max(80, int(1000 * gif_stride / max(1, fps))),
        loop=0,
    )


def render_archive_period_video(
    zarr_path: Path,
    out_dir: Path,
    period_tag: str,
    period_label: str,
    start_idx: int = 0,
    num_5min_slots: int = 36,
    render_every_n_slots: int = 1,
    warmup_slots: int = 12,
    fps: int = 3,
) -> dict[str, object]:
    """Run CMA, CT, and RDT_original on a Zarr archive window and render MP4 + GIF + Peak Preview PNG."""
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

    with xr.open_zarr(zarr_path, consolidated=True, chunks=None) as ds_x:
        raw_times = pd.DatetimeIndex(ds_x.time.values)

    sorted_order = np.argsort(raw_times.asi8)
    all_times = raw_times[sorted_order]

    # For 20200917.zarr (Medicane Ianos), apply the documented payload-order swap at 162:193
    is_ianos_20200917 = zarr_path.name == "20200917.zarr" and len(all_times) == 288
    if is_ianos_20200917:
        corrected_order = np.concatenate(
            [
                sorted_order[:162],
                sorted_order[181:193],
                sorted_order[162:181],
                sorted_order[193:],
            ]
        ).astype(np.int64)
        payload_times = raw_times[corrected_order]
    else:
        payload_times = all_times

    window_times = all_times[start_idx : start_idx + num_5min_slots]
    window_payload_times = payload_times[start_idx : start_idx + num_5min_slots]
    if len(window_times) <= warmup_slots:
        raise ValueError(f"Not enough timestamps in {zarr_path} for window starting at {start_idx}")

    reader = DomainFrameReader(zarr_path.parent.parent)
    ct_cfg = NwcSafCmaCtConfig.full_domain_pure_ir()
    ct_tracker = NwcSafCmaCtTracker(ct_cfg)
    rdt_cfg = NwcSafRdtConfig(
        enforce_stretched_mask=False,
        nwp_conv_filter=False,
        discri_scheme="GEN",
        parallax=False,
    )
    rdt_tracker = NwcSafRdtTracker(
        rdt_cfg,
        use_ancillary_ct=True,
        use_hrw_amv=False,
        use_parallax_correction=False,
        separate_observation_support=True,
    )

    video_path = out_dir / f"three_heads_probing_{period_tag}.mp4"
    gif_path = out_dir / f"three_heads_probing_{period_tag}.gif"

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        "1920x1080",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-g",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    assert ffmpeg_proc.stdin is not None

    total_expected_render = ((len(window_times) - 1 - warmup_slots) // render_every_n_slots) + 1
    gif_stride = max(1, total_expected_render // 72)
    gif_frames: list[Image.Image] = []
    rendered_frames_count = 0

    peak_conv_pixels = -1
    peak_score = -1
    peak_frame: np.ndarray | None = None
    peak_ts_str = ""

    total_conv_pixels = 0
    total_conv_in_ho = 0
    total_conv_in_ho_or_mid = 0
    total_conv_in_cma = 0
    total_ot_pixels = 0
    total_cell_pixels = 0
    total_cool15_pixels = 0

    print(
        f"\n>>> Rendering {period_tag} ({period_label}): "
        f"{window_times[0]} -> {window_times[-1]} ({len(window_times)} slots, warmup={warmup_slots})",
        flush=True,
    )

    for idx, (ts, load_ts) in enumerate(zip(window_times, window_payload_times)):
        cube = reader.load(load_ts)
        ct_res = ct_tracker.step(cube, lat_deg=lat_deg, lon_deg=lon_deg, land_mask=land_mask)
        tropo_k = _scene_tropopause_temp_k(cube)
        rdt_res = rdt_tracker.step_frame(
            cube,
            ts,
            area_km2,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            tropopause_temp_k=tropo_k,
        )

        cma_3 = map_cma_to_3_classes(ct_res.cma, ct_res.cma_cloudsnow)
        ct_5 = map_ct_to_5_regimes(ct_res.ct)
        rdt_5 = map_rdt_semantic_to_5_classes(rdt_res["semantic_label"])

        if idx >= warmup_slots:
            conv_m = rdt_5 >= 2
            n_conv = int(conv_m.sum())
            n_ot = int((rdt_5 == 4).sum())
            total_conv_pixels += n_conv
            total_ot_pixels += n_ot
            total_conv_in_ho += int((conv_m & (ct_5 == 3)).sum())
            total_conv_in_ho_or_mid += int((conv_m & np.isin(ct_5, [2, 3])).sum())
            total_conv_in_cma += int((conv_m & (ct_res.cma_cloudsnow == 1)).sum())
            for c_info in rdt_res["cells"].values():
                npx = int(c_info["pixel_count"])
                total_cell_pixels += npx
                if np.isfinite(c_info.get("cooling_15m_k", np.nan)):
                    total_cool15_pixels += npx

            if (idx - warmup_slots) % render_every_n_slots == 0:
                ts_str = ts.strftime("%Y-%m-%d %H:%M")
                frame = compose_three_heads_frame(
                    cube=cube,
                    cma_3=cma_3,
                    ct_5=ct_5,
                    rdt_5=rdt_5,
                    is_earth_disk=is_earth_disk,
                    is_coast_or_grid=is_coast,
                    timestamp_str=ts_str,
                    period_label=period_label,
                )
                ffmpeg_proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
                if rendered_frames_count % gif_stride == 0:
                    gif_frames.append(
                        Image.fromarray(frame).resize((960, 540), resample=Image.Resampling.BOX)
                    )
                rendered_frames_count += 1

                # Prioritize frames with strong convective activity AND confirmed Overshooting Tops for the preview PNG
                frame_score = n_conv + 150 * n_ot
                if frame_score > peak_score:
                    peak_score = frame_score
                    peak_conv_pixels = n_conv
                    peak_frame = frame.copy()
                    peak_ts_str = ts.strftime("%Y%m%dT%H%MZ")
                del frame

            if (idx + 1) % 36 == 0:
                print(
                    f"    ... slot {idx + 1}/{len(window_times)} ({ts.strftime('%H:%M UTC')}) | "
                    f"cumul conv={total_conv_pixels:,}, OT={total_ot_pixels:,}",
                    flush=True,
                )

        del ct_res, rdt_res, cube, cma_3, ct_5, rdt_5
        if (idx + 1) % 6 == 0:
            gc.collect()

    reader.close()
    del ct_tracker, rdt_tracker, reader
    gc.collect()

    ffmpeg_proc.stdin.close()
    err_bytes = ffmpeg_proc.stderr.read() if ffmpeg_proc.stderr is not None else b""
    ffmpeg_proc.wait()
    if ffmpeg_proc.returncode != 0:
        err = err_bytes.decode("utf-8", errors="replace") if err_bytes else ""
        raise RuntimeError(f"FFmpeg failed ({ffmpeg_proc.returncode}): {err}")

    preview_path = out_dir / f"three_heads_preview_{period_tag}_{peak_ts_str}.png"
    if peak_frame is not None:
        Image.fromarray(peak_frame).save(preview_path)
        del peak_frame

    if gif_frames:
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=max(80, int(1000 * gif_stride / max(1, fps))),
            loop=0,
        )
    del gif_frames
    gc.collect()

    ho_recall = (total_conv_in_ho / max(1, total_conv_pixels)) * 100.0
    ho_mid_recall = (total_conv_in_ho_or_mid / max(1, total_conv_pixels)) * 100.0
    cma_recall = (total_conv_in_cma / max(1, total_conv_pixels)) * 100.0
    cool15_pct = (total_cool15_pixels / max(1, total_cell_pixels)) * 100.0
    info = {
        "period_tag": period_tag,
        "period_label": period_label,
        "zarr_archive": str(zarr_path),
        "warmup_start_utc": str(window_times[0]),
        "start_utc": str(window_times[warmup_slots]),
        "end_utc": str(window_times[-1]),
        "rendered_frames": rendered_frames_count,
        "fps": int(fps),
        "step_minutes": int(5 * render_every_n_slots),
        "total_convective_pixels": int(total_conv_pixels),
        "peak_frame_convective_pixels": int(peak_conv_pixels),
        "total_overshooting_top_pixels": int(total_ot_pixels),
        "convective_recall_in_high_opaque_pct": float(ho_recall),
        "convective_recall_in_high_or_mid_opaque_pct": float(ho_mid_recall),
        "convective_recall_in_cma_pct": float(cma_recall),
        "valid_15m_cooling_cell_pixels_pct": float(cool15_pct),
        "video_mp4": str(video_path),
        "animated_gif": str(gif_path),
        "peak_preview_png": str(preview_path),
    }
    print(
        f"  [OK] {period_tag}: {rendered_frames_count} frames | "
        f"Conv pixels: {total_conv_pixels:,} (Peak: {peak_conv_pixels:,}, OT: {total_ot_pixels:,}) | "
        f"HO Recall: {ho_recall:.2f}% | Video: {video_path.name}",
        flush=True,
    )
    return info


def main():
    staging_root = Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_cma_ct_pilot")
    out_dir = staging_root / "three_heads_convective_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Also update the canonical 20261027 directory so old links point to the clean 3-head version
    legacy_out_dir = staging_root / "probing_spec_video_20261027"
    legacy_out_dir.mkdir(parents=True, exist_ok=True)

    # Curated list of long multi-hour / full-day intense convective periods from 2020 and 2022
    # Format: (zarr_path, period_tag, period_label, start_idx, num_5min_slots, render_every_n_slots, warmup_slots, fps)
    periods = [
        (
            Path(r"E:\Datasets\MSG-SEVIRI\2020\20200917.zarr"),
            "20200917_medicane_ianos",
            "2020-09-17 Medicane Ianos (Mar Ionio — Giornata Intera 24h)",
            0,    # 00:00 UTC -> 23:55 UTC (full 288-slot day, matching old Ianos video)
            288,
            1,
            0,    # Render all 288 slots from 00:00 UTC
            3,
        ),
        (
            Path(r"E:\Datasets\MSG-SEVIRI\2020\20200711.zarr"),
            "20200711_summer_convective_outbreak",
            "2020-07-11 Outbreak Temporalesco Estivo (Giornata Intera 24h)",
            0,    # 00:00 UTC -> 23:55 UTC (full 288-slot day)
            288,
            1,
            0,
            3,
        ),
        (
            Path(r"E:\Datasets\MSG-SEVIRI\2020\20200715.zarr"),
            "20200715_alpine_med_storms",
            "2020-07-15 Sistemi Convettivi Alpini & Mediterranei (Giornata Intera 24h)",
            0,    # 00:00 UTC -> 23:55 UTC (full 288-slot day)
            288,
            1,
            0,
            3,
        ),
        (
            Path(r"E:\Datasets\MSG-SEVIRI\2022\20220924.zarr"),
            "20220924_med_mcs_outbreak",
            "2022-09-24 Outbreak MCS V-Shaped Mediterraneo & Italia (17h)",
            72,   # 06:00 UTC start (1h spin-up -> renders 07:00 to 23:55 UTC = 204 frames)
            216,
            1,
            12,
            3,
        ),
        (
            Path(r"E:\Datasets\MSG-SEVIRI\2022\20220925.zarr"),
            "20220925_extreme_ot_mcs",
            "2022-09-25 MCS Estremo Tirreno/Adriatico — Top 199 K (13.5h)",
            0,    # 00:00 UTC start (30m spin-up -> renders 00:30 to 13:55 UTC = 162 frames)
            168,
            1,
            6,
            3,
        ),
        (
            Path(r"E:\Datasets\MSG-SEVIRI\2022\20220522.zarr"),
            "20220522_euro_supercells",
            "2022-05-22 Outbreak Europeo di Supercelle & MCS (13h)",
            120,  # 10:00 UTC start, 2.5h before initiation (1h spin-up -> renders 11:00 to 23:55 UTC = 156 frames)
            168,
            1,
            12,
            3,
        ),
        (
            Path(r"E:\Datasets\MSG-SEVIRI\2022\20220912.zarr"),
            "20220912_deep_tropopause_storms",
            "2022-09-12 Convezione Profonda & Overshooting 197.5 K (13h)",
            120,  # 10:00 UTC start, 2.5h before initiation (1h spin-up -> renders 11:00 to 23:55 UTC = 156 frames)
            168,
            1,
            12,
            3,
        ),
        (
            Path(r"E:\Datasets\MSG-SEVIRI\2022\20220528.zarr"),
            "20220528_med_continental_mcs",
            "2022-05-28 Sistemi Convettivi Mesoscala & Temporali Diurni (13h)",
            96,   # 08:00 UTC start, 2h before initiation (1h spin-up -> renders 09:00 to 21:55 UTC = 156 frames)
            168,
            1,
            12,
            3,
        ),
        (
            Path(r"E:\Datasets\MSG-SEVIRI\2022\20220506.zarr"),
            "20220506_widespread_med_convection",
            "2022-05-06 Ciclone Convettivo Mediterraneo Esteso (13.5h)",
            0,    # 00:00 UTC start (30m spin-up -> renders 00:30 to 13:55 UTC = 162 frames)
            168,
            1,
            6,
            3,
        ),
        (
            Path(r"E:\Datasets\MSG-SEVIRI\2022\20220628.zarr"),
            "20220628_alpine_west_med_severe",
            "2022-06-28 Supercelle Alpine & Mediterraneo Occidentale (13h)",
            108,  # 09:00 UTC start, 2h before initiation (1h spin-up -> renders 10:00 to 22:55 UTC = 156 frames)
            168,
            1,
            12,
            3,
        ),
    ]

    summary_json = out_dir / "three_heads_videos_summary.json"
    existing_summaries = {}
    if summary_json.exists():
        try:
            existing_doc = json.loads(summary_json.read_text(encoding="utf-8"))
            for p in existing_doc.get("periods", []):
                existing_summaries[p["period_tag"]] = p
        except Exception:
            pass

    for zarr_path, tag, label, s_idx, n_slots, step_n, w_slots, fps_val in periods:
        if not zarr_path.exists():
            print(f"Skipping missing archive: {zarr_path}", flush=True)
            continue
        v_path = out_dir / f"three_heads_probing_{tag}.mp4"
        if v_path.exists() and tag in existing_summaries and tag != "20220924_med_mcs_outbreak":
            print(f"Skipping already rendered video: {v_path.name}", flush=True)
            continue
        info = render_archive_period_video(
            zarr_path=zarr_path,
            out_dir=out_dir,
            period_tag=tag,
            period_label=label,
            start_idx=s_idx,
            num_5min_slots=n_slots,
            render_every_n_slots=step_n,
            warmup_slots=w_slots,
            fps=fps_val,
        )
        existing_summaries[tag] = info

        summary_doc = {
            "title": "Multi-Period 3-Head VideoMAE Probing Videos (CMA + CT + RDT_original)",
            "piano_b_excluded": True,
            "heads": [
                "Testa 1: Cloud Mask (CMA Pure-IR)",
                "Testa 2: Cloud Type (0=Clear Sky + 4 Cloud Regimes)",
                "Testa 3: RDT_original Convective Lifecycle (5 Classes, NwcSafRdtTracker)",
            ],
            "num_periods_rendered": len(existing_summaries),
            "periods": list(existing_summaries.values()),
        }
        summary_json.write_text(json.dumps(summary_doc, indent=2), encoding="utf-8")

    print(f"\nAll {len(existing_summaries)} multi-period 3-head videos generated! Summary: {summary_json}", flush=True)


if __name__ == "__main__":
    main()
