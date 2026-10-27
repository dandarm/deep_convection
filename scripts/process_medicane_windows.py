#!/usr/bin/env python3
"""Acquire SEVIRI RSS data for Medicane event windows and render Pure RDT videos.

This script reads known Mediterranean cyclone events from medicanes_new_windows.csv,
downloads/converts missing days into standard daily Zarr archives (7 thermal channels,
5-min RSS, 763x1762 geostationary domain), runs the 100% autonomous Pure RDT pipeline
(NWC SAF GEN discrimination + IR ancillary CT/HRW + 3D parallax correction), and
produces high-definition, fluid MP4 videos with intermediate caching.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer
from scipy import ndimage
import xarray as xr
import zarr

ROOT = Path(__file__).resolve().parents[1]
RDT_ORIG_ROOT = ROOT / "RDT_original"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(RDT_ORIG_ROOT))

from emma_gpm.file_lock import exclusive_file_lock
from acquire_seviri_year import (
    acquire_day,
    pin_selection_policy,
    verify_store,
)
from download_msg_seviri import credentials, DEFAULT_COLLECTION

from rdt_nwcsaf.ancillary import (
    apply_parallax_correction_2d,
    compute_hrw_amv_guess_field,
    compute_nwcsaf_cloud_type,
)
from rdt_nwcsaf.config import NwcSafRdtConfig
from rdt_nwcsaf.detection import detect_rdt_cells
from rdt_nwcsaf.discrimination import discriminate_and_classify_cell
from rdt_nwcsaf.otd import detect_cell_overshooting_tops
from rdt_nwcsaf.tracking import track_rdt_cells

# Default paths
DEFAULT_CSV = Path(r"c:\Users\Daniele.LOKI\Documents\Progetti\Demetra\moduli\videomae\medicane_data_input\medicanes_new_windows.csv")
DEFAULT_ZARR_ROOT = Path(r"E:\Datasets\MSG-SEVIRI")
DEFAULT_OUT_DIR = RDT_ORIG_ROOT / "results" / "medicanes"
ARTIFACTS_DIR = Path(r"C:\Users\Daniele.LOKI\.gemini\antigravity\brain\6ce9d192-9e7a-46f3-923e-ada02993fbe9")

KNOWN_NAMES: dict[int, str] = {
    39299: "Apollo",
    46218: "Daniel",
    82258: "Blas",
    83565: "Juliette",
    7001283: "Medicane_1283",
    7001328: "Rolf",
    7001358: "Medicane_1358",
    7001421: "Cleopatra",
    7001461: "Qendresa",
    7001466: "Medicane_1466",
    7001500: "Medicane_1500",
    7001521: "Medicane_1521",
    7001542: "Medicane_1542",
    7001575: "Numa",
    7001674: "Detlef",
    7001702: "Ianos",
    7001715: "Medicane_1715",
    7001716: "Medicane_1716",
}


@dataclass
class MedicaneEvent:
    id_final: int
    name: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    days: list[str]  # ['YYYYMMDD', ...]
    records: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def load_medicane_events(csv_path: Path) -> list[MedicaneEvent]:
    df = pd.read_csv(csv_path)
    events: list[MedicaneEvent] = []
    for fid, grp in df.groupby("id_final"):
        fid_int = int(fid)
        named = grp["name"].dropna()
        if len(named) > 0 and str(named.iloc[0]).strip() and str(named.iloc[0]) != "nan":
            ev_name = str(named.iloc[0]).strip()
        else:
            ev_name = KNOWN_NAMES.get(fid_int, f"Event_{fid_int}")

        st_series = grp["start_time"].dropna()
        et_series = grp["end_time"].dropna()
        st = pd.to_datetime(st_series.iloc[0] if len(st_series) > 0 else grp["time"].min())
        et = pd.to_datetime(et_series.iloc[0] if len(et_series) > 0 else grp["time"].max())

        days_range = pd.date_range(st.floor("D"), et.floor("D"), freq="D")
        days_list = [d.strftime("%Y%m%d") for d in days_range]

        events.append(
            MedicaneEvent(
                id_final=fid_int,
                name=ev_name,
                start_time=st,
                end_time=et,
                days=days_list,
                records=len(grp),
                lat_min=float(grp["lat"].min()),
                lat_max=float(grp["lat"].max()),
                lon_min=float(grp["lon"].min()),
                lon_max=float(grp["lon"].max()),
            )
        )
    events.sort(key=lambda e: e.start_time)
    return events


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
    area_km2 = np.where(np.isfinite(area_km2) & (area_km2 > 0.0) & (area_km2 < 250.0), area_km2, 12.0).astype(np.float32)

    return {
        "x_raw": x_raw,
        "y_raw": y_raw,
        "x_m": x_m,
        "y_m": y_m,
        "area_km2": area_km2,
        "lat_deg": np.asarray(lat_deg, dtype=np.float32),
        "lon_deg": np.asarray(lon_deg, dtype=np.float32),
    }


def fill_internal_mask_holes(mask_2d: np.ndarray, max_hole_size: int = 16) -> np.ndarray:
    if not np.any(mask_2d > 0):
        return mask_2d
    is_cloud = mask_2d > 0
    filled = ndimage.binary_fill_holes(is_cloud)
    holes = filled & (~is_cloud)
    if not np.any(holes):
        return mask_2d
    lbl, _ = ndimage.label(holes, structure=np.ones((3, 3), dtype=bool))
    counts = np.bincount(lbl.ravel())
    small_holes = np.isin(lbl, np.where((counts > 0) & (counts <= max_hole_size))[0]) & (lbl > 0)
    out = mask_2d.copy()
    out[small_holes] = ndimage.maximum_filter(mask_2d, size=3)[small_holes]
    return out


def _ir108_to_rgb_base(ir108_k: np.ndarray) -> np.ndarray:
    norm = np.clip((300.0 - np.nan_to_num(ir108_k, nan=295.0)) / (300.0 - 202.0), 0.0, 1.0)
    gray = (30.0 + 205.0 * norm).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


class SmoothSemanticRenderer:
    def __init__(self, shape: tuple[int, int], gamma: float = 0.38):
        self.gamma = gamma
        self.conv_color_buffer = np.zeros((*shape, 3), dtype=np.float32)
        self.conv_active = np.zeros(shape, dtype=bool)

    def render(self, base_rgb: np.ndarray, sem_label: np.ndarray) -> np.ndarray:
        out = base_rgb.astype(np.float32).copy()

        # 1. Non-convective cold cloud (Class 1 -> Steel Blue)
        cirrus = sem_label == 1
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
                m = sem_label == cls_id
                if np.any(m):
                    target_rgb[m] = [r, g, b]

            smooth_pixels = conv_mask & self.conv_active
            self.conv_color_buffer[smooth_pixels] = (
                (1.0 - self.gamma) * self.conv_color_buffer[smooth_pixels]
                + self.gamma * target_rgb[smooth_pixels]
            )
            new_pixels = conv_mask & (~self.conv_active)
            self.conv_color_buffer[new_pixels] = target_rgb[new_pixels]

            alpha = 0.82
            out[conv_mask] = (1.0 - alpha) * out[conv_mask] + alpha * self.conv_color_buffer[conv_mask]

        self.conv_active = conv_mask

        # 3. Overshooting Tops (Class 6 -> Vivid Magenta with 1-px dilation)
        ot_mask = sem_label == 6
        if np.any(ot_mask):
            ot_dil = ndimage.binary_dilation(ot_mask, iterations=1)
            alpha_ot = 0.96
            out[ot_dil, 0] = (1.0 - alpha_ot) * out[ot_dil, 0] + alpha_ot * 245.0
            out[ot_dil, 1] = (1.0 - alpha_ot) * out[ot_dil, 1] + alpha_ot * 105.0
            out[ot_dil, 2] = (1.0 - alpha_ot) * out[ot_dil, 2] + alpha_ot * 255.0

        return np.clip(out, 0, 255).astype(np.uint8)


def _compose_single_full_domain_frame(rgb_map: np.ndarray, title: str) -> np.ndarray:
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

    draw.text((pad + 4, 14), title, fill=(248, 250, 252), font=font_h1)

    panel_img = Image.fromarray(rgb_map)
    canvas.paste(panel_img, (pad, header_h))
    draw.rectangle([pad - 1, header_h - 1, pad + w, header_h + h], outline=(71, 85, 105), width=1)

    fy = header_h + h + 13
    lx = pad + 6
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
    def __init__(self, out_path: Path, width: int, height: int, fps: float = 24.0):
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


def ensure_day_zarr(
    day_str: str,
    zarr_root: Path,
    key: str,
    secret: str,
    collection: Any,
) -> Path:
    """Download and crop SEVIRI RSS native data into daily Zarr if missing."""
    year_str = day_str[:4]
    store = zarr_root / year_str / f"{day_str}.zarr"
    if store.exists():
        try:
            with xr.open_zarr(store, consolidated=True) as ds:
                if "seviri" in ds and ds.seviri.shape[0] >= 200:
                    print(f"[Acquisition] Found existing Zarr {store.name} ({ds.seviri.shape[0]} frames).", flush=True)
                    return store
        except Exception:
            pass

    day_dt = datetime.strptime(day_str, "%Y%m%d")
    work = zarr_root / "_acquisition" / year_str
    work.mkdir(parents=True, exist_ok=True)

    class AcquisitionArgs:
        zarr_root_path = zarr_root
        year = day_dt.year
        workers = 4
        conversion_workers = 1
        conversion_threads = 1
        conversion_prefetch = 1
        crop_batch_size = 2
        preferred_satellite = None
        satellite_priority = ["MSG4", "MSG3", "MSG2", "MSG1"]
        update_selection_policy = True
        recover_native_read_errors = True
        exclude_native = []

    args = AcquisitionArgs()
    args.zarr_root = zarr_root

    with exclusive_file_lock(work / "acquisition.lock"):
        pin_selection_policy(work, args.preferred_satellite, args.satellite_priority, allow_update=True)
        print(f"[Acquisition] Downloading and processing SEVIRI RSS day {day_str}...", flush=True)
        acquire_day(day_dt, collection, args, work, key, secret)

    if not store.exists():
        raise RuntimeError(f"Failed to produce daily store {store}")
    return store


def _make_snapshot(cell: dict[str, Any], cid: int, event: str, frame_idx: int) -> dict[str, Any]:
    """Extract only lightweight scalar attributes to avoid keeping full masks and hierarchy in memory."""
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


def process_single_day_pure_rdt(
    zarr_path: Path,
    geom: dict[str, Any],
    out_dir: Path,
    fps: float = 24.0,
    max_frames: int | None = None,
    time_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> Path | None:
    """Compute Pure RDT for one day, write _cache_{day}_pure.npz, and render MP4."""
    day_str = zarr_path.stem
    day_pure_mp4 = out_dir / f"_day_{day_str}_pure.mp4"
    cache_file = out_dir / f"_cache_{day_str}_pure.npz"

    area_km2 = geom["area_km2"]
    lat_deg = geom["lat_deg"]
    lon_deg = geom["lon_deg"]
    h, w = area_km2.shape

    pad = 12
    total_w = w + 2 * pad
    total_h = 48 + h + 44 + 2 * pad
    if total_w % 2 == 1:
        total_w += 1
    if total_h % 2 == 1:
        total_h += 1

    g = zarr.open(str(zarr_path), mode="r")
    if "time" in g:
        raw_unix = np.asarray(g["time"][:], dtype=np.int64)
        chronological_indices = np.argsort(raw_unix)
        times_all = pd.to_datetime(raw_unix[chronological_indices], unit="s", utc=True)
    else:
        chronological_indices = np.arange(len(g["seviri"]))
        times_all = pd.date_range(day_str, periods=len(g["seviri"]), freq="5min", tz="UTC")

    orig_day_indices = np.arange(len(chronological_indices))
    if time_window is not None:
        st_raw, et_raw = time_window
        st_win = pd.Timestamp(st_raw).tz_localize("UTC") if pd.Timestamp(st_raw).tzinfo is None else pd.Timestamp(st_raw).tz_convert("UTC")
        et_win = pd.Timestamp(et_raw).tz_localize("UTC") if pd.Timestamp(et_raw).tzinfo is None else pd.Timestamp(et_raw).tz_convert("UTC")
        in_win = (times_all >= st_win) & (times_all <= et_win)
        if not np.any(in_win):
            print(f"[Pure RDT] Day {day_str} completely outside window ({st_win} to {et_win}), skipping.", flush=True)
            return None
        orig_day_indices = orig_day_indices[in_win]
        chronological_indices = chronological_indices[in_win]
        times_all = times_all[in_win]

    num_frames = len(chronological_indices)
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)
        orig_day_indices = orig_day_indices[:num_frames]
        chronological_indices = chronological_indices[:num_frames]
        times_all = times_all[:num_frames]

    # Check for existing pre-computed cache (in ev_dir or historical comparison dir)
    cached_sem = None
    alt_cache = RDT_ORIG_ROOT / "results" / "comparison_seviri_2020" / f"_cache_{day_str}.npz"
    source_cache = cache_file if cache_file.exists() else (alt_cache if alt_cache.exists() else None)
    if source_cache is not None:
        try:
            with np.load(source_cache) as c:
                key = "semantic_pure" if "semantic_pure" in c else ("sem_pure" if "sem_pure" in c else None)
                if key is not None:
                    cached_sem = np.asarray(c[key], dtype=np.uint8)
        except Exception:
            cached_sem = None

    # Resume from cache if video already rendered
    if cached_sem is not None and day_pure_mp4.exists():
        print(f"[Pure RDT] Day {day_str} already complete ({num_frames} frames). Resuming!", flush=True)
        return day_pure_mp4

    # Fast rendering directly from pre-computed semantic cache if available
    if cached_sem is not None:
        print(f"[Pure RDT] Fast-rendering day {day_str} from cached masks ({num_frames} frames)...", flush=True)
        writer = FfmpegStreamWriter(day_pure_mp4, total_w, total_h, fps=fps)
        renderer_pure = SmoothSemanticRenderer((h, w), gamma=0.38)
        t_start = time.time()
        for step_idx in range(num_frames):
            raw_idx = int(chronological_indices[step_idx])
            ts_curr = times_all[step_idx]
            orig_idx = int(orig_day_indices[step_idx])
            sem_pure = cached_sem[orig_idx]

            raw_curr = np.asarray(g["seviri"][raw_idx], dtype=np.float32)
            f_curr, _, _ = _orient_north_up_west_left(raw_curr, geom["y_raw"], geom["x_raw"])

            cold_c = np.isfinite(f_curr[4]) & (f_curr[4] < 235.0)
            t_tr_c = (
                float(0.6 * np.nanpercentile(f_curr[0][cold_c], 10) + 0.4 * np.nanpercentile(f_curr[4][cold_c], 5))
                if int(np.count_nonzero(cold_c)) >= 50
                else 215.0
            )
            pure_tropo_c = np.full(f_curr[4].shape, float(np.clip(t_tr_c, 205.0, 226.0)), dtype=np.float32)

            ir_pure_ground = apply_parallax_correction_2d(
                f_curr[4], f_curr[4], lat_deg, lon_deg, t_surf_k=292.0, t_tropo_k=pure_tropo_c, fill_value=290.0
            )
            base_rgb_pure = _ir108_to_rgb_base(ir_pure_ground)
            rgb_pure = renderer_pure.render(base_rgb_pure, sem_pure)
            ts_str = ts_curr.strftime("%Y-%m-%d %H:%M")
            composed = _compose_single_full_domain_frame(
                rgb_pure, f"RDT puro (NWC SAF GEN, IR ancillary, parallasse), {ts_str} UTC"
            )
            writer.write_frame(composed)

            if (step_idx + 1) % 50 == 0 or (step_idx + 1) == num_frames:
                rate = (step_idx + 1) / (time.time() - t_start)
                print(f"   [{day_str}] frame {step_idx + 1}/{num_frames} ({rate:.2f} fps)", flush=True)

        writer.close()
        if not cache_file.exists():
            np.savez_compressed(cache_file, semantic_pure=cached_sem)
        print(f"[Pure RDT] Rendered {day_pure_mp4.name}", flush=True)
        return day_pure_mp4

    print(f"[Pure RDT] Computing day {day_str} ({num_frames} frames)...", flush=True)
    writer = FfmpegStreamWriter(day_pure_mp4, total_w, total_h, fps=fps)
    renderer_pure = SmoothSemanticRenderer((h, w), gamma=0.38)

    pure_cfg = NwcSafRdtConfig(enforce_stretched_mask=False, nwp_conv_filter=False, discri_scheme="GEN")

    prev_f: np.ndarray | None = None
    prev_labels: np.ndarray | None = None
    prev_hierarchy = None
    prev_cells_pure: dict[int, dict[str, Any]] = {}
    hist_pure: dict[int, list[dict[str, Any]]] = {}

    daily_sem_pure = np.zeros((num_frames, h, w), dtype=np.uint8)
    daily_times_unix = np.zeros((num_frames,), dtype=np.int64)

    t_start = time.time()
    for step_idx in range(num_frames):
        raw_idx = int(chronological_indices[step_idx])
        ts_curr = times_all[step_idx]
        daily_times_unix[step_idx] = int(ts_curr.timestamp())

        raw_curr = np.asarray(g["seviri"][raw_idx], dtype=np.float32)
        f_curr, _, _ = _orient_north_up_west_left(raw_curr, geom["y_raw"], geom["x_raw"])

        cold_c = np.isfinite(f_curr[4]) & (f_curr[4] < 235.0)
        t_tr_c = (
            float(0.6 * np.nanpercentile(f_curr[0][cold_c], 10) + 0.4 * np.nanpercentile(f_curr[4][cold_c], 5))
            if int(np.count_nonzero(cold_c)) >= 50
            else 215.0
        )
        pure_tropo_c = np.full(f_curr[4].shape, float(np.clip(t_tr_c, 205.0, 226.0)), dtype=np.float32)
        pure_tropo_c_scalar = float(np.clip(t_tr_c, 205.0, 226.0))

        ct_curr = compute_nwcsaf_cloud_type(
            f_curr[4],
            ir120=f_curr[5],
            ir087=f_curr[2],
            wv062=f_curr[0],
            wv073=f_curr[1],
            t_tropo_k=pure_tropo_c,
            t_surf_k=292.0,
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
                prev_cells_pure,
                prev_labels,
                prev_hierarchy,
                prev_f[4],
                curr_cells_base,
                curr_labels,
                curr_hierarchy,
                f_curr[4],
                area_km2,
                pure_cfg,
                guess_motion_yx=guess_motion_yx,
            )
        else:
            parents, children, edges, primary_parent = {}, {}, {}, {}
            events = {cid: "new" for cid in curr_cells_base}

        curr_cells_pure: dict[int, dict[str, Any]] = {}
        new_hist_pure: dict[int, list[dict[str, Any]]] = {}
        sem_pure = np.zeros((h, w), dtype=np.uint8)

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

            cp = dict(base_c)
            prev_p_c = prev_cells_pure.get(p_main) if p_main is not None else None
            conv_p_p = prev_p_c if (prev_p_c is not None and prev_p_c.get("is_convective")) else None
            if conv_p_p is None:
                for pid in sorted(p_list):
                    if prev_cells_pure.get(pid, {}).get("is_convective"):
                        conv_p_p = prev_cells_pure[pid]
                        break

            hp = (
                hist_pure.get(p_main, [])
                if (p_main is not None and ev in ("continuation", "merge_main", "split_main"))
                else []
            )
            dp = discriminate_and_classify_cell(
                cp,
                prev_p_c,
                conv_p_p,
                hp,
                prev_hierarchy,
                curr_hierarchy,
                ot_c,
                5.0,
                pure_cfg,
                nwp_mask_value=2,
                lightning_strokes=0,
                cell_lat_deg=c_lat,
                calibrated_model=None,
                cell_tropo_temp_k=pure_tropo_c_scalar,
            )
            cp.update(dp)
            curr_cells_pure[cid] = cp
            new_hist_pure[cid] = (hp + [_make_snapshot(cp, cid, ev, step_idx)])[-12:]
            sem_pure[bbox][mask] = int(dp["class_id"])
            if dp["is_convective"]:
                if np.any(ot_conf_loc):
                    sem_pure[bbox][ot_conf_loc] = 6

        sem_pure = apply_parallax_correction_2d(
            sem_pure, f_curr[4], lat_deg, lon_deg, t_surf_k=292.0, t_tropo_k=pure_tropo_c, fill_value=0
        )
        sem_pure = fill_internal_mask_holes(sem_pure, max_hole_size=16)

        prev_f = f_curr
        prev_labels = curr_labels
        prev_hierarchy = curr_hierarchy
        prev_cells_pure = curr_cells_pure
        hist_pure = new_hist_pure

        daily_sem_pure[step_idx] = sem_pure

        # Render frame
        ir_pure_ground = apply_parallax_correction_2d(
            f_curr[4], f_curr[4], lat_deg, lon_deg, t_surf_k=292.0, t_tropo_k=pure_tropo_c, fill_value=290.0
        )
        base_rgb_pure = _ir108_to_rgb_base(ir_pure_ground)
        rgb_pure = renderer_pure.render(base_rgb_pure, sem_pure)
        ts_str = ts_curr.strftime("%Y-%m-%d %H:%M")
        composed = _compose_single_full_domain_frame(
            rgb_pure, f"RDT puro (NWC SAF GEN, IR ancillary, parallasse), {ts_str} UTC"
        )
        writer.write_frame(composed)

        if (step_idx + 1) % 20 == 0:
            gc.collect()

        if (step_idx + 1) % 50 == 0 or (step_idx + 1) == num_frames:
            rate = (step_idx + 1) / (time.time() - t_start)
            print(f"   [{day_str}] frame {step_idx + 1}/{num_frames} ({rate:.2f} fps)", flush=True)

        if step_idx % 25 == 0:
            gc.collect()

    writer.close()
    gc.collect()

    # Save cache atomically
    tmp_cache = cache_file.with_name(f"{cache_file.stem}_part.npz")
    np.savez_compressed(
        tmp_cache,
        semantic_pure=daily_sem_pure,
        timestamps_unix=daily_times_unix,
    )
    tmp_cache.replace(cache_file)
    print(f"[Pure RDT] Cached {cache_file.name} and rendered {day_pure_mp4.name}", flush=True)
    return day_pure_mp4


def process_event(
    event: MedicaneEvent,
    zarr_root: Path,
    out_dir: Path,
    key: str,
    secret: str,
    collection: Any,
    fps: float = 24.0,
    download_only: bool = False,
    render_only: bool = False,
    max_frames: int | None = None,
    tight_window: bool = False,
) -> Path | None:
    event_slug = f"{event.name}_{event.id_final}_{event.start_time:%Y%m%d}_{event.end_time:%Y%m%d}"
    ev_dir = out_dir / event_slug
    ev_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"PROCESSING MEDICANE EVENT: {event.name} (ID: {event.id_final})")
    print(f"Window: {event.start_time} UTC to {event.end_time} UTC (tight={tight_window})")
    print(f"Days: {event.days}")
    print(f"Target dir: {ev_dir}")
    print("=" * 80, flush=True)

    # 1. Download / Verify daily Zarr archives
    day_zarrs: list[Path] = []
    for day_str in event.days:
        if not render_only:
            zp = ensure_day_zarr(day_str, zarr_root, key, secret, collection)
        else:
            zp = zarr_root / day_str[:4] / f"{day_str}.zarr"
            if not zp.exists():
                raise FileNotFoundError(f"Missing required Zarr store {zp} for render-only mode")
        day_zarrs.append(zp)

    if download_only:
        print(f"[Done] Download completed for all {len(day_zarrs)} days of event {event.name}.", flush=True)
        return None

    # 2. Geometry from first Zarr
    geom = _load_full_domain_geometry(day_zarrs[0])

    # 3. Process each day with Pure RDT
    gc.collect()
    day_mp4s: list[Path] = []
    for zp in day_zarrs:
        d_mp4 = process_single_day_pure_rdt(
            zp,
            geom,
            ev_dir,
            fps=fps,
            max_frames=max_frames,
            time_window=(event.start_time, event.end_time) if tight_window else None,
        )
        if d_mp4 is not None:
            day_mp4s.append(d_mp4)

    # 4. Concatenate if multi-day
    final_mp4 = ev_dir / f"medicane_{event_slug}_pure_rdt.mp4"
    if len(day_mp4s) == 1:
        shutil.copy2(day_mp4s[0], final_mp4)
    else:
        list_file = ev_dir / "_concat_list.txt"
        with list_file.open("w", encoding="utf-8") as f:
            for p in day_mp4s:
                f.write(f"file '{p.name}'\n")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(final_mp4),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(ev_dir))
        list_file.unlink(missing_ok=True)

    # 5. Mirror to artifacts directory
    if ARTIFACTS_DIR.exists():
        mirror_target = ARTIFACTS_DIR / final_mp4.name
        shutil.copy2(final_mp4, mirror_target)
        print(f"[Artifact] Mirrored video to: {mirror_target}", flush=True)

    print(f"\n[COMPLETED] Video ready: {final_mp4}", flush=True)
    return final_mp4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to medicanes CSV")
    parser.add_argument("--zarr-root", type=Path, default=DEFAULT_ZARR_ROOT, help="Root path for Zarr stores")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for results")
    parser.add_argument("--list", action="store_true", help="List all events and status")
    parser.add_argument("--event", type=str, default=None, help="Process event by name or id_final")
    parser.add_argument("--all", action="store_true", help="Process all events sequentially")
    parser.add_argument("--fps", type=float, default=24.0, help="Output video frames per second")
    parser.add_argument("--download-only", action="store_true", help="Only download/convert data without rendering")
    parser.add_argument("--render-only", action="store_true", help="Only render without downloading")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per day for testing")
    parser.add_argument("--tight-window", action="store_true", default=False, help="Crop strictly between CSV start_time and end_time")
    args = parser.parse_args()

    events = load_medicane_events(args.csv)

    if args.list:
        print("\n" + "=" * 100)
        print(f"{'ID':<10} {'Name':<15} {'Start Time':<20} {'End Time':<20} {'Days':<6} {'Existing Zarrs'}")
        print("=" * 100)
        for ev in events:
            existing = [d for d in ev.days if (args.zarr_root / d[:4] / f"{d}.zarr").exists()]
            stat_str = f"{len(existing)}/{len(ev.days)} available"
            if len(existing) == len(ev.days):
                stat_str += " (ALL READY)"
            print(f"{ev.id_final:<10} {ev.name:<15} {str(ev.start_time):<20} {str(ev.end_time):<20} {len(ev.days):<6} {stat_str}")
        print("=" * 100)
        return

    key, secret = credentials()
    import eumdac
    token = eumdac.AccessToken((key, secret))
    datastore = eumdac.DataStore(token)
    collection = datastore.get_collection(DEFAULT_COLLECTION)

    target_events: list[MedicaneEvent] = []
    if args.event:
        ev_query = args.event.strip().lower()
        for ev in events:
            if ev_query == str(ev.id_final).lower() or ev_query == ev.name.lower():
                target_events.append(ev)
        if not target_events:
            print(f"Error: Event '{args.event}' not found in {args.csv}")
            sys.exit(1)
    elif args.all:
        target_events = events
    else:
        print("Please specify --event <NAME or ID>, --all, or --list")
        sys.exit(0)

    for ev in target_events:
        process_event(
            ev,
            args.zarr_root,
            args.out_dir,
            key,
            secret,
            collection,
            fps=args.fps,
            download_only=args.download_only,
            render_only=args.render_only,
            max_frames=args.max_frames,
            tight_window=args.tight_window,
        )


if __name__ == "__main__":
    main()
