#!/usr/bin/env python3
"""Acquire SEVIRI RSS data for Medicane Ianos and render Pure GEO-CI Convection Initiation videos.

Applies the official EUMETSAT NWC SAF GEO-CI algorithm in Pure-SEVIRI IR mode
(24/7, zero visible or NWP dependency) across the entire Medicane Ianos week
(2020-09-14 through 2020-09-18) over the full MSG-SEVIRI Rapid Scan domain (763 x 1762).

Generates:
1. Daily caches: _cache_{YYYYMMDD}_pure_ci.npz
2. Daily videos: _day_{YYYYMMDD}_pure_ci.mp4
3. Full concatenated 5-day video: medicane_Ianos_7001702_20200914_20200918_pure_ci.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer
from scipy import ndimage
import xarray as xr
import zarr

# Add repository root and CI package
REPO_ROOT = Path(__file__).resolve().parents[2]
CI_ORIG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(CI_ORIG_ROOT))

from ci_nwcsaf import (
    NwcSafCiConfig,
    NwcSafCiTracker,
    CiProbabilityClass,
)


DEFAULT_ZARR_ROOT = Path(r"E:\Datasets\MSG-SEVIRI")
DEFAULT_OUT_DIR = CI_ORIG_ROOT / "results" / "medicanes"
ARTIFACTS_DIR = Path(r"C:\Users\Daniele.LOKI\.gemini\antigravity\brain\4b947dd6-3fc2-4948-9ed6-9a2067fb2a1a")

# Official NWC SAF CI visualization color palette:
# Class 0: No prob -> Transparent
# Class 1: 0-25%   -> Yellow (VLOWPROB)
# Class 2: 25-50%  -> Orange (LOWPROB)
# Class 3: 50-75%  -> Red (MODPROB)
# Class 4: 75-100% -> Magenta (HIGHPROB)
CI_CLASS_COLORS: Dict[int, Tuple[float, float, float]] = {
    1: (250.0, 204.0, 21.0),   # Yellow
    2: (249.0, 115.0, 22.0),   # Orange
    3: (220.0, 38.0, 38.0),    # Red
    4: (245.0, 105.0, 255.0),  # Magenta
}

CI_CLASS_ALPHA: Dict[int, float] = {
    1: 0.65,
    2: 0.78,
    3: 0.86,
    4: 0.94,
}


@dataclass
class MedicaneEvent:
    id_final: int
    name: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    days: List[str]


IANOS_EVENT = MedicaneEvent(
    id_final=7001702,
    name="Ianos",
    start_time=pd.Timestamp("2020-09-14 23:50:00", tz="UTC"),
    end_time=pd.Timestamp("2020-09-18 19:40:00", tz="UTC"),
    days=["20200914", "20200915", "20200916", "20200917", "20200918"],
)


def _orient_north_up_west_left(
    data: np.ndarray, y_raw: np.ndarray, x_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _load_full_domain_geometry(zarr_path: Path) -> Dict[str, Any]:
    with xr.open_zarr(zarr_path, consolidated=True) as ds:
        x_raw = ds.x.values.copy()
        y_raw = ds.y.values.copy()
        proj_wkt = ds.attrs.get("projection_wkt", "")

    dummy = np.zeros((1, len(y_raw), len(x_raw)), dtype=np.float32)
    _, y_m, x_m = _orient_north_up_west_left(dummy, y_raw, x_raw)

    return {
        "x_raw": x_raw,
        "y_raw": y_raw,
        "x_m": x_m,
        "y_m": y_m,
    }


def _ir108_to_rgb_base(ir108_k: np.ndarray) -> np.ndarray:
    """Enhanced grayscale IR 10.8 background: cold tops bright, warm ground dark."""
    norm = np.clip((300.0 - np.nan_to_num(ir108_k, nan=295.0)) / (300.0 - 202.0), 0.0, 1.0)
    gray = (30.0 + 205.0 * norm).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


class SmoothCiRenderer:
    """Renders CI probability masks with temporal smoothing to prevent single-frame flickering."""

    def __init__(self, shape: Tuple[int, int], gamma: float = 0.40):
        h, w = shape
        self.gamma = gamma
        self.ci_color_buffer = np.zeros((h, w, 3), dtype=np.float32)
        self.ci_active = np.zeros((h, w), dtype=bool)

    def render(self, base_rgb: np.ndarray, ci_prob: np.ndarray) -> np.ndarray:
        out = base_rgb.astype(np.float32).copy()
        active = (ci_prob >= 1) & (ci_prob <= 4)

        if np.any(active):
            target_rgb = np.zeros_like(self.ci_color_buffer)
            target_alpha = np.zeros(ci_prob.shape, dtype=np.float32)

            for cls_id in (1, 2, 3, 4):
                m = (ci_prob == cls_id)
                if np.any(m):
                    target_rgb[m] = CI_CLASS_COLORS[cls_id]
                    target_alpha[m] = CI_CLASS_ALPHA[cls_id]

            smooth_pixels = active & self.ci_active
            self.ci_color_buffer[smooth_pixels] = (
                (1.0 - self.gamma) * self.ci_color_buffer[smooth_pixels]
                + self.gamma * target_rgb[smooth_pixels]
            )
            new_pixels = active & (~self.ci_active)
            self.ci_color_buffer[new_pixels] = target_rgb[new_pixels]

            alpha_expanded = target_alpha[active, None]
            out[active] = (1.0 - alpha_expanded) * out[active] + alpha_expanded * self.ci_color_buffer[active]

        self.ci_active = active
        return np.clip(out, 0, 255).astype(np.uint8)

    def render_hybrid(
        self,
        base_rgb: np.ndarray,
        base_diag: np.ndarray,
        forecast_corridor: np.ndarray,
    ) -> np.ndarray:
        """Renders Hybrid CI:
        1. Continuous forecast corridor (0-30m) rendered with soft semi-transparent alpha.
        2. Instantaneous initiation cores (base_diag) overlaid with solid, crisp opacity.
        """
        out = base_rgb.astype(np.float32).copy()

        # 1. Overlay smooth continuous corridor
        active_corridor = (forecast_corridor >= 1) & (forecast_corridor <= 4)
        if np.any(active_corridor):
            corridor_rgb = np.zeros_like(self.ci_color_buffer)
            corridor_alpha = np.zeros(forecast_corridor.shape, dtype=np.float32)
            for cls_id in (1, 2, 3, 4):
                m = (forecast_corridor == cls_id)
                if np.any(m):
                    corridor_rgb[m] = CI_CLASS_COLORS[cls_id]
                    corridor_alpha[m] = CI_CLASS_ALPHA[cls_id] * 0.45

            smooth_p = active_corridor & self.ci_active
            self.ci_color_buffer[smooth_p] = (
                (1.0 - self.gamma) * self.ci_color_buffer[smooth_p]
                + self.gamma * corridor_rgb[smooth_p]
            )
            new_p = active_corridor & (~self.ci_active)
            self.ci_color_buffer[new_p] = corridor_rgb[new_p]

            a_exp = corridor_alpha[active_corridor, None]
            out[active_corridor] = (1.0 - a_exp) * out[active_corridor] + a_exp * self.ci_color_buffer[active_corridor]

        self.ci_active = active_corridor

        # 2. Overlay instantaneous cores at full solid opacity
        active_cores = (base_diag >= 1) & (base_diag <= 4)
        if np.any(active_cores):
            core_rgb = np.zeros_like(out)
            core_alpha = np.zeros(base_diag.shape, dtype=np.float32)
            for cls_id in (1, 2, 3, 4):
                m = (base_diag == cls_id)
                if np.any(m):
                    core_rgb[m] = CI_CLASS_COLORS[cls_id]
                    core_alpha[m] = 0.94

            a_core = core_alpha[active_cores, None]
            out[active_cores] = (1.0 - a_core) * out[active_cores] + a_core * core_rgb[active_cores]

        return np.clip(out, 0, 255).astype(np.uint8)


def _compose_single_full_domain_frame(rgb_map: np.ndarray, title: str, note: Optional[str] = None) -> np.ndarray:
    """Adds header, border, and NWC SAF CI color legend matching RDT video layout."""
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
        font_note = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font_h1 = ImageFont.load_default()
        font_leg = ImageFont.load_default()
        font_note = ImageFont.load_default()

    draw.text((pad + 4, 14), title, fill=(248, 250, 252), font=font_h1)

    panel_img = Image.fromarray(rgb_map)
    canvas.paste(panel_img, (pad, header_h))
    draw.rectangle([pad - 1, header_h - 1, pad + w, header_h + h], outline=(71, 85, 105), width=1)

    fy = header_h + h + 13
    lx = pad + 6
    legend_items = [
        ((250, 204, 21), "Iniziazione 0-25% (VLOWPROB)"),
        ((249, 115, 22), "Iniziazione 25-50% (LOWPROB)"),
        ((220, 38, 38), "Iniziazione 50-75% (MODPROB)"),
        ((245, 105, 255), "Iniziazione 75-100% (HIGHPROB)"),
    ]
    for color, label in legend_items:
        draw.rectangle([lx, fy, lx + 18, fy + 16], fill=color, outline=(255, 255, 255))
        draw.text((lx + 24, fy + 1), label, fill=(226, 232, 240), font=font_leg)
        lx += 24 + int(len(label) * 7.5) + 24

    if note is None:
        note = "Iniziazione ibrida: nucleo al tempo t (solido) + previsione 0-30 min continua (sfumata)"
    draw.text((total_w - pad - 500, fy + 2), note, fill=(148, 163, 184), font=font_note)

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
        if self.proc and self.proc.stdin:
            self.proc.stdin.close()
            self.proc.wait()


def process_single_day_pure_ci(
    zarr_path: Path,
    geom: Dict[str, Any],
    out_dir: Path,
    tracker: NwcSafCiTracker,
    fps: float = 24.0,
    max_frames: Optional[int] = None,
    time_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    render_mode: str = "hybrid",
) -> Optional[Path]:
    """Processes one day of SEVIRI observations with NwcSafCiTracker, caches results, and renders video."""
    day_str = zarr_path.stem
    suffix = f"_{render_mode}" if render_mode != "forecast" else ""
    day_ci_mp4 = out_dir / f"_day_{day_str}_pure_ci{suffix}.mp4"
    cache_file = out_dir / f"_cache_{day_str}_pure_ci{suffix}.npz"

    h = len(geom["y_raw"])
    w = len(geom["x_raw"])

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

    # Correct known 17 September permutation if present
    if day_str == "20200917" and len(chronological_indices) == 288:
        chronological_indices = np.concatenate([
            chronological_indices[:162],
            chronological_indices[181:193],
            chronological_indices[162:181],
            chronological_indices[193:],
        ]).astype(np.int64)
        times_all = pd.date_range("2020-09-17 00:00:00", periods=288, freq="5min", tz="UTC")

    orig_day_indices = np.arange(len(chronological_indices))
    if time_window is not None:
        st_raw, et_raw = time_window
        st_win = pd.Timestamp(st_raw).tz_localize("UTC") if pd.Timestamp(st_raw).tzinfo is None else pd.Timestamp(st_raw).tz_convert("UTC")
        et_win = pd.Timestamp(et_raw).tz_localize("UTC") if pd.Timestamp(et_raw).tzinfo is None else pd.Timestamp(et_raw).tz_convert("UTC")
        in_win = (times_all >= st_win) & (times_all <= et_win)
        if not np.any(in_win):
            print(f"[Pure CI] Day {day_str} outside window, skipping.", flush=True)
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

    # Resume if already complete
    if cache_file.exists() and day_ci_mp4.exists():
        print(f"[Pure CI] Day {day_str} already complete ({num_frames} frames). Resuming!", flush=True)
        return day_ci_mp4

    print(f"[Pure CI] Computing & rendering day {day_str} ({num_frames} frames, mode={render_mode})...", flush=True)
    writer = FfmpegStreamWriter(day_ci_mp4, total_w, total_h, fps=fps)
    renderer_ci = SmoothCiRenderer((h, w), gamma=0.40)

    daily_base_diag = np.zeros((num_frames, h, w), dtype=np.uint8)
    daily_prob30 = np.zeros((num_frames, h, w), dtype=np.uint8)
    daily_prob60 = np.zeros((num_frames, h, w), dtype=np.uint8)
    daily_times_unix = np.zeros((num_frames,), dtype=np.int64)

    t_start = time.time()
    for step_idx in range(num_frames):
        raw_idx = int(chronological_indices[step_idx])
        ts_curr = times_all[step_idx]
        daily_times_unix[step_idx] = int(ts_curr.timestamp())

        raw_curr = np.asarray(g["seviri"][raw_idx], dtype=np.float32)
        f_curr, _, _ = _orient_north_up_west_left(raw_curr, geom["y_raw"], geom["x_raw"])

        # Execute Pure-SEVIRI CI step
        step_res = tracker.step(f_curr, timestamp=ts_curr)
        prob30 = step_res.ci_prob30
        prob60 = step_res.ci_prob60
        base_diag = step_res.base_diagnosis

        daily_base_diag[step_idx] = base_diag
        daily_prob30[step_idx] = prob30
        daily_prob60[step_idx] = prob60

        # Base image: enhanced IR 10.8
        base_rgb = _ir108_to_rgb_base(f_curr[4])

        # Overlay CI convective initiation
        if render_mode == "hybrid":
            rgb_ci = renderer_ci.render_hybrid(base_rgb, base_diag, prob30)
            sub_note = "Iniziazione ibrida: nucleo al tempo t (solido) + previsione 0-30 min continua (sfumata)"
        elif render_mode == "instantaneous":
            rgb_ci = renderer_ci.render(base_rgb, base_diag)
            sub_note = "Iniziazione convettiva istantanea (tempo t) · Solo 7 canali termici SEVIRI"
        else:
            rgb_ci = renderer_ci.render(base_rgb, prob30)
            sub_note = "Previsione 0-30 min continua · Solo 7 canali termici SEVIRI (24/7)"

        ts_str = ts_curr.strftime("%Y-%m-%d %H:%M")
        composed = _compose_single_full_domain_frame(
            rgb_ci, f"NWC SAF Convection Initiation (GEO-CI puro IR), {ts_str} UTC", note=sub_note
        )
        writer.write_frame(composed)

        if (step_idx + 1) % 25 == 0 or (step_idx + 1) == num_frames:
            rate = (step_idx + 1) / max(0.1, time.time() - t_start)
            active_pct = float(np.count_nonzero(prob30 > 0) / (h * w) * 100.0)
            print(
                f"   [{day_str}] frame {step_idx + 1}/{num_frames} "
                f"({rate:.2f} fps) | CI prob30 active: {active_pct:.2f}%",
                flush=True,
            )

        if step_idx % 30 == 0:
            gc.collect()

    writer.close()
    gc.collect()

    # Save cache atomically
    tmp_cache = cache_file.with_name(f"{cache_file.stem}_part.npz")
    np.savez_compressed(
        tmp_cache,
        base_diagnosis=daily_base_diag,
        ci_prob30=daily_prob30,
        ci_prob60=daily_prob60,
        timestamps_unix=daily_times_unix,
    )
    tmp_cache.replace(cache_file)
    print(f"[Pure CI] Cached {cache_file.name} and rendered {day_ci_mp4.name}", flush=True)
    return day_ci_mp4


def process_event_ci(
    event: MedicaneEvent,
    zarr_root: Path,
    out_dir: Path,
    fps: float = 24.0,
    max_frames: Optional[int] = None,
    tight_window: bool = False,
    render_mode: str = "hybrid",
) -> Path:
    event_slug = f"{event.name}_{event.id_final}_{event.days[0]}_{event.days[-1]}"
    ev_dir = out_dir / event_slug
    ev_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"PROCESSING MEDICANE CONVECTION INITIATION: {event.name} (ID: {event.id_final}) [mode={render_mode}]")
    print(f"Window: {event.start_time} UTC to {event.end_time} UTC (tight={tight_window})")
    print(f"Days: {event.days}")
    print(f"Target dir: {ev_dir}")
    print("=" * 80, flush=True)

    day_zarrs: List[Path] = []
    for day_str in event.days:
        zp = zarr_root / day_str[:4] / f"{day_str}.zarr"
        if not zp.exists():
            raise FileNotFoundError(f"Missing required Zarr store {zp}")
        day_zarrs.append(zp)

    geom = _load_full_domain_geometry(day_zarrs[0])

    # Shared tracker for continuous stateful tracking across consecutive days
    ci_config = NwcSafCiConfig(mode="night_or_pure_ir")
    tracker = NwcSafCiTracker(config=ci_config)

    day_mp4s: List[Path] = []
    for zp in day_zarrs:
        d_mp4 = process_single_day_pure_ci(
            zp,
            geom,
            ev_dir,
            tracker=tracker,
            fps=fps,
            max_frames=max_frames,
            time_window=(event.start_time, event.end_time) if tight_window else None,
            render_mode=render_mode,
        )
        if d_mp4 is not None:
            day_mp4s.append(d_mp4)

    suffix = f"_{render_mode}" if render_mode != "forecast" else ""
    final_mp4 = ev_dir / f"medicane_{event_slug}_pure_ci{suffix}.mp4"
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

    if ARTIFACTS_DIR.exists():
        mirror_target = ARTIFACTS_DIR / final_mp4.name
        shutil.copy2(final_mp4, mirror_target)
        print(f"[Artifact] Mirrored video to: {mirror_target}", flush=True)

    print(f"\n[COMPLETED] CI Video ready: {final_mp4}", flush=True)
    return final_mp4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-root", type=Path, default=DEFAULT_ZARR_ROOT, help="Root path for Zarr stores")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for results")
    parser.add_argument("--fps", type=float, default=24.0, help="Output video frames per second")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per day for testing")
    parser.add_argument("--tight-window", action="store_true", default=False, help="Crop strictly between start_time and end_time")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "instantaneous", "forecast"],
        help="Rendering mode: hybrid (solid core + continuous 30m corridor), instantaneous, or forecast",
    )
    args = parser.parse_args()

    process_event_ci(
        IANOS_EVENT,
        args.zarr_root,
        args.out_dir,
        fps=args.fps,
        max_frames=args.max_frames,
        tight_window=args.tight_window,
        render_mode=args.render_mode,
    )


if __name__ == "__main__":
    main()
