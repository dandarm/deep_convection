#!/usr/bin/env python3
"""Render the newly computed pure-GEN/object-local-UTLS weekly product.

Every observation is read again from the local seven-channel SEVIRI archive.
The IR 10.8 background uses one fixed 190--300 K display scale for the whole
week.  Labels remain on the native satellite grid and are overlaid with the
original detailed RDT lifecycle classes (0..6).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import zarr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.build_rdt_pure_gen_utls_week import _day_inventory, _geometry, _orient  # noqa: E402


CLASS_NAMES = {
    0: "sfondo",
    1: "nube fredda non convettiva",
    2: "triggering",
    3: "growing",
    4: "mature",
    5: "decaying",
    6: "overshooting top persistente",
}
CLASS_COLORS = np.asarray(
    [
        [0, 0, 0],
        [59, 130, 246],
        [250, 204, 21],
        [249, 115, 22],
        [220, 38, 38],
        [153, 27, 27],
        [245, 105, 255],
    ],
    dtype=np.float32,
)
CLASS_ALPHA = np.asarray([0.0, 0.55, 0.82, 0.82, 0.82, 0.82, 0.96], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"E:\Datasets\MSG-SEVIRI\2020"),
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=ROOT / "results" / "rdt_pure_gen_utls_week_20200914_20200920",
    )
    parser.add_argument("--first-day", default="20200914")
    parser.add_argument("--last-day", default="20200920")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "figures"
            / "videomae_pretraining"
            / "sept2020_pure_gen_utls_v1_full_domain.mp4"
        ),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _contour_segments(field: np.ndarray, levels: list[float]) -> dict[float, list[np.ndarray]]:
    figure, axis = plt.subplots(figsize=(2, 2))
    contour = axis.contour(np.ma.masked_invalid(field), levels=levels)
    segments = {
        float(level): [segment for segment in group if len(segment) >= 2]
        for level, group in zip(contour.levels, contour.allsegs)
    }
    plt.close(figure)
    return segments


def _format_lon(value: float) -> str:
    if value < 0:
        return f"{abs(int(value))}°W"
    if value > 0:
        return f"{int(value)}°E"
    return "0°"


def _label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    *,
    anchor: str = "mm",
) -> None:
    bounds = draw.textbbox(xy, text, font=font, anchor=anchor, stroke_width=2)
    draw.rounded_rectangle(
        [bounds[0] - 3, bounds[1] - 2, bounds[2] + 3, bounds[3] + 2],
        radius=3,
        fill=(4, 11, 22, 190),
    )
    draw.text(
        xy,
        text,
        font=font,
        anchor=anchor,
        fill=(241, 245, 249, 255),
        stroke_width=2,
        stroke_fill=(4, 11, 22, 235),
    )


def _static_overlay(geometry: dict, canvas_size: tuple[int, int]) -> Image.Image:
    width, height = canvas_size
    map_left = 12
    map_top = 80
    longitude = np.asarray(geometry["longitude"])
    latitude = np.asarray(geometry["latitude"])
    map_height, map_width = latitude.shape
    overlay = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    tick_font = _font(14, bold=True)
    longitude_levels = [-20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0]
    latitude_levels = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
    for segments in _contour_segments(longitude, longitude_levels).values():
        for segment in segments:
            draw.line(
                [(map_left + float(x), map_top + float(y)) for x, y in segment],
                fill=(45, 212, 191, 110),
                width=1,
            )
    for segments in _contour_segments(latitude, latitude_levels).values():
        for segment in segments:
            draw.line(
                [(map_left + float(x), map_top + float(y)) for x, y in segment],
                fill=(253, 224, 71, 100),
                width=1,
            )

    southern = longitude[-1]
    for value in longitude_levels:
        distance = np.where(np.isfinite(southern), np.abs(southern - value), np.inf)
        column = int(np.argmin(distance))
        if distance[column] <= 0.5:
            _label(
                draw,
                (
                    float(np.clip(map_left + column, 42, width - 42)),
                    map_top + map_height - 17,
                ),
                _format_lon(value),
                tick_font,
            )
    central_latitude = latitude[:, map_width // 2]
    for value in latitude_levels:
        distance = np.where(
            np.isfinite(central_latitude), np.abs(central_latitude - value), np.inf
        )
        row = int(np.argmin(distance))
        if distance[row] <= 0.5:
            _label(draw, (map_left + 31, map_top + row), f"{int(value)}°N", tick_font)
    draw.rectangle(
        [map_left, map_top, map_left + map_width - 1, map_top + map_height - 1],
        outline=(226, 232, 240, 150),
        width=1,
    )

    legend_font = _font(14, bold=True)
    footer_y = map_top + map_height + 21
    x = map_left + 6
    for class_id in range(1, 7):
        color = tuple(int(v) for v in CLASS_COLORS[class_id]) + (255,)
        draw.rectangle([x, footer_y, x + 18, footer_y + 18], fill=color)
        draw.text(
            (x + 25, footer_y + 9),
            f"{class_id} {CLASS_NAMES[class_id]}",
            font=legend_font,
            anchor="lm",
            fill=(226, 232, 240, 255),
        )
        x += 266 if class_id == 1 else 225
    note_font = _font(13)
    draw.text(
        (width - 14, height - 16),
        "Dominio RSS locale completo · griglia satellitare nativa · nessuna parallasse · BT assoluta 190–300 K",
        font=note_font,
        anchor="rs",
        fill=(148, 163, 184, 255),
    )
    return overlay


def _radiance_composite(ir108: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # One absolute display transform for all pixels, channels are never
    # independently stretched.  Cold tops are bright, warm surfaces dark.
    finite = valid & np.isfinite(ir108)
    safe_ir = np.where(finite, ir108, 300.0)
    gray = np.clip((300.0 - safe_ir) / 110.0, 0.0, 1.0) * 255.0
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    rgb[~finite] = np.asarray([4.0, 11.0, 22.0])
    alpha = CLASS_ALPHA[labels][..., None]
    rgb = rgb * (1.0 - alpha) + CLASS_COLORS[labels] * alpha
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _preview_index(metrics: list[dict], total_frames: int) -> int:
    if not metrics:
        return max(0, total_frames // 2)
    return int(
        max(
            metrics[:total_frames],
            key=lambda row: (
                row["class_counts_0_to_6"][6],
                sum(row["class_counts_0_to_6"][3:6]),
            ),
        )["frame"]
    )


def _video_metadata(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    args = parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffmpeg and ffprobe are required")
    progress_path = args.labels_root / "progress.json"
    labels_path = args.labels_root / "labels.zarr"
    metrics_path = args.labels_root / "frame_metrics.json"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    if args.max_frames is None and progress.get("status") != "complete":
        raise RuntimeError("The weekly label product is not complete")

    paths = sorted(args.source_root.glob("*.zarr"))
    paths = [path for path in paths if args.first_day <= path.stem <= args.last_day]
    inventory = _day_inventory(paths)
    geometry = _geometry(paths[0])
    labels_group = zarr.open_group(str(labels_path), mode="r")
    total_available = len(paths) * 288
    total_frames = total_available if args.max_frames is None else min(args.max_frames, total_available)
    if labels_group["semantic_label"].shape[0] < total_frames:
        raise ValueError("Label store is shorter than the requested render")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else []
    preview_index = _preview_index(metrics, total_frames)

    map_height, map_width = np.asarray(geometry["latitude"]).shape
    canvas_width = map_width + 24
    canvas_height = map_height + 165
    if canvas_height % 2:
        canvas_height += 1
    canvas_size = (canvas_width, canvas_height)
    static = _static_overlay(geometry, canvas_size)
    valid = np.asarray(geometry["valid_geolocation"])
    header_font = _font(25, bold=True)
    sub_font = _font(15)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{canvas_width}x{canvas_height}",
        "-r",
        f"{args.fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-metadata",
        "title=RDT-CW-inspired pure GEN plus object-local UTLS proxy",
        "-metadata",
        "comment=Seven-channel SEVIRI only; native grid; no CAL, NWP, OPERA, lightning or parallax",
        str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("Unable to open ffmpeg input pipe")
    preview_path = args.output.with_name(args.output.stem + "_preview.png")
    global_index = 0
    try:
        for day in inventory:
            source = zarr.open_group(str(day["path"]), mode="r")["seviri"]
            for local_index in range(288):
                if global_index >= total_frames:
                    break
                raw_index = int(day["indices"][local_index])
                frame = np.asarray(source[raw_index], dtype=np.float32)
                frame, _, _ = _orient(
                    frame,
                    np.asarray(geometry["source_x"]),
                    np.asarray(geometry["source_y"]),
                )
                labels = np.asarray(labels_group["semantic_label"][global_index], dtype=np.uint8)
                map_rgb = _radiance_composite(frame[4], labels, valid)
                canvas = Image.new("RGB", canvas_size, (4, 11, 22))
                canvas.paste(Image.fromarray(map_rgb, mode="RGB"), (12, 80))
                canvas = Image.alpha_composite(canvas.convert("RGBA"), static)
                draw = ImageDraw.Draw(canvas, "RGBA")
                stamp = pd.Timestamp(day["times"][local_index])
                draw.text(
                    (14, 13),
                    f"{stamp:%Y-%m-%d %H:%M} UTC  ·  frame {global_index + 1}/{total_frames}",
                    font=header_font,
                    fill=(248, 250, 252, 255),
                )
                draw.text(
                    (canvas_width - 14, 19),
                    "PURE GEN + proxy UTLS locale multispettrale · classi RDT 0–6",
                    font=sub_font,
                    anchor="ra",
                    fill=(94, 234, 212, 255),
                )
                rgb_canvas = canvas.convert("RGB")
                if global_index == preview_index:
                    rgb_canvas.save(preview_path)
                process.stdin.write(np.asarray(rgb_canvas, dtype=np.uint8).tobytes())
                global_index += 1
            if global_index >= total_frames:
                break
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

    class_totals = np.zeros(7, dtype=np.int64)
    for row in metrics[:total_frames]:
        class_totals += np.asarray(row["class_counts_0_to_6"], dtype=np.int64)
    summary = {
        "video": str(args.output.resolve()),
        "preview": str(preview_path.resolve()),
        "algorithm": labels_group.attrs.get("algorithm"),
        "config_hash": labels_group.attrs.get("config_hash"),
        "frames": total_frames,
        "observation_cadence_minutes": 5,
        "first_time": pd.Timestamp(inventory[0]["times"][0]).isoformat(),
        "last_time": pd.Timestamp(int(labels_group["time_ns"][total_frames - 1])).isoformat(),
        "semantic_class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "class_palette_rgb_0_to_6": CLASS_COLORS.astype(int).tolist(),
        "ot_rendering": "exact semantic label pixels; no dilation, halos or rings",
        "class_pixel_totals_0_to_6": class_totals.tolist(),
        "preview_frame": preview_index,
        "preview_time": (
            metrics[preview_index]["time"]
            if len(metrics) > preview_index
            else pd.Timestamp(int(labels_group["time_ns"][preview_index])).isoformat()
        ),
        "uses_cal": False,
        "uses_opera": False,
        "uses_nwp": False,
        "uses_lightning": False,
        "uses_parallax": False,
        "grid": "native SEVIRI RSS archive, north-up/west-left",
        "ir108_display_range_k": [190.0, 300.0],
        "scope_note": (
            "Complete 763x1762 domain in the local RSS archive, not the complete MSG Earth disk."
        ),
        "video_metadata": _video_metadata(args.output),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
