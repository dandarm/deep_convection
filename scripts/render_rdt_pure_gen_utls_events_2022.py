#!/usr/bin/env python3
"""Render the selected 2022 RDT events on the complete local SEVIRI domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import zarr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.build_rdt_pure_gen_utls_week import _geometry, _orient  # noqa: E402
from scripts.render_rdt_pure_gen_utls_week import (  # noqa: E402
    CLASS_COLORS,
    CLASS_NAMES,
    _font,
    _radiance_composite,
    _static_overlay,
    _video_metadata,
)
from scripts.search_rdt_pure_gen_utls_events_2022 import _ordered_inventory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(r"E:\Datasets\MSG-SEVIRI\2022"))
    parser.add_argument(
        "--search-root",
        type=Path,
        default=ROOT / "results" / "rdt_pure_gen_utls_events_2022",
    )
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "figures"
            / "videomae_pretraining"
            / "rdt_pure_gen_utls_2022_30_events_full_domain.mp4"
        ),
    )
    return parser.parse_args()


def _peak_panel(canvas: Image.Image, event: dict, ordinal: int) -> Image.Image:
    panel = canvas.copy()
    draw = ImageDraw.Draw(panel, "RGBA")
    font = _font(24, bold=True)
    draw.rounded_rectangle([18, 91, 420, 132], radius=7, fill=(4, 11, 22, 220))
    draw.text(
        (31, 111),
        f"{ordinal:02d}  {pd.Timestamp(event['peak_time']):%Y-%m-%d %H:%M UTC}",
        font=font,
        anchor="lm",
        fill=(248, 250, 252, 255),
    )
    return panel


def _contact_sheet(peaks: list[Image.Image], events: list[dict], path: Path) -> None:
    columns = 3
    thumb_width = 620
    thumb_height = int(round(peaks[0].height * thumb_width / peaks[0].width))
    rows = (len(peaks) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), (4, 11, 22))
    for index, image in enumerate(peaks):
        thumb = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        sheet.paste(thumb, ((index % columns) * thumb_width, (index // columns) * thumb_height))
    metadata = sheet.getexif()
    metadata[270] = (
        "Thirty full-domain peak frames selected from distinct important-convection tracks; "
        "RDT_original palette; OT is exact pixel coverage without dilation."
    )
    sheet.save(path, exif=metadata)


def main() -> int:
    args = parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffmpeg and ffprobe are required")
    manifest_path = args.search_root / "selected_events.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = manifest["events"]
    if not events:
        raise ValueError("The event manifest contains no selected events")

    first_day = events[0]["day"]
    geometry = _geometry(args.source_root / f"{first_day}.zarr")
    map_height, map_width = np.asarray(geometry["latitude"]).shape
    canvas_width = map_width + 24
    canvas_height = map_height + 165
    if canvas_height % 2:
        canvas_height += 1
    canvas_size = (canvas_width, canvas_height)
    static = _static_overlay(geometry, canvas_size)
    valid = np.asarray(geometry["valid_geolocation"])
    source_x = np.asarray(geometry["source_x"])
    source_y = np.asarray(geometry["source_y"])
    header_font = _font(24, bold=True)
    sub_font = _font(14)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_frames = sum(
        int(event["clip_stop_frame_exclusive"]) - int(event["clip_start_frame"])
        for event in events
    )
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
        "title=Thirty 2022 important-convection samples — pure GEN plus UTLS proxy",
        "-metadata",
        "comment=RDT_original palette; exact OT pixels; no rings or dilation",
        str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("Unable to open ffmpeg input pipe")
    peaks: list[Image.Image] = []
    written = 0
    try:
        for event_index, event in enumerate(events, start=1):
            day = event["day"]
            source_group = zarr.open_group(str(args.source_root / f"{day}.zarr"), mode="r")
            source = source_group["seviri"]
            times, indices = _ordered_inventory(source_group)
            labels_group = zarr.open_group(
                str(args.search_root / "daily" / day / "labels.zarr"), mode="r"
            )
            labels = labels_group["semantic_label"]
            start = int(event["clip_start_frame"])
            stop = int(event["clip_stop_frame_exclusive"])
            for clip_index, local_frame in enumerate(range(start, stop), start=1):
                frame = np.asarray(source[int(indices[local_frame])], dtype=np.float32)
                frame, _, _ = _orient(frame, source_x, source_y)
                semantic = np.asarray(labels[local_frame], dtype=np.uint8)
                map_rgb = _radiance_composite(frame[4], semantic, valid)
                canvas = Image.new("RGB", canvas_size, (4, 11, 22))
                canvas.paste(Image.fromarray(map_rgb, mode="RGB"), (12, 80))
                canvas = Image.alpha_composite(canvas.convert("RGBA"), static)
                draw = ImageDraw.Draw(canvas, "RGBA")
                stamp = times[local_frame]
                draw.text(
                    (14, 13),
                    f"{stamp:%Y-%m-%d %H:%M} UTC  ·  evento {event_index:02d}/{len(events):02d}  ·  frame {clip_index:02d}/{stop-start:02d}",
                    font=header_font,
                    fill=(248, 250, 252, 255),
                )
                draw.text(
                    (canvas_width - 14, 18),
                    (
                        f"picco: classe {event['class_id']} · {event['area_km2']:.0f} km² · "
                        f"OT {event['ot_pixels']} px · {event['centre_lat']:.1f}°, {event['centre_lon']:.1f}°"
                    ),
                    font=sub_font,
                    anchor="ra",
                    fill=(94, 234, 212, 255),
                )
                y0, y1, x0, x1 = [int(value) for value in event["bbox_yx"]]
                target_box = [12 + x0, 80 + y0, 12 + x1 - 1, 80 + y1 - 1]
                is_peak = local_frame == int(event["local_frame"])
                draw.rectangle(
                    target_box,
                    outline=(45, 212, 191, 255 if is_peak else 190),
                    width=4 if is_peak else 2,
                )
                target_font = _font(13, bold=True)
                target_text = "TARGET" + (" · PICCO" if is_peak else "")
                text_y = max(82, target_box[1] - 18)
                draw.text(
                    (target_box[0], text_y),
                    target_text,
                    font=target_font,
                    fill=(94, 234, 212, 255),
                    stroke_width=2,
                    stroke_fill=(4, 11, 22, 230),
                )
                rgb = canvas.convert("RGB")
                if is_peak:
                    peaks.append(_peak_panel(rgb, event, event_index))
                process.stdin.write(np.asarray(rgb, dtype=np.uint8).tobytes())
                written += 1
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    if written != total_frames:
        raise RuntimeError(f"Expected {total_frames} frames, wrote {written}")

    sheet_path = args.output.with_name(args.output.stem + "_contact_sheet.jpg")
    _contact_sheet(peaks, events, sheet_path)
    summary = {
        "video": str(args.output.resolve()),
        "contact_sheet": str(sheet_path.resolve()),
        "events": len(events),
        "frames": written,
        "fps": args.fps,
        "clip_frames": manifest["clip_shape_tchw"][0],
        "class_palette_rgb_0_to_6": CLASS_COLORS.astype(int).tolist(),
        "semantic_class_names": {str(key): value for key, value in CLASS_NAMES.items()},
        "ot_rendering": "exact semantic label pixels; no dilation, halos or rings",
        "target_annotation": "cyan bounding box; visualization only; not part of the labels",
        "grid": "complete local SEVIRI RSS archive domain, native north-up/west-left grid",
        "ir108_display_range_k": [190.0, 300.0],
        "video_metadata": _video_metadata(args.output),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
