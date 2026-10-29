#!/usr/bin/env python3
"""Render a synchronized NWC SAF GIF / local RDT-CW-inspired comparison.

The upper panel is the unmodified public archive GIF.  The lower panel shows
the local IR 10.8 observation and only lifecycle classes 2--5.  It is sampled
onto the calibrated MSG-0 GIF view, including the selected cloud-height
parallax hypothesis, so the panels share the same geographic canvas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.nwcsaf_rdt_archive import (  # noqa: E402
    GifGeoReference,
    TARGET_CLASSES,
    differential_view_parallax,
    sha256_file,
)
from emma_gpm.rdt.pure_gen_utls import PureGenUtlsTracker  # noqa: E402
from emma_gpm.rdt.storage import DomainFrameReader  # noqa: E402
from emma_gpm.rdt_teacher import projected_pixel_areas  # noqa: E402


LOCAL_CLASS_NAMES = {
    2: "triggering",
    3: "growing",
    4: "mature",
    5: "decay",
}
# Project-standard pure-GEN palette used by render_rdt_pure_gen_utls_week.py.
LOCAL_CLASS_RGB = {
    2: (250, 204, 21),
    3: (249, 115, 22),
    4: (220, 38, 38),
    5: (153, 27, 27),
}
LOCAL_CLASS_ALPHA = 0.82


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


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _find_ffmpeg(explicit: str | None) -> str:
    if explicit:
        return explicit
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError("Install FFmpeg or imageio-ffmpeg, or pass --ffmpeg") from error


def _manifest_index(path: Path) -> dict[pd.Timestamp, Path]:
    table = pd.read_csv(path)
    required = {"time", "status", "path", "sha256"}
    if not required.issubset(table):
        raise ValueError(f"Archive manifest lacks columns {sorted(required - set(table))}")
    result: dict[pd.Timestamp, Path] = {}
    for row in table.itertuples(index=False):
        if row.status == "missing" or not isinstance(row.path, str):
            continue
        gif = Path(row.path)
        if not gif.exists() or sha256_file(gif) != str(row.sha256):
            raise ValueError(f"Missing or changed GIF declared by manifest: {gif}")
        result[pd.Timestamp(row.time)] = gif
    return result


def _available_times(zarr_root: Path, times: pd.DatetimeIndex) -> set[pd.Timestamp]:
    available: set[pd.Timestamp] = set()
    for day in times.normalize().unique():
        path = zarr_root / f"{day:%Y}" / f"{day:%Y%m%d}.zarr"
        if not path.exists():
            continue
        with xr.open_zarr(path, consolidated=True, chunks=None) as dataset:
            available.update(pd.DatetimeIndex(dataset.time.values))
    return available


def _oriented_geometry(
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    projection: str,
) -> dict[str, np.ndarray | str]:
    area = projected_pixel_areas(raw_x, raw_y, projection, allow_invalid=True)
    x, y = raw_x.copy(), raw_y.copy()
    if y[0] < y[-1]:
        y = y[::-1]
        area = area[::-1]
    if x[0] > x[-1]:
        x = x[::-1]
        area = area[:, ::-1]
    xx, yy = np.meshgrid(x, y)
    to_lonlat = Transformer.from_crs(CRS.from_user_input(projection), "EPSG:4326", always_xy=True)
    longitude, latitude = to_lonlat.transform(xx, yy)
    return {
        "x": x,
        "y": y,
        "area_km2": np.ascontiguousarray(area, dtype=np.float32),
        "longitude": np.asarray(longitude, dtype=np.float32),
        "latitude": np.asarray(latitude, dtype=np.float32),
        "projection": projection,
    }


def _geostationary_longitude(crs: CRS, fallback: float) -> float:
    operation = crs.coordinate_operation
    if operation is not None:
        for parameter in operation.params:
            if parameter.name.lower() == "longitude of natural origin":
                return float(parameter.value)
    return fallback


def _target_to_source_lookup(
    georef: GifGeoReference,
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_crs: CRS | str,
    *,
    cloud_height_km: float,
    chunk_rows: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each GIF-map pixel to the nearest north-up/west-left RSS pixel."""

    x0, y0, x1, y1 = georef.map_bbox_xyxy
    height, width = y1 - y0, x1 - x0
    source_rows = np.zeros((height, width), dtype=np.int32)
    source_columns = np.zeros((height, width), dtype=np.int32)
    valid = np.zeros((height, width), dtype=bool)
    source_projection = CRS.from_user_input(source_crs)
    source_longitude = _geostationary_longitude(source_projection, 9.5)
    reference_longitude = _geostationary_longitude(georef.geos_crs, 0.0)
    reference_to_lonlat = Transformer.from_crs(georef.geos_crs, "EPSG:4326", always_xy=True)
    lonlat_to_source = Transformer.from_crs("EPSG:4326", source_projection, always_xy=True)
    columns = np.arange(x0, x1, dtype=np.float64)

    for output_start in range(0, height, chunk_rows):
        output_stop = min(output_start + chunk_rows, height)
        rows = np.arange(y0 + output_start, y0 + output_stop, dtype=np.float64)
        cc, rr = np.meshgrid(columns, rows)
        target_x, target_y = georef.pixel_to_projected(rr, cc)
        target_lon, target_lat = reference_to_lonlat.transform(target_x, target_y)
        source_lon, source_lat, parallax_valid = differential_view_parallax(
            target_lon,
            target_lat,
            source_longitude_deg=reference_longitude,
            target_longitude_deg=source_longitude,
            cloud_height_km=cloud_height_km,
        )
        sx, sy = lonlat_to_source.transform(source_lon, source_lat)
        source_column_float = (sx - source_x[0]) / (source_x[-1] - source_x[0]) * (len(source_x) - 1)
        source_row_float = (sy - source_y[0]) / (source_y[-1] - source_y[0]) * (len(source_y) - 1)
        inside = (
            parallax_valid
            & np.isfinite(source_column_float)
            & np.isfinite(source_row_float)
            & (source_column_float >= -0.5)
            & (source_column_float < len(source_x) - 0.5)
            & (source_row_float >= -0.5)
            & (source_row_float < len(source_y) - 0.5)
        )
        safe_row_float = np.where(np.isfinite(source_row_float), source_row_float, 0.0)
        safe_column_float = np.where(np.isfinite(source_column_float), source_column_float, 0.0)
        local_rows = np.clip(np.rint(safe_row_float), 0, len(source_y) - 1).astype(np.int32)
        local_columns = np.clip(np.rint(safe_column_float), 0, len(source_x) - 1).astype(np.int32)
        selection = slice(output_start, output_stop)
        source_rows[selection] = local_rows
        source_columns[selection] = local_columns
        valid[selection] = inside
    return source_rows, source_columns, valid


def _local_map_rgb(
    ir108: np.ndarray,
    labels: np.ndarray,
    observation_valid: np.ndarray,
    source_rows: np.ndarray,
    source_columns: np.ndarray,
    mapping_valid: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    sampled_ir = ir108[source_rows, source_columns]
    sampled_labels = labels[source_rows, source_columns]
    valid = mapping_valid & observation_valid[source_rows, source_columns] & np.isfinite(sampled_ir)
    safe_ir = np.where(valid, sampled_ir, 300.0)
    gray = np.clip((300.0 - safe_ir) / 110.0, 0.0, 1.0) * 255.0
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    rgb[~valid] = np.asarray([4.0, 11.0, 22.0])
    counts: dict[int, int] = {}
    for class_id in TARGET_CLASSES:
        selected = valid & (sampled_labels == class_id)
        counts[class_id] = int(selected.sum())
        color = np.asarray(LOCAL_CLASS_RGB[class_id], dtype=np.float32)
        rgb[selected] = rgb[selected] * (1.0 - LOCAL_CLASS_ALPHA) + color * LOCAL_CLASS_ALPHA
    return np.clip(rgb, 0, 255).astype(np.uint8), counts


def _render_local_panel(
    map_rgb: np.ndarray,
    georef: GifGeoReference,
    timestamp: pd.Timestamp,
    cloud_height_km: float,
) -> Image.Image:
    panel = Image.new("RGB", (georef.image_width, georef.image_height), (245, 247, 250))
    x0, y0, x1, y1 = georef.map_bbox_xyxy
    panel.paste(Image.fromarray(map_rgb, mode="RGB"), (x0, y0))
    draw = ImageDraw.Draw(panel)
    title_font = _font(30, bold=True)
    legend_font = _font(22, bold=True)
    note_font = _font(18)
    draw.text(
        (georef.image_width // 2, 18),
        f"RDT-CW-inspired locale — {timestamp:%Y-%m-%d %H:%M} UTC",
        font=title_font,
        anchor="mm",
        fill=(15, 23, 42),
    )
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(100, 116, 139), width=2)
    legend_y = y1 + 35
    x = x0 + 26
    for class_id in TARGET_CLASSES:
        color = LOCAL_CLASS_RGB[class_id]
        draw.rectangle((x, legend_y, x + 30, legend_y + 30), fill=color)
        draw.text(
            (x + 42, legend_y + 15),
            LOCAL_CLASS_NAMES[class_id],
            font=legend_font,
            anchor="lm",
            fill=(15, 23, 42),
        )
        x += 275
    draw.text(
        (x0 + 26, legend_y + 66),
        f"IR 10.8: 190–300 K · vista MSG-0 · parallasse {cloud_height_km:g} km",
        font=note_font,
        fill=(51, 65, 85),
    )
    draw.text(
        (x0 + 26, legend_y + 94),
        "Mostrate solo le classi 2–5; nube fredda non convettiva e OT esclusi",
        font=note_font,
        fill=(51, 65, 85),
    )
    return panel


def _stack_panels(top: Image.Image, bottom: Image.Image, output_width: int) -> Image.Image:
    if top.size != bottom.size:
        width_difference = abs(top.width - bottom.width)
        height_difference = abs(top.height - bottom.height)
        if width_difference > 4 or height_difference > 4:
            raise ValueError(f"Panel size mismatch: {top.size} versus {bottom.size}")
        # A few public archive GIFs differ by one terminal row or column from
        # the calibrated canvas.  Keep the calibrated local canvas as the
        # fixed video size and align the public panel at the common upper-left
        # map origin, padding or cropping only the terminal edge.
        aligned = Image.new("RGB", bottom.size, (245, 247, 250))
        common_width = min(top.width, bottom.width)
        common_height = min(top.height, bottom.height)
        aligned.paste(top.crop((0, 0, common_width, common_height)), (0, 0))
        top = aligned
    band_height = 52
    full = Image.new("RGB", (top.width, 2 * top.height + 2 * band_height), (4, 11, 22))
    draw = ImageDraw.Draw(full)
    font = _font(26, bold=True)
    draw.text(
        (top.width // 2, band_height // 2),
        "NWC SAF RDT-CW — GIF pubblico",
        font=font,
        anchor="mm",
        fill=(241, 245, 249),
    )
    full.paste(top, (0, band_height))
    second_band_y = band_height + top.height
    draw.text(
        (top.width // 2, second_band_y + band_height // 2),
        "Teacher locale RDT-CW-inspired — stessa osservazione UTC",
        font=font,
        anchor="mm",
        fill=(241, 245, 249),
    )
    full.paste(bottom, (0, second_band_y + band_height))
    output_height = int(round(full.height * output_width / full.width))
    output_height += output_height % 2
    return full.resize((output_width, output_height), Image.Resampling.LANCZOS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-root", type=Path, required=True)
    parser.add_argument("--gif-manifest", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "configs" / "nwcsaf_rdt_gif_v2025_msg.json",
    )
    parser.add_argument("--start", required=True, help="Teacher sequence start, UTC")
    parser.add_argument("--evaluation-start", required=True, help="First rendered quarter-hour slot")
    parser.add_argument("--end", required=True, help="Exclusive UTC end")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--cloud-height-km", type=float, default=12.0)
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    evaluation_start = pd.Timestamp(args.evaluation_start)
    end = pd.Timestamp(args.end)
    if not start < evaluation_start < end:
        parser.error("Require --start < --evaluation-start < --end")
    if args.fps <= 0 or args.width < 640 or args.width % 2 or args.cloud_height_km < 0:
        parser.error("FPS and cloud height must be valid; width must be even and at least 640")
    times = pd.date_range(start, end, freq="5min", inclusive="left")
    evaluation_times = [stamp for stamp in times if stamp >= evaluation_start and stamp.minute % 15 == 0]
    if not evaluation_times:
        raise ValueError("No quarter-hour render slots in the requested interval")
    available_times = _available_times(args.zarr_root, times)
    missing_data = [stamp for stamp in times if stamp not in available_times]
    if missing_data:
        raise ValueError(f"Incomplete five-minute teacher sequence: {missing_data[:5]}")
    gif_by_time = _manifest_index(args.gif_manifest)
    missing_gifs = [stamp for stamp in evaluation_times if stamp not in gif_by_time]
    if missing_gifs:
        raise ValueError(f"Missing co-temporal archive GIFs: {missing_gifs[:5]}")
    if args.output.exists():
        raise FileExistsError(args.output)

    georef = GifGeoReference.from_json(args.calibration)
    reader = DomainFrameReader(args.zarr_root)
    first_frame = reader.load(times[0])
    if reader.geometry is None:
        raise RuntimeError("Reader did not expose source geometry")
    raw_x, raw_y, projection = reader.geometry
    geometry = _oriented_geometry(raw_x, raw_y, projection)
    source_rows, source_columns, mapping_valid = _target_to_source_lookup(
        georef,
        np.asarray(geometry["x"]),
        np.asarray(geometry["y"]),
        str(geometry["projection"]),
        cloud_height_km=args.cloud_height_km,
    )

    output_height = int(round((2 * georef.image_height + 104) * args.width / georef.image_width))
    output_height += output_height % 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.stem + ".part.mp4")
    if partial.exists():
        partial.unlink()
    log_path = args.output.with_suffix(".ffmpeg.log")
    preview_path = args.output.with_suffix(".preview.png")
    metadata_path = args.output.with_suffix(".json")
    command = [
        _find_ffmpeg(args.ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{args.width}x{output_height}",
        "-r",
        str(args.fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial),
    ]

    tracker = PureGenUtlsTracker()
    encoder: subprocess.Popen[bytes] | None = None
    frame_rows: list[dict] = []
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            if encoder.stdin is None:
                raise RuntimeError("Unable to open FFmpeg input pipe")
            for index, stamp in enumerate(times):
                frame = first_frame if index == 0 else reader.load(stamp)
                result = tracker.step_frame(
                    frame,
                    stamp,
                    np.asarray(geometry["area_km2"]),
                    lat_deg=np.asarray(geometry["latitude"]),
                    lon_deg=np.asarray(geometry["longitude"]),
                )
                if stamp not in evaluation_times:
                    continue
                map_rgb, counts = _local_map_rgb(
                    frame[4],
                    np.asarray(result["semantic_label"], dtype=np.uint8),
                    np.asarray(result["valid_observation"], dtype=bool),
                    source_rows,
                    source_columns,
                    mapping_valid,
                )
                local_panel = _render_local_panel(map_rgb, georef, stamp, args.cloud_height_km)
                with Image.open(gif_by_time[stamp]) as gif:
                    top_panel = gif.convert("RGB")
                composed = _stack_panels(top_panel, local_panel, args.width)
                if composed.size != (args.width, output_height):
                    raise RuntimeError(f"Unexpected composed frame size {composed.size}")
                if not frame_rows:
                    composed.save(preview_path)
                encoder.stdin.write(np.asarray(composed, dtype=np.uint8).tobytes())
                frame_rows.append(
                    {
                        "time": stamp.isoformat(),
                        "gif": str(gif_by_time[stamp].resolve()),
                        "local_class_pixels_on_render_grid": {
                            str(class_id): counts[class_id] for class_id in TARGET_CLASSES
                        },
                    }
                )
                print(
                    f"rendered={len(frame_rows)}/{len(evaluation_times)} time={stamp.isoformat()} "
                    f"teacher_frame={index + 1}/{len(times)} elapsed_seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )
            encoder.stdin.close()
            return_code = encoder.wait()
            if return_code:
                raise RuntimeError(f"FFmpeg failed with code {return_code}; see {log_path}")
        partial.replace(args.output)
    except BaseException:
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        if partial.exists():
            partial.unlink()
        raise
    finally:
        reader.close()

    metadata = {
        "status": "complete",
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "bytes": args.output.stat().st_size,
        "width": args.width,
        "height": output_height,
        "fps": args.fps,
        "frames": len(frame_rows),
        "duration_seconds": len(frame_rows) / args.fps,
        "teacher_sequence_start": start.isoformat(),
        "evaluation_start": evaluation_start.isoformat(),
        "end_exclusive": end.isoformat(),
        "teacher_cadence_minutes": 5,
        "render_cadence_minutes": 15,
        "cloud_height_km": args.cloud_height_km,
        "classes_shown": {str(key): value for key, value in LOCAL_CLASS_NAMES.items()},
        "classes_excluded": ["non_convective_cold_cloud", "overshooting_top", "triggering_from_split"],
        "top_panel": "public NWC SAF archive GIF; terminal edge normalized by at most four pixels to the calibrated canvas",
        "bottom_panel": "local IR 10.8 plus project-standard lifecycle masks, remapped to calibrated MSG-0 GIF view",
        "gif_manifest": str(args.gif_manifest.resolve()),
        "gif_manifest_sha256": sha256_file(args.gif_manifest),
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": sha256_file(args.calibration),
        "zarr_root": str(args.zarr_root.resolve()),
        "mapping_valid_pixels": int(mapping_valid.sum()),
        "mapping_total_pixels": int(mapping_valid.size),
        "elapsed_seconds": time.monotonic() - started,
        "preview": str(preview_path.resolve()),
        "frames_detail": frame_rows,
    }
    _atomic_json(metadata_path, metadata)
    print(json.dumps({key: metadata[key] for key in ("status", "output", "frames", "duration_seconds", "bytes")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
