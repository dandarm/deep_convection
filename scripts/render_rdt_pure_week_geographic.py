#!/usr/bin/env python3
"""Add a labelled geographic graticule to the cached full-domain pure-GEN video.

This is an exploratory case-selection product.  It deliberately preserves the
RDT_original 0..6 lifecycle schema and the already-computed pure-GEN labels;
it does not claim to contain the planned object-local UTLS proxy upgrade.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-video",
        type=Path,
        default=(
            ROOT
            / "RDT_original"
            / "results"
            / "comparison_seviri_2020"
            / "sept2020_full_domain_3_RDT_puro.mp4"
        ),
    )
    parser.add_argument(
        "--geometry-zarr",
        type=Path,
        default=Path(r"E:\Datasets\MSG-SEVIRI\2020\20200914.zarr"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "figures"
            / "videomae_pretraining"
            / "sept2020_pure_gen_full_domain_geographic.mp4"
        ),
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=12.0,
        help="Output playback rate. All 2016 five-minute observations are retained.",
    )
    return parser.parse_args()


def _video_metadata(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    stream = metadata["streams"][0]
    numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "source_fps": numerator / denominator,
        "frames": int(stream["nb_frames"]),
        "duration_s": float(metadata["format"]["duration"]),
    }


def _orient_north_up_west_left(
    longitude: np.ndarray,
    latitude: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if y[0] < y[-1]:
        longitude = longitude[::-1]
        latitude = latitude[::-1]
    if x[0] > x[-1]:
        longitude = longitude[:, ::-1]
        latitude = latitude[:, ::-1]
    return np.ascontiguousarray(longitude), np.ascontiguousarray(latitude)


def _load_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_zarr(path, consolidated=True) as dataset:
        x = dataset.x.values.copy()
        y = dataset.y.values.copy()
        projection_wkt = dataset.attrs["projection_wkt"]
    xx, yy = np.meshgrid(x, y)
    transformer = Transformer.from_crs(
        CRS.from_wkt(projection_wkt), CRS.from_epsg(4326), always_xy=True
    )
    longitude, latitude = transformer.transform(xx, yy)
    longitude = np.where(np.isfinite(longitude), longitude, np.nan)
    latitude = np.where(np.isfinite(latitude), latitude, np.nan)
    return _orient_north_up_west_left(longitude, latitude, x, y)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    anchor: str = "mm",
) -> None:
    bbox = draw.textbbox(xy, text, font=font, anchor=anchor, stroke_width=2)
    pad = 2
    draw.rounded_rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        radius=2,
        fill=(5, 12, 24, 176),
    )
    draw.text(
        xy,
        text,
        font=font,
        fill=fill,
        anchor=anchor,
        stroke_width=2,
        stroke_fill=(5, 12, 24, 230),
    )


def _contour_segments(field: np.ndarray, levels: list[float]) -> dict[float, list[np.ndarray]]:
    masked = np.ma.masked_invalid(field)
    fig, axis = plt.subplots(figsize=(2, 2))
    contour = axis.contour(masked, levels=levels)
    output = {
        float(level): [segment for segment in segments if len(segment) >= 2]
        for level, segments in zip(contour.levels, contour.allsegs)
    }
    plt.close(fig)
    return output


def _format_longitude(value: float) -> str:
    if value < 0:
        return f"{abs(int(value))}°W"
    if value > 0:
        return f"{int(value)}°E"
    return "0°"


def _create_overlay(
    longitude: np.ndarray,
    latitude: np.ndarray,
    video_width: int,
    video_height: int,
    output: Path,
) -> dict:
    map_height, map_width = latitude.shape
    map_left = 12
    map_top = 48
    if map_left + map_width > video_width or map_top + map_height > video_height:
        raise ValueError("The geometry does not fit the source video's documented map panel")

    overlay = Image.new("RGBA", (video_width, video_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    label_font = _font(14, bold=True)
    note_font = _font(13, bold=True)
    longitude_levels = [-20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0]
    latitude_levels = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]

    longitude_segments = _contour_segments(longitude, longitude_levels)
    latitude_segments = _contour_segments(latitude, latitude_levels)
    for level, segments in longitude_segments.items():
        for segment in segments:
            points = [(map_left + float(x), map_top + float(y)) for x, y in segment]
            draw.line(points, fill=(64, 224, 208, 95), width=1)
    for level, segments in latitude_segments.items():
        for segment in segments:
            points = [(map_left + float(x), map_top + float(y)) for x, y in segment]
            draw.line(points, fill=(250, 204, 21, 86), width=1)

    # Longitude labels follow the southern, fully observed edge of the domain.
    southern_longitude = longitude[-1]
    for value in longitude_levels:
        finite = np.isfinite(southern_longitude)
        if not finite.any():
            continue
        differences = np.where(finite, np.abs(southern_longitude - value), np.inf)
        column = int(np.argmin(differences))
        if differences[column] > 0.5:
            continue
        _draw_label(
            draw,
            (map_left + column, map_top + map_height - 17),
            _format_longitude(value),
            font=label_font,
            fill=(153, 246, 228, 255),
        )

    # Latitude tick positions are evaluated along the central projection column.
    centre_column = map_width // 2
    central_latitude = latitude[:, centre_column]
    for value in latitude_levels:
        finite = np.isfinite(central_latitude)
        differences = np.where(finite, np.abs(central_latitude - value), np.inf)
        row = int(np.argmin(differences))
        if differences[row] > 0.5:
            continue
        _draw_label(
            draw,
            (map_left + 31, map_top + row),
            f"{int(value)}°N",
            font=label_font,
            fill=(254, 240, 138, 255),
        )

    note = "Graticule: lon 10° · lat 5° | intero dominio RSS archiviato, non full-disk MSG"
    note_bbox = draw.textbbox((0, 0), note, font=note_font)
    note_width = note_bbox[2] - note_bbox[0]
    _draw_label(
        draw,
        (video_width - 18, 24),
        note,
        font=note_font,
        fill=(226, 232, 240, 255),
        anchor="rm",
    )
    draw.rectangle(
        [map_left, map_top, map_left + map_width - 1, map_top + map_height - 1],
        outline=(226, 232, 240, 145),
        width=1,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output)

    finite = np.isfinite(longitude) & np.isfinite(latitude)
    return {
        "map_shape": [map_height, map_width],
        "longitude_range_deg": [float(np.nanmin(longitude)), float(np.nanmax(longitude))],
        "latitude_range_deg": [float(np.nanmin(latitude)), float(np.nanmax(latitude))],
        "finite_geolocation_fraction": float(finite.mean()),
        "longitude_levels_deg": longitude_levels,
        "latitude_levels_deg": latitude_levels,
        "header_note_width_px": note_width,
    }


def _encode(
    input_video: Path,
    overlay: Path,
    output: Path,
    source_fps: float,
    playback_fps: float,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg is required to render the weekly video")
    slowdown = source_fps / playback_fps
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-loop",
            "1",
            "-i",
            str(overlay),
            "-filter_complex",
            f"[0:v][1:v]overlay=0:0:shortest=1,setpts={slowdown:.12f}*PTS[v]",
            "-map",
            "[v]",
            "-r",
            f"{playback_fps:.6f}",
            "-frames:v",
            "2016",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-metadata",
            "title=RDT-CW-inspired pure GEN - September 2020 archived RSS domain",
            "-metadata",
            "comment=Exploratory case-selection video; cached pure-GEN baseline with original 0..6 classes",
            str(output),
        ],
        check=True,
    )


def _render_preview(video: Path, output: Path, timestamp_s: float = 91.0) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp_s:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    args = parse_args()
    if not args.input_video.exists():
        raise FileNotFoundError(args.input_video)
    if not args.geometry_zarr.exists():
        raise FileNotFoundError(args.geometry_zarr)
    metadata = _video_metadata(args.input_video)
    if metadata["frames"] != 2016:
        raise ValueError(f"Expected 2016 five-minute frames, found {metadata['frames']}")
    longitude, latitude = _load_geometry(args.geometry_zarr)
    overlay_path = args.output.with_name(args.output.stem + "_graticule.png")
    preview_path = args.output.with_name(args.output.stem + "_preview.png")
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    geometry = _create_overlay(
        longitude,
        latitude,
        metadata["width"],
        metadata["height"],
        overlay_path,
    )
    _encode(
        args.input_video,
        overlay_path,
        args.output,
        metadata["source_fps"],
        args.playback_fps,
    )
    output_metadata = _video_metadata(args.output)
    _render_preview(args.output, preview_path, timestamp_s=output_metadata["duration_s"] * 0.55)
    summary = {
        "video": str(args.output.resolve()),
        "preview": str(preview_path.resolve()),
        "source_video": str(args.input_video.resolve()),
        "geometry_source": str(args.geometry_zarr.resolve()),
        "algorithm": "rdt_cw_inspired_seviri_pure_gen_cached_baseline",
        "semantic_schema": {
            "0": "background",
            "1": "non-convective cold cloud",
            "2": "triggering",
            "3": "growing",
            "4": "mature",
            "5": "decaying",
            "6": "overshooting top",
        },
        "input": metadata,
        "output": output_metadata,
        "all_input_frames_retained": output_metadata["frames"] == metadata["frames"],
        "observational_interval": "2020-09-14T00:00:00Z/2020-09-20T23:55:00Z",
        "observation_cadence_minutes": 5,
        "geometry": geometry,
        "scope_warning": (
            "The local archive covers the complete 763x1762 RSS subset used by the project, "
            "not the full MSG Earth disk. Its southern edge spans approximately 20 W-40 E; "
            "the curved valid footprint broadens toward high latitudes and reaches about "
            "29-75 N and 62 W-82 E at its finite extremes."
        ),
        "method_warning": (
            "This visualization uses the existing cached pure-GEN labels, including their "
            "experimental parallax remapping and scene-global radiative tropopause proxy. The "
            "planned object-local multispectral UTLS proxy requires a new label run and is not "
            "claimed here."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
