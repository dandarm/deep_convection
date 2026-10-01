#!/usr/bin/env python3
"""Render one full-domain SEVIRI sequence as eleven separate MP4 files."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import os
from pathlib import Path
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from pyproj import Transformer
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.seviri import (  # noqa: E402
    RSS_CRS,
    THERMAL_CHANNELS,
    geographic_bbox_to_geos_bounds,
)


LON_MIN = -20.0
LON_MAX = 40.0
LAT_MIN = 30.0
LAT_MAX = 60.0
BT_MIN_K = 180.0
BT_MAX_K = 330.0
NUM_FRAMES = 16
CADENCE_MINUTES = 5

CHANNEL_LABELS = (
    "WV 6.2 µm",
    "WV 7.3 µm",
    "IR 8.7 µm",
    "IR 9.7 µm",
    "IR 10.8 µm",
    "IR 12.0 µm",
    "IR 13.4 µm",
)

PRODUCTS = (
    ("01_WV_062", "WV 6.2 µm", 0, None, BT_MIN_K, BT_MAX_K),
    ("02_WV_073", "WV 7.3 µm", 1, None, BT_MIN_K, BT_MAX_K),
    ("03_IR_087", "IR 8.7 µm", 2, None, BT_MIN_K, BT_MAX_K),
    ("04_IR_097", "IR 9.7 µm", 3, None, BT_MIN_K, BT_MAX_K),
    ("05_IR_108", "IR 10.8 µm", 4, None, BT_MIN_K, BT_MAX_K),
    ("06_IR_120", "IR 12.0 µm", 5, None, BT_MIN_K, BT_MAX_K),
    ("07_IR_134", "IR 13.4 µm", 6, None, BT_MIN_K, BT_MAX_K),
    ("08_WV_062_minus_IR_108", "WV 6.2 − IR 10.8", 0, 4, -80.0, 20.0),
    ("09_IR_087_minus_IR_108", "IR 8.7 − IR 10.8", 2, 4, -15.0, 15.0),
    ("10_IR_108_minus_IR_120", "IR 10.8 − IR 12.0", 4, 5, -10.0, 10.0),
    ("11_IR_108_minus_IR_134", "IR 10.8 − IR 13.4", 4, 6, -20.0, 40.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path(
            os.environ.get(
                "SEVIRI_ZARR_ROOT",
                r"E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N",
            )
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "catalogs" / "seviri_pretraining_random_100.csv",
    )
    parser.add_argument("--clip-id", default="random_000")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def _utc_naive(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _projection() -> ccrs.Geostationary:
    globe = ccrs.Globe(semimajor_axis=6378169.0, inverse_flattening=295.488065897014)
    return ccrs.Geostationary(
        central_longitude=9.5,
        satellite_height=35785831.0,
        sweep_axis="y",
        globe=globe,
    )


def load_domain_sequence(
    root: Path, start_time: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load ``(T,C,Y,X)`` over the exact geographic domain."""

    expected = pd.date_range(
        start_time,
        periods=NUM_FRAMES,
        freq=f"{CADENCE_MINUTES}min",
    )
    pieces: list[np.ndarray] = []
    selected_x: np.ndarray | None = None
    selected_y: np.ndarray | None = None
    bounds = geographic_bbox_to_geos_bounds(
        LON_MIN, LON_MAX, LAT_MIN, LAT_MAX
    )

    with ExitStack() as stack:
        for normalized_day in expected.normalize().unique():
            day = pd.Timestamp(normalized_day)
            day_times = expected[expected.normalize() == day]
            store = root / f"{day:%Y}" / f"{day:%Y%m%d}.zarr"
            if not store.exists():
                raise FileNotFoundError(f"Missing daily Zarr store: {store}")
            dataset = stack.enter_context(xr.open_zarr(store, consolidated=True)).sortby(
                "time"
            )
            channels = tuple(str(value) for value in dataset.channel.values)
            if channels != THERMAL_CHANNELS:
                raise RuntimeError(f"Unexpected channel order in {store}: {channels}")
            available = pd.DatetimeIndex(dataset.time.values)
            indices = available.get_indexer(day_times)
            if np.any(indices < 0):
                missing = day_times[indices < 0]
                raise RuntimeError(f"Missing frames in {store}: {list(missing)}")

            x_values = np.asarray(dataset.x.values)
            y_values = np.asarray(dataset.y.values)
            x_indices = np.flatnonzero(
                (x_values >= bounds.x_min) & (x_values <= bounds.x_max)
            )
            y_indices = np.flatnonzero(
                (y_values >= bounds.y_min) & (y_values <= bounds.y_max)
            )
            if not len(x_indices) or not len(y_indices):
                raise RuntimeError("Requested Mediterranean domain is outside the Zarr grid.")
            piece = (
                dataset["seviri"]
                .isel(time=indices, y=y_indices, x=x_indices)
                .transpose("time", "channel", "y", "x")
                .values
            )
            current_x = x_values[x_indices]
            current_y = y_values[y_indices]
            if selected_x is None:
                selected_x, selected_y = current_x, current_y
            elif not np.array_equal(selected_x, current_x) or not np.array_equal(
                selected_y, current_y
            ):
                raise RuntimeError("Daily stores do not share the same projected grid.")
            pieces.append(np.asarray(piece, dtype=np.float32))

    assert selected_x is not None and selected_y is not None
    data = np.concatenate(pieces, axis=0)
    if selected_x[0] > selected_x[-1]:
        selected_x = selected_x[::-1]
        data = data[..., ::-1]
    if selected_y[0] > selected_y[-1]:
        selected_y = selected_y[::-1]
        data = data[..., ::-1, :]

    xx, yy = np.meshgrid(selected_x, selected_y)
    longitude, latitude = Transformer.from_crs(
        RSS_CRS, "EPSG:4326", always_xy=True
    ).transform(xx, yy)
    geographic_mask = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= LON_MIN)
        & (longitude <= LON_MAX)
        & (latitude >= LAT_MIN)
        & (latitude <= LAT_MAX)
    )
    return data, selected_x, selected_y, geographic_mask, expected


def _extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    return x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2


def render_product(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    times: pd.DatetimeIndex,
    *,
    output: Path,
    title: str,
    first_channel: int,
    second_channel: int | None,
    vmin: float,
    vmax: float,
    fps: float,
    dpi: int,
) -> None:
    geos = _projection()
    plate = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 7.2))
    axis = fig.add_subplot(1, 1, 1, projection=geos)
    axis.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=plate)
    axis.set_facecolor("#dbe6ec")
    axis.coastlines(resolution="50m", color="#20252b", linewidth=0.75)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="#4b5563")
    gridlines = axis.gridlines(
        crs=plate,
        draw_labels=True,
        linewidth=0.35,
        color="#374151",
        alpha=0.5,
        linestyle=":",
        x_inline=False,
        y_inline=False,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False

    def values(frame: int) -> np.ndarray:
        result = data[frame, first_channel]
        if second_channel is not None:
            result = result - data[frame, second_channel]
        return np.where(mask, result, np.nan)

    if second_channel is None:
        image = axis.imshow(
            values(0),
            origin="lower",
            extent=_extent(x, y),
            transform=geos,
            cmap="turbo_r",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        colorbar_label = "Brightness temperature [K]"
    else:
        image = axis.imshow(
            values(0),
            origin="lower",
            extent=_extent(x, y),
            transform=geos,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
            interpolation="nearest",
        )
        colorbar_label = "Brightness-temperature difference [K]"
    colorbar = fig.colorbar(image, ax=axis, orientation="horizontal", pad=0.055, shrink=0.78)
    colorbar.set_label(colorbar_label)
    timestamp_title = axis.set_title("", fontsize=14, pad=12)
    fig.suptitle(
        f"SEVIRI Mediterranean domain — {title} — 20°W–40°E, 30–60°N",
        fontsize=15,
        y=0.985,
    )
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.12, top=0.90)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=5500,
        metadata={
            "title": f"SEVIRI Mediterranean domain: {title}",
            "artist": "deep_convection",
            "comment": "16 frames at the native five-minute RSS cadence",
        },
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(output), dpi=dpi):
        for frame, timestamp in enumerate(times):
            image.set_data(values(frame))
            timestamp_title.set_text(
                f"{timestamp:%Y-%m-%d %H:%M} UTC — frame {frame + 1}/{len(times)}"
            )
            writer.grab_frame()
    plt.close(fig)


def main() -> int:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    selected = manifest.loc[manifest.clip_id == args.clip_id]
    if len(selected) != 1:
        raise SystemExit(
            f"Expected exactly one manifest row for {args.clip_id!r}; found {len(selected)}."
        )
    start_time = _utc_naive(selected.iloc[0].start_time)
    output_dir = args.output_dir or (
        ROOT
        / "figures"
        / "videomae_pretraining"
        / f"mediterranean_{args.clip_id}"
    )
    data, x, y, mask, times = load_domain_sequence(args.zarr_root, start_time)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for filename, title, first, second, vmin, vmax in PRODUCTS:
        output = output_dir / f"{filename}.mp4"
        print(f"Rendering {output.name} ...", flush=True)
        render_product(
            data,
            x,
            y,
            mask,
            times,
            output=output,
            title=title,
            first_channel=first,
            second_channel=second,
            vmin=vmin,
            vmax=vmax,
            fps=args.fps,
            dpi=args.dpi,
        )
        rows.append(
            {
                "product": filename,
                "title": title,
                "path": output.name,
                "vmin_k": vmin,
                "vmax_k": vmax,
                "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "frames": NUM_FRAMES,
                "cadence_minutes": CADENCE_MINUTES,
                "lon_min": LON_MIN,
                "lon_max": LON_MAX,
                "lat_min": LAT_MIN,
                "lat_max": LAT_MAX,
            }
        )
    with (output_dir / "video_index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} videos and index to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
