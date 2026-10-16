#!/usr/bin/env python3
"""Render an explanatory animation of adaptive RDT cell detection.

The animation uses one real SEVIRI IR 10.8 frame over the Mediterranean.  It
visualizes regional minima, marker-controlled thermal flooding, and the
cell-specific saddle temperature at which each basin stops growing.  It is a
scientific illustration of the planned detector, not a validated RDT product.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_seviri_mediterranean_videos import (  # noqa: E402
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    _extent,
    _projection,
    _utc_naive,
    load_domain_sequence,
)


IR108_INDEX = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path(
            os.environ.get(
                "SEVIRI_ZARR_ROOT",
                r"E:\Datasets\MSG-SEVIRI",
            )
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "catalogs" / "seviri_pretraining_random_100.csv",
    )
    parser.add_argument("--clip-id", default="random_000")
    parser.add_argument("--frame-index", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "figures"
        / "videomae_pretraining"
        / "rdt_step1_adaptive_detection_mediterranean.mp4",
    )
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=125)
    parser.add_argument("--max-seeds", type=int, default=12)
    return parser.parse_args()


def _select_regional_minima(
    temperature: np.ndarray,
    valid: np.ndarray,
    *,
    max_seeds: int,
    filter_size: int = 17,
    minimum_distance: int = 55,
) -> list[tuple[int, int]]:
    """Select deep, spatially separated regional minima for visualization."""

    filled = np.where(valid, temperature, np.inf)
    local_minimum = filled == ndimage.minimum_filter(
        filled, size=filter_size, mode="nearest"
    )
    cold_limit = float(np.nanquantile(temperature[valid], 0.025))
    candidates = np.argwhere(local_minimum & valid & (temperature <= cold_limit))
    order = np.argsort(temperature[candidates[:, 0], candidates[:, 1]])
    selected: list[tuple[int, int]] = []
    for index in order:
        row, column = (int(value) for value in candidates[index])
        if all(
            (row - other_row) ** 2 + (column - other_column) ** 2
            >= minimum_distance**2
            for other_row, other_column in selected
        ):
            selected.append((row, column))
        if len(selected) >= max_seeds:
            break
    if len(selected) < 2:
        raise RuntimeError("Could not identify enough separated thermal minima.")
    return selected


def _watershed_labels(
    temperature: np.ndarray,
    valid: np.ndarray,
    seeds: list[tuple[int, int]],
) -> np.ndarray:
    """Return marker-controlled watershed labels on the thermal surface."""

    low, high = np.nanquantile(temperature[valid], [0.001, 0.995])
    scaled = np.clip((temperature - low) / max(high - low, 1.0), 0.0, 1.0)
    topography = np.asarray(np.rint(scaled * 254.0), dtype=np.uint8)
    topography[~valid] = 255
    markers = np.zeros(temperature.shape, dtype=np.int32)
    for label, (row, column) in enumerate(seeds, start=1):
        markers[row, column] = label
    labels = ndimage.watershed_ift(topography, markers)
    labels[~valid] = 0
    return labels


def _cell_saddle_temperatures(
    temperature: np.ndarray,
    labels: np.ndarray,
    seeds: list[tuple[int, int]],
) -> np.ndarray:
    """Estimate the first merge saddle for every marker-controlled basin."""

    count = len(seeds)
    saddles = np.full(count + 1, np.inf, dtype=np.float32)
    for row_shift, column_shift in ((1, 0), (0, 1)):
        first = labels[
            : labels.shape[0] - row_shift or None,
            : labels.shape[1] - column_shift or None,
        ]
        second = labels[row_shift:, column_shift:]
        first_t = temperature[
            : temperature.shape[0] - row_shift or None,
            : temperature.shape[1] - column_shift or None,
        ]
        second_t = temperature[row_shift:, column_shift:]
        boundary = (first > 0) & (second > 0) & (first != second)
        if not np.any(boundary):
            continue
        left_labels = first[boundary]
        right_labels = second[boundary]
        contact_temperature = np.maximum(first_t[boundary], second_t[boundary])
        np.minimum.at(saddles, left_labels, contact_temperature)
        np.minimum.at(saddles, right_labels, contact_temperature)

    fallback = float(np.nanquantile(temperature[labels > 0], 0.18))
    for label, (row, column) in enumerate(seeds, start=1):
        if not np.isfinite(saddles[label]):
            saddles[label] = fallback
        saddles[label] = max(saddles[label], float(temperature[row, column]))
    return saddles


def _frozen_cells(
    temperature: np.ndarray,
    labels: np.ndarray,
    saddles: np.ndarray,
    flood_temperature: float,
    *,
    minimum_pixels: int,
) -> np.ndarray:
    output = np.zeros(labels.shape, dtype=np.int32)
    for label in range(1, len(saddles)):
        limit = min(float(flood_temperature), float(saddles[label]))
        cell = (labels == label) & (temperature <= limit)
        if np.count_nonzero(cell) >= minimum_pixels:
            output[cell] = label
    return output


def _boundary(mask: np.ndarray) -> np.ndarray:
    inside = mask > 0
    eroded = ndimage.binary_erosion(inside, structure=np.ones((3, 3)))
    return inside & ~eroded


def render_animation(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    selected = manifest.loc[manifest.clip_id == args.clip_id]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one row for clip {args.clip_id!r}; found {len(selected)}."
        )
    start_time = _utc_naive(selected.iloc[0].start_time)
    data, x, y, geographic_mask, times = load_domain_sequence(
        args.zarr_root, start_time
    )
    if not 0 <= args.frame_index < len(times):
        raise ValueError(f"frame-index must be between 0 and {len(times) - 1}")

    raw = np.where(
        geographic_mask,
        data[args.frame_index, IR108_INDEX],
        np.nan,
    )
    valid = geographic_mask & np.isfinite(raw)
    smoothed = ndimage.gaussian_filter(
        np.where(valid, raw, np.nanmedian(raw[valid])), sigma=1.0
    )
    seeds = _select_regional_minima(
        smoothed, valid, max_seeds=args.max_seeds
    )
    labels = _watershed_labels(smoothed, valid, seeds)
    saddles = _cell_saddle_temperatures(smoothed, labels, seeds)

    pixel_area_km2 = abs(float(np.median(np.diff(x)) * np.median(np.diff(y)))) / 1e6
    minimum_pixels = max(1, int(np.ceil(100.0 / pixel_area_km2)))
    seed_temperatures = np.asarray([smoothed[row, column] for row, column in seeds])
    flood_start = float(np.min(seed_temperatures))
    flood_end = float(np.nanmax(saddles[1:]))
    flood_levels = np.linspace(flood_start, flood_end, 42)

    palette = ListedColormap(
        [
            "#e63946",
            "#ff9f1c",
            "#ffd166",
            "#06d6a0",
            "#00b4d8",
            "#4361ee",
            "#7209b7",
            "#f72585",
            "#8ac926",
            "#1982c4",
            "#6a4c93",
            "#ff595e",
        ][: len(seeds)]
    )
    geos = _projection()
    plate = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12.8, 7.6), facecolor="#0b1020")
    axis = fig.add_subplot(1, 1, 1, projection=geos)
    axis.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=plate)
    axis.set_facecolor("#dbe6ec")
    base = axis.imshow(
        raw,
        origin="lower",
        extent=_extent(x, y),
        transform=geos,
        cmap="turbo_r",
        vmin=190.0,
        vmax=300.0,
        interpolation="nearest",
    )
    overlay = axis.imshow(
        np.ma.masked_all(raw.shape),
        origin="lower",
        extent=_extent(x, y),
        transform=geos,
        cmap=palette,
        vmin=1,
        vmax=max(len(seeds), 2),
        interpolation="nearest",
        alpha=0.52,
    )
    outline = axis.imshow(
        np.ma.masked_all(raw.shape),
        origin="lower",
        extent=_extent(x, y),
        transform=geos,
        cmap=ListedColormap(["white"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        alpha=0.95,
    )
    seed_artist = axis.scatter(
        [],
        [],
        transform=geos,
        marker="x",
        s=55,
        linewidths=1.8,
        color="white",
        zorder=5,
    )
    axis.coastlines(resolution="50m", color="#111827", linewidth=0.8, zorder=6)
    axis.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.35,
        edgecolor="#374151",
        zorder=6,
    )
    gridlines = axis.gridlines(
        crs=plate,
        draw_labels=True,
        linewidth=0.3,
        color="#d1d5db",
        alpha=0.65,
        linestyle=":",
        x_inline=False,
        y_inline=False,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    colorbar = fig.colorbar(
        base, ax=axis, orientation="horizontal", pad=0.055, shrink=0.77
    )
    colorbar.set_label("Brightness temperature IR 10.8 [K]", color="white")
    colorbar.ax.tick_params(colors="white")
    title = axis.set_title("", fontsize=15, pad=12, color="white")
    subtitle = fig.text(
        0.5,
        0.955,
        "",
        ha="center",
        va="top",
        fontsize=12,
        color="#dbeafe",
    )
    fig.suptitle(
        "Step 1 · Rilevamento adattivo delle celle in IR 10.8",
        fontsize=18,
        y=0.995,
        color="white",
    )
    note = fig.text(
        0.99,
        0.018,
        "Prototipo visuale RDT-CW-inspired · non ancora validato",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#94a3b8",
    )
    note.set_in_layout(False)
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.115, top=0.89)

    seed_offsets = np.asarray([(x[column], y[row]) for row, column in seeds])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=6000,
        metadata={
            "title": "RDT step 1 adaptive thermal cell detection",
            "artist": "deep_convection",
            "comment": "Scientific illustration using real SEVIRI IR 10.8 data",
        },
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    empty = np.ma.masked_all(raw.shape)
    final_cells = _frozen_cells(
        smoothed,
        labels,
        saddles,
        flood_end,
        minimum_pixels=minimum_pixels,
    )
    with writer.saving(fig, str(args.output), dpi=args.dpi):
        for _ in range(18):
            overlay.set_data(empty)
            outline.set_data(empty)
            seed_artist.set_offsets(np.empty((0, 2)))
            title.set_text(f"SEVIRI · {times[args.frame_index]:%Y-%m-%d %H:%M} UTC")
            subtitle.set_text("Campo osservato sull'intero bacino mediterraneo")
            writer.grab_frame()

        for _ in range(20):
            seed_artist.set_offsets(seed_offsets)
            subtitle.set_text(
                "Minimi regionali freddi e spazialmente separati · marker delle torri"
            )
            writer.grab_frame()

        for flood_temperature in flood_levels:
            cells = _frozen_cells(
                smoothed,
                labels,
                saddles,
                float(flood_temperature),
                minimum_pixels=minimum_pixels,
            )
            overlay.set_data(np.ma.masked_where(cells == 0, cells))
            subtitle.set_text(
                "Flooding verso temperature più calde · ogni colore segue un minimo "
                f"· livello {flood_temperature:.1f} K"
            )
            writer.grab_frame()

        final_boundary = _boundary(final_cells)
        outline.set_data(np.ma.masked_where(~final_boundary, final_boundary.astype(float)))
        for _ in range(28):
            subtitle.set_text(
                "T_edge specifica per cella · crescita fermata al primo saddle "
                "· componenti < 100 km² escluse"
            )
            writer.grab_frame()
    plt.close(fig)


def main() -> int:
    args = parse_args()
    render_animation(args)
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
