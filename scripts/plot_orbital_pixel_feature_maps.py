#!/usr/bin/env python3
"""Plot resampled pixel masks for every selected GMI/DPR orbital scene.

Each scene is shown in two equivalent representations: a binary union of all
precipitation features, and the same pixels separated by connected-component
identity.  Both are nearest-centre resamplings to the EMMA grid and therefore
are explicitly exploratory footprint approximations, not exact beam polygons.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import os
import sys

import cartopy
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path
from emma_gpm.emma import emma_files, load_emma_mask, parse_emma_time, read_grid


EMMA_FILL = "#b79ac8"
EMMA_EDGE = "#59366e"
SENSOR_COLORS = {"GMI": "#2878b5", "DPR": "#e07a27"}
TIME_TOLERANCE = np.timedelta64(30, "m")

_cartopy_root = Path(os.environ.get("EMMA_GPM_DATA_ROOT", ROOT / "data")) / "interim" / "cartopy"
_cartopy_root.mkdir(parents=True, exist_ok=True)
cartopy.config["data_dir"] = str(_cartopy_root)


@lru_cache(maxsize=1)
def coastline_segments() -> tuple[np.ndarray, ...]:
    """Load Natural Earth coastlines once as ordinary lon/lat line segments."""
    path = shpreader.natural_earth(
        resolution="50m", category="physical", name="coastline"
    )
    segments: list[np.ndarray] = []

    def collect(geometry) -> None:
        if geometry.geom_type in {"LineString", "LinearRing"}:
            segments.append(np.asarray(geometry.coords))
        elif hasattr(geometry, "geoms"):
            for child in geometry.geoms:
                collect(child)

    for geometry in shpreader.Reader(path).geometries():
        collect(geometry)
    return tuple(segments)


def draw_coastlines(ax) -> None:
    for segment in coastline_segments():
        ax.plot(
            segment[:, 0], segment[:, 1], color="#202020", linewidth=0.65,
            alpha=0.85, zorder=6,
        )


def save_figure_atomic(fig, destination: Path, *, dpi: int = 180) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.stem}.part{destination.suffix}")
    fig.savefig(partial, dpi=dpi, facecolor="white")
    partial.replace(destination)


def nearest_index(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    right = np.searchsorted(grid, values, side="left")
    right = np.clip(right, 0, len(grid) - 1)
    left = np.clip(right - 1, 0, len(grid) - 1)
    choose_left = np.abs(values - grid[left]) <= np.abs(values - grid[right])
    return np.where(choose_left, left, right)


def component_palette(count: int) -> ListedColormap:
    if count == 0:
        return ListedColormap(["#cccccc"])
    # Golden-ratio hue stepping makes adjacent component IDs visually distinct.
    hues = (0.11 + np.arange(count) * 0.61803398875) % 1.0
    hsv = np.column_stack([hues, np.full(count, 0.70), np.full(count, 0.88)])
    return ListedColormap(plt.cm.hsv(hsv[:, 0]))


def rasterize_components(
    frame: pd.DataFrame,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    shape = (len(lat), len(lon))
    if frame.empty:
        return np.zeros(shape, dtype=bool), np.full(shape, np.nan), []
    work = frame[["lat", "lon", "component_uid"]].copy()
    work["iy"] = nearest_index(work.lat.to_numpy(), lat)
    work["ix"] = nearest_index(work.lon.to_numpy(), lon)
    # If multiple native components land in one EMMA cell, retain the component
    # contributing the largest number of native pixel centres to that cell.
    counts = (
        work.groupby(["iy", "ix", "component_uid"], observed=True)
        .size()
        .rename("native_pixels")
        .reset_index()
        .sort_values("native_pixels", ascending=False)
        .drop_duplicates(["iy", "ix"])
    )
    identifiers = sorted(counts.component_uid.unique())
    code = {identifier: index for index, identifier in enumerate(identifiers)}
    component = np.full(shape, np.nan)
    component[
        counts.iy.to_numpy(dtype=int), counts.ix.to_numpy(dtype=int)
    ] = counts.component_uid.map(code).to_numpy(dtype=float)
    return np.isfinite(component), component, identifiers


def plot_base(
    ax,
    emma: np.ndarray,
    grid,
    domain: dict[str, float],
    *,
    emma_fill: str = EMMA_FILL,
    emma_edge: str = EMMA_EDGE,
    show_emma: bool = True,
) -> None:
    emma_union = emma > 0
    if show_emma:
        ax.pcolormesh(
            grid.pilot_lon,
            grid.pilot_lat,
            np.where(emma_union, 1.0, np.nan),
            cmap=ListedColormap([emma_fill]),
            shading="nearest",
            alpha=0.34,
            rasterized=True,
            zorder=1,
        )
        if emma_union.any():
            ax.contour(
                grid.pilot_lon,
                grid.pilot_lat,
                emma_union.astype(float),
                levels=[0.5],
                colors=[emma_edge],
                linewidths=1.0,
                zorder=4,
            )
    ax.set_xlim(domain["lon_min"], domain["lon_max"])
    ax.set_ylim(domain["lat_min"], domain["lat_max"])
    mean_lat = (domain["lat_min"] + domain["lat_max"]) / 2.0
    ax.set_aspect(1 / np.cos(np.deg2rad(mean_lat)))
    ax.grid(color="#777777", alpha=0.18, linewidth=0.6)
    draw_coastlines(ax)
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")


def plot_union(ax, union: np.ndarray, sensor: str, grid) -> None:
    ax.pcolormesh(
        grid.pilot_lon,
        grid.pilot_lat,
        np.where(union, 1.0, np.nan),
        cmap=ListedColormap([SENSOR_COLORS[sensor]]),
        shading="nearest",
        alpha=0.78,
        rasterized=True,
        zorder=3,
    )


def plot_components(ax, components: np.ndarray, count: int, grid) -> None:
    ax.pcolormesh(
        grid.pilot_lon,
        grid.pilot_lat,
        components,
        cmap=component_palette(count),
        vmin=-0.5,
        vmax=max(count - 0.5, 0.5),
        shading="nearest",
        alpha=0.80,
        rasterized=True,
        zorder=3,
    )


def prepare_scene(
    orbit: int,
    emma_time: pd.Timestamp,
    pixels: pd.DataFrame,
    emma: np.ndarray,
    grid,
) -> tuple[dict[str, dict], list[dict]]:
    prepared: dict[str, dict] = {}
    rows: list[dict] = []
    emma_union = emma > 0
    for sensor in ("GMI", "DPR"):
        sensor_pixels = pixels[
            pixels.orbit.eq(orbit)
            & pixels.sensor.eq(sensor)
            & (pixels.observation_time.sub(emma_time).abs() <= TIME_TOLERANCE)
        ].copy()
        union, components, identifiers = rasterize_components(
            sensor_pixels, grid.pilot_lat, grid.pilot_lon
        )
        intersection = union & emma_union
        combined = union | emma_union
        prepared[sensor] = {
            "pixels": sensor_pixels,
            "union": union,
            "components": components,
            "identifiers": identifiers,
        }
        rows.append(
            {
                "orbit": orbit,
                "emma_time": emma_time,
                "sensor": sensor,
                "native_precipitating_pixel_centres": len(sensor_pixels),
                "native_connected_features_in_window": int(sensor_pixels.component_uid.nunique()),
                "connected_features_represented_after_resampling": len(identifiers),
                "resampled_gpm_cells": int(union.sum()),
                "emma_mcs_cells": int(emma_union.sum()),
                "intersection_cells": int(intersection.sum()),
                "union_cells": int(combined.sum()),
                "gpm_fraction_inside_emma": (
                    float(intersection.sum() / union.sum()) if union.any() else np.nan
                ),
                "emma_fraction_covered_by_gpm": (
                    float(intersection.sum() / emma_union.sum()) if emma_union.any() else np.nan
                ),
                "pixel_mask_iou": (
                    float(intersection.sum() / combined.sum()) if combined.any() else np.nan
                ),
                "native_mask_rule": (
                    "PRECIP>0.1 mm h-1" if sensor == "GMI" else "NSPRECIP>0.1 mm h-1"
                ),
                "connectivity": "8-neighbour on full native swath",
                "resampling": "nearest native pixel centre to EMMA grid cell; majority component on collisions",
            }
        )
    return prepared, rows


def render_scene(
    orbit: int,
    emma_time: pd.Timestamp,
    prepared: dict[str, dict],
    emma: np.ndarray,
    grid,
    domain: dict[str, float],
    destination: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True, sharex=True, sharey=True)
    for row_index, sensor in enumerate(("GMI", "DPR")):
        data = prepared[sensor]
        for column_index in range(2):
            ax = axes[row_index, column_index]
            plot_base(ax, emma, grid, domain)
            if column_index == 0:
                plot_union(ax, data["union"], sensor, grid)
                ax.set_title(
                    f"{sensor}: unione di tutte le feature\n"
                    f"{int(data['union'].sum()):,} celle ricampionate",
                    fontsize=12,
                )
            else:
                plot_components(ax, data["components"], len(data["identifiers"]), grid)
                ax.set_title(
                    f"{sensor}: oggetti PF connessi distinti\n"
                    f"{len(data['identifiers']):,} oggetti; colori identificativi, non fisici",
                    fontsize=12,
                )
    handles = [
        Patch(facecolor=EMMA_FILL, edgecolor=EMMA_EDGE, label="Unione delle maschere MCS EMMA all’ora indicata"),
        Patch(facecolor=SENSOR_COLORS["GMI"], label="Unione dei pixel precipitanti GMI"),
        Patch(facecolor=SENSOR_COLORS["DPR"], label="Unione dei pixel precipitanti DPR"),
        Patch(facecolor="#47a35a", label="Pannelli destri: ogni colore identifica un oggetto PF, non intensità o classe"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=True, fontsize=10)
    fig.suptitle(
        f"Maschere pixel-level ricampionate · orbita {orbit:06d} · EMMA {emma_time:%Y-%m-%d %H:%MZ}\n"
        "GMI: PRECIP > 0,1 mm/h · DPR: NSPRECIP > 0,1 mm/h · finestra rispetto a EMMA: ±30 min",
        fontsize=15,
    )
    save_figure_atomic(fig, destination)
    plt.close(fig)


def render_overview(
    scenes: list[dict],
    representation: str,
    grid,
    domain: dict[str, float],
    destination: Path,
) -> None:
    fig, axes = plt.subplots(
        len(scenes), 2, figsize=(17, 4.2 * len(scenes)), constrained_layout=True, sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    for row_index, scene in enumerate(scenes):
        for column_index, sensor in enumerate(("GMI", "DPR")):
            ax = axes[row_index, column_index]
            data = scene["prepared"][sensor]
            plot_base(ax, scene["emma"], grid, domain)
            if representation == "union":
                plot_union(ax, data["union"], sensor, grid)
                descriptor = f"{int(data['union'].sum()):,} celle"
            else:
                plot_components(ax, data["components"], len(data["identifiers"]), grid)
                descriptor = f"{len(data['identifiers']):,} feature"
            ax.set_title(
                f"{scene['emma_time']:%Y-%m-%d %H:%MZ} · orbita {scene['orbit']:06d} · {sensor} · {descriptor}",
                fontsize=9,
            )
    title = (
        "Unione binaria di tutte le precipitation features"
        if representation == "union"
        else "Le stesse maschere separate in oggetti PF connessi: colori identificativi, non variabili fisiche"
    )
    fig.suptitle(
        title + "\nPixel nativi ricampionati sulla griglia EMMA; lilla = MCS EMMA; stesso dominio in ogni pannello",
        fontsize=15,
    )
    save_figure_atomic(fig, destination, dpi=160)
    plt.close(fig)


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    pixels = pd.read_parquet(interim / "orbital_pixel_feature_masks_2020.parquet")
    pixels["observation_time"] = pd.to_datetime(pixels.observation_time, utc=True)
    manifest = pd.read_csv(ROOT / "catalogs/visual_case_manifest_2020.csv")
    manifest["emma_time"] = pd.to_datetime(manifest.emma_time, utc=True)
    scene_rows = (
        manifest[["orbit", "emma_time"]]
        .dropna()
        .drop_duplicates()
        .assign(orbit=lambda table: table.orbit.astype(int))
        .sort_values("emma_time")
    )

    paths = emma_files(resolve_project_path(ROOT, config["emma"]["extracted_root"]))
    path_by_time = {parse_emma_time(path): path for path in paths}
    grid = read_grid(paths[0], config["domain"])
    scenes: list[dict] = []
    summary_rows: list[dict] = []
    output_dir = ROOT / "figures" / "pixel_feature_maps"
    for row in scene_rows.itertuples(index=False):
        emma, _ = load_emma_mask(path_by_time[row.emma_time], grid)
        prepared, metrics = prepare_scene(row.orbit, row.emma_time, pixels, emma, grid)
        summary_rows.extend(metrics)
        scene = {"orbit": row.orbit, "emma_time": row.emma_time, "emma": emma, "prepared": prepared}
        scenes.append(scene)
        destination = output_dir / f"{row.emma_time:%Y%m%dT%H}_{row.orbit:06d}_pixel_masks.png"
        render_scene(row.orbit, row.emma_time, prepared, emma, grid, config["domain"], destination)
        print(f"wrote {destination}", flush=True)

    render_overview(
        scenes,
        "union",
        grid,
        config["domain"],
        ROOT / "figures" / "pixel_feature_union_overview.png",
    )
    render_overview(
        scenes,
        "components",
        grid,
        config["domain"],
        ROOT / "figures" / "pixel_feature_components_overview.png",
    )
    pd.DataFrame(summary_rows).to_csv(ROOT / "results/pixel_feature_map_summary.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
