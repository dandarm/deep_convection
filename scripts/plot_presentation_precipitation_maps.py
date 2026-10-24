#!/usr/bin/env python3
"""Create individual presentation maps of continuous GMI and DPR retrievals."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from emma_gpm.config import load_config, resolve_project_path
from emma_gpm.emma import emma_files, load_emma_mask, parse_emma_time, read_grid
from plot_orbital_pixel_feature_maps import (
    TIME_TOLERANCE,
    plot_base,
    save_figure_atomic,
)
from plot_pixel_physical_composition import raster_max


PRESENTATION_EMMA_FILL = "#ef5b5b"
PRESENTATION_EMMA_EDGE = "#8b0000"


def scene_pixels(
    pixels: pd.DataFrame, orbit: int, emma_time: pd.Timestamp, sensor: str
) -> pd.DataFrame:
    return pixels[
        pixels.orbit.eq(orbit)
        & pixels.sensor.eq(sensor)
        & (pixels.observation_time.sub(emma_time).abs() <= TIME_TOLERANCE)
    ]


def plot_rate(ax, raster: np.ndarray, emma: np.ndarray, grid, domain, norm, title: str):
    plot_base(
        ax,
        emma,
        grid,
        domain,
        emma_fill=PRESENTATION_EMMA_FILL,
        emma_edge=PRESENTATION_EMMA_EDGE,
        show_emma=False,
    )
    mesh = ax.pcolormesh(
        grid.pilot_lon,
        grid.pilot_lat,
        np.ma.masked_invalid(raster),
        cmap="viridis",
        norm=norm,
        shading="nearest",
        rasterized=True,
        zorder=3,
    )
    emma_union = emma > 0
    ax.pcolormesh(
        grid.pilot_lon,
        grid.pilot_lat,
        np.where(emma_union, 1.0, np.nan),
        cmap=ListedColormap([PRESENTATION_EMMA_FILL]),
        shading="nearest",
        alpha=0.32,
        rasterized=True,
        zorder=7,
    )
    if emma_union.any():
        ax.contour(
            grid.pilot_lon,
            grid.pilot_lat,
            emma_union.astype(float),
            levels=[0.5],
            colors=[PRESENTATION_EMMA_EDGE],
            linewidths=1.8,
            zorder=8,
        )
    ax.set_title(title)
    return mesh


def render_rate_scene(scene: dict, grid, domain, norm, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.4), constrained_layout=True, sharex=True, sharey=True)
    mesh = plot_rate(
        axes[0], scene["GMI"], scene["emma"], grid, domain, norm,
        "Precipitation-rate retrieval GMI\nGMI PRECIP [mm/h]",
    )
    plot_rate(
        axes[1], scene["DPR"], scene["emma"], grid, domain, norm,
        "Near-surface precipitation-rate retrieval DPR\nDPR NSPRECIP [mm/h]",
    )
    fig.colorbar(
        mesh, ax=axes, label="Rate di precipitazione [mm/h]",
        extend="neither", shrink=0.82,
    )
    fig.legend(
        handles=[
            Patch(
                facecolor=PRESENTATION_EMMA_FILL,
                edgecolor=PRESENTATION_EMMA_EDGE,
                label="Maschere MCS EMMA",
            )
        ],
        loc="outside lower center", frameon=True,
    )
    fig.suptitle(
        f"Campi pixel-level di precipitazione · orbita {scene['orbit']:06d} · "
        f"EMMA {scene['emma_time']:%Y-%m-%d %H:%MZ}",
        fontsize=15,
    )
    save_figure_atomic(fig, destination)
    plt.close(fig)


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    pixels = pd.read_parquet(interim / "orbital_pixel_feature_masks_2020.parquet")
    pixels["observation_time"] = pd.to_datetime(pixels.observation_time, utc=True)
    manifest_path = ROOT / "catalogs" / "presentation_scene_manifest_2020.csv"
    if not manifest_path.exists():
        manifest_path = ROOT / "catalogs" / "visual_case_manifest_2020.csv"
    manifest = pd.read_csv(manifest_path)
    manifest["emma_time"] = pd.to_datetime(manifest.emma_time, utc=True)
    selected = (
        manifest[["orbit", "emma_time"]].dropna().drop_duplicates()
        .assign(orbit=lambda table: table.orbit.astype(int)).sort_values("emma_time")
    )
    paths = emma_files(resolve_project_path(ROOT, config["emma"]["extracted_root"]))
    path_by_time = {parse_emma_time(path): path for path in paths}
    grid = read_grid(paths[0], config["domain"])

    # Use the actual full selected-data range rather than a presentation-specific
    # clipping percentile.  Both sensors therefore share exactly the same norm.
    rate_max = float(pixels.precipitation_rate_mm_h.max())
    norm = LogNorm(vmin=0.1, vmax=rate_max)
    scenes: list[dict] = []
    figure_rows: list[dict] = []
    rate_dir = ROOT / "figures" / "presentation_precipitation_rate"
    for row in selected.itertuples(index=False):
        emma, _ = load_emma_mask(path_by_time[row.emma_time], grid)
        gmi = scene_pixels(pixels, row.orbit, row.emma_time, "GMI")
        dpr = scene_pixels(pixels, row.orbit, row.emma_time, "DPR")
        scene = {
            "orbit": row.orbit,
            "emma_time": row.emma_time,
            "emma": emma,
            "GMI": raster_max(gmi, "precipitation_rate_mm_h", grid),
            "DPR": raster_max(dpr, "precipitation_rate_mm_h", grid),
        }
        scenes.append(scene)
        stem = f"{row.emma_time:%Y%m%dT%H}_{row.orbit:06d}"
        destination = rate_dir / f"{stem}_rate.png"
        render_rate_scene(scene, grid, config["domain"], norm, destination)
        figure_rows.append(
            {
                "orbit": row.orbit,
                "emma_time": row.emma_time.isoformat(),
                "figure_path": str(destination.resolve()),
            }
        )
        print(f"wrote presentation maps for {stem}", flush=True)

    pd.DataFrame(
        [{"shared_rate_vmin_mm_h": 0.1, "shared_rate_vmax_mm_h": rate_max, "scenes": len(scenes)}]
    ).to_csv(ROOT / "results/presentation_precipitation_map_settings.csv", index=False)
    pd.DataFrame(figure_rows).to_csv(
        ROOT / "results/presentation_precipitation_figure_manifest.csv", index=False
    )


if __name__ == "__main__":
    main()
