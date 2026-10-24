#!/usr/bin/env python3
"""Create presentation maps of mask union and sensor-specific physical content."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from emma_gpm.config import load_config, resolve_project_path
from emma_gpm.emma import emma_files, load_emma_mask, parse_emma_time, read_grid
from plot_orbital_pixel_feature_maps import (
    EMMA_EDGE,
    EMMA_FILL,
    SENSOR_COLORS,
    TIME_TOLERANCE,
    nearest_index,
    plot_base,
    plot_union,
    prepare_scene,
    save_figure_atomic,
)


DPR_COLORS = {
    "stratiform": "#3478b8",
    "other": "#e4bd3e",
    "convective": "#d84a45",
}


def raster_max(frame: pd.DataFrame, value: str, grid) -> np.ndarray:
    raster = np.full((len(grid.pilot_lat), len(grid.pilot_lon)), np.nan)
    if frame.empty:
        return raster
    work = frame[["lat", "lon", value]].copy()
    work["iy"] = nearest_index(work.lat.to_numpy(), grid.pilot_lat)
    work["ix"] = nearest_index(work.lon.to_numpy(), grid.pilot_lon)
    maximum = work.groupby(["iy", "ix"], observed=True)[value].max().reset_index()
    raster[maximum.iy.to_numpy(dtype=int), maximum.ix.to_numpy(dtype=int)] = maximum[value]
    return raster


def raster_dpr_class(frame: pd.DataFrame, grid) -> np.ndarray:
    raster = np.full((len(grid.pilot_lat), len(grid.pilot_lon)), np.nan)
    if frame.empty:
        return raster
    work = frame[frame.near_surface_precip.gt(0.1)][
        ["lat", "lon", "type_precip_label"]
    ].copy()
    work["iy"] = nearest_index(work.lat.to_numpy(), grid.pilot_lat)
    work["ix"] = nearest_index(work.lon.to_numpy(), grid.pilot_lon)
    # Convective takes display priority over other, which takes priority over
    # stratiform, only when several native centres collide in one EMMA cell.
    priority = {"stratiform": 1, "other": 2, "no_precip_or_missing": 2, "convective": 3}
    work["code"] = work.type_precip_label.map(priority).fillna(2).astype(int)
    selected = work.groupby(["iy", "ix"], observed=True).code.max().reset_index()
    raster[selected.iy.to_numpy(dtype=int), selected.ix.to_numpy(dtype=int)] = selected.code
    return raster


def render_scene(
    scene: dict,
    raw_dpr: pd.DataFrame,
    grid,
    domain: dict[str, float],
    destination: Path,
) -> None:
    orbit = scene["orbit"]
    emma_time = scene["emma_time"]
    prepared = scene["prepared"]
    emma = scene["emma"]
    gmi_rate = raster_max(
        prepared["GMI"]["pixels"], "precipitation_rate_mm_h", grid
    )
    dpr_here = raw_dpr[
        raw_dpr.orbit.eq(orbit)
        & (raw_dpr.observation_time.sub(emma_time).abs() <= TIME_TOLERANCE)
    ]
    dpr_class = raster_dpr_class(dpr_here, grid)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True, sharex=True, sharey=True)
    for row_index, sensor in enumerate(("GMI", "DPR")):
        for column_index in range(2):
            plot_base(axes[row_index, column_index], emma, grid, domain)
        plot_union(axes[row_index, 0], prepared[sensor]["union"], sensor, grid)
        axes[row_index, 0].set_title(
            f"{sensor}: area-unione dei pixel precipitanti\n"
            f"{int(prepared[sensor]['union'].sum()):,} celle sulla griglia EMMA"
        )

    gmi_mesh = axes[0, 1].pcolormesh(
        grid.pilot_lon,
        grid.pilot_lat,
        np.ma.masked_invalid(gmi_rate),
        cmap="viridis",
        norm=LogNorm(vmin=0.1, vmax=50),
        shading="nearest",
        rasterized=True,
        zorder=3,
    )
    axes[0, 1].set_title("GMI: la stessa area, colorata per PRECIP\nrate recuperato [mm/h], scala logaritmica")
    fig.colorbar(gmi_mesh, ax=axes[0, 1], label="GMI PRECIP [mm/h]", extend="max", shrink=0.82)

    class_cmap = ListedColormap(
        [DPR_COLORS["stratiform"], DPR_COLORS["other"], DPR_COLORS["convective"]]
    )
    axes[1, 1].pcolormesh(
        grid.pilot_lon,
        grid.pilot_lat,
        np.ma.masked_invalid(dpr_class),
        cmap=class_cmap,
        norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5], class_cmap.N),
        shading="nearest",
        rasterized=True,
        zorder=3,
    )
    axes[1, 1].set_title("DPR: la stessa area, colorata per tipo di precipitazione\nblu=stratiforme · rosso=convettiva · giallo=altra/non classificata")

    handles = [
        Patch(facecolor=EMMA_FILL, edgecolor=EMMA_EDGE, label="Maschere MCS EMMA all’ora indicata"),
        Patch(facecolor=SENSOR_COLORS["GMI"], label="Unione binaria dei pixel GMI"),
        Patch(facecolor=SENSOR_COLORS["DPR"], label="Unione binaria dei pixel DPR"),
        Patch(facecolor=DPR_COLORS["stratiform"], label="DPR stratiforme: eco più esteso e meno intenso"),
        Patch(facecolor=DPR_COLORS["convective"], label="DPR convettiva: colonne con moti verticali intensi"),
        Patch(facecolor=DPR_COLORS["other"], label="DPR altra o non classificata nelle due categorie principali"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=True, fontsize=9)
    fig.suptitle(
        f"Area precipitante e contenuto fisico · orbita {orbit:06d} · EMMA {emma_time:%Y-%m-%d %H:%MZ}\n"
        "Le ellissi PF non sono usate; i pixel nativi sono ricampionati sulla griglia EMMA",
        fontsize=15,
    )
    save_figure_atomic(fig, destination)
    plt.close(fig)


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    pixels = pd.read_parquet(interim / "orbital_pixel_feature_masks_2020.parquet")
    pixels["observation_time"] = pd.to_datetime(pixels.observation_time, utc=True)
    raw_dpr = pd.read_parquet(interim / "raw_dpr_visual_case_pixels.parquet")
    raw_dpr["observation_time"] = pd.to_datetime(raw_dpr.observation_time, utc=True)
    manifest = pd.read_csv(ROOT / "catalogs/visual_case_manifest_2020.csv")
    manifest["emma_time"] = pd.to_datetime(manifest.emma_time, utc=True)
    selected = (
        manifest[["orbit", "emma_time"]]
        .dropna()
        .drop_duplicates()
        .assign(orbit=lambda table: table.orbit.astype(int))
        .sort_values("emma_time")
    )
    paths = emma_files(resolve_project_path(ROOT, config["emma"]["extracted_root"]))
    path_by_time = {parse_emma_time(path): path for path in paths}
    grid = read_grid(paths[0], config["domain"])
    destination_dir = ROOT / "figures" / "pixel_physical_composition"

    for row in selected.itertuples(index=False):
        emma, _ = load_emma_mask(path_by_time[row.emma_time], grid)
        prepared, _ = prepare_scene(row.orbit, row.emma_time, pixels, emma, grid)
        scene = {"orbit": row.orbit, "emma_time": row.emma_time, "emma": emma, "prepared": prepared}
        destination = destination_dir / f"{row.emma_time:%Y%m%dT%H}_{row.orbit:06d}_physical.png"
        render_scene(scene, raw_dpr, grid, config["domain"], destination)
        print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
