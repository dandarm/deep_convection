#!/usr/bin/env python3
"""Plot GMI and DPR as peer observations against EMMA using continuous IoU."""
from __future__ import annotations

from pathlib import Path
import math
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path
from emma_gpm.emma import emma_files, load_emma_mask, parse_emma_time, read_grid


MASK_COLOR = "#8c62aa"
SENSOR_COLORS = {"DPR": "#d95f02", "GMI": "#1f77b4"}


def save_figure_atomic(fig, destination: Path, *, dpi: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.stem}.part{destination.suffix}")
    fig.savefig(partial, dpi=dpi)
    partial.replace(destination)


def ellipse_bounds(row: pd.Series) -> tuple[float, float, float, float]:
    lat0, lon0 = float(row.r_lat), float(row.r_lon)
    width = float(row.r_major) / (111.320 * max(np.cos(np.deg2rad(lat0)), 0.2))
    height = float(row.r_minor) / 110.574
    angle = np.deg2rad(90.0 - (float(row.r_orientation) if np.isfinite(row.r_orientation) else 0.0))
    # Match the exact axis-aligned bounds of the Ellipse patch as Matplotlib
    # draws it in longitude/latitude coordinates, so it is never clipped.
    lon_radius = np.sqrt(((width / 2.0) * np.cos(angle)) ** 2 + ((height / 2.0) * np.sin(angle)) ** 2)
    lat_radius = np.sqrt(((width / 2.0) * np.sin(angle)) ** 2 + ((height / 2.0) * np.cos(angle)) ** 2)
    return lon0 - lon_radius, lon0 + lon_radius, lat0 - lat_radius, lat0 + lat_radius


def select_low_examples(low: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for sensor in ("DPR", "GMI"):
        for month in range(5, 10):
            subset = low[
                low.sensor.eq(sensor) & low.observation_time.dt.month.eq(month)
            ].sort_values("target_emma_pf_iou")
            if subset.empty:
                continue
            positions = np.unique(
                np.rint(np.linspace(0, len(subset) - 1, min(2, len(subset)))).astype(int)
            )
            selected.append(subset.iloc[positions])
    return pd.concat(selected, ignore_index=True).sort_values(["sensor", "observation_time"])


def prepare(row: pd.Series, objects: pd.DataFrame, path_by_time, grid) -> dict:
    target = objects[
        objects.track_uid.eq(row.track_uid) & objects.time.eq(row.emma_time)
    ].iloc[0]
    mcs = load_emma_mask(path_by_time[row.emma_time], grid)[0]
    ellipse = ellipse_bounds(row)
    pad = 0.6
    extent = (
        min(float(target.lon_min), ellipse[0]) - pad,
        max(float(target.lon_max), ellipse[1]) + pad,
        min(float(target.lat_min), ellipse[2]) - pad,
        max(float(target.lat_max), ellipse[3]) + pad,
    )
    return {"row": row, "target": target, "mcs": mcs, "extent": extent}


def ellipse_patch(row: pd.Series) -> Ellipse:
    lat0, lon0 = float(row.r_lat), float(row.r_lon)
    return Ellipse(
        (lon0, lat0),
        width=float(row.r_major) / (111.320 * max(np.cos(np.deg2rad(lat0)), 0.2)),
        height=float(row.r_minor) / 110.574,
        angle=90.0 - float(row.r_orientation),
        fill=False,
        edgecolor=SENSOR_COLORS[str(row.sensor)],
        linewidth=2.4,
        zorder=5,
    )


def plot_one(ax, prepared: dict, grid, *, compact: bool) -> None:
    row = prepared["row"]
    lon_min, lon_max, lat_min, lat_max = prepared["extent"]
    target_id = int(row.mcs_id)
    ax.pcolormesh(
        grid.pilot_lon,
        grid.pilot_lat,
        np.where(prepared["mcs"] == target_id, 1.0, np.nan),
        cmap=ListedColormap([MASK_COLOR]),
        shading="nearest",
        alpha=0.85,
        rasterized=True,
    )
    ax.add_patch(ellipse_patch(row))
    ax.plot(
        float(row.r_lon), float(row.r_lat), marker="+",
        color=SENSOR_COLORS[str(row.sensor)], markersize=7, markeredgewidth=1.5, zorder=6,
    )
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1 / max(np.cos(np.deg2rad((lat_min + lat_max) / 2.0)), 0.3))
    ax.set_title(
        f"{row.sensor} · IoU={100 * row.target_emma_pf_iou:.2f}%\n"
        f"PF dentro EMMA={100 * row.target_emma_fraction_of_pf_ellipse:.1f}% · "
        f"EMMA coperta={100 * row.target_emma_fraction_covered_by_pf:.1f}%",
        fontsize=8 if compact else 12,
    )
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")
    ax.grid(alpha=0.2)


def handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=MASK_COLOR, markersize=10, label="Maschera dell’MCS EMMA associato"),
        Line2D([0], [0], color=SENSOR_COLORS["DPR"], lw=2.5, label="Ellisse della feature DPR"),
        Line2D([0], [0], color=SENSOR_COLORS["GMI"], lw=2.5, label="Ellisse della feature GMI"),
    ]


def render_group(
    prepared_rows: list[dict],
    group_name: str,
    group_definition: str,
    total_dpr: int,
    total_gmi: int,
    grid,
) -> None:
    legend = handles()
    individual_dir = ROOT / "figures" / "continuous_overlap_examples" / group_name
    for index, prepared in enumerate(prepared_rows, start=1):
        row = prepared["row"]
        fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
        plot_one(ax, prepared, grid, compact=False)
        fig.suptitle(
            f"{group_definition} · esempio {index:02d} · {row.feature_uid}\n"
            f"Ora PF {row.observation_time:%Y-%m-%d %H:%MZ}; ora EMMA {row.emma_time:%Y-%m-%d %H:%MZ}; "
            f"offset {row.time_offset_minutes:+.2f} min",
            fontsize=13,
        )
        fig.legend(handles=legend, loc="outside lower center", ncol=3, frameon=True)
        save_figure_atomic(fig, individual_dir / f"{index:02d}_{row.sensor}_{row.feature_uid.replace(':', '_')}.png", dpi=180)
        plt.close(fig)

    columns = 5
    rows = math.ceil(len(prepared_rows) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(24, 4.4 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, prepared in zip(axes, prepared_rows):
        plot_one(ax, prepared, grid, compact=True)
    for ax in axes[len(prepared_rows):]:
        ax.axis("off")
    fig.legend(handles=legend, loc="outside lower center", ncol=3, frameon=True)
    fig.suptitle(
        f"{group_definition}\n"
        f"Totale nel pilot: DPR={total_dpr:,} feature; GMI={total_gmi:,} feature. "
        "IoU = intersezione / unione fra ellisse PF e maschera EMMA.",
        fontsize=16,
    )
    save_figure_atomic(fig, ROOT / "figures" / f"continuous_overlap_{group_name}.png", dpi=170)
    plt.close(fig)


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    metrics = pd.read_parquet(ROOT / "catalogs/gpm_emma_continuous_overlap_2020.parquet")
    objects = pd.read_parquet(interim / "emma_hourly_objects_2020.parquet")
    paths = emma_files(resolve_project_path(ROOT, config["emma"]["extracted_root"]))
    path_by_time = {parse_emma_time(path): path for path in paths}
    grid = read_grid(paths[0], config["domain"])
    positive = metrics[metrics.target_emma_pf_iou.notna()].copy()

    high = positive[positive.target_emma_pf_iou.ge(0.50)].sort_values(
        ["sensor", "target_emma_pf_iou"], ascending=[True, False]
    )
    low_all = positive[
        positive.target_emma_pf_iou.gt(0) & positive.target_emma_pf_iou.le(0.05)
    ]
    low = select_low_examples(low_all)

    groups = [
        (
            high,
            "high",
            "Concordanza geometrica alta: IoU ≥ 50%",
            int(high.sensor.eq("DPR").sum()),
            int(high.sensor.eq("GMI").sum()),
        ),
        (
            low,
            "low",
            "Concordanza geometrica molto bassa ma positiva: 0% < IoU ≤ 5%",
            int(low_all.sensor.eq("DPR").sum()),
            int(low_all.sensor.eq("GMI").sum()),
        ),
    ]
    selected_rows = []
    for frame, name, definition, total_dpr, total_gmi in groups:
        prepared_rows = [prepare(row, objects, path_by_time, grid) for _, row in frame.iterrows()]
        render_group(prepared_rows, name, definition, total_dpr, total_gmi, grid)
        chosen = frame.copy()
        chosen["display_group"] = name
        selected_rows.append(chosen)
    pd.concat(selected_rows, ignore_index=True).to_csv(
        ROOT / "results/continuous_overlap_figure_examples.csv", index=False
    )
    print(
        f"high IoU: DPR={int(high.sensor.eq('DPR').sum())}, GMI={int(high.sensor.eq('GMI').sum())}"
    )
    print(
        f"low positive IoU: DPR={int(low_all.sensor.eq('DPR').sum())}, "
        f"GMI={int(low_all.sensor.eq('GMI').sum())}; displayed={len(low)}"
    )


if __name__ == "__main__":
    main()
