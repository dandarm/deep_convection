#!/usr/bin/env python3
"""Plot the seasonal OPERA-first screening used to request N1 sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MONTH_NAMES = {5: "mag", 6: "giu", 7: "lug", 8: "ago", 9: "set"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--times", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tiles = pd.read_csv(args.tiles, parse_dates=["central_time"])
    times = pd.read_csv(args.times, parse_dates=["central_time"])
    selected = pd.read_csv(args.manifest, parse_dates=["central_time"])
    selected_times = set(selected.central_time)
    times["selected"] = times.central_time.isin(selected_times)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(
        "Screening OPERA-first — MJJAS 2020",
        fontsize=17,
        weight="bold",
    )

    ax = axes[0, 0]
    ax.scatter(
        times.central_time,
        times.central_time.dt.hour,
        s=20,
        c="#c7c7c7",
        label="tempi OPERA esaminati",
        zorder=1,
    )
    chosen = times[times.selected]
    scatter = ax.scatter(
        chosen.central_time,
        chosen.central_time.dt.hour,
        s=30 + 0.8 * chosen.quiet_tiles,
        c=chosen.quiet_tiles,
        cmap="viridis",
        vmin=1,
        zorder=2,
        edgecolor="black",
        linewidth=0.35,
        label="sequenze SEVIRI selezionate",
    )
    fig.colorbar(scatter, ax=ax, label="tessere radar-quiet (128 km)")
    ax.set(title="A · campionamento", ylabel="ora UTC", ylim=(-0.8, 23.8))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    quiet = tiles[tiles.radar_quiet_candidate].copy()
    spatial = (
        quiet.groupby(["tile_y", "tile_x"], as_index=False)
        .agg(
            center_lon=("center_lon", "median"),
            center_lat=("center_lat", "median"),
            quiet_occurrences=("central_time", "nunique"),
        )
    )
    spatial_scatter = ax.scatter(
        spatial.center_lon,
        spatial.center_lat,
        c=spatial.quiet_occurrences,
        s=12 + 1.2 * spatial.quiet_occurrences,
        cmap="magma_r",
        edgecolor="black",
        linewidth=0.25,
    )
    fig.colorbar(spatial_scatter, ax=ax, label="numero di tempi radar-quiet")
    ax.set(
        title="B · frequenza spaziale",
        xlabel="longitudine",
        ylabel="latitudine",
        xlim=(-21, 41),
        ylim=(29, 71),
        aspect="equal",
    )
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    month_counts = pd.DataFrame(
        {
            "esaminati": times.groupby(times.central_time.dt.month).size(),
            "selezionati": chosen.groupby(chosen.central_time.dt.month).size(),
        }
    ).fillna(0)
    x = np.arange(len(month_counts))
    ax.bar(x - 0.18, month_counts.esaminati, width=0.36, color="#bdbdbd", label="esaminati")
    ax.bar(x + 0.18, month_counts.selezionati, width=0.36, color="#238b45", label="selezionati")
    ax.set_xticks(x, [MONTH_NAMES.get(int(value), str(value)) for value in month_counts.index])
    ax.set(title="C · campionamento mensile", ylabel="numero di tempi")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    observable = times.observable_tiles.to_numpy(float)
    quiet_count = times.quiet_tiles.to_numpy(float)
    ax.scatter(
        observable,
        quiet_count,
        c=np.where(times.selected, "#238b45", "#bdbdbd"),
        s=np.where(times.selected, 35, 18),
        alpha=0.85,
    )
    ax.axhline(1, color="#238b45", linestyle="--", linewidth=1)
    ax.set(
        title="D · osservabilità e quiete",
        xlabel="tessere con copertura/QIND adeguati",
        ylabel="tessere senza eco ≥30 dBZ in 5 slot",
    )
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.97,
        f"153 tempi OPERA\n{len(chosen)} sequenze selezionate\n"
        f"{int(selected.expected_frames.sum())} frame RSS previsti",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#888888"},
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
