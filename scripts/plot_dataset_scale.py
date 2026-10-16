#!/usr/bin/env python3
"""Plot the temporal extent and positive-sample capacity of EMMA/SEVIRI/OPERA."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.inventory)

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.0), constrained_layout=True)
    ax = axes[0]
    colors = ["#bdbdbd" if year < 2008 else "#5aa9e6" if year < 2013 else "#2a9d8f"
              for year in data.year]
    ax.bar(data.year, data.object_hours, color=colors, width=0.82)
    ax.set_ylabel("oggetti-orari EMMA")
    ax.set_title("A · positivi disponibili per anno, maggio-settembre")
    ax.set_xticks(data.year[::2])
    second = ax.twinx()
    second.plot(data.year, data.tracks, color="#4b3869", marker="o", markersize=3,
                linewidth=1.2)
    second.set_ylabel("track annuali", color="#4b3869")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    labels = ["EMMA 1998-2024", "EMMA + RSS 2008-2024", "EMMA + RSS + OPERA 2013-2024"]
    values = [
        int(data.object_hours.sum()),
        int(data.loc[data.year >= 2008, "object_hours"].sum()),
        int(data.loc[data.year >= 2013, "object_hours"].sum()),
    ]
    bars = ax.barh(labels, values, color=["#bdbdbd", "#5aa9e6", "#2a9d8f"])
    ax.invert_yaxis()
    ax.set_xlabel("oggetti-orari EMMA nel periodo comune")
    ax.set_title("B · capacità supervisionata teorica prima degli scarti QC")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(value + 900, bar.get_y() + bar.get_height() / 2, f"{value:,}".replace(",", "."),
                va="center", fontsize=11, weight="bold")
    ax.set_xlim(0, max(values) * 1.16)
    fig.suptitle("Scala temporale potenziale del dataset", fontsize=16, weight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
