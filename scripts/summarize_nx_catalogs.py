#!/usr/bin/env python3
"""Create tabular and visual statistics for the implemented Nx verification."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ["N0_candidate", "N1_candidate", "N2_candidate", "NU"]
COLORS = {
    "N0_candidate": "#16a34a",
    "N1_candidate": "#2563eb",
    "N2_candidate": "#dc2626",
    "NU": "#8b5cf6",
}


def count_table(frame: pd.DataFrame, column: str, catalog: str) -> pd.DataFrame:
    counts = frame[column].value_counts().reindex(ORDER, fill_value=0)
    return pd.DataFrame(
        {
            "catalog": catalog,
            "class": counts.index,
            "count": counts.values,
            "fraction": counts.values / max(int(counts.sum()), 1),
            "catalog_total": int(counts.sum()),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasonal", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    seasonal = pd.read_csv(args.seasonal, parse_dates=["central_time"])
    legacy = pd.read_csv(args.legacy)
    positives = pd.read_csv(args.positives)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.concat(
        [
            count_table(seasonal, "class", "seasonal_cold_shield"),
            count_table(legacy, "provisional_class", "legacy_180_crop_check"),
        ],
        ignore_index=True,
    )
    summary.to_csv(args.results_dir / "nx_statistics_summary.csv", index=False)

    seasonal["month"] = seasonal.central_time.dt.month
    monthly = (
        seasonal.groupby(["month", "class"]).size().unstack(fill_value=0).reindex(columns=ORDER, fill_value=0)
    )
    monthly.to_csv(args.results_dir / "nx_statistics_by_month.csv")

    numeric = [
        "central_shield_pixels",
        "central_cold_component_pixels",
        "opera_inside_fraction",
        "opera_valid_fraction_median",
        "opera_good_fraction_median",
        "opera_max_dbzh_5slots",
        "opera_max_area35_km2_5slots",
        "opera_max_area40_km2_5slots",
        "video_precentral_min_bt_slope_k_per_min",
    ]
    quantile_rows = []
    for category, group in seasonal.groupby("class"):
        for variable in numeric:
            values = pd.to_numeric(group[variable], errors="coerce").dropna()
            if values.empty:
                continue
            for quantile in (0.10, 0.25, 0.50, 0.75, 0.90):
                quantile_rows.append(
                    {
                        "class": category,
                        "variable": variable,
                        "quantile": quantile,
                        "value": float(values.quantile(quantile)),
                        "finite_count": len(values),
                    }
                )
    pd.DataFrame(quantile_rows).to_csv(
        args.results_dir / "nx_statistics_quantiles.csv", index=False
    )
    positive_counts = positives.positive_evidence_class.value_counts().rename_axis(
        "positive_evidence_class"
    ).reset_index(name="count")
    positive_counts["fraction"] = positive_counts["count"] / len(positives)
    positive_counts.to_csv(
        args.results_dir / "emma_positive_evidence_summary.csv", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Statistiche dei campioni Nx", fontsize=17, weight="bold")

    ax = axes[0, 0]
    seasonal_counts = summary[summary.catalog == "seasonal_cold_shield"].set_index("class")
    bars = ax.bar(
        np.arange(len(ORDER)),
        seasonal_counts.loc[ORDER, "count"],
        color=[COLORS[value] for value in ORDER],
    )
    ax.bar_label(bars, padding=3)
    ax.set_xticks(np.arange(len(ORDER)), [value.replace("_candidate", "") for value in ORDER])
    ax.set(title="A · catalogo stagionale", ylabel="oggetti")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    bottom = np.zeros(len(monthly))
    for category in ORDER:
        values = monthly[category].to_numpy()
        ax.bar(monthly.index, values, bottom=bottom, color=COLORS[category], label=category.replace("_candidate", ""))
        bottom += values
    ax.set(title="B · distribuzione mensile", xlabel="mese", ylabel="oggetti")
    ax.set_xticks(monthly.index)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    for category in ORDER:
        group = seasonal[seasonal["class"] == category]
        if group.empty:
            continue
        ax.scatter(
            group.central_shield_pixels,
            group.opera_max_dbzh_5slots,
            s=22,
            alpha=0.65,
            color=COLORS[category],
            label=category.replace("_candidate", ""),
        )
    ax.axhline(30, color="#374151", linestyle="--", linewidth=0.9)
    ax.axhline(40, color="#991b1b", linestyle="--", linewidth=0.9)
    ax.set_xscale("log")
    ax.set(
        title="C · dimensione e riflettività",
        xlabel="pixel dello scudo <253 K",
        ylabel="massimo DBZH [dBZ]",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    pcolors = {
        "P_radar_confirmed": "#dc2626",
        "P_radar_quiet": "#2563eb",
        "P_unresolved": "#8b5cf6",
    }
    positive_counts = positive_counts.sort_values("positive_evidence_class")
    bars = ax.bar(
        np.arange(len(positive_counts)),
        positive_counts["count"],
        color=[pcolors.get(value, "#6b7280") for value in positive_counts.positive_evidence_class],
    )
    ax.bar_label(bars, padding=3)
    ax.set_xticks(
        np.arange(len(positive_counts)),
        [value.replace("P_", "") for value in positive_counts.positive_evidence_class],
        rotation=15,
    )
    ax.set(title="D · positivi EMMA", ylabel="casi")
    ax.grid(axis="y", alpha=0.2)

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(positive_counts.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
