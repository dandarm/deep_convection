#!/usr/bin/env python3
"""Compute reproducible summary tables and diagnostic plots for the PF pilot.

These metrics describe precipitation-feature co-occurrences, not complete GPM
swath coverage.  That distinction is repeated in every machine-readable summary.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path


INTERPRETATION = (
    "PF-feature co-occurrence lower bound; absence is not proof of no sensor overpass"
)


def metric(rows: list[dict], name: str, value, unit: str = "count", sensor: str = "ALL"):
    rows.append(
        {
            "metric": name,
            "sensor": sensor,
            "value": value,
            "unit": unit,
            "interpretation": INTERPRETATION,
        }
    )


def main() -> None:
    results = ROOT / "results"
    figures = ROOT / "figures"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    config = load_config(ROOT / "config/pilot_2020.json")
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    matches = pd.read_parquet(ROOT / "catalogs/gpm_to_emma_pf_2020.parquet")
    lifecycle = pd.read_parquet(ROOT / "catalogs/emma_to_gpm_pf_2020.parquet")
    objects = pd.read_parquet(interim / "emma_hourly_objects_2020.parquet")

    rows: list[dict] = []
    metric(rows, "emma_mcs_tracks_intersecting_pilot_domain", len(lifecycle))
    metric(
        rows,
        "emma_tracks_with_official_robust_phase_in_domain",
        int(lifecycle.has_official_robust_mask_in_domain.sum()),
    )
    metric(rows, "emma_hourly_objects", len(objects))
    metric(rows, "emma_hourly_objects_official_robust", int(objects.official_robust_flag.sum()))
    metric(
        rows,
        "emma_hourly_objects_nonrobust_development_or_decay",
        int((~objects.official_robust_flag).sum()),
    )

    for sensor in ("GMI", "DPR"):
        sensor_lower = sensor.lower()
        subset = matches[matches.sensor.eq(sensor)]
        hit_column = f"observed_by_{sensor_lower}_pf"
        n_hit = int(lifecycle[hit_column].sum())
        metric(rows, "pf_features_in_pilot", len(subset), sensor=sensor)
        metric(rows, "emma_tracks_with_pf_cooccurrence", n_hit, sensor=sensor)
        metric(
            rows,
            "emma_track_pf_cooccurrence_fraction",
            n_hit / len(lifecycle),
            unit="fraction",
            sensor=sensor,
        )
        for class_name, count in subset.emma_class.value_counts().items():
            metric(rows, f"pf_features_{class_name}", int(count), sensor=sensor)

    offset = matches.time_offset_minutes.dropna()
    for name, value in {
        "time_offset_min_minutes": offset.min(),
        "time_offset_p05_minutes": offset.quantile(0.05),
        "time_offset_median_minutes": offset.median(),
        "time_offset_mean_minutes": offset.mean(),
        "time_offset_p95_minutes": offset.quantile(0.95),
        "time_offset_max_minutes": offset.max(),
    }.items():
        metric(rows, name, float(value), unit="minutes")

    summary = pd.DataFrame(rows)
    summary.to_csv(results / "metrics_summary.csv", index=False)

    by_month = (
        matches.assign(month=matches.observation_time.dt.month)
        .groupby(["sensor", "month", "emma_class"], observed=True)
        .size()
        .rename("feature_count")
        .reset_index()
    )
    by_month["interpretation"] = INTERPRETATION
    by_month.to_csv(results / "counts_by_month.csv", index=False)

    lon_edges = np.arange(-10, 50, 5)
    lat_edges = np.arange(25, 60, 5)
    geo = matches.assign(
        lon_bin=pd.cut(matches.lon, lon_edges, right=False),
        lat_bin=pd.cut(matches.lat, lat_edges, right=False),
    )
    by_geo = (
        geo.groupby(["sensor", "lat_bin", "lon_bin", "emma_class"], observed=True)
        .size()
        .rename("feature_count")
        .reset_index()
    )
    by_geo["lat_min"] = by_geo.lat_bin.map(lambda value: float(value.left))
    by_geo["lat_max"] = by_geo.lat_bin.map(lambda value: float(value.right))
    by_geo["lon_min"] = by_geo.lon_bin.map(lambda value: float(value.left))
    by_geo["lon_max"] = by_geo.lon_bin.map(lambda value: float(value.right))
    by_geo.drop(columns=["lat_bin", "lon_bin"]).to_csv(
        results / "counts_by_geo_5deg.csv", index=False
    )

    bins = np.arange(-30, 32, 2)
    histogram_frames = []
    for sensor, subset in matches.groupby("sensor"):
        counts, edges = np.histogram(subset.time_offset_minutes, bins=bins)
        histogram_frames.append(
            pd.DataFrame(
                {
                    "sensor": sensor,
                    "offset_left_minutes": edges[:-1],
                    "offset_right_minutes": edges[1:],
                    "feature_count": counts,
                }
            )
        )
    offset_hist = pd.concat(histogram_frames, ignore_index=True)
    offset_hist.to_csv(results / "time_offset_histogram.csv", index=False)

    positive = matches[matches.track_uid.notna()].copy()
    duplicates = (
        positive.groupby(["sensor", "track_uid", "orbit"], observed=True)
        .agg(
            feature_count=("feature_uid", "nunique"),
            earliest_observation=("observation_time", "min"),
            latest_observation=("observation_time", "max"),
            robust_feature_count=("robust_mcs_id", lambda values: int(values.fillna(0).gt(0).sum())),
        )
        .reset_index()
    )
    duplicates["is_duplicate_system_orbit"] = duplicates.feature_count.gt(1)
    duplicates.to_csv(results / "duplicates_by_system_orbit.csv", index=False)
    for sensor, subset in duplicates.groupby("sensor"):
        duplicate_groups = int(subset.is_duplicate_system_orbit.sum())
        extra_features = int((subset.feature_count - 1).clip(lower=0).sum())
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    [
                        {
                            "metric": "duplicate_system_orbit_groups",
                            "sensor": sensor,
                            "value": duplicate_groups,
                            "unit": "count",
                            "interpretation": INTERPRETATION,
                        },
                        {
                            "metric": "extra_pf_features_within_system_orbit",
                            "sensor": sensor,
                            "value": extra_features,
                            "unit": "count",
                            "interpretation": INTERPRETATION,
                        },
                        {
                            "metric": "max_pf_features_for_one_system_orbit",
                            "sensor": sensor,
                            "value": int(subset.feature_count.max()),
                            "unit": "count",
                            "interpretation": INTERPRETATION,
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
    summary.to_csv(results / "metrics_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sensor, subset in matches.groupby("sensor"):
        ax.hist(
            subset.time_offset_minutes,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            label=f"{sensor} (n={len(subset):,})",
        )
    ax.axvline(-30, color="0.4", linestyle=":")
    ax.axvline(30, color="0.4", linestyle=":")
    ax.set(xlabel="GPM time − nearest EMMA hour (minutes)", ylabel="PF feature count")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "time_offset_distribution.png", dpi=180)
    plt.close(fig)

    monthly_total = (
        matches.assign(month=matches.observation_time.dt.month)
        .groupby(["sensor", "month"])
        .size()
        .unstack("sensor", fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    monthly_total.plot(kind="bar", ax=ax)
    ax.set(xlabel="Month (2020)", ylabel="PF feature count")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "monthly_pf_feature_counts.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.8), sharex=True, sharey=True, constrained_layout=True
    )
    meshes = []
    positive_counts = []
    histograms = {}
    for sensor in ("GMI", "DPR"):
        subset = matches[matches.sensor.eq(sensor)]
        histogram, _, _ = np.histogram2d(
            subset.lat, subset.lon, bins=[lat_edges, lon_edges]
        )
        histograms[sensor] = histogram
        positive_counts.extend(histogram[histogram > 0].tolist())
    common_norm = LogNorm(vmin=max(1, min(positive_counts)), vmax=max(positive_counts))
    for ax, sensor in zip(axes, ("GMI", "DPR")):
        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            np.ma.masked_equal(histograms[sensor], 0),
            norm=common_norm,
            cmap="viridis",
            shading="flat",
        )
        meshes.append(mesh)
        ax.set_title(sensor)
        ax.set_xlabel("Longitude")
        ax.set_aspect(1 / np.cos(np.deg2rad(40)))
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Latitude")
    colorbar = fig.colorbar(
        meshes[0], ax=axes, label="PF feature count per 5° × 5° bin", pad=0.025
    )
    colorbar.ax.minorticks_off()
    fig.suptitle("GPM precipitation features in the requested pilot domain")
    fig.savefig(figures / "pf_feature_counts_by_geo_5deg.png", dpi=180)
    plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
