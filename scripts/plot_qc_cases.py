#!/usr/bin/env python3
"""Overlay ten EMMA masks with actual DPR pixel centres for visual QA."""
from pathlib import Path
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


CLASS_COLORS = {
    "no_precip_or_missing": "#9aa0a6",
    "stratiform": "#3274a1",
    "convective": "#d62728",
    "other": "#f2b134",
}


def nearest_grid_index(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    insertion = np.searchsorted(grid, values)
    right = np.clip(insertion, 0, len(grid) - 1)
    left = np.clip(insertion - 1, 0, len(grid) - 1)
    return np.where(np.abs(values - grid[left]) <= np.abs(values - grid[right]), left, right)


def inside_pf_ellipse(frame: pd.DataFrame, case: pd.Series) -> np.ndarray:
    """Test pixel centres against the same PF fitted-ellipse convention as matching."""
    major = float(case.r_major)
    minor = float(case.r_minor)
    if not np.isfinite(major) or not np.isfinite(minor) or major <= 0 or minor <= 0:
        return np.zeros(len(frame), dtype=bool)
    lat0, lon0 = float(case.r_lat), float(case.r_lon)
    east = (frame.lon.to_numpy() - lon0) * 111.320 * np.cos(np.deg2rad(lat0))
    north = (frame.lat.to_numpy() - lat0) * 110.574
    theta = np.deg2rad(float(case.r_orientation) if np.isfinite(case.r_orientation) else 0.0)
    along = east * np.sin(theta) + north * np.cos(theta)
    across = east * np.cos(theta) - north * np.sin(theta)
    return (along / max(major / 2, 2.5)) ** 2 + (across / max(minor / 2, 2.5)) ** 2 <= 1


def case_data(
    case: pd.Series,
    pixels: pd.DataFrame,
    objects: pd.DataFrame,
    path_by_time: dict[pd.Timestamp, Path],
    grid,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    timestamp = pd.Timestamp(case.emma_time)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    is_hit = case.case_kind == "hit"
    target_id = int(str(case.target_track_uid).split(":")[-1]) if is_hit else 0
    if is_hit:
        target = objects[
            objects.track_uid.eq(case.target_track_uid) & objects.time.eq(timestamp)
        ].iloc[0]
        pad_lon, pad_lat = 2.0, 1.5
        extent = {
            "lon_min": max(float(grid.pilot_lon.min()), float(target.lon_min) - pad_lon),
            "lon_max": min(float(grid.pilot_lon.max()), float(target.lon_max) + pad_lon),
            "lat_min": max(float(grid.pilot_lat.min()), float(target.lat_min) - pad_lat),
            "lat_max": min(float(grid.pilot_lat.max()), float(target.lat_max) + pad_lat),
        }
    else:
        # For a GPM→EMMA miss, centre the map on the fitted PF envelope itself.
        # The nearest MCS stored in the manifest is diagnostic and is not used
        # to declare either a hit or a miss.
        lat0, lon0 = float(case.r_lat), float(case.r_lon)
        radius_lat = max(float(case.r_major) / 2 / 110.574, 0.4) + 1.0
        radius_lon = max(
            float(case.r_major) / 2 / (111.320 * max(np.cos(np.deg2rad(lat0)), 0.2)),
            0.4,
        ) + 1.0
        extent = {
            "lon_min": max(float(grid.pilot_lon.min()), lon0 - radius_lon),
            "lon_max": min(float(grid.pilot_lon.max()), lon0 + radius_lon),
            "lat_min": max(float(grid.pilot_lat.min()), lat0 - radius_lat),
            "lat_max": min(float(grid.pilot_lat.max()), lat0 + radius_lat),
        }
    orbit_pixels = pixels[pixels.orbit.eq(int(case.orbit))].copy()
    pixel_times = pd.to_datetime(orbit_pixels.observation_time, utc=True)
    time_keep = (pixel_times - timestamp).abs() <= pd.Timedelta(minutes=30)
    space_keep = (
        orbit_pixels.lon.between(extent["lon_min"], extent["lon_max"])
        & orbit_pixels.lat.between(extent["lat_min"], extent["lat_max"])
    )
    selected = orbit_pixels[time_keep & space_keep].copy()
    selected["time_offset_minutes"] = (
        (pd.to_datetime(selected.observation_time, utc=True) - timestamp).dt.total_seconds() / 60
    )
    mcs, robust = load_emma_mask(path_by_time[timestamp], grid)
    if len(selected):
        iy = nearest_grid_index(selected.lat.to_numpy(), grid.pilot_lat)
        ix = nearest_grid_index(selected.lon.to_numpy(), grid.pilot_lon)
        selected["emma_mcs_id_at_pixel_center"] = mcs[iy, ix]
        selected["emma_robust_id_at_pixel_center"] = robust[iy, ix]
    else:
        selected["emma_mcs_id_at_pixel_center"] = pd.Series(dtype=int)
        selected["emma_robust_id_at_pixel_center"] = pd.Series(dtype=int)
    selected["inside_feature_ellipse"] = inside_pf_ellipse(selected, case)
    selected["inside_target_mcs"] = selected.emma_mcs_id_at_pixel_center.eq(target_id) if is_hit else False
    selected["inside_target_robust"] = selected.emma_robust_id_at_pixel_center.eq(target_id) if is_hit else False
    ellipse_pixels = selected[selected.inside_feature_ellipse]
    ellipse_emma_hits = int(ellipse_pixels.emma_mcs_id_at_pixel_center.gt(0).sum())
    metrics = {
        "case_id": case.case_id,
        "pf_case_kind": case.case_kind,
        "orbit": int(case.orbit),
        "emma_time": timestamp,
        "target_track_uid": case.target_track_uid,
        "pf_time_offset_minutes": float(case.time_offset_minutes),
        "plotted_dpr_pixel_centres": len(selected),
        "dpr_centres_inside_target_mcs": int(selected.inside_target_mcs.sum()),
        "dpr_centres_inside_target_robust": int(selected.inside_target_robust.sum()),
        "dpr_centres_inside_feature_ellipse": len(ellipse_pixels),
        "feature_ellipse_centres_inside_any_emma": ellipse_emma_hits,
        "raw_pixel_center_hit": bool(selected.inside_target_mcs.any()) if is_hit else False,
        "case_validation_passed": (
            bool(selected.inside_target_mcs.any()) if is_hit else ellipse_emma_hits == 0
        ),
        "min_raw_pixel_offset_minutes": float(selected.time_offset_minutes.min()) if len(selected) else np.nan,
        "max_raw_pixel_offset_minutes": float(selected.time_offset_minutes.max()) if len(selected) else np.nan,
        **extent,
        "footprint_approximation": "actual DPR geolocation represented by pixel centre",
    }
    return selected, mcs, robust, metrics


def plot_panel(ax, case, selected, mcs, robust, metrics, grid) -> None:
    extent = [metrics["lon_min"], metrics["lon_max"], metrics["lat_min"], metrics["lat_max"]]
    is_hit = case.case_kind == "hit"
    target_id = int(str(case.target_track_uid).split(":")[-1]) if is_hit else 0
    any_mcs = np.where(mcs > 0, 1.0, np.nan)
    target = np.where(mcs == target_id, 1.0, np.nan)
    target_robust = np.where(robust == target_id, 1.0, np.nan)
    ax.pcolormesh(
        grid.pilot_lon,
        grid.pilot_lat,
        any_mcs,
        cmap=ListedColormap(["#ded6eb"]),
        shading="nearest",
        alpha=0.55,
        rasterized=True,
    )
    if is_hit:
        ax.pcolormesh(
            grid.pilot_lon,
            grid.pilot_lat,
            target,
            cmap=ListedColormap(["#7c4d9e"]),
            shading="nearest",
            alpha=0.65,
            rasterized=True,
        )
        ax.contour(
            grid.pilot_lon,
            grid.pilot_lat,
            np.where(target_robust == 1, 1, 0),
            levels=[0.5],
            colors=["#32154f"],
            linewidths=1.0,
        )
    for label, color in CLASS_COLORS.items():
        subset = selected[selected.type_precip_label.eq(label)]
        if len(subset):
            ax.scatter(
                subset.lon,
                subset.lat,
                s=4.0 if label == "no_precip_or_missing" else 7.0,
                c=color,
                alpha=0.38 if label == "no_precip_or_missing" else 0.8,
                linewidths=0,
                rasterized=True,
            )
    # PF R_major/R_minor are documented fitted axes.  Longitude conversion is
    # local and the patch is diagnostic, not a geodetic footprint polygon.
    lat0, lon0 = float(case.r_lat), float(case.r_lon)
    width = float(case.r_major) / (111.320 * max(np.cos(np.deg2rad(lat0)), 0.2))
    height = float(case.r_minor) / 110.574
    ax.add_patch(
        Ellipse(
            (lon0, lat0),
            width=width,
            height=height,
            angle=90.0 - float(case.r_orientation),
            fill=False,
            edgecolor="black",
            linewidth=1.1,
            linestyle="--",
            zorder=6,
        )
    )
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_aspect(1 / max(np.cos(np.deg2rad(np.mean(extent[2:]))), 0.3))
    outcome = "VALIDATED" if metrics["case_validation_passed"] else "CHECK"
    ax.set_title(
        f"{case.case_id} · PF {case.case_kind.upper()} · {outcome}\n"
        f"orbit {int(case.orbit)} · EMMA {pd.Timestamp(case.emma_time):%Y-%m-%d %H:%MZ}",
        fontsize=8.5,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.18)


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    manifest = pd.read_csv(ROOT / "catalogs/visual_case_manifest_2020.csv", parse_dates=["observation_time", "emma_time"])
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    pixels = pd.read_parquet(interim / "raw_dpr_visual_case_pixels.parquet")
    objects = pd.read_parquet(interim / "emma_hourly_objects_2020.parquet")
    paths = emma_files(resolve_project_path(ROOT, config["emma"]["extracted_root"]))
    path_by_time = {parse_emma_time(path): path for path in paths}
    grid = read_grid(paths[0], config["domain"])
    output_dir = ROOT / "figures/qc_cases"
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = []
    metrics_rows = []
    for _, case in manifest.iterrows():
        selected, mcs, robust, metrics = case_data(case, pixels, objects, path_by_time, grid)
        prepared.append((case, selected, mcs, robust, metrics))
        metrics_rows.append(metrics)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        plot_panel(ax, case, selected, mcs, robust, metrics, grid)
        fig.tight_layout()
        fig.savefig(output_dir / f"{case.case_id}.png", dpi=190)
        plt.close(fig)

    case_metrics = pd.DataFrame(metrics_rows)
    case_metrics.to_csv(ROOT / "results/qc_case_metrics.csv", index=False)
    manifest.merge(case_metrics, on=["case_id", "orbit", "emma_time", "target_track_uid"], how="left").to_csv(
        ROOT / "catalogs/visual_case_manifest_2020.csv", index=False
    )

    fig, axes = plt.subplots(2, 5, figsize=(21, 9), constrained_layout=True)
    for ax, prepared_case in zip(axes.flat, prepared):
        plot_panel(ax, *prepared_case, grid)
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#7c4d9e", markersize=9, label="target EMMA MCS"),
        Line2D([0], [0], color="#32154f", lw=1.5, label="target robust boundary"),
        Line2D([0], [0], color="black", lw=1.2, ls="--", label="PF fitted ellipse"),
    ] + [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=6, label=label)
        for label, color in CLASS_COLORS.items()
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=7, frameon=False)
    fig.suptitle(
        "EMMA masks and actual DPR pixel centres · centre is the explicit footprint approximation",
        fontsize=14,
    )
    fig.savefig(ROOT / "figures/qc_cases_overview.png", dpi=180)
    plt.close(fig)
    print(case_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
