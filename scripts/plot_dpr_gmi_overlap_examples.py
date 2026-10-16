#!/usr/bin/env python3
"""Plot representative EMMA overlaps with DPR- and GMI-PF fitted ellipses.

These figures visualize the feature-level matching geometry. They are separate
from the DPR pixel-centre QA, which tests how well the fitted DPR ellipse
represents the underlying radar pixels.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path
from emma_gpm.emma import emma_files, load_emma_mask, parse_emma_time, read_grid


MASK_COLOR = "#8c62aa"
DPR_COLOR = "#d95f02"
GMI_COLOR = "#1f77b4"


def save_figure_atomic(fig, destination: Path, *, dpi: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.stem}.part{destination.suffix}")
    fig.savefig(partial, dpi=dpi)
    partial.replace(destination)


def ellipse_bounds(row: pd.Series, pad_degrees: float = 0.0) -> tuple[float, float, float, float]:
    lat0, lon0 = float(row.r_lat), float(row.r_lon)
    a = max(float(row.r_major) / 2.0, 2.5)
    b = max(float(row.r_minor) / 2.0, 2.5)
    theta = np.deg2rad(float(row.r_orientation) if np.isfinite(row.r_orientation) else 0.0)
    east_radius = np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
    north_radius = np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)
    lon_radius = east_radius / (111.320 * max(np.cos(np.deg2rad(lat0)), 0.2))
    lat_radius = north_radius / 110.574
    return (
        lon0 - lon_radius - pad_degrees,
        lon0 + lon_radius + pad_degrees,
        lat0 - lat_radius - pad_degrees,
        lat0 + lat_radius + pad_degrees,
    )


def nearest_object(row: pd.Series, objects: pd.DataFrame) -> pd.Series | None:
    same_time = objects[objects.time.eq(row.emma_time)]
    if same_time.empty:
        return None
    dlat = np.maximum.reduce([
        same_time.lat_min.to_numpy() - row.r_lat,
        row.r_lat - same_time.lat_max.to_numpy(),
        np.zeros(len(same_time)),
    ])
    dlon = np.maximum.reduce([
        same_time.lon_min.to_numpy() - row.r_lon,
        row.r_lon - same_time.lon_max.to_numpy(),
        np.zeros(len(same_time)),
    ])
    distance = np.hypot(
        110.574 * dlat,
        111.320 * np.cos(np.deg2rad(float(row.r_lat))) * dlon,
    )
    return same_time.iloc[int(np.argmin(distance))]


def quantile_examples(frame: pd.DataFrame, metric: str, count: int = 4) -> pd.DataFrame:
    ordered = frame.dropna(subset=[metric]).sort_values(metric)
    if ordered.empty:
        return ordered
    positions = np.unique(np.rint(np.linspace(0, len(ordered) - 1, min(count, len(ordered)))).astype(int))
    return ordered.iloc[positions]


def choose_examples(dpr: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "hit":
        eligible = dpr[dpr.mcs_id.fillna(0).gt(0)]
        metric = "n_overlapped_emma_grid_centres"
    else:
        eligible = dpr[dpr.emma_class.eq("outside_emma") & dpr.r_major.gt(0)]
        metric = "r_major"
    selected = []
    for month in range(5, 10):
        monthly = eligible[eligible.observation_time.dt.month.eq(month)]
        chosen = quantile_examples(monthly, metric, count=4).copy()
        chosen["selection_metric"] = metric
        chosen["selection_rank"] = np.arange(1, len(chosen) + 1)
        selected.append(chosen)
    out = pd.concat(selected, ignore_index=True)
    out["example_kind"] = kind
    out["example_id"] = [
        f"{kind}_{int(row.observation_time.month):02d}_{int(row.selection_rank):02d}"
        for row in out.itertuples()
    ]
    return out


def nearby_gmi_candidates(case: pd.Series, gmi: pd.DataFrame) -> pd.DataFrame:
    candidates = gmi[
        gmi.orbit.eq(case.orbit)
        & ((gmi.observation_time - case.observation_time).abs() <= pd.Timedelta(minutes=30))
    ].copy()
    if candidates.empty:
        return candidates
    east = (candidates.r_lon - case.r_lon) * 111.320 * np.cos(np.deg2rad(float(case.r_lat)))
    north = (candidates.r_lat - case.r_lat) * 110.574
    centre_distance = np.hypot(east, north)
    dpr_radius = max(float(case.r_major) / 2.0, 2.5)
    gmi_radius = candidates.r_major.fillna(0).clip(lower=5.0) / 2.0
    candidates["centre_distance_to_dpr_km"] = centre_distance
    candidates["candidate_envelopes_overlap"] = centre_distance <= dpr_radius + gmi_radius
    return candidates[candidates.candidate_envelopes_overlap].copy()


def union_extent(
    case: pd.Series,
    target: pd.Series | None,
    gmi_candidates: pd.DataFrame,
    grid,
) -> tuple[float, float, float, float]:
    bounds = [ellipse_bounds(case, pad_degrees=0.7)]
    if target is not None:
        bounds.append((
            float(target.lon_min) - 0.7,
            float(target.lon_max) + 0.7,
            float(target.lat_min) - 0.7,
            float(target.lat_max) + 0.7,
        ))
    for _, feature in gmi_candidates.iterrows():
        bounds.append(ellipse_bounds(feature, pad_degrees=0.2))
    lon_min = max(float(grid.pilot_lon.min()), min(item[0] for item in bounds))
    lon_max = min(float(grid.pilot_lon.max()), max(item[1] for item in bounds))
    lat_min = max(float(grid.pilot_lat.min()), min(item[2] for item in bounds))
    lat_max = min(float(grid.pilot_lat.max()), max(item[3] for item in bounds))
    return lon_min, lon_max, lat_min, lat_max


def ellipse_patch(row: pd.Series, color: str, linewidth: float, alpha: float = 1.0) -> Ellipse:
    lat0, lon0 = float(row.r_lat), float(row.r_lon)
    return Ellipse(
        (lon0, lat0),
        width=float(row.r_major) / (111.320 * max(np.cos(np.deg2rad(lat0)), 0.2)),
        height=float(row.r_minor) / 110.574,
        angle=90.0 - float(row.r_orientation),
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=5,
    )


def prepare_example(case: pd.Series, gmi: pd.DataFrame, objects: pd.DataFrame, path_by_time, grid) -> dict:
    target = None
    if pd.notna(case.mcs_id) and case.mcs_id > 0:
        selected = objects[objects.track_uid.eq(case.track_uid) & objects.time.eq(case.emma_time)]
        if len(selected):
            target = selected.iloc[0]
    else:
        target = nearest_object(case, objects)
    target_uid = str(target.track_uid) if target is not None else ""
    gmi_candidates = nearby_gmi_candidates(case, gmi)
    any_gmi_emma = bool(gmi_candidates.mcs_id.fillna(0).gt(0).any()) if len(gmi_candidates) else False
    same_target = bool(gmi_candidates.track_uid.eq(target_uid).any()) if len(gmi_candidates) and target_uid else False
    mcs, _ = load_emma_mask(path_by_time[case.emma_time], grid)
    return {
        "case": case,
        "target": target,
        "target_uid": target_uid,
        "gmi": gmi_candidates,
        "any_gmi_emma": any_gmi_emma,
        "same_target": same_target,
        "mcs": mcs,
        "extent": union_extent(case, target, gmi_candidates, grid),
    }


def plot_example(ax, prepared: dict, grid, *, compact: bool) -> None:
    case = prepared["case"]
    mcs = prepared["mcs"]
    lon_min, lon_max, lat_min, lat_max = prepared["extent"]
    ax.pcolormesh(
        grid.pilot_lon,
        grid.pilot_lat,
        np.where(mcs > 0, 1.0, np.nan),
        cmap=ListedColormap([MASK_COLOR]),
        shading="nearest",
        alpha=0.82,
        rasterized=True,
    )
    for _, feature in prepared["gmi"].iterrows():
        ax.add_patch(ellipse_patch(feature, GMI_COLOR, 1.1 if compact else 1.5, alpha=0.75))
    ax.add_patch(ellipse_patch(case, DPR_COLOR, 2.1 if compact else 2.6))
    ax.plot(float(case.r_lon), float(case.r_lat), marker="+", color=DPR_COLOR, markersize=7, zorder=7)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1 / max(np.cos(np.deg2rad((lat_min + lat_max) / 2.0)), 0.3))
    recovery = "YES" if prepared["any_gmi_emma"] else "NO"
    same = "YES" if prepared["same_target"] else "NO"
    ax.set_title(
        f"{case.example_id} · DPR {case.example_kind.upper()} · EMMA centres={int(case.n_overlapped_emma_grid_centres)}\n"
        f"GMI candidates={len(prepared['gmi'])} · GMI→any MCS {recovery} · same MCS {same}",
        fontsize=8 if compact else 11,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=MASK_COLOR, markersize=10, label="EMMA MCS mask (same color in both figures)"),
        Line2D([0], [0], color=DPR_COLOR, lw=2.5, label="Selected DPR-PF fitted ellipse"),
        Line2D([0], [0], color=GMI_COLOR, lw=1.7, label="Nearby GMI-PF candidate ellipse"),
        Line2D([0], [0], marker="+", color=DPR_COLOR, lw=0, markersize=8, label="DPR-PF centre"),
    ]


def render_group(prepared_examples: list[dict], kind: str, totals: dict, grid) -> None:
    output_dir = ROOT / "figures" / "dpr_gmi_overlap_examples" / kind
    handles = legend_handles()
    for prepared in prepared_examples:
        case = prepared["case"]
        fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
        plot_example(ax, prepared, grid, compact=False)
        fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=True)
        save_figure_atomic(fig, output_dir / f"{case.example_id}.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(24, 18), constrained_layout=True)
    for ax, prepared in zip(axes.flat, prepared_examples):
        plot_example(ax, prepared, grid, compact=True)
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=True)
    label = "COINCIDENCE" if kind == "hit" else "NO COINCIDENCE"
    fig.suptitle(
        f"{label}: representative DPR-PF examples with nearby GMI-PF ellipses\n"
        f"Full pilot within EMMA coverage — DPR: {totals['dpr_hit']:,} coincidences / {totals['dpr_miss']:,} no coincidences; "
        f"GMI: {totals['gmi_hit']:,} coincidences / {totals['gmi_miss']:,} no coincidences",
        fontsize=16,
    )
    save_figure_atomic(fig, ROOT / "figures" / f"dpr_gmi_{kind}_examples.png", dpi=170)
    plt.close(fig)


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    matches = pd.read_parquet(ROOT / "catalogs/gpm_to_emma_pf_2020.parquet")
    objects = pd.read_parquet(interim / "emma_hourly_objects_2020.parquet")
    paths = emma_files(resolve_project_path(ROOT, config["emma"]["extracted_root"]))
    path_by_time = {parse_emma_time(path): path for path in paths}
    grid = read_grid(paths[0], config["domain"])
    dpr = matches[matches.sensor.eq("DPR")].copy()
    gmi = matches[matches.sensor.eq("GMI")].copy()
    totals = {
        "dpr_hit": int(dpr.mcs_id.fillna(0).gt(0).sum()),
        "dpr_miss": int(dpr.emma_class.eq("outside_emma").sum()),
        "gmi_hit": int(gmi.mcs_id.fillna(0).gt(0).sum()),
        "gmi_miss": int(gmi.emma_class.eq("outside_emma").sum()),
    }

    metric_rows = []
    for kind in ("hit", "miss"):
        examples = choose_examples(dpr, kind)
        prepared_examples = [
            prepare_example(row, gmi, objects, path_by_time, grid)
            for _, row in examples.iterrows()
        ]
        render_group(prepared_examples, kind, totals, grid)
        for prepared in prepared_examples:
            case = prepared["case"]
            metric_rows.append({
                "example_id": case.example_id,
                "example_kind": kind,
                "feature_uid": case.feature_uid,
                "orbit": int(case.orbit),
                "observation_time": case.observation_time,
                "emma_time": case.emma_time,
                "dpr_overlapped_emma_grid_centres": int(case.n_overlapped_emma_grid_centres),
                "reference_track_uid": prepared["target_uid"],
                "nearby_gmi_candidate_count": len(prepared["gmi"]),
                "nearby_gmi_matches_any_emma": prepared["any_gmi_emma"],
                "nearby_gmi_matches_reference_mcs": prepared["same_target"],
                "selection_metric": case.selection_metric,
                "dpr_r_major_km": float(case.r_major),
                "dpr_r_minor_km": float(case.r_minor),
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(ROOT / "results/dpr_gmi_overlap_examples.csv", index=False)
    print(totals)
    print(metrics.groupby(["example_kind", "nearby_gmi_matches_any_emma"]).size())


if __name__ == "__main__":
    main()
