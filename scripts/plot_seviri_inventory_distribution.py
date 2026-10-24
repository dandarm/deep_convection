#!/usr/bin/env python3
"""Simple temporal summary of SEVIRI downloads and EMMA mask locations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--emma-objects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest["central_time"] = pd.to_datetime(manifest["central_time"])
    emma = pd.read_parquet(args.emma_objects)
    lon = emma["centroid_lon_gridmean"].to_numpy(dtype=float)
    lat = emma["centroid_lat_gridmean"].to_numpy(dtype=float)

    month_counts = manifest.central_time.dt.month.value_counts().sort_index()
    hour_counts = manifest.central_time.dt.hour.value_counts().reindex(range(24), fill_value=0)
    summary = pd.concat(
        [
            month_counts.rename("count").rename_axis("value").reset_index().assign(dimension="month"),
            hour_counts.rename("count").rename_axis("value").reset_index().assign(dimension="utc_hour"),
        ],
        ignore_index=True,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    emma[["time", "track_uid", "centroid_lon_gridmean", "centroid_lat_gridmean"]].to_csv(
        args.summary_output.with_name("emma_mask_centroids_2020.csv"), index=False
    )

    fig = plt.figure(figsize=(13.2, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35])
    ax_month = fig.add_subplot(grid[0, 0])
    ax_hour = fig.add_subplot(grid[0, 1])
    ax_map = fig.add_subplot(grid[1, :], projection=ccrs.PlateCarree())

    months = ["mag", "giu", "lug", "ago", "set"]
    ax_month.bar(months, month_counts.reindex(range(5, 10), fill_value=0), color="#2878b5")
    ax_month.set_ylabel("sequenze complete")
    ax_month.set_title("A · distribuzione mensile")
    for i, value in enumerate(month_counts.reindex(range(5, 10), fill_value=0)):
        ax_month.text(i, value + 0.4, str(int(value)), ha="center", fontsize=9)

    ax_hour.bar(hour_counts.index, hour_counts.values, color="#5aa9e6")
    ax_hour.set_xticks(range(0, 24, 3))
    ax_hour.set_xlabel("ora centrale UTC")
    ax_hour.set_ylabel("sequenze complete")
    ax_hour.set_title("B · distribuzione diurna")

    finite = np.isfinite(lon) & np.isfinite(lat)
    ax_map.set_extent([-20, 40, 30, 70], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#eef5fa")
    ax_map.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f7f5ef")
    ax_map.coastlines(resolution="110m", linewidth=0.75, color="#252525")
    ax_map.add_feature(
        cfeature.BORDERS.with_scale("110m"), linewidth=0.35,
        edgecolor="#777777", linestyle=":"
    )
    ax_map.scatter(
        lon[finite], lat[finite], s=8, c="#7b3294", alpha=0.36,
        linewidths=0, transform=ccrs.PlateCarree(), rasterized=True,
    )
    gl = ax_map.gridlines(
        draw_labels=True, linewidth=0.35, color="#777777", alpha=0.4,
        linestyle=":"
    )
    gl.top_labels = False
    gl.right_labels = False
    ax_map.set_title(
        f"C · posizione delle {finite.sum():,} maschere-oggetto EMMA del 2020".replace(",", ".")
    )
    fig.suptitle("SEVIRI scaricato — MJJAS 2020", fontsize=16, weight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
