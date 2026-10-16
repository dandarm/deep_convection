#!/usr/bin/env python3
"""Plot the exact OPERA geometric grid footprint and candidate EMMA pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


OPERA_CRS = CRS.from_proj4(
    "+proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000 "
    "+datum=WGS84 +units=m +no_defs"
)
X_MIN, X_MAX = 0.0, 3_800_000.0
Y_MIN, Y_MAX = -4_400_000.0, 0.0


def boundary_lonlat(points_per_edge: int = 500) -> tuple[np.ndarray, np.ndarray]:
    top_x = np.linspace(X_MIN, X_MAX, points_per_edge)
    right_y = np.linspace(Y_MAX, Y_MIN, points_per_edge)
    bottom_x = np.linspace(X_MAX, X_MIN, points_per_edge)
    left_y = np.linspace(Y_MIN, Y_MAX, points_per_edge)
    x = np.concatenate(
        [top_x, np.full_like(right_y, X_MAX), bottom_x, np.full_like(left_y, X_MIN)]
    )
    y = np.concatenate(
        [np.full_like(top_x, Y_MAX), right_y, np.full_like(bottom_x, Y_MIN), left_y]
    )
    lon, lat = Transformer.from_crs(OPERA_CRS, "EPSG:4326", always_xy=True).transform(x, y)
    return np.asarray(lon), np.asarray(lat)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidates = pd.read_csv(args.shortlist).head(4)
    boundary_lon, boundary_lat = boundary_lonlat()
    projection = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=55)
    figure = plt.figure(figsize=(11.5, 9.2), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1, projection=projection)
    axis.set_extent([-43, 61, 29, 72], crs=ccrs.PlateCarree())
    axis.add_feature(cfeature.LAND, facecolor="#f3f0e8")
    axis.add_feature(cfeature.OCEAN, facecolor="#dcecf4")
    axis.add_feature(cfeature.BORDERS, linewidth=0.45, edgecolor="#777777")
    axis.coastlines(resolution="50m", linewidth=0.65, color="#4b5563")
    grid = axis.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4,
        color="#64748b", alpha=0.45, linestyle="--"
    )
    grid.top_labels = False
    grid.right_labels = False
    axis.fill(
        boundary_lon,
        boundary_lat,
        transform=ccrs.PlateCarree(),
        facecolor="#38bdf8",
        edgecolor="#0369a1",
        linewidth=2.0,
        alpha=0.16,
        label="griglia geometrica OPERA 2200x1900, 2 km",
    )

    # Current pair: the negative centre lies just east of the OPERA grid.
    axis.scatter(
        [-9.5663], [51.8602], transform=ccrs.PlateCarree(), marker="*", s=180,
        color="#16a34a", edgecolor="white", linewidth=0.8, zorder=5,
        label="positivo attuale EMMA"
    )
    axis.scatter(
        [32.8437], [41.9941], transform=ccrs.PlateCarree(), marker="X", s=105,
        color="#dc2626", edgecolor="white", linewidth=0.8, zorder=5,
        label="senza-EMMA attuale (fuori griglia)"
    )

    # Show each distinct positive and the strongest geometrically eligible negatives.
    shown_positive: set[tuple[float, float]] = set()
    for rank, row in candidates.reset_index(drop=True).iterrows():
        positive = (round(float(row.plon), 4), round(float(row.plat), 4))
        if positive not in shown_positive:
            axis.scatter(
                [row.plon], [row.plat], transform=ccrs.PlateCarree(), marker="*", s=135,
                color="#15803d", edgecolor="white", linewidth=0.6, zorder=5
            )
            shown_positive.add(positive)
        axis.scatter(
            [row.nlon], [row.nlat], transform=ccrs.PlateCarree(), marker="o", s=70,
            color="#f59e0b", edgecolor="#7c2d12", linewidth=0.7, zorder=5
        )
        axis.text(
            row.nlon + 0.7,
            row.nlat + 0.3,
            f"N{rank + 1}",
            transform=ccrs.PlateCarree(),
            fontsize=9,
            weight="bold",
            color="#7c2d12",
            zorder=6,
        )

    axis.set_title(
        "Dominio geometrico OPERA e candidati borderline\n"
        "La copertura radar effettiva e i quality flag variano nello spazio e nel tempo",
        fontsize=14,
    )
    axis.legend(loc="lower left", framealpha=0.94, fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=190)
    plt.close(figure)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
