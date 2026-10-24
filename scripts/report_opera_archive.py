#!/usr/bin/env python3
"""Inventory a local OPERA year and render temporal/spatial coverage diagnostics."""

from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path
import re

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import cartopy.crs as ccrs
import cartopy.feature as cfeature


NAME = re.compile(r"OPERA@(\d{8})T(\d{4})@0@DBZH_QIND\.h5$")


def scan_archive(root: Path, year: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in (root / str(year)).rglob("*.h5"):
        match = NAME.match(path.name)
        if not match:
            continue
        timestamp = pd.to_datetime("".join(match.groups()), format="%Y%m%d%H%M")
        rows.append(
            {
                "timestamp_utc": timestamp,
                "path": str(path),
                "bytes": path.stat().st_size,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["timestamp_utc", "path", "bytes"])
    return pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)


def odim_domain(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        where = handle["where"].attrs
        projection = where["projdef"]
        if isinstance(projection, bytes):
            projection = projection.decode("ascii")
        corners = {
            key: [float(where[f"{key}_lon"]), float(where[f"{key}_lat"])]
            for key in ("LL", "LR", "UR", "UL")
        }
        return {
            "projection": str(projection),
            "xsize": int(where["xsize"]),
            "ysize": int(where["ysize"]),
            "xscale_m": float(where["xscale"]),
            "yscale_m": float(where["yscale"]),
            "corners_lon_lat": corners,
        }


def monthly_summary(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    actual = (
        frame.assign(month=frame.timestamp_utc.dt.month)
        .groupby("month")
        .agg(files=("timestamp_utc", "size"), bytes=("bytes", "sum"))
        .reindex(range(1, 13), fill_value=0)
    )
    actual.index.name = "month"
    actual = actual.reset_index()
    actual["expected_slots"] = [
        calendar.monthrange(year, month)[1] * 96 for month in actual.month
    ]
    actual["coverage_fraction"] = actual.files / actual.expected_slots
    actual["size_gib"] = actual.bytes / 1024**3
    return actual


def render_plot(
    frame: pd.DataFrame,
    monthly: pd.DataFrame,
    domain: dict[str, object],
    year: int,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.85, 1.15))
    ax_month = fig.add_subplot(grid[0, 0])
    ax_time = fig.add_subplot(grid[1, 0])
    ax_map = fig.add_subplot(grid[:, 1], projection=ccrs.PlateCarree())

    labels = [calendar.month_abbr[value] for value in monthly.month]
    ax_month.bar(labels, monthly.coverage_fraction * 100, color="#2563eb")
    ax_month.axhline(100, color="#111827", lw=0.8, ls="--")
    ax_month.set_ylim(0, 105)
    ax_month.set_ylabel("slot locali / slot nominali [%]")
    ax_month.set_title("Copertura mensile a 15 minuti")
    ax_month.grid(axis="y", alpha=0.25)

    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    matrix = np.zeros((96, len(days)), dtype=np.uint8)
    if not frame.empty:
        day_index = (frame.timestamp_utc.dt.normalize() - days[0]).dt.days.to_numpy()
        slot_index = (
            frame.timestamp_utc.dt.hour.to_numpy() * 4
            + frame.timestamp_utc.dt.minute.to_numpy() // 15
        )
        matrix[slot_index, day_index] = 1
    ax_time.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
        extent=(0, len(days), 0, 24),
    )
    starts = [pd.Timestamp(year, month, 1) for month in range(1, 13)]
    ticks = [(value - days[0]).days for value in starts]
    ax_time.set_xticks(ticks, [calendar.month_abbr[value.month] for value in starts])
    ax_time.set_yticks(range(0, 25, 4))
    ax_time.set_xlabel("giorno dell’anno")
    ax_time.set_ylabel("ora UTC")
    ax_time.set_title("Disponibilità di ciascuno slot")

    corners = domain["corners_lon_lat"]
    order = ["LL", "LR", "UR", "UL", "LL"]
    polygon = np.asarray([corners[key] for key in order])
    geographic = ccrs.PlateCarree()
    ax_map.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f3f4f6")
    ax_map.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbeafe")
    ax_map.coastlines(resolution="50m", linewidth=0.75, color="#374151")
    ax_map.add_feature(
        cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="#6b7280"
    )
    ax_map.fill(
        polygon[:, 0], polygon[:, 1], color="#ef4444", alpha=0.18,
        transform=geographic,
    )
    ax_map.plot(
        polygon[:, 0], polygon[:, 1], color="#b91c1c", lw=2, label="OPERA",
        transform=geographic,
    )
    seviri = np.asarray([[-20, 30], [40, 30], [40, 60], [-20, 60], [-20, 30]])
    ax_map.plot(
        seviri[:, 0], seviri[:, 1], color="#2563eb", lw=2, ls="--",
        label="dominio SEVIRI 20°W–40°E, 30–60°N",
        transform=geographic,
    )
    for key in order[:-1]:
        lon, lat = corners[key]
        ax_map.scatter(lon, lat, color="#b91c1c", s=28, transform=geographic)
        ax_map.text(
            lon, lat, f" {key}\n {lon:.1f}°, {lat:.1f}°", fontsize=8,
            transform=geographic,
        )
    ax_map.set_extent((-50, 65, 25, 72), crs=geographic)
    gridlines = ax_map.gridlines(
        crs=geographic, draw_labels=True, linewidth=0.4, color="#6b7280",
        alpha=0.35, linestyle="--",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    ax_map.set_title("Dominio nominale del composito OPERA")
    ax_map.legend(loc="lower right")

    expected = 96 * (366 if calendar.isleap(year) else 365)
    fig.suptitle(
        f"OPERA DBZH/QIND {year} · {len(frame):,}/{expected:,} slot "
        f"({len(frame) / expected:.1%}) · {frame.bytes.sum() / 1024**3:.2f} GiB",
        fontsize=15,
        weight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"E:\Datasets\Deep_convection\raw\opera"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/opera_2020"))
    args = parser.parse_args()

    frame = scan_archive(args.root, args.year)
    if frame.empty:
        raise SystemExit(f"No OPERA files found below {args.root / str(args.year)}")
    duplicates = int(frame.timestamp_utc.duplicated().sum())
    unique = frame.drop_duplicates("timestamp_utc", keep="last").copy()
    domain = odim_domain(Path(unique.iloc[0].path))
    monthly = monthly_summary(unique, args.year)
    expected = 96 * (366 if calendar.isleap(args.year) else 365)
    slots = pd.date_range(
        f"{args.year}-01-01", f"{args.year + 1}-01-01", freq="15min", inclusive="left"
    )
    available = pd.DatetimeIndex(unique.timestamp_utc)
    missing = slots.difference(available)
    summary = {
        "year": args.year,
        "product": "DBZH_QIND",
        "cadence_minutes": 15,
        "files": int(len(unique)),
        "duplicates": duplicates,
        "expected_slots": expected,
        "coverage_fraction": float(len(unique) / expected),
        "total_bytes": int(unique.bytes.sum()),
        "first_timestamp_utc": unique.timestamp_utc.min().isoformat(),
        "last_timestamp_utc": unique.timestamp_utc.max().isoformat(),
        "missing_slots": int(len(missing)),
        "domain": domain,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    unique.to_csv(args.out_dir / "opera_inventory.csv", index=False)
    monthly.to_csv(args.out_dir / "opera_monthly_coverage.csv", index=False)
    pd.DataFrame({"timestamp_utc": missing}).to_csv(
        args.out_dir / "opera_missing_slots.csv", index=False
    )
    (args.out_dir / "opera_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    render_plot(
        unique,
        monthly,
        domain,
        args.year,
        args.out_dir / "opera_temporal_spatial_coverage.png",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
