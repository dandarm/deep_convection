#!/usr/bin/env python3
"""Plot a co-temporal MSG/SEVIRI IR scene with DeepStorm-like targets and EMMA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from satpy import Scene


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path
from emma_gpm.emma import load_emma_mask, read_grid


def nearest_emma_path(root: Path, timestamp: pd.Timestamp) -> Path:
    hour = timestamp.round("h")
    path = root / f"{hour:%m}" / f"tracking_{hour:%Y%m%dT%H}.nc"
    if not path.exists():
        raise FileNotFoundError(f"No EMMA mask at nearest hour: {path}")
    return path


def pixel_coordinates(area, longitude: np.ndarray, latitude: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = area.get_array_coordinates_from_lonlat(longitude, latitude)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    return x, y, valid


def crop_bounds(area, longitude: np.ndarray, latitude: np.ndarray, pad_degrees: float) -> tuple[slice, slice]:
    lon_min, lon_max = longitude.min() - pad_degrees, longitude.max() + pad_degrees
    lat_min, lat_max = latitude.min() - pad_degrees, latitude.max() + pad_degrees
    corners_lon = np.array([lon_min, lon_min, lon_max, lon_max])
    corners_lat = np.array([lat_min, lat_max, lat_min, lat_max])
    x, y, valid = pixel_coordinates(area, corners_lon, corners_lat)
    if valid.sum() != 4:
        raise ValueError("Selected target domain is outside the SEVIRI scene")
    x0, x1 = np.clip([np.floor(x.min()), np.ceil(x.max())], 0, area.shape[1] - 1).astype(int)
    y0, y1 = np.clip([np.floor(y.min()), np.ceil(y.max())], 0, area.shape[0] - 1).astype(int)
    return slice(y0, y1 + 1), slice(x0, x1 + 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("native_scene", type=Path)
    parser.add_argument("--source-file", required=True, help="2B-CSATGPM filename from the catalog")
    parser.add_argument("--config", default="config/deepstorm_probe.json")
    parser.add_argument("--pilot-config", default="config/pilot_2020.json")
    parser.add_argument("--pad-degrees", type=float, default=4.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_config = load_config(args.config)
    pilot_config = load_config(args.pilot_config)
    catalog = resolve_project_path(ROOT, data_config["outputs"]["catalog"])
    targets = pd.read_parquet(catalog)
    targets = targets.loc[targets["source_file"] == args.source_file].copy()
    if targets.empty:
        raise ValueError(f"No rows for source file: {args.source_file}")
    target_time = pd.Timestamp(targets["observation_time"].median())

    scene = Scene(reader="seviri_l1b_native", filenames=[str(args.native_scene)])
    scene.load(["IR_108"])
    channel = scene["IR_108"]
    area = channel.attrs["area"]
    image = np.asarray(channel.values, dtype=float)
    row_slice, col_slice = crop_bounds(
        area,
        targets["longitude"].to_numpy(),
        targets["latitude"].to_numpy(),
        args.pad_degrees,
    )
    cropped = image[row_slice, col_slice]
    target_x, target_y, target_valid = pixel_coordinates(
        area, targets["longitude"].to_numpy(), targets["latitude"].to_numpy()
    )

    emma_root = resolve_project_path(ROOT, pilot_config["emma"]["extracted_root"])
    sample_emma = next(emma_root.rglob("tracking_*.nc"))
    emma_grid = read_grid(sample_emma, pilot_config["domain"])
    emma_path = nearest_emma_path(emma_root, target_time)
    emma_ids, _ = load_emma_mask(emma_path, emma_grid)
    emma_lat, emma_lon = np.meshgrid(emma_grid.pilot_lat, emma_grid.pilot_lon, indexing="ij")
    emma_mask = emma_ids > 0
    emma_x, emma_y, emma_valid = pixel_coordinates(
        area, emma_lon[emma_mask], emma_lat[emma_mask]
    )
    in_crop_emma = (
        emma_valid
        & (emma_x >= col_slice.start)
        & (emma_x < col_slice.stop)
        & (emma_y >= row_slice.start)
        & (emma_y < row_slice.stop)
    )

    output = args.output or ROOT / "figures" / "deepstorm_msg_overlay" / (
        Path(args.source_file).stem + "_ir108_deepstorm_emma.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ir = axis.imshow(cropped, cmap="gray_r", vmin=190, vmax=300, origin="upper")
    figure.colorbar(ir, ax=axis, label="SEVIRI IR10.8 brightness temperature [K]")
    if in_crop_emma.any():
        axis.scatter(
            emma_x[in_crop_emma] - col_slice.start,
            emma_y[in_crop_emma] - row_slice.start,
            s=10,
            marker="s",
            c="#c77cff",
            alpha=0.22,
            linewidths=0,
            label="EMMA MCS cells (nearest hour)",
            zorder=2,
        )
    inside = (
        target_valid
        & (target_x >= col_slice.start)
        & (target_x < col_slice.stop)
        & (target_y >= row_slice.start)
        & (target_y < row_slice.stop)
    )
    scatter = axis.scatter(
        target_x[inside] - col_slice.start,
        target_y[inside] - row_slice.start,
        c=targets.loc[inside, "target_iwp_max_kg_m2"],
        cmap="viridis",
        norm=Normalize(vmin=0, vmax=max(1.0, float(targets["target_iwp_max_kg_m2"].max()))),
        s=48,
        edgecolors="white",
        linewidths=0.7,
        label="DeepStorm-like footprint (color: IWP)",
        zorder=4,
    )
    figure.colorbar(scatter, ax=axis, label="CloudSat IWP max [kg m$^{-2}$]")
    dc = targets["target_dc_any"].fillna(0).astype(bool).to_numpy() & inside
    if dc.any():
        axis.scatter(
            target_x[dc] - col_slice.start,
            target_y[dc] - row_slice.start,
            marker="x",
            s=110,
            c="#ff3b30",
            linewidths=2.0,
            label="CloudSat deep convection",
            zorder=5,
        )
    axis.set_title(
        "MSG/SEVIRI IR10.8 with CloudSat–GPM targets\n"
        f"DeepStorm {target_time:%Y-%m-%d %H:%M:%SZ}; EMMA {pd.Timestamp(emma_path.stem.removeprefix('tracking_')):%Y-%m-%d %H:00Z}"
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(loc="lower left", fontsize=8, framealpha=0.85)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"output={output}")
    print(f"target_rows={len(targets)} dc_positive={int(targets['target_dc_any'].sum())}")
    print(f"emma_cells_in_crop={int(in_crop_emma.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
