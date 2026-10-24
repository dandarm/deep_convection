#!/usr/bin/env python3
"""Make label-resolution and 30-case visual checks for EMMA/SEVIRI crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.seviri import RSS_CRS  # noqa: E402

NATIVE_PIXEL_METERS = 3000.403165817


def utc_naive(value: object) -> pd.Timestamp:
    time = pd.Timestamp(value)
    if time.tzinfo is not None:
        time = time.tz_convert("UTC").tz_localize(None)
    return pd.Timestamp(time.value, unit="ns")


def nearest_index(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    inserted = np.searchsorted(values, coordinate)
    left = np.clip(inserted - 1, 0, len(values) - 1)
    right = np.clip(inserted, 0, len(values) - 1)
    return np.where(
        np.abs(coordinate - values[left]) <= np.abs(values[right] - coordinate),
        left,
        right,
    )


def lookup_for_scene(x: np.ndarray, y: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    xx, yy = np.meshgrid(x, y)
    transformer = Transformer.from_crs(RSS_CRS, "EPSG:4326", always_xy=True)
    grid_lon, grid_lat = transformer.transform(xx, yy)
    valid = (
        np.isfinite(grid_lon)
        & np.isfinite(grid_lat)
        & (grid_lon >= lon.min())
        & (grid_lon <= lon.max())
        & (grid_lat >= lat.min())
        & (grid_lat <= lat.max())
    )
    row = np.zeros(valid.shape, dtype=np.int16)
    col = np.zeros(valid.shape, dtype=np.int16)
    row[valid] = nearest_index(lat, grid_lat[valid]).astype(np.int16)
    col[valid] = nearest_index(lon, grid_lon[valid]).astype(np.int16)
    return valid, row, col


def load_case_mask(emma_root: Path, target_time: pd.Timestamp, mcs_id: int, x, y, lookup_cache):
    path = emma_root / f"{target_time:%Y}" / f"{target_time:%m}" / f"tracking_{target_time:%Y%m%dT%H}.nc"
    with h5py.File(path, "r") as handle:
        ids = np.asarray(handle["mcs_id"][0])
        lat = np.asarray(handle["lat"][:])
        lon = np.asarray(handle["lon"][:])
    key = (len(x), len(y), float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    if key not in lookup_cache:
        lookup_cache[key] = lookup_for_scene(x, y, lat, lon)
    valid, row, col = lookup_cache[key]
    mask = np.zeros(valid.shape, dtype=bool)
    mask[valid] = ids[row[valid], col[valid]] == mcs_id
    return mask


def load_ir108(zarr_root: Path, timestamp: pd.Timestamp, cache: dict[Path, xr.Dataset]):
    store = zarr_root / f"{timestamp:%Y}" / f"{timestamp:%Y%m%d}.zarr"
    if store not in cache:
        cache[store] = xr.open_zarr(store, consolidated=True)
    ds = cache[store]
    indices = np.flatnonzero(pd.DatetimeIndex(ds.time.values) == np.datetime64(timestamp.to_datetime64(), "ns"))
    if len(indices) != 1:
        raise KeyError(f"SEVIRI frame unavailable: {timestamp.isoformat()}")
    channel = [str(value) for value in ds.channel.values].index("IR_108")
    return (
        np.asarray(ds.seviri.isel(time=int(indices[0]), channel=channel).values, dtype=np.float32),
        np.asarray(ds.x.values),
        np.asarray(ds.y.values),
    )


def centered_crop(mask: np.ndarray, crop_size: int = 224):
    yy, xx = np.where(mask)
    if not len(xx):
        raise ValueError("Selected EMMA object projects to zero SEVIRI pixels")
    height, width = mask.shape
    centre_y = int(round((int(yy.min()) + int(yy.max())) / 2))
    centre_x = int(round((int(xx.min()) + int(xx.max())) / 2))
    y0 = int(np.clip(centre_y - crop_size // 2, 0, max(0, height - crop_size)))
    x0 = int(np.clip(centre_x - crop_size // 2, 0, max(0, width - crop_size)))
    y1, x1 = y0 + crop_size, x0 + crop_size
    contained = yy.min() >= y0 and yy.max() < y1 and xx.min() >= x0 and xx.max() < x1
    return slice(y0, y1), slice(x0, x1), bool(contained)


def occupancy(mask: np.ndarray, output_size: int) -> np.ndarray:
    block = mask.shape[0] // output_size
    if mask.shape != (224, 224) or 224 % output_size:
        raise ValueError("Heatmap target needs a 224×224 crop and a divisor of 224")
    return mask.reshape(output_size, block, output_size, block).mean(axis=(1, 3))


def projected_bbox_extent_px(objects: pd.DataFrame) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4326", RSS_CRS, always_xy=True)
    extents = []
    for row in objects.itertuples(index=False):
        # Sample each bbox edge; using only its four corners is not safe in a
        # geostationary projection at high latitude.
        lon_bottom = np.linspace(row.lon_min, row.lon_max, 17)
        lon_top = lon_bottom
        lat_left = np.linspace(row.lat_min, row.lat_max, 17)
        lat_right = lat_left
        lon = np.concatenate([lon_bottom, lon_top, np.full(17, row.lon_min), np.full(17, row.lon_max)])
        lat = np.concatenate([np.full(17, row.lat_min), np.full(17, row.lat_max), lat_left, lat_right])
        x, y = transformer.transform(lon, lat)
        extents.append(max(np.ptp(x), np.ptp(y)) / NATIVE_PIXEL_METERS)
    return np.asarray(extents)


def plot_size_distribution(objects: pd.DataFrame, out: Path) -> dict[str, float]:
    extents = projected_bbox_extent_px(objects)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].hist(objects.n_emma_grid_cells, bins=55, color="#3b82f6", alpha=0.85)
    axes[0].set_xlabel("EMMA mask cells per object-hour [0.1° cells]")
    axes[0].set_ylabel("Object-hours")
    axes[0].set_title("All EMMA object-hour masks (n=3,150)")
    axes[1].hist(extents, bins=55, color="#f97316", alpha=0.85)
    axes[1].axvline(224, color="#dc2626", linewidth=2, label="224-pixel crop")
    axes[1].set_xlabel("Maximum projected EMMA bbox extent [SEVIRI pixels]")
    axes[1].set_ylabel("Object-hours")
    axes[1].set_title("Same masks in MSG RSS geometry")
    axes[1].legend()
    figure.savefig(out, dpi=180)
    plt.close(figure)
    return {
        "objects": int(len(objects)),
        "grid_cells_p50": float(objects.n_emma_grid_cells.quantile(0.5)),
        "grid_cells_p95": float(objects.n_emma_grid_cells.quantile(0.95)),
        "bbox_extent_px_p50": float(np.quantile(extents, 0.5)),
        "bbox_extent_px_p95": float(np.quantile(extents, 0.95)),
        "bbox_extent_px_p99": float(np.quantile(extents, 0.99)),
        "fraction_fits_224": float(np.mean(extents <= 224)),
    }


def plot_resolution_comparison(image, mask, row, out: Path) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.3), constrained_layout=True)
    axes[0].imshow(image, cmap="gray_r", vmin=190, vmax=300)
    axes[0].contour(mask, levels=[0.5], colors="#ec4899", linewidths=1.5)
    axes[0].set_title("IR10.8 + EMMA object")
    for axis, output_size in zip(axes[1:], (14, 28, 56)):
        target = occupancy(mask, output_size)
        rendered = axis.imshow(target, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        axis.contour(target, levels=[0.05], colors="white", linewidths=0.6)
        axis.set_title(f"Artificial soft target {output_size}×{output_size}\ncell {224 // output_size}×{224 // output_size} px")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(rendered, ax=axes[1:], label="EMMA occupancy fraction")
    figure.suptitle(f"{row.target_time} · track {row.track_uid} · MCS {int(row.mcs_id)}", fontsize=12)
    figure.savefig(out, dpi=200)
    plt.close(figure)


def plot_contact_sheet(cases: list[dict[str, object]], out: Path) -> None:
    figure, axes = plt.subplots(6, 5, figsize=(15, 18), constrained_layout=True)
    for axis, case in zip(axes.flat, cases):
        axis.imshow(case["image"], cmap="gray_r", vmin=190, vmax=300)
        axis.contour(case["mask"], levels=[0.5], colors="#ec4899", linewidths=0.8)
        axis.set_title(
            f"{case['target_time']:%m-%d %H} · {case['size_bin']}\n"
            f"{case['n_cells']} cells · {'contained' if case['contained'] else 'truncated'}",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(cases):]:
        axis.axis("off")
    figure.suptitle("Thirty EMMA MCS object-hours: IR10.8 at selected SEVIRI lag, EMMA contour", fontsize=14)
    figure.savefig(out, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--emma-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    objects = pd.read_parquet(args.objects)
    summary = plot_size_distribution(objects, args.out_dir / "emma_object_size_distribution_2020.png")
    selection = pd.read_csv(args.manifest)
    datasets: dict[Path, xr.Dataset] = {}
    lookup_cache = {}
    cases: list[dict[str, object]] = []
    for row in selection.itertuples(index=False):
        target_time = utc_naive(row.target_time)
        seviri_time = utc_naive(row.seviri_time)
        image, x, y = load_ir108(args.seviri_root, seviri_time, datasets)
        full_mask = load_case_mask(args.emma_root, target_time, int(row.mcs_id), x, y, lookup_cache)
        ys, xs, contained = centered_crop(full_mask)
        cases.append(
            {
                "image": image[ys, xs],
                "mask": full_mask[ys, xs],
                "target_time": target_time,
                "seviri_time": seviri_time,
                "track_uid": row.track_uid,
                "mcs_id": int(row.mcs_id),
                "size_bin": row.size_bin,
                "n_cells": int(row.n_emma_grid_cells),
                "contained": contained,
            }
        )
    for dataset in datasets.values():
        dataset.close()
    if len(cases) != 30:
        raise RuntimeError(f"Expected 30 cases, found {len(cases)}")
    comparison = next(case for case in cases if case["size_bin"] == "large")
    row = selection[selection.mcs_id == comparison["mcs_id"]].iloc[0]
    plot_resolution_comparison(
        comparison["image"], comparison["mask"], row,
        args.out_dir / "emma_heatmap_resolution_comparison.png",
    )
    plot_contact_sheet(cases, args.out_dir / "emma_training_examples_30.png")
    pd.DataFrame([{key: value for key, value in case.items() if key not in {"image", "mask"}} for case in cases]).to_csv(
        args.out_dir / "emma_training_examples_30.csv", index=False
    )
    (args.out_dir / "emma_object_size_summary_2020.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"examples={len(cases)} out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
