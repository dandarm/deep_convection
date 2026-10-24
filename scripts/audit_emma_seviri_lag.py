#!/usr/bin/env python3
"""Measure the temporal correspondence of EMMA masks and MSG RSS IR10.8.

This is deliberately a small, reproducible audit, not a training-label builder.
For each selected EMMA object-hour it compares the labelled footprint with a
nearby cloud-free-ish annulus at every available 5-minute SEVIRI lag.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.seviri import RSS_CRS  # noqa: E402


def utc_naive(value: object) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if value.tzinfo is not None:
        value = value.tz_convert("UTC").tz_localize(None)
    # Normalize resolution too: pandas 3 can preserve a generic NumPy unit
    # from CSV parsing, which emits deprecation warnings in arithmetic.
    return pd.Timestamp(value.value, unit="ns")


def nearest_index(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    """Nearest grid-centre index for a monotonic regular coordinate."""
    insert = np.searchsorted(values, coordinate)
    left = np.clip(insert - 1, 0, len(values) - 1)
    right = np.clip(insert, 0, len(values) - 1)
    return np.where(
        np.abs(coordinate - values[left]) <= np.abs(values[right] - coordinate),
        left,
        right,
    )


def build_emma_lookup(
    x: np.ndarray, y: np.ndarray, emma_lat: np.ndarray, emma_lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map the fixed SEVIRI crop grid once to nearest EMMA cells."""
    xx, yy = np.meshgrid(x, y)
    transformer = Transformer.from_crs(RSS_CRS, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (lon >= emma_lon.min())
        & (lon <= emma_lon.max())
        & (lat >= emma_lat.min())
        & (lat <= emma_lat.max())
    )
    row = np.zeros(lat.shape, dtype=np.int16)
    col = np.zeros(lon.shape, dtype=np.int16)
    row[valid] = nearest_index(emma_lat, lat[valid]).astype(np.int16)
    col[valid] = nearest_index(emma_lon, lon[valid]).astype(np.int16)
    return valid, row, col


def open_day(root: Path, timestamp: pd.Timestamp, cache: dict[Path, xr.Dataset]) -> xr.Dataset:
    store = root / f"{timestamp:%Y}" / f"{timestamp:%Y%m%d}.zarr"
    if store not in cache:
        if not store.exists():
            raise FileNotFoundError(store)
        cache[store] = xr.open_zarr(store, consolidated=True)
    return cache[store]


def ir108_at(
    root: Path, timestamp: pd.Timestamp, cache: dict[Path, xr.Dataset]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = open_day(root, timestamp, cache)
    times = pd.DatetimeIndex(ds.time.values)
    where = np.flatnonzero(times == np.datetime64(timestamp.to_datetime64(), "ns"))
    if len(where) != 1:
        raise KeyError(f"frame_not_available:{timestamp.isoformat()}")
    channels = [str(value) for value in ds.channel.values]
    index = channels.index("IR_108")
    return (
        np.asarray(ds.seviri.isel(time=int(where[0]), channel=index).values, dtype=np.float32),
        np.asarray(ds.x.values),
        np.asarray(ds.y.values),
    )


def emma_id_mask(
    root: Path,
    timestamp: pd.Timestamp,
    mcs_id: int,
    x: np.ndarray,
    y: np.ndarray,
    lookup_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    path = root / f"{timestamp:%Y}" / f"{timestamp:%m}" / f"tracking_{timestamp:%Y%m%dT%H}.nc"
    with h5py.File(path, "r") as handle:
        # EMMA's hourly files retain a singleton time dimension.
        ids = np.asarray(handle["mcs_id"][0])
        lat = np.asarray(handle["lat"][:])
        lon = np.asarray(handle["lon"][:])
    cache_key = f"{x.shape}:{y.shape}:{x[0]}:{x[-1]}:{y[0]}:{y[-1]}"
    if cache_key not in lookup_cache:
        lookup_cache[cache_key] = build_emma_lookup(x, y, lat, lon)
    valid, row, col = lookup_cache[cache_key]
    mask = np.zeros(valid.shape, dtype=bool)
    mask[valid] = ids[row[valid], col[valid]] == mcs_id
    return mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--emma-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-lag", type=int, default=-30)
    parser.add_argument("--max-lag", type=int, default=60)
    parser.add_argument("--annulus-inner-pixels", type=int, default=3)
    parser.add_argument("--annulus-outer-pixels", type=int, default=20)
    args = parser.parse_args()
    if args.min_lag > args.max_lag or args.min_lag % 5 or args.max_lag % 5:
        raise SystemExit("Lag bounds must be ordered multiples of five minutes")

    selected = pd.read_csv(args.manifest)
    lags = list(range(args.min_lag, args.max_lag + 1, 5))
    datasets: dict[Path, xr.Dataset] = {}
    lookup_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, object]] = []

    for selected_row in selected.itertuples(index=False):
        target = utc_naive(selected_row.target_time)
        record = {
            "target_time": target.isoformat(),
            "track_uid": str(selected_row.track_uid),
            "mcs_id": int(selected_row.mcs_id),
        }
        try:
            frames = []
            x = y = None
            # A clip is valid only if its *entire* temporal window is present.
            # Loading before adding rows prevents partially audited cases.
            for lag in lags:
                frame, frame_x, frame_y = ir108_at(
                    args.seviri_root, target + timedelta(minutes=lag), datasets
                )
                frames.append((lag, frame))
                if x is None:
                    x, y = frame_x, frame_y
            assert x is not None and y is not None
            mask = emma_id_mask(args.emma_root, target, int(selected_row.mcs_id), x, y, lookup_cache)
            if mask.sum() < 16:
                raise ValueError(f"too_few_projected_mask_pixels:{int(mask.sum())}")
            inner = binary_dilation(mask, iterations=args.annulus_inner_pixels)
            outer = binary_dilation(mask, iterations=args.annulus_outer_pixels)
            annulus = outer & ~inner
            for lag, frame in frames:
                finite = np.isfinite(frame)
                inside = frame[mask & finite]
                background = frame[annulus & finite]
                if len(inside) < 16 or len(background) < 16:
                    raise ValueError("insufficient_finite_ir108_pixels")
                rows.append(
                    record
                    | {
                        "status": "ok",
                        "reason": "",
                        "lag_minutes": lag,
                        "mask_pixels": int(mask.sum()),
                        "inside_median_k": float(np.median(inside)),
                        "annulus_median_k": float(np.median(background)),
                        "contrast_k": float(np.median(background) - np.median(inside)),
                        "inside_fraction_lt241k": float(np.mean(inside < 241.0)),
                        "inside_fraction_lt235k": float(np.mean(inside < 235.0)),
                    }
                )
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            rows.append(record | {"status": "skipped", "reason": str(exc), "lag_minutes": np.nan})

    for dataset in datasets.values():
        dataset.close()
    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    complete = result[result.status == "ok"].copy()
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    if not complete.empty:
        summary = (
            complete.groupby("lag_minutes", as_index=False)
            .agg(
                cases=("target_time", "nunique"),
                median_inside_k=("inside_median_k", "median"),
                median_contrast_k=("contrast_k", "median"),
                mean_fraction_lt241k=("inside_fraction_lt241k", "mean"),
                mean_fraction_lt235k=("inside_fraction_lt235k", "mean"),
            )
            .sort_values("lag_minutes")
        )
        summary.to_csv(summary_path, index=False)
        best = summary.loc[summary.median_contrast_k.idxmax()]
        print(
            f"complete_cases={complete.target_time.nunique()} skipped_cases={len(result[result.status == 'skipped'])} "
            f"best_median_contrast_lag={int(best.lag_minutes)}m contrast={best.median_contrast_k:.2f}K"
        )
    else:
        pd.DataFrame().to_csv(summary_path, index=False)
        print(f"complete_cases=0 skipped_cases={len(result)}")
    print(f"detail={args.out} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
