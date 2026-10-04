#!/usr/bin/env python3
"""Crop native EUMETSAT SEVIRI RSS files and retain seven thermal channels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from numcodecs import Blosc
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.seviri import (  # noqa: E402
    RSS_CRS,
    THERMAL_CHANNELS,
    geographic_bbox_to_geos_bounds,
    load_native_crop,
    native_slot_start_from_name,
)

TIME_ENCODING = {
    "dtype": "int64",
    "units": "seconds since 1970-01-01 00:00:00",
    "calendar": "proleptic_gregorian",
}


def output_dataset(crops, dtype: str) -> xr.Dataset:
    first = crops[0]
    data = np.stack([crop.data for crop in crops]).astype(dtype)
    dataset = xr.Dataset(
        data_vars={"seviri": (("time", "channel", "y", "x"), data)},
        coords={
            "time": np.asarray([np.datetime64(crop.start_time) for crop in crops]),
            "channel": list(first.channels),
            "y": first.y,
            "x": first.x,
        },
        attrs={
            "title": "Locally cropped EUMETSAT SEVIRI Rapid Scanning Service",
            "source_collection": "EO:EUM:DAT:MSG:MSG15-RSS",
            "source_format": "EUMETSAT native L1.5",
            "calibration": first.calibration,
            "units": first.units,
            "projection_wkt": RSS_CRS.to_wkt(),
            "channels_retained": " ".join(first.channels),
            "airmass_rgb_created": False,
        },
    )
    # Keep the encoding identical for the initial write and every append.
    # Without an explicit unit xarray may create a one-frame store in ``days``
    # and subsequently append a minute offset as a day count.
    dataset.time.encoding.update(TIME_ENCODING)
    return dataset


def append_batch(store: Path, crops, dtype: str) -> None:
    dataset = output_dataset(crops, dtype).sortby("time")
    if store.exists():
        dataset.to_zarr(store, mode="a", append_dim="time", consolidated=True)
        return
    store.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(
        store,
        mode="w",
        consolidated=True,
        zarr_format=2,
        encoding={
            "seviri": {
                "dtype": dtype,
                "chunks": (1, len(THERMAL_CHANNELS), 224, 224),
                "compressor": Blosc(
                    cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE
                ),
            },
            "time": TIME_ENCODING,
        },
    )


def existing_times(store: Path) -> set[pd.Timestamp]:
    if not store.exists():
        return set()
    with xr.open_zarr(store, consolidated=True) as dataset:
        return set(pd.DatetimeIndex(dataset.time.values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        choices=["counts", "radiance", "brightness_temperature"],
        default="brightness_temperature",
        help=(
            "Physical representation retained in Zarr. The default explicitly "
            "matches the implicit Satpy choice used by DeMeTra for IR/WV bands."
        ),
    )
    parser.add_argument("--lon-min", type=float, default=-20.0)
    parser.add_argument("--lon-max", type=float, default=40.0)
    parser.add_argument("--lat-min", type=float, default=30.0)
    parser.add_argument("--lat-max", type=float, default=70.0)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    paths = sorted(args.input.rglob("*.nat"))
    if not paths:
        raise SystemExit(f"No native files found below {args.input}")
    bounds = geographic_bbox_to_geos_bounds(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max
    )
    dtype = "uint16" if args.calibration == "counts" else "float16"
    pending = defaultdict(list)
    processed = 0
    skipped = 0
    stores_seen: set[Path] = set()
    seen_times_by_store: dict[Path, set[pd.Timestamp]] = {}

    for path in paths:
        slot_start = native_slot_start_from_name(path)
        store = args.out / f"{slot_start:%Y}" / f"{slot_start:%Y%m%d}.zarr"
        stores_seen.add(store)
        seen_times = seen_times_by_store.setdefault(store, existing_times(store))
        if pd.Timestamp(slot_start) in seen_times:
            skipped += 1
            continue
        crop = load_native_crop(path, bounds, calibration=args.calibration)
        if crop.start_time != slot_start:
            raise RuntimeError(
                f"Filename/Satpy time mismatch for {path}: {slot_start} != {crop.start_time}"
            )
        pending[store].append(crop)
        seen_times.add(pd.Timestamp(slot_start))
        if len(pending[store]) >= args.batch_size:
            append_batch(store, pending.pop(store), dtype)
            processed += args.batch_size
            print(f"appended {store} processed={processed}", flush=True)

    for store, crops in pending.items():
        append_batch(store, crops, dtype)
        processed += len(crops)
        print(f"appended {store} processed={processed}", flush=True)

    for store in sorted(stores_seen):
        dataset = xr.open_zarr(store, consolidated=True)
        times = pd.DatetimeIndex(dataset.time.values).sort_values()
        expected = pd.date_range(times.min(), times.max(), freq="5min")
        missing = expected.difference(times)
        size_bytes = sum(path.stat().st_size for path in store.rglob("*") if path.is_file())
        summary = {
            "path": str(store),
            "frame_count": len(times),
            "first_time": times.min().isoformat(),
            "last_time": times.max().isoformat(),
            "internal_missing_times": [value.isoformat() for value in missing],
            "shape": [int(value) for value in dataset.seviri.shape],
            "dtype": str(dataset.seviri.dtype),
            "calibration": args.calibration,
            "units": dataset.attrs["units"],
            "size_bytes": size_bytes,
            "requested_bbox": {
                "lon_min": args.lon_min,
                "lon_max": args.lon_max,
                "lat_min": args.lat_min,
                "lat_max": args.lat_max,
            },
        }
        store.with_suffix(".json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"verified {store} frames={len(times)} missing={len(missing)} "
            f"size_gib={size_bytes / 1024**3:.2f}",
            flush=True,
        )
    print(f"done processed={processed} skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
