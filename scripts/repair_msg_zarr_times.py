#!/usr/bin/env python3
"""Repair time coordinates in local MSG Zarr stores from their native inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import xarray as xr
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.seviri import native_slot_start_from_name  # noqa: E402

TIME_UNITS = "seconds since 1970-01-01 00:00:00"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, action="append", required=True)
    parser.add_argument("--zarr-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Write repaired CF time coordinates")
    args = parser.parse_args()

    slots_by_day: dict[object, set[object]] = defaultdict(set)
    for root in args.raw_root:
        for path in root.rglob("*.nat"):
            slot = native_slot_start_from_name(path)
            slots_by_day[slot.date()].add(slot)

    repaired = 0
    for store in sorted(args.zarr_root.rglob("*.zarr")):
        try:
            day = __import__("datetime").datetime.strptime(store.stem, "%Y%m%d").date()
        except ValueError:
            continue
        expected = sorted(slots_by_day[day])
        dataset = xr.open_zarr(store, consolidated=True, decode_times=False)
        count = dataset.sizes["time"]
        dataset.close()
        if len(expected) != count:
            raise RuntimeError(
                f"Cannot safely repair {store}: Zarr has {count} frames, native set has {len(expected)}"
            )
        seconds = np.asarray(
            [np.datetime64(slot, "s").astype("int64") for slot in expected], dtype=np.int64
        )
        print(f"{'repair' if args.apply else 'check'} {store} frames={count}")
        if args.apply:
            group = zarr.open_group(store, mode="r+")
            group["time"][:] = seconds
            group["time"].attrs["units"] = TIME_UNITS
            group["time"].attrs["calendar"] = "proleptic_gregorian"
            # xarray opens these stores with consolidated metadata; update the
            # metadata index after direct Zarr coordinate surgery.
            zarr.consolidate_metadata(store)
            repaired += 1
    print(f"stores={'repaired' if args.apply else 'checked'}={repaired if args.apply else len(list(args.zarr_root.rglob('*.zarr')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
