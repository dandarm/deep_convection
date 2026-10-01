#!/usr/bin/env python3
"""Build a reproducible manifest of random complete SEVIRI VideoMAE clips."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.seviri_clips import (  # noqa: E402
    VIDEO_MAE_CHANNELS,
    IncompleteSeviriClipError,
    SeviriZarrClipReader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path(
            os.environ.get(
                "SEVIRI_ZARR_ROOT",
                r"E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N",
            )
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "catalogs" / "seviri_pretraining_random_100.csv",
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--cadence-minutes", type=int, default=5)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--max-attempts", type=int, default=5000)
    return parser.parse_args()


def inventory(root: Path, crop_size: int) -> tuple[pd.DatetimeIndex, int, int]:
    """Read only Zarr metadata and return all timestamps plus common grid size."""

    stores = sorted(root.rglob("*.zarr"))
    if not stores:
        raise RuntimeError(f"No Zarr stores found below {root}")
    times: list[np.datetime64] = []
    grid_shapes: set[tuple[int, int]] = set()
    for store in stores:
        with xr.open_zarr(store, consolidated=True) as dataset:
            channels = tuple(str(value) for value in dataset.channel.values)
            if channels != VIDEO_MAE_CHANNELS:
                raise RuntimeError(
                    f"Unexpected channels in {store}: {channels}; "
                    f"expected {VIDEO_MAE_CHANNELS}."
                )
            height, width = int(dataset.sizes["y"]), int(dataset.sizes["x"])
            if height < crop_size or width < crop_size:
                raise RuntimeError(
                    f"Grid {(height, width)} in {store} is smaller than crop {crop_size}."
                )
            grid_shapes.add((height, width))
            times.extend(np.asarray(dataset.time.values).tolist())
    if len(grid_shapes) != 1:
        raise RuntimeError(f"Random sampling requires one common grid; found {grid_shapes}.")
    height, width = grid_shapes.pop()
    return pd.DatetimeIndex(times).drop_duplicates().sort_values(), height, width


def complete_start_times(
    times: pd.DatetimeIndex,
    *,
    num_frames: int,
    cadence_minutes: int,
) -> pd.DatetimeIndex:
    """Return starts for which every native-cadence frame exists exactly."""

    cadence_ns = cadence_minutes * 60 * 1_000_000_000
    available = set(int(value) for value in times.asi8)
    starts = [
        timestamp
        for timestamp in times
        if all(
            int(timestamp.value) + frame * cadence_ns in available
            for frame in range(num_frames)
        )
    ]
    return pd.DatetimeIndex(starts)


def utc_iso(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    args = parse_args()
    if args.count < 1 or args.num_frames < 1 or args.crop_size < 1:
        raise SystemExit("count, num-frames, and crop-size must be positive")
    times, height, width = inventory(args.zarr_root, args.crop_size)
    starts = complete_start_times(
        times,
        num_frames=args.num_frames,
        cadence_minutes=args.cadence_minutes,
    )
    if starts.empty:
        raise SystemExit("No complete temporal windows are available.")

    generator = np.random.default_rng(args.seed)
    start_order = generator.permutation(len(starts)).tolist()
    accepted: list[dict[str, object]] = []
    seen: set[tuple[int, int, int]] = set()
    attempts = 0
    with SeviriZarrClipReader(
        args.zarr_root,
        num_frames=args.num_frames,
        cadence_minutes=args.cadence_minutes,
        crop_size=args.crop_size,
    ) as reader:
        while len(accepted) < args.count and attempts < args.max_attempts:
            if not start_order:
                start_order = generator.permutation(len(starts)).tolist()
            timestamp = starts[start_order.pop()]
            origin_y = int(generator.integers(0, height - args.crop_size + 1))
            origin_x = int(generator.integers(0, width - args.crop_size + 1))
            key = (int(timestamp.value), origin_y, origin_x)
            attempts += 1
            if key in seen:
                continue
            seen.add(key)
            try:
                reader.load(timestamp, origin_y=origin_y, origin_x=origin_x)
            except IncompleteSeviriClipError:
                continue
            accepted.append(
                {
                    "clip_id": f"random_{len(accepted):03d}",
                    "start_time": utc_iso(timestamp),
                    "origin_y": origin_y,
                    "origin_x": origin_x,
                    "selection": "uniform_random_complete_window_and_crop",
                    "seed": args.seed,
                }
            )

    if len(accepted) < args.count:
        raise SystemExit(
            f"Only {len(accepted)}/{args.count} valid clips after {attempts} attempts."
        )
    frame = pd.DataFrame(accepted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(
        f"Wrote {len(frame)} clips to {args.output.resolve()} from "
        f"{len(starts)} complete temporal starts on a {height}x{width} grid "
        f"(seed={args.seed}, attempts={attempts})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
