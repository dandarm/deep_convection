#!/usr/bin/env python3
"""Build a reproducible manifest of random complete SEVIRI VideoMAE clips."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stats-output", type=Path,
                        help="Save shared scalar moments of all accepted training pixels.")
    parser.add_argument("--starts-file", type=Path,
                        help="CSV of allowed start_time values from a temporal split.")
    return parser.parse_args()


def inventory(root: Path, crop_size: int) -> tuple[pd.DatetimeIndex, int, int]:
    """Read only Zarr metadata and return all timestamps plus common grid size."""

    stores = sorted(root.rglob("*.zarr"))
    if not stores:
        raise RuntimeError(f"No Zarr stores found below {root}")
    times: list[np.datetime64] = []
    grid_shapes: set[tuple[int, int]] = set()
    for store in stores:
        with xr.open_zarr(store, consolidated=True, chunks=None) as dataset:
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


def candidates(starts, height, width, args):
    generator = np.random.default_rng(args.seed)
    start_order = []
    seen = set()
    for _ in range(args.max_attempts):
        if not start_order:
            start_order = generator.permutation(len(starts)).tolist()
        timestamp = starts[start_order.pop()]
        y = int(generator.integers(0, height - args.crop_size + 1))
        x = int(generator.integers(0, width - args.crop_size + 1))
        key = (int(timestamp.value), y, x)
        if key not in seen:
            seen.add(key)
            yield (utc_iso(timestamp), y, x)


def initialize_validator(root, num_frames, cadence_minutes, crop_size, moments):
    global _validation_reader, _collect_moments
    import zarr
    from numcodecs import blosc

    zarr.config.set({"async.concurrency": 8, "threading.max_workers": 4})
    blosc.set_nthreads(1)
    _validation_reader = SeviriZarrClipReader(
        root, num_frames=num_frames, cadence_minutes=cadence_minutes, crop_size=crop_size)
    _collect_moments = moments


def validate_candidate(candidate):
    timestamp, y, x = candidate
    try:
        clip = _validation_reader.load(timestamp, origin_y=y, origin_x=x)
    except IncompleteSeviriClipError:
        return None
    if _collect_moments:
        values = clip.astype(np.float64)
        moments = (float(values.sum()), float(np.square(values).sum()), values.size)
    else:
        moments = (0.0, 0.0, 0)
    return candidate, moments


def parallel_candidates(stream, workers, init_args):
    """Bound pending reads and let workers finish gracefully on early stop."""
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize_validator, initargs=init_args) as executor:
        pending = deque()
        for _ in range(2 * workers):
            candidate = next(stream, None)
            if candidate is None:
                break
            pending.append(executor.submit(validate_candidate, candidate))
        while pending:
            yield pending.popleft().result()
            candidate = next(stream, None)
            if candidate is not None:
                pending.append(executor.submit(validate_candidate, candidate))


def main() -> int:
    args = parse_args()
    if args.count < 1 or args.num_frames < 1 or args.crop_size < 1:
        raise SystemExit("count, num-frames, and crop-size must be positive")
    if args.num_workers < 0:
        raise SystemExit("num-workers must be nonnegative")
    if args.stats_output and (args.num_frames != 16 or args.cadence_minutes != 5 or args.crop_size != 224):
        raise SystemExit("VideoMAE training statistics require 16 frames, 5-minute cadence and 224 crops")
    times, height, width = inventory(args.zarr_root, args.crop_size)
    starts = complete_start_times(
        times,
        num_frames=args.num_frames,
        cadence_minutes=args.cadence_minutes,
    )
    if args.starts_file:
        allowed = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(args.starts_file).start_time, utc=True)).tz_localize(None)
        if not allowed.is_unique or not allowed.isin(starts).all():
            raise SystemExit("Split contains duplicate or incomplete temporal starts")
        starts = starts[starts.isin(allowed)]
    if starts.empty:
        raise SystemExit("No complete temporal windows are available.")

    accepted: list[dict[str, object]] = []
    attempts = 0
    total = squared_total = 0.0
    pixel_count = 0
    init_args = (args.zarr_root, args.num_frames, args.cadence_minutes, args.crop_size,
                 bool(args.stats_output))
    results = None
    try:
        stream = candidates(starts, height, width, args)
        if args.num_workers:
            results = parallel_candidates(stream, args.num_workers, init_args)
        else:
            initialize_validator(*init_args)
            results = map(validate_candidate, stream)
        for result in results:
            attempts += 1
            if result is None:
                continue
            (timestamp, origin_y, origin_x), (clip_sum, clip_square_sum, clip_count) = result
            total += clip_sum
            squared_total += clip_square_sum
            pixel_count += clip_count
            accepted.append(
                {
                    "clip_id": f"random_{len(accepted):03d}",
                    "start_time": timestamp,
                    "origin_y": origin_y,
                    "origin_x": origin_x,
                    "selection": "uniform_random_complete_window_and_crop",
                    "seed": args.seed,
                }
            )
            if len(accepted) % 256 == 0:
                print(f"validated={len(accepted)}/{args.count} attempts={attempts}", flush=True)
            if len(accepted) == args.count:
                break
    finally:
        if args.num_workers and results is not None:
            results.close()
        else:
            _validation_reader.close()

    if len(accepted) < args.count:
        raise SystemExit(
            f"Only {len(accepted)}/{args.count} valid clips after {attempts} attempts."
        )
    frame = pd.DataFrame(accepted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    if args.stats_output:
        mean = total / pixel_count
        std = float(np.sqrt(max(squared_total / pixel_count - mean * mean, 0.0)))
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise RuntimeError("Invalid training statistics")
        stats = dict(channels=list(VIDEO_MAE_CHANNELS),
                     normalization="single_global_affine_transform",
                     global_mean_k=mean, global_std_k=std,
                     scope="all pixels, channels, and clips in the training split",
                     manifest_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                     samples=len(frame), pixel_count=pixel_count,
                     zarr_root=str(args.zarr_root.resolve()))
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(frame)} clips to {args.output.resolve()} from "
        f"{len(starts)} complete temporal starts on a {height}x{width} grid "
        f"(seed={args.seed}, attempts={attempts})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
