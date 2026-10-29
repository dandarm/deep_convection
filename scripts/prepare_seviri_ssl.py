#!/usr/bin/env python3
"""Validate existing temporal manifests and optionally subsample by variability."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.forecasting import ForecastWindow, variability_statistics
from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.seviri_clips import SeviriZarrClipReader
from emma_gpm.ssl_sampling import compensated_variability, stratified_indices
from scripts.pretrain_seviri_ssl import digest, validate_manifests, write_json


def prepare(args):
    training, validation = pd.read_csv(args.manifest), pd.read_csv(args.validation_manifest)
    reserved = json.loads(args.reserved_days.read_text())["days"] if args.reserved_days else []
    validate_manifests(training, validation, reserved)
    window = ForecastWindow()
    if any(path.name != ".lock" for path in args.output_dir.iterdir()):
        raise ValueError("Preparation requires an empty output directory")
    count = len(training) if args.count is None else args.count
    if not 1 <= count <= len(training):
        raise ValueError("Count must be within the candidate training pool")
    torch.set_num_threads(args.cpu_threads)
    descriptors, moments = [], []
    # No clip cache or dataset materialization; at most one raw clip in memory.
    with SeviriZarrClipReader(args.zarr_root) as reader:
        for index, row in enumerate(training.itertuples(index=False)):
            raw = reader.load(row.start_time, origin_y=int(row.origin_y), origin_x=int(row.origin_x))
            history, _ = window.split(torch.from_numpy(raw)[None])
            values = {key: float(value[0]) for key, value in variability_statistics(history).items()}
            if args.compensated:
                values["compensated_mean_abs_change_k"] = float(compensated_variability(history, args.max_motion_pixels)[0])
            descriptors.append(dict(candidate_index=index, **values))
            cube = raw.astype(np.float64)
            moments.append((cube.sum(), np.square(cube).sum(), cube.size))
            if (index + 1) % 32 == 0:
                print(f"validated training candidates {index + 1}/{len(training)}", flush=True)
        for row in validation.itertuples(index=False):
            reader.load(row.start_time, origin_y=int(row.origin_y), origin_x=int(row.origin_x))
    descriptors = pd.DataFrame(descriptors)
    score_name = "compensated_mean_abs_change_k" if args.compensated else "mean_abs_change_k"
    if args.count is None:
        indices = np.arange(len(training))
        sampling = dict(mode="unchanged", samples=len(training))
    else:
        indices, sampling = stratified_indices(descriptors[score_name].to_numpy(), count,
            uniform_fraction=args.uniform_fraction, seed=args.seed)
        sampling["mode"] = "uniform_plus_variability_stratified_without_replacement"
    selected = training.iloc[indices].reset_index(drop=True)
    selected.to_csv(args.output_dir / "train.csv", index=False)
    validation.to_csv(args.output_dir / "validation.csv", index=False)
    descriptors["selected"] = descriptors.candidate_index.isin(indices)
    descriptors.to_csv(args.output_dir / "candidate_variability.csv", index=False)
    total, squares, pixels = np.asarray(moments)[indices].sum(0)
    mean = total / pixels
    std = float(np.sqrt(max(squares / pixels - mean ** 2, 0.)))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid global training statistics")
    stats = dict(global_mean_k=float(mean), global_std_k=std, samples=len(selected), pixel_count=int(pixels),
                 normalization="single_global_affine_transform", channels=list(THERMAL_CHANNELS),
                 manifest_sha256=digest(args.output_dir / "train.csv"), zarr_root=str(args.zarr_root.resolve()),
                 scope="all pixels and channels in the selected 16-frame training clips")
    write_json(args.output_dir / "channel_stats.json", stats)
    write_json(args.output_dir / "protocol.json", dict(sampling=sampling, score=score_name,
        source_training_sha256=digest(args.manifest), source_validation_sha256=digest(args.validation_manifest),
        training_sha256=digest(args.output_dir / "train.csv"), validation_sha256=digest(args.output_dir / "validation.csv"),
        embargo_hours=24, reserved_days=reserved, historical_frames=window.history_frames,
        max_motion_pixels=args.max_motion_pixels, teacher_labels_used=False, validation_sampling="unchanged",
        preparation_source_sha256=digest(__file__)))
    print(f"Prepared {len(selected)} training and {len(validation)} validation clips in {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Existing training-only candidate manifest")
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--zarr-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reserved-days", type=Path)
    parser.add_argument("--count", type=int, help="Omit to retain all candidates in original order")
    parser.add_argument("--uniform-fraction", type=float, default=.5)
    parser.add_argument("--compensated", action="store_true")
    parser.add_argument("--max-motion-pixels", type=int, default=12)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.max_motion_pixels < 0 or not 0 < args.uniform_fraction <= 1:
        parser.error("Invalid threads, motion bound or uniform fraction")
    with exclusive_file_lock(args.output_dir / ".lock"):
        prepare(args)


if __name__ == "__main__":
    main()
