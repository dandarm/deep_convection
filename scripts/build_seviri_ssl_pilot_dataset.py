#!/usr/bin/env python3
"""Build a frozen, spatially balanced pilot from physically uploaded Zarr chunks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.seviri_clips import SeviriZarrClipReader, IncompleteSeviriClipError
from emma_gpm.ssl_dataset import (inventory_upload, available_candidates, temporal_partition,
    balanced_candidate_order, source_files_for_manifests, file_digest, verify_snapshot)
from emma_gpm.ssl_sampling import stratified_indices
from scripts.pretrain_seviri_ssl import write_json, validate_manifests
from scripts.pretrain_videomae import initialize_loader_worker


def init_worker(root):
    global READER
    initialize_loader_worker(0)
    READER = SeviriZarrClipReader(root)


def validate_candidate(row):
    try:
        raw = READER.load(row["start_time"], origin_y=row["origin_y"], origin_x=row["origin_x"])
        if raw.min() < 100 or raw.max() > 400:
            raise ValueError("Radiometry outside [100,400] K")
        difference = np.abs(np.diff(raw[:10, 4], axis=0))
        mean = float(raw.mean(dtype=np.float64))
        variance = float(raw.var(dtype=np.float64))
        return dict(**row, valid=True, mean_k=mean, variance_k2=variance,
                    mean_abs_change_k=float(difference.mean(dtype=np.float64)),
                    p90_abs_change_k=float(np.quantile(difference, .9)),
                    fraction_change_gt2k=float(np.mean(difference > 2)))
    except (IncompleteSeviriClipError, ValueError, OSError) as exc:
        return dict(**row, valid=False, rejection=str(exc))


def collect_valid(executor, candidates, count, output, split):
    accepted, rejected = [], []
    rows = candidates.to_dict("records")
    for offset in range(0, len(rows), 256):
        for result in executor.map(validate_candidate, rows[offset:offset + 256], chunksize=2):
            (accepted if result["valid"] else rejected).append(result)
        write_json(output / "progress.json", dict(status="validating", split=split,
                   accepted=len(accepted), requested=count, rejected=len(rejected)))
        print(f"{split}: {len(accepted)}/{count} valid, {len(rejected)} rejected", flush=True)
        if len(accepted) >= count:
            return pd.DataFrame(accepted[:count]), rejected
    raise ValueError(f"Only {len(accepted)} finite {split} clips; requested {count}")


def build(args):
    output = args.output_dir
    if any(p.name != ".lock" for p in output.iterdir()):
        raise ValueError("Choose an empty output directory; datasets are frozen, not silently rebuilt")
    inventory = inventory_upload(args.zarr_root, args.stable_seconds)
    write_json(output / "inventory.json", inventory)
    candidates = available_candidates(inventory, args.stride)
    train_pool, val_pool, boundary = temporal_partition(candidates, args.validation_hours)
    train_pool = balanced_candidate_order(train_pool, args.seed)
    val_pool = balanced_candidate_order(val_pool, args.seed + 1)
    # Bind every candidate we might read before validation. Adding other chunks
    # during the upload is harmless; replacing a selected chunk is not.
    paths = source_files_for_manifests(inventory, [train_pool, val_pool])
    hashes = {name: file_digest(args.zarr_root / name) for name in paths}
    print(f"Frozen {len(paths)} source files; candidates train={len(train_pool)}, val={len(val_pool)}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(args.zarr_root,)) as executor:
        training, rejected_train = collect_valid(executor, train_pool, args.candidates, output, "training_pool")
        validation, rejected_val = collect_valid(executor, val_pool, args.validation_count, output, "validation")
    chosen, sampling = stratified_indices(training.mean_abs_change_k, args.count,
                                          uniform_fraction=.5, seed=args.seed)
    selected = training.iloc[chosen].reset_index(drop=True)
    validate_manifests(selected, validation)
    mean = float(selected.mean_k.mean())
    std = float(np.sqrt(np.mean(selected.variance_k2 + (selected.mean_k - mean)**2)))
    train_columns = ["start_time", "origin_y", "origin_x", "day", "time_bin", "store",
                     "mean_abs_change_k", "p90_abs_change_k", "fraction_change_gt2k"]
    selected[train_columns].to_csv(output / "train.csv", index=False)
    validation[train_columns].to_csv(output / "validation.csv", index=False)
    training["selected"] = training.index.isin(chosen)
    training.to_csv(output / "candidate_variability.csv", index=False)
    pd.DataFrame(rejected_train + rejected_val).to_csv(output / "rejections.csv", index=False)
    verify_snapshot(args.zarr_root, hashes)
    selected_paths = source_files_for_manifests(inventory, [selected, validation])
    write_json(output / "source_snapshot.json", {name: hashes[name] for name in selected_paths})
    stats = dict(global_mean_k=mean, global_std_k=std, samples=len(selected),
        pixel_count=len(selected)*16*7*224**2, normalization="single_global_affine_transform",
        channels=list(THERMAL_CHANNELS), manifest_sha256=file_digest(output / "train.csv"),
        zarr_root=str(args.zarr_root.resolve()), scope="all pixels/channels/16 frames of selected training clips")
    write_json(output / "channel_stats.json", stats)
    report = dict(status="completed", created_at_utc=datetime.now(timezone.utc).isoformat(),
        kind="small-data engineering pilot; overlapping clips are not independent events",
        train_samples=len(selected), validation_samples=len(validation),
        train_days=selected.groupby("day").size().to_dict(), validation_days=validation.groupby("day").size().to_dict(),
        train_unique_starts=int(selected.start_time.nunique()), validation_unique_starts=int(validation.start_time.nunique()),
        train_spatial_sites=len(selected[["origin_y", "origin_x"]].drop_duplicates()),
        validation_spatial_sites=len(validation[["origin_y", "origin_x"]].drop_duplicates()),
        train_last_frame=(pd.to_datetime(selected.start_time, utc=True).max()+pd.Timedelta(minutes=75)).isoformat(),
        validation_first_frame=pd.to_datetime(validation.start_time, utc=True).min().isoformat(),
        validation_boundary=boundary, embargo_hours=24, spatial_stride=args.stride,
        geometric_candidates=len(candidates), validated_training_pool=len(training),
        sampling=sampling, validation_sampling="balanced across spatial sites, days, six-hour bins; no variability weighting",
        rejected_training=len(rejected_train), rejected_validation=len(rejected_val),
        teacher_labels_used=False, seed=args.seed, source_files=len(selected_paths),
        manifests={name:file_digest(output / name) for name in ("train.csv", "validation.csv")},
        source_sha256={name:file_digest(ROOT / name) for name in (
            "src/emma_gpm/ssl_dataset.py", "scripts/build_seviri_ssl_pilot_dataset.py", "src/emma_gpm/ssl_sampling.py")})
    write_json(output / "protocol.json", report)
    write_json(output / "progress.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--candidates", type=int, default=8192)
    parser.add_argument("--validation-count", type=int, default=512)
    parser.add_argument("--validation-hours", type=int, default=12)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stable-seconds", type=float, default=5)
    parser.add_argument("--seed", type=int, default=20261024)
    args=parser.parse_args()
    if min(args.count,args.validation_count,args.workers,args.stride,args.validation_hours)<1 or args.candidates<args.count or args.stable_seconds<0:
        parser.error("Invalid count, pool size, stride, workers, holdout or stability interval")
    with exclusive_file_lock(args.output_dir / ".lock"):
        build(args)


if __name__ == "__main__":
    main()
