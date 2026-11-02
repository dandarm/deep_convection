#!/usr/bin/env python3
"""Prepare six frozen, single-GPU Leonardo runs from a verified data transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def relocated_arguments(original, dataset, output, runtime, model_size, objective):
    """Keep canonical data identities; change only their physical locations."""
    values = dict(original)
    values.update(config=None, model_size=model_size, objective=objective,
                  manifest=str(dataset / "dataset/train.csv"),
                  validation_manifest=str(dataset / "dataset/validation.csv"),
                  global_stats=str(dataset / "dataset/channel_stats.json"),
                  variability_reference_manifest=str(dataset / "dataset/train.csv"),
                  output_dir=str(output), cuda_runtime=str(runtime),
                  execution_revision=None, resume=None, profile_epoch=None)
    return values


def prepare(dataset, seed, output):
    from scripts.pretrain_seviri_ssl import source_hashes

    dataset, seed, output = [Path(p).resolve() for p in (dataset, seed, output)]
    verification = json.loads((dataset / "verification.json").read_text())
    if verification.get("status") != "verified" or verification["transfer_plan_sha256"] != digest(dataset / "transfer_plan.json"):
        raise ValueError("A completed, checksum-verified dataset transfer is required")
    transfer = json.loads((dataset / "transfer_plan.json").read_text())
    for name, entry in transfer["payload"].items():
        if digest(dataset / name) != entry["sha256"]:
            raise ValueError(f"Transferred manifest changed: {name}")
    for name, expected in json.loads((seed / "seed_manifest.json").read_text())["files"].items():
        if digest(seed / name) != expected:
            raise ValueError(f"Seed checkpoint or provenance changed: {name}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("HPC preparation requires an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((seed / "run_config.json").read_text())["arguments"]
    if original["reserved_days"] or original["encoder_from"] or original["clip_root"]:
        raise ValueError("This migration expects the frozen native-frame SSL study")
    if original["frame_cache"] != transfer["canonical_frame_root"] or original["zarr_root"] != transfer["canonical_zarr_root"]:
        raise ValueError("Training and transferred data roots differ")
    mapping = output / "storage_roots.json"
    write_json(mapping, dict(version=1, roots={
        transfer["canonical_frame_root"]: str(dataset / "frames"),
        transfer["canonical_zarr_root"]: str(dataset / "sources"),
    }, source_provenance=dict(root=str(dataset / "sources"),
        snapshot=str(dataset / "dataset/source_snapshot.json"),
        snapshot_sha256=digest(dataset / "dataset/source_snapshot.json"),
        timestamps=str(dataset / "transport_times.json"),
        timestamps_sha256=digest(dataset / "transport_times.json"))))
    archive = output / "source_archive"
    for directory in ("src", "scripts"):
        shutil.copytree(ROOT / directory, archive / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    reservation = ROOT / 'configs/ssl_validation_extended.json'
    if reservation.exists():
        (archive / 'configs').mkdir(exist_ok=True)
        shutil.copy2(reservation, archive / 'configs' / reservation.name)
    sources = source_hashes()
    for name, expected in sources.items():
        if digest(archive / name) != expected:
            raise ValueError("Source changed while freezing the HPC run")
    runtime = output / "cuda_runtime.json"
    shutil.copy2(seed / "cuda_runtime.json", runtime)
    parent = json.loads((seed / "checkpoint_metadata.json").read_text())
    checkpoint = seed / "small_n131072/mvm/checkpoint_latest.pt"
    if digest(checkpoint) != parent["checkpoint_sha256"]:
        raise ValueError("Checkpoint metadata belongs to another checkpoint")
    runs = []
    for model in ("small", "base"):
        for objective in ("mvm", "future", "joint"):
            index = len(runs)
            run_output = output / f"{model}_n131072" / objective
            config = relocated_arguments(original, dataset, run_output, runtime, model, objective)
            stop = output / "stop_requests" / f"{index}.stop"
            config["stop_after_epoch_file"] = str(stop)
            if model == "small" and objective == "mvm":
                shutil.copytree(seed / "small_n131072/mvm", run_output)
                revision = output / "execution_revision.json"
                write_json(revision, dict(version=3, reason="Move the paused run to Leonardo",
                    parent_sources=parent["identity"]["sources"],
                    parent_checkpoints={parent["checkpoint_sha256"]: str(checkpoint)}, sources=sources,
                    runtime={key: config[key] for key in ("batch_size", "loader_batch_size", "gradient_accumulation", "num_workers")},
                    cuda_runtime=json.loads(runtime.read_text()),
                    clip_metadata_sha256=digest(dataset / "frames/metadata.json"),
                    storage_mapping_sha256=digest(mapping)))
                config["execution_revision"] = str(revision)
            config_path = output / "configs" / f"{index}.json"
            write_json(config_path, config)
            runs.append(dict(index=index, model_size=model, objective=objective,
                             config=str(config_path), config_sha256=digest(config_path),
                             output=str(run_output), stop_file=str(stop)))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    frozen_files = {str(p.relative_to(archive)): digest(p) for p in archive.rglob("*") if p.is_file()}
    write_json(output / "hpc_plan.json", dict(version=1, commit=commit, runs=runs,
        source_archive=str(archive), frozen_files=frozen_files, storage_roots=str(mapping),
        storage_roots_sha256=digest(mapping), dataset=str(dataset),
        transfer_plan_sha256=verification["transfer_plan_sha256"],
        cuda_runtime_sha256=digest(runtime), initial_completed_epochs=parent["completed_epochs"],
        initial_optimizer_steps=parent["optimizer_steps"]))
    (output / "stop_requests").mkdir(exist_ok=True)
    (output / "slurm_logs").mkdir(exist_ok=True)
    print(output / "hpc_plan.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dataset_root, args.seed_root, args.output_root)


if __name__ == "__main__":
    main()
