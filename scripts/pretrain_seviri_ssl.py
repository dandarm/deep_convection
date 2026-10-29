#!/usr/bin/env python3
"""Causal SEVIRI SSL pilot; independent of the existing VideoMAE trainer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from emma_gpm.advection import advection_forecast
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.forecasting import ForecastWindow, temporal_ablation, variability_statistics
from emma_gpm.pretraining_validation import assert_temporal_holdout
from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.seviri_clips import SeviriVideoMAEDataset, SeviriZarrClipReader
from emma_gpm.seviri_clip_files import ClipFileDataset, ClipBatchLoader
from emma_gpm.seviri_frame_dataset import FrameFileDataset
from emma_gpm.ssl_execution import restore_with_revision
from emma_gpm.ssl_runtime import (CudaRuntime, configure_runtime, make_optimizer,
    apply_optimizer_runtime, compile_model_regions)
from emma_gpm.ssl_evaluation import evaluate_ssl
from emma_gpm.ssl_masking import make_ssl_mask
from emma_gpm.ssl_model import SeviriSSL, encoder_source_config, history_config, load_encoder_weights, ssl_loss
from emma_gpm.training_checkpoint import save_checkpoint
from emma_gpm.videomae_v2 import running_cell_mask
from scripts.pretrain_videomae import (build_loader, choose_device, estimate_global_stats,
    initialize_loader_worker, iter_training_batches, load_training_stats, seed_everything)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON defaults; CLI flags override them")
    parser.add_argument("--objective", choices=("mvm", "future", "joint", "residual"), default="future")
    parser.add_argument("--model-size", choices=("small", "base"), default="small")
    parser.add_argument("--frame-cache", type=Path, help="Verified lossless daily frame files")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--zarr-root", type=Path, default=Path(os.environ.get("SEVIRI_ZARR_ROOT", ROOT.parent / "deep_convection_data/MSG-SEVIRI")))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-stats", type=Path)
    parser.add_argument("--clip-root", type=Path, help="Verified lossless per-clip files")
    parser.add_argument("--execution-revision", type=Path, help="Explicit checksum-bound runtime migration")
    parser.add_argument("--cuda-runtime", type=Path, help="JSON with explicit CUDA execution settings")
    parser.add_argument("--profile-epoch", type=int, help="Capture this full training epoch with Nsight")
    parser.add_argument("--variability-reference-manifest", type=Path,
                        help="Training subset defining common variability groups across dataset sizes")
    parser.add_argument("--reserved-days", type=Path, help="JSON with additional reserved UTC days under 'days'")
    parser.add_argument("--history-frames", type=int, default=10)
    parser.add_argument("--horizons-minutes", type=int, nargs="+", default=[15, 30])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--huber-delta", type=float, default=1.)
    parser.add_argument("--lambda-mvm", type=float, default=1.)
    parser.add_argument("--lambda-future", type=float, default=1.)
    parser.add_argument("--mask-ratio", type=float, default=.9)
    parser.add_argument("--mask-strategy", choices=("tube", "random", "block"), default="tube")
    parser.add_argument("--max-motion-pixels", type=int, default=12)
    parser.add_argument("--head-width", type=int, default=192)
    parser.add_argument("--encoder-from", type=Path)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--temporal-ablation", choices=("chronological", "shuffle", "last_frame"), default="chronological",
                        help="Evaluation-only encoder intervention; baselines and last frame stay fixed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--loader-batch-size", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--synthetic-smoke", action="store_true", help="Tiny encoder and generated 32px clips; engineering test only")
    early, _ = parser.parse_known_args(argv)
    if early.config:
        defaults = json.loads(early.config.read_text())
        valid = {action.dest for action in parser._actions}
        if set(defaults) - valid:
            parser.error(f"Unknown config keys: {sorted(set(defaults) - valid)}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    # JSON defaults can also supply path fields, without depending on argparse's
    # treatment of non-string defaults.
    for name in ("manifest", "validation_manifest", "zarr_root", "output_dir", "global_stats", "reserved_days", "encoder_from", "resume", "variability_reference_manifest", "clip_root", "frame_cache", "execution_revision", "cuda_runtime"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value))
    return args


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_manifests(training, validation, reserved_days=()):
    for frame in (training, validation):
        if frame.empty or not {"start_time", "origin_y", "origin_x"}.issubset(frame):
            raise ValueError("Nonempty manifests require start_time, origin_y and origin_x")
        stamps = pd.to_datetime(frame.start_time, utc=True, errors="raise")
        if stamps.isna().any() or ((stamps.astype("int64") % (5 * 60 * 10**9)) != 0).any():
            raise ValueError("Manifest timestamps must be exact five-minute UTC slots")
        for column in ("origin_y", "origin_x"):
            values = pd.to_numeric(frame[column], errors="raise").to_numpy()
            if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
                raise ValueError("Crop origins must be finite nonnegative integers")
        if frame.duplicated(["start_time", "origin_y", "origin_x"]).any():
            raise ValueError("Duplicate clips in a manifest")
    assert_temporal_holdout(training, validation, embargo_hours=24)
    # These dates are already used by the historical embedding study. Never
    # silently reuse them when its unversioned reservation JSON is absent.
    days = set(pd.date_range("2020-03-03", "2020-03-09", tz="UTC"))
    days.update(pd.to_datetime(list(reserved_days), utc=True).normalize())
    for start in pd.to_datetime(training.start_time, utc=True):
        if any(day in days for day in pd.date_range(start.normalize(), (start + pd.Timedelta(minutes=75)).normalize(), freq="D")):
            raise ValueError("Training clip intersects a reserved diagnostic day")


class SyntheticClips:
    """Deterministic translated/cooling thermal fields, never scientific data."""
    def __init__(self, count, seed, mean=0., std=1.):
        self.count, self.seed, self.mean, self.std = count, seed, mean, std

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)
        y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
        centre = torch.rand(2, generator=generator) * 6 + 10
        channels = torch.tensor([-30., -20., -2., -12., 0., 1., -8.])[:, None, None]
        frames = []
        for time_index in range(16):
            cloud = torch.exp(-((x - centre[0] - time_index * .4) ** 2 + (y - centre[1]) ** 2) / 30)
            frames.append(282 + channels - (40 + time_index * .6) * cloud)
        return {"pixel_values": (torch.stack(frames) - self.mean) / self.std}


def prepare_data(args):
    if args.synthetic_smoke:
        raw = SyntheticClips(4, args.seed)
        pixels = torch.stack([raw[index]["pixel_values"] for index in range(len(raw))]).double()
        mean, std = float(pixels.mean()), float(pixels.std(correction=0))
        stats = dict(global_mean_k=mean, global_std_k=std, synthetic=True,
                     normalization="single_global_affine_transform", channels=list(THERMAL_CHANNELS))
        return SyntheticClips(4, args.seed, mean, std), SyntheticClips(2, args.seed + 10000, mean, std), stats
    if not args.manifest or not args.validation_manifest:
        raise ValueError("Real runs require --manifest and --validation-manifest with a 24h temporal embargo")
    training, validation = pd.read_csv(args.manifest), pd.read_csv(args.validation_manifest)
    days = json.loads(args.reserved_days.read_text())["days"] if args.reserved_days else []
    validate_manifests(training, validation, days)
    reader = SeviriZarrClipReader(args.zarr_root)
    stats_path = args.global_stats
    if stats_path is None and args.resume:
        stats_path = args.output_dir / "channel_stats.json"
    if stats_path:
        stats = load_training_stats(stats_path, args.manifest, args.zarr_root)
        if stats.get("samples") != len(training) or stats.get("pixel_count") != len(training) * 16 * 7 * 224 ** 2:
            raise ValueError("Statistics must cover all training pixels")
    else:
        mean, std = estimate_global_stats(training, reader)
        stats = dict(global_mean_k=mean, global_std_k=std, samples=len(training),
                     pixel_count=len(training) * 16 * 7 * 224 ** 2, channels=list(THERMAL_CHANNELS),
                     normalization="single_global_affine_transform", manifest_sha256=digest(args.manifest),
                     zarr_root=str(args.zarr_root.resolve()), scope="all 16 frames, pixels and channels of training only")
    reader.close()
    if getattr(args, 'frame_cache', None):
        return (FrameFileDataset(args.frame_cache, args.manifest, zarr_root=args.zarr_root),
                FrameFileDataset(args.frame_cache, args.validation_manifest, zarr_root=args.zarr_root), stats)
    if getattr(args, 'clip_root', None):
        return (ClipFileDataset(args.clip_root, args.manifest, zarr_root=args.zarr_root),
                ClipFileDataset(args.clip_root, args.validation_manifest, zarr_root=args.zarr_root), stats)
    params = dict(global_mean=stats["global_mean_k"], global_std=stats["global_std_k"])
    return (SeviriVideoMAEDataset(training, reader, **params),
            SeviriVideoMAEDataset(validation, SeviriZarrClipReader(args.zarr_root), **params), stats)


def source_hashes():
    files = [Path(__file__), ROOT / "scripts/pretrain_videomae.py"]
    files += [ROOT / "src/emma_gpm" / name for name in (
        "forecasting.py", "advection.py", "ssl_model.py", "ssl_masking.py", "ssl_evaluation.py",
        "forecast_metrics.py", "videomae.py", "videomae_v2.py", "seviri_clips.py", "training_checkpoint.py",
        "seviri_clip_files.py", "seviri_frame_cache.py", "seviri_frame_dataset.py", "ssl_execution.py", "ssl_runtime.py")]
    return {str(path.relative_to(ROOT)): digest(path) for path in files}


def reference_variability(training, reference_path):
    """Use descriptors measured on an explicitly shared subset of training clips."""
    keys = ["start_time", "origin_y", "origin_x"]
    reference = pd.read_csv(reference_path)
    if reference.empty or not set(keys).issubset(reference) or "mean_abs_change_k" not in training:
        raise ValueError("Variability reference requires prepared training clips")
    training = training.copy()
    reference = reference[keys].copy()
    for frame in (training, reference):
        frame["start_time"] = pd.to_datetime(frame.start_time, utc=True)
    selected = reference.merge(training[keys + ["mean_abs_change_k"]], on=keys,
                               how="left", validate="one_to_one")
    scores = selected.mean_abs_change_k.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Variability reference must be a subset of the training manifest with valid descriptors")
    return scores


def run(args):
    if (min(args.epochs, args.batch_size, args.gradient_accumulation, args.prefetch_factor, args.cpu_threads, args.head_width) < 1
            or args.num_workers < 0 or args.max_motion_pixels < 0 or args.learning_rate <= 0 or args.weight_decay < 0
            or not 0 < args.mask_ratio < 1 or args.lambda_mvm < 0 or args.lambda_future < 0 or args.huber_delta <= 0):
        raise ValueError("Invalid training parameters")
    if args.loader_batch_size < 0 or (args.loader_batch_size and args.batch_size % args.loader_batch_size):
        raise ValueError("Loader batch size must divide training batch size")
    if args.temporal_ablation != "chronological" and not args.evaluate_only:
        raise ValueError("Temporal ablations are explicit evaluation-only interventions")
    if args.evaluate_only and not args.resume:
        raise ValueError("Evaluation requires --resume CHECKPOINT")
    if args.clip_root and args.frame_cache:
        raise ValueError("Choose one input backend")
    if (args.clip_root or args.frame_cache) and (args.synthetic_smoke or args.num_workers < 1 or args.loader_batch_size not in (0, args.batch_size)):
        raise ValueError("Clip-file loading requires real data, worker threads and complete batches")
    if args.synthetic_smoke and (args.manifest or args.validation_manifest or args.global_stats or args.encoder_from or args.variability_reference_manifest):
        raise ValueError("Synthetic tests cannot use real manifests, statistics or pretrained weights")
    active_weights = ([args.lambda_mvm] if args.objective == "mvm" else
                      [args.lambda_mvm, args.lambda_future] if args.objective == "joint" else [args.lambda_future])
    if not any(weight > 0 for weight in active_weights):
        raise ValueError("At least one active loss weight must be positive")
    window = ForecastWindow(args.history_frames, tuple(args.horizons_minutes))
    if args.variability_reference_manifest and args.history_frames != 10:
        raise ValueError("Prepared variability references require the ten-frame history")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    device = choose_device(args.device)
    runtime = CudaRuntime(**json.loads(args.cuda_runtime.read_text())) if args.cuda_runtime else CudaRuntime()
    configure_runtime(runtime, device)
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 requires a compatible CUDA device")
    existing = [path for path in args.output_dir.iterdir() if path.name != ".lock"]
    if existing and not args.resume:
        raise ValueError("Output directory is not empty; use a new directory or --resume")
    training, validation, stats = prepare_data(args)
    mean, std = stats["global_mean_k"], stats["global_std_k"]
    validation_generator = torch.Generator().manual_seed(314159)
    if args.clip_root or args.frame_cache:
        loader_options = dict(batch_size=args.batch_size, workers=args.num_workers,
            prefetch=args.prefetch_factor, mean=mean, std=std, device=device)
        loader = ClipBatchLoader(training, seed=args.seed, shuffle=True, **loader_options)
        val_loader = ClipBatchLoader(validation, seed=314159, shuffle=False, **loader_options)
        validation_generator = val_loader.generator
    else:
        loader = build_loader(training, args, device)
        workers = min(args.num_workers, 4)
        val_loader = torch.utils.data.DataLoader(validation, batch_size=args.batch_size, shuffle=False,
            num_workers=workers, pin_memory=loader.pin_memory, generator=validation_generator,
            **(dict(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=args.prefetch_factor,
                    worker_init_fn=initialize_loader_worker) if workers else {}))
    overrides = dict(image_size=32, patch_size=8, hidden_size=48, num_hidden_layers=1,
                     num_attention_heads=3, intermediate_size=96, decoder_hidden_size=24,
                     decoder_num_hidden_layers=1, decoder_num_attention_heads=3, decoder_intermediate_size=48) if args.synthetic_smoke else {}
    config = encoder_source_config(args.encoder_from, window) if args.encoder_from else history_config(window, model_size=args.model_size, **overrides)
    model = SeviriSSL(config, len(window.steps), args.objective, args.head_width, args.freeze_encoder).to(device)
    initialization = None
    if args.encoder_from:
        # Trusted local checkpoint paths only; never download model weights.
        initialization = load_encoder_weights(model.encoder, args.encoder_from)
    optimizer = make_optimizer(model, runtime, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    identity = dict(window=asdict(window), model_config=model.config.to_dict(), objective=args.objective,
        stats=stats, train_sha256=digest(args.manifest) if args.manifest else None,
        validation_sha256=digest(args.validation_manifest) if args.validation_manifest else None,
        variability_reference_sha256=digest(args.variability_reference_manifest) if args.variability_reference_manifest else None,
        seed=args.seed, batch_size=args.batch_size, loader_batch_size=loader.batch_size,
        gradient_accumulation=args.gradient_accumulation, num_workers=args.num_workers,
        precision=args.precision, device_type=device.type, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        loss=args.loss, huber_delta=args.huber_delta, lambda_mvm=args.lambda_mvm, lambda_future=args.lambda_future,
        mask_strategy=args.mask_strategy, mask_ratio=args.mask_ratio, head_width=args.head_width,
        max_motion_pixels=args.max_motion_pixels, frozen_encoder=args.freeze_encoder,
        synthetic=args.synthetic_smoke, initialization=initialization, sources=source_hashes())
    identity["cuda_runtime"] = runtime.to_dict()
    if args.clip_root:
        identity['input_pipeline'] = dict(backend='lossless_clip_files',
            root=str(args.clip_root.resolve()), metadata_sha256=training.metadata_sha256,
            normalization_device=device.type, prefetch=args.prefetch_factor,
            execution_revision_sha256=digest(args.execution_revision) if args.execution_revision else None)
    if args.frame_cache:
        identity['input_pipeline'] = dict(backend='lossless_native_frame_files',
            root=str(args.frame_cache.resolve()), metadata_sha256=training.metadata_sha256,
            normalization_device=device.type, prefetch=args.prefetch_factor)
    generators = dict(loader=loader.generator, sampler=loader.sampler.generator, validation=validation_generator,
                      mask=torch.Generator().manual_seed(args.seed + 1))
    progress = dict(completed_epochs=0, optimizer_steps=0, best_score=None, epochs=[], variability_edges_k=None)
    if args.resume:
        progress = restore_with_revision(args.resume, model, optimizer, scheduler, generators, identity, args.execution_revision)
    apply_optimizer_runtime(optimizer, runtime)
    compiled_regions = compile_model_regions(model, runtime)
    if not args.evaluate_only and args.epochs < progress["completed_epochs"]:
        raise ValueError("Requested epochs cannot precede the completed checkpoint epoch")
    if args.evaluate_only:
        report = evaluate_ssl(model, val_loader, window, device, mean, std, mask_ratio=args.mask_ratio,
            mask_strategy=args.mask_strategy, precision=args.precision, max_shift=args.max_motion_pixels,
            ablation=args.temporal_ablation, variability_edges=progress["variability_edges_k"],
            example_path=args.output_dir / f"evaluation_{args.temporal_ablation}.png")
        write_json(args.output_dir / f"evaluation_{args.temporal_ablation}.json", report)
        return report
    if progress["variability_edges_k"] is None:
        # Streaming descriptors use history only; radiometric statistics retain
        # all training pixels for compatibility with the MVM normalization.
        changes = []
        if args.variability_reference_manifest:
            changes = reference_variability(training.manifest, args.variability_reference_manifest)
        elif (window.history_frames == 10 and hasattr(training, "manifest")
                and "mean_abs_change_k" in training.manifest):
            # Frozen preparation already measured this exact raw-Kelvin
            # historical descriptor. It never uses validation observations.
            changes = training.manifest.mean_abs_change_k.to_numpy(dtype=float)
            if not np.isfinite(changes).all() or (changes < 0).any():
                raise ValueError("Invalid prepared training variability descriptors")
        else:
            for batch in torch.utils.data.DataLoader(training, batch_size=args.batch_size):
                history, _ = window.split(batch["pixel_values"])
                changes.extend(variability_statistics(history * std + mean)["mean_abs_change_k"].tolist())
        progress["variability_edges_k"] = np.quantile(changes, [1/3, 2/3]).tolist()
        if hasattr(training, "reader"):
            training.reader.close()
    write_json(args.output_dir / "channel_stats.json", stats)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    write_json(args.output_dir / "run_config.json", dict(arguments={key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()}, identity=identity, initialization=initialization, git_commit=commit,
        cuda_runtime=runtime.to_dict(), compiled_regions=compiled_regions,
        attention_implementation=model.encoder.config._attn_implementation,
        torch_version=torch.__version__, numpy_version=np.__version__,
        hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        data_kind="synthetic engineering smoke" if args.synthetic_smoke else "SEVIRI pilot"))
    step_log = args.output_dir / "loss_steps.csv"
    if args.resume and step_log.exists():
        logged = pd.read_csv(step_log)
        logged[logged.epoch <= progress["completed_epochs"]].to_csv(step_log, index=False)
    model.train()
    write_json(args.output_dir / "progress.json", dict(status="running", **progress))
    for epoch in range(progress["completed_epochs"], args.epochs):
        capturing = args.profile_epoch == epoch + 1
        if capturing:
            torch.cuda.synchronize(device)
            torch.cuda.profiler.start()
            torch.cuda.nvtx.range_push('MEASURED_TRAINING_WINDOW')
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        totals = {"total": 0., "mvm": 0., "future": 0.}
        samples = 0
        optimizer.zero_grad(set_to_none=True)
        micro_steps = (len(training) + args.batch_size - 1) // args.batch_size
        for step, batch in enumerate(iter_training_batches(loader, args.batch_size), start=1):
            history, target = window.split(batch["pixel_values"].to(device, non_blocking=loader.pin_memory))
            mask = decode = adv = None
            if model.mae is not None:
                mask = make_ssl_mask(config, len(history), args.mask_ratio, args.mask_strategy,
                                     generator=generators["mask"], device=device)
                decode = running_cell_mask(config, len(history), .5, device="cpu", generator=generators["mask"]).to(device)
            if args.objective == "residual":
                adv, _, _ = advection_forecast(history, window.steps, args.max_motion_pixels)
            if compiled_regions:
                torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                output = model(history, mask=mask, decode_mask=decode, advection=adv)
                loss, terms = ssl_loss(output, target, loss=args.loss, huber_delta=args.huber_delta,
                                       lambda_mvm=args.lambda_mvm, lambda_future=args.lambda_future)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            group_start = ((step - 1) // args.gradient_accumulation) * args.gradient_accumulation * args.batch_size
            group_samples = min(args.batch_size * args.gradient_accumulation, len(training) - group_start)
            (loss * len(history) / group_samples).backward()
            if step % args.gradient_accumulation == 0 or step == micro_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                progress["optimizer_steps"] += 1
            values = {"total": float(loss.detach()), **{name: float(value.detach()) for name, value in terms.items()}}
            for name, value in values.items():
                totals[name] += value * len(history)
            samples += len(history)
            row = dict(epoch=epoch + 1, step=step, samples=len(history), loss=values["total"],
                       mvm_loss=values.get("mvm"), future_loss=values.get("future"))
            pd.DataFrame([row]).to_csv(step_log, mode="a", header=not step_log.exists(), index=False)
            if step % 16 == 0 or step == micro_steps:
                write_json(args.output_dir / "progress.json", dict(status="training", current_epoch=epoch+1,
                    current_step=step, steps_per_epoch=micro_steps, **progress))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if capturing:
            torch.cuda.nvtx.range_pop()
            torch.cuda.profiler.stop()
        write_json(args.output_dir / "progress.json", dict(status="validating", current_epoch=epoch+1, **progress))
        validation_started = time.perf_counter()
        report = evaluate_ssl(model, val_loader, window, device, mean, std, mask_ratio=args.mask_ratio,
            mask_strategy=args.mask_strategy, precision=args.precision, max_shift=args.max_motion_pixels,
            variability_edges=progress["variability_edges_k"], example_path=args.output_dir / "forecast_latest.png")
        score = report["mvm_mse"] if args.objective == "mvm" else report["neural"]["mse"]
        improved = progress["best_score"] is None or score < progress["best_score"]
        if improved:
            progress["best_score"] = score
        metrics = dict(epoch=epoch + 1, **{f"train_{key}": value / samples for key, value in totals.items()},
                       validation_score=score, train_seconds=elapsed, clips_per_second=samples / elapsed,
                       validation_mvm=report["mvm_mse"],
                       validation_future=report.get("neural", {}).get("mse"),
                       validation_rmse_k=report.get("neural", {}).get("rmse_k"),
                       persistence_rmse_k=report["persistence"]["rmse_k"],
                       advection_rmse_k=report["advection"]["rmse_k"],
                       validation_seconds=time.perf_counter()-validation_started,
                       train_batches=micro_steps, validation_batches=len(val_loader),
                       batch_size=args.batch_size,
                       effective_batch_size=args.batch_size * args.gradient_accumulation,
                       optimizer_steps=progress["optimizer_steps"],
                       peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                       peak_gpu_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None)
        progress["completed_epochs"] = epoch + 1
        progress["epochs"].append(metrics)
        scheduler.step()
        # Publish recoverable state before convenience logs and figures. An
        # interrupted epoch is replayed from the previous committed checkpoint.
        saving_started = time.perf_counter()
        if improved:
            save_checkpoint(args.output_dir / "checkpoint_best.pt", model, optimizer, scheduler, generators, progress, identity)
            write_json(args.output_dir / "validation_best.json", report)
        save_checkpoint(args.output_dir / "checkpoint_latest.pt", model, optimizer, scheduler, generators, progress, identity)
        write_json(args.output_dir / "validation_latest.json", report)
        write_json(args.output_dir / f"validation_epochs/epoch_{epoch+1:03d}.json", report)
        write_json(args.output_dir / "progress.json", dict(status="running", **progress))
        pd.DataFrame(progress["epochs"]).to_csv(args.output_dir / "loss_epochs.csv", index=False)
        write_json(args.output_dir / f"timing_epochs/epoch_{epoch+1:03d}.json", dict(
            epoch=epoch+1, train_seconds=elapsed, validation_seconds=metrics['validation_seconds'],
            checkpoint_and_logs_seconds=time.perf_counter()-saving_started,
            epoch_wall_seconds=time.perf_counter()-started))
        print(json.dumps(metrics), flush=True)
    model.encoder.save_pretrained(args.output_dir / "encoder")
    write_json(args.output_dir / "progress.json", dict(status="completed", **progress))
    for dataset in (training, validation):
        if hasattr(dataset, "reader"):
            dataset.reader.close()
        elif hasattr(dataset, "close"):
            dataset.close()
    return progress


def main(argv=None):
    args = parse_args(argv)
    with exclusive_file_lock(args.output_dir / ".lock"):
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
