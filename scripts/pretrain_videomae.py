#!/usr/bin/env python3
"""Self-supervised seven-channel VideoMAE pretraining on SEVIRI clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import shutil
import sys
import time
from copy import copy

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.seviri import THERMAL_CHANNELS  # noqa: E402
from emma_gpm.seviri_clips import (  # noqa: E402
    SeviriVideoMAEDataset,
    SeviriZarrClipReader,
    MaterializedSeviriDataset,
)
from emma_gpm.videomae import (  # noqa: E402
    build_videomae_7ch_from_rgb_checkpoint,
    build_videomae_small_7ch_from_scratch,
    forward_seviri_mae,
    reconstruction_values_per_tubelet,
)
from emma_gpm.training_checkpoint import save_checkpoint, restore_checkpoint
from emma_gpm.pretraining_validation import assert_temporal_holdout, evaluate_reconstruction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--manifest",
        type=Path,
        default=ROOT / "catalogs" / "seviri_pretraining_smoke.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "videomae_pretraining_smoke",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--validation-mask-ratio", type=float, default=.9)
    parser.add_argument("--validation-seed", type=int, default=314159)
    parser.add_argument("--precision", choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--warm-start", type=Path, help="Continue a legacy run's weights, optimizer and epoch count; reseed missing legacy RNG.")
    parser.add_argument("--resume", type=Path, help="Resume an atomic epoch-boundary checkpoint.")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--global-stats", type=Path,
                        help="Reuse manifest-verified scalar training statistics.")
    parser.add_argument("--sample-root", type=Path,
                        help="Read materialized Kelvin samples instead of Zarr crops.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.9)
    parser.add_argument('--model-version', choices=['v1', 'v2'], default='v1')
    parser.add_argument('--decoder-mask-ratio', type=float, choices=[.25, .5, .75], default=.5)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--loader-batch-size", type=int, default=0,
                        help="Clips per worker task; zero uses the training batch size.")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--initialization", choices=["scratch", "rgb"], default="scratch"
    )
    parser.add_argument(
        "--rgb-checkpoint",
        default="MCG-NJU/videomae-small-finetuned-kinetics",
    )
    parser.add_argument(
        "--cpu-smoke-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the real VideoMAE-Small encoder but use one 96-wide decoder "
            "layer for the CPU integration test. Disable for the full MAE."
        ),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str):
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_loader_worker(worker_id: int) -> None:
    """Keep Zarr reads local to spawned workers without nested CPU pools."""
    import zarr
    import torch
    from numcodecs import blosc

    torch.set_num_threads(1)
    zarr.config.set({"async.concurrency": 8, "threading.max_workers": 4})
    blosc.set_nthreads(1)
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_loader(dataset, args, device):
    from torch.utils.data import DataLoader, RandomSampler
    import torch

    pin_memory = device.type == "cuda" if args.pin_memory is None else args.pin_memory
    options = dict(
        batch_size=getattr(args, "loader_batch_size", 0) or args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(args.seed),
    )
    # Keep sampler RNG independent of worker creation, including after resume.
    sampler_generator = torch.Generator().manual_seed(args.seed)
    torch.empty((), dtype=torch.int64).random_(generator=sampler_generator)
    options["sampler"] = RandomSampler(dataset, generator=sampler_generator)
    if args.num_workers > 0:
        options.update(
            multiprocessing_context="spawn",
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
            worker_init_fn=initialize_loader_worker,
        )
    return DataLoader(dataset, **options)


def iter_training_batches(loader, batch_size: int):
    """Assemble small parallel I/O tasks into unchanged optimizer batches."""
    import torch

    parts = []
    count = 0

    def assemble():
        if len(parts) == 1:
            return {"pixel_values": parts[0]}
        buffer = torch.empty((count, *parts[0].shape[1:]), dtype=parts[0].dtype,
                             pin_memory=loader.pin_memory)
        torch.cat(parts, dim=0, out=buffer)
        return {"pixel_values": buffer}

    for batch in loader:
        pixels = batch["pixel_values"]
        parts.append(pixels)
        count += pixels.shape[0]
        if count == batch_size:
            yield assemble()
            parts = []
            count = 0
        elif count > batch_size:
            raise ValueError("Loader batch size must divide the training batch size.")
    if parts:
        yield assemble()


def estimate_global_stats(
    manifest: pd.DataFrame, reader: SeviriZarrClipReader
) -> tuple[float, float]:
    """Calculate one pair of moments over all training pixels and channels."""

    total = 0.0
    squared_total = 0.0
    count = 0
    for row in manifest.itertuples(index=False):
        clip = reader.load(
            row.start_time,
            origin_y=int(row.origin_y),
            origin_x=int(row.origin_x),
        )
        values = clip.astype(np.float64, copy=False)
        total += float(values.sum())
        squared_total += float(np.square(values).sum())
        count += values.size
    mean = total / count
    variance = max(squared_total / count - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    if std <= 0 or not np.isfinite(mean) or not np.isfinite(std):
        raise RuntimeError("Invalid global statistics in training manifest.")
    return float(mean), std


def make_tube_mask(config, batch_size: int, mask_ratio: float, device):
    """Mask the same spatial patch locations in every temporal tubelet."""

    import torch

    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be strictly between zero and one.")
    image_size = int(config.image_size)
    patch_size = int(config.patch_size)
    spatial_tokens = (image_size // patch_size) ** 2
    temporal_tokens = int(config.num_frames) // int(config.tubelet_size)
    masked_spatial = int(round(spatial_tokens * mask_ratio))
    masked_spatial = min(max(masked_spatial, 1), spatial_tokens - 1)
    rows = []
    for _ in range(batch_size):
        spatial = torch.zeros(spatial_tokens, dtype=torch.bool, device=device)
        selected = torch.randperm(spatial_tokens, device=device)[:masked_spatial]
        spatial[selected] = True
        rows.append(spatial.repeat(temporal_tokens))
    return torch.stack(rows)


def load_training_stats(path: Path, manifest_path: Path, zarr_root: Path) -> dict:
    stats = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if stats.get("manifest_sha256") != expected_hash:
        raise ValueError("Global statistics belong to a different manifest.")
    if stats.get("channels") != list(THERMAL_CHANNELS):
        raise ValueError("Global statistics have an unexpected channel order.")
    if Path(stats.get("zarr_root", "")).resolve() != zarr_root.resolve():
        raise ValueError("Global statistics belong to a different Zarr root.")
    if stats.get("normalization") != "single_global_affine_transform":
        raise ValueError("Expected one shared scalar affine transform.")
    mean, std = np.asarray(stats["global_mean_k"]), np.asarray(stats["global_std_k"])
    if mean.ndim or std.ndim or not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Global statistics must be finite scalars with positive std.")
    return stats


def build_model(args: argparse.Namespace):
    if getattr(args, 'model_version', 'v1') == 'v2' and args.initialization != 'scratch':
        raise ValueError('V2 experiments require scratch initialization')
    if args.initialization == "rgb":
        if args.cpu_smoke_decoder:
            raise ValueError(
                "--cpu-smoke-decoder is only available with scratch initialization. "
                "Use --no-cpu-smoke-decoder for an RGB checkpoint."
            )
        return build_videomae_7ch_from_rgb_checkpoint(args.rgb_checkpoint)
    overrides = {}
    if args.cpu_smoke_decoder:
        overrides.update(
            decoder_hidden_size=96,
            decoder_num_hidden_layers=1,
            decoder_num_attention_heads=3,
            decoder_intermediate_size=384,
        )
    if getattr(args, 'model_version', 'v1') == 'v2':
        from emma_gpm.videomae_v2 import build_videomae_v2_small_7ch_from_scratch
        model = build_videomae_v2_small_7ch_from_scratch(**overrides)
        model.config.decoder_mask_ratio = args.decoder_mask_ratio
        return model
    return build_videomae_small_7ch_from_scratch(**overrides)


def save_epoch_loss_plot(epoch_metrics: list[dict], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(epoch_metrics)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(frame["epoch"], frame["mean_loss"], marker="o")
    if 'validation_own_mse' in frame:
        axis.plot(frame['epoch'], frame['validation_own_mse'], '--', label='Validation, training mask ratio')
        axis.lines[0].set_label('Training (run mask ratio)')
        axis.legend()
    axis.set(xlabel="Epoch", ylabel="Mean training MSE (sample weighted)",
             title="VideoMAE-Small — SEVIRI integration test")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_per_epoch.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.warm_start and args.resume:
        raise SystemExit("Choose either warm-start or resume")
    if args.model_version == 'v2' and args.warm_start:
        raise SystemExit('V2 starts from scratch; legacy warm-start is not a V2 run')
    if args.checkpoint_every < 0:
        raise SystemExit("checkpoint-every must be nonnegative")
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size must be positive")
    if args.gradient_accumulation < 1 or not 0 < args.mask_ratio < 1 or not 0 < args.validation_mask_ratio < 1:
        raise SystemExit('Invalid accumulation or mask ratio')
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise SystemExit("num-workers must be nonnegative and prefetch-factor positive")
    if args.loader_batch_size < 0 or (args.loader_batch_size and args.batch_size % args.loader_batch_size):
        raise SystemExit("loader-batch-size must be zero or a positive divisor of batch-size")
    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "Training dependencies are missing. Install requirements-training.txt."
        ) from error

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    device = choose_device(args.device)
    if args.precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise SystemExit('bf16 requires a supported CUDA device')
    manifest = pd.read_csv(args.manifest)
    if manifest.empty:
        raise SystemExit(f"Empty clip manifest: {args.manifest}")

    reader = SeviriZarrClipReader(args.zarr_root)
    persisted_stats = None
    if args.global_stats:
        persisted_stats = load_training_stats(args.global_stats, args.manifest, args.zarr_root)
        expected_pixels = len(manifest) * 16 * 7 * 224 * 224
        if persisted_stats.get("samples") != len(manifest) or persisted_stats.get("pixel_count") != expected_pixels:
            raise ValueError("Global statistics do not cover every training pixel.")
        mean, std = persisted_stats["global_mean_k"], persisted_stats["global_std_k"]
    else:
        mean, std = estimate_global_stats(manifest, reader)
    # Spawned workers reopen their own stores; never serialize open Zarr handles.
    reader.close()
    if args.sample_root:
        dataset = MaterializedSeviriDataset(
            args.sample_root, args.manifest, zarr_root=args.zarr_root,
            global_mean=mean, global_std=std)
    else:
        dataset = SeviriVideoMAEDataset(
            manifest, reader, global_mean=mean, global_std=std)
    loader = build_loader(dataset, args, device)
    validation_loader = None
    validation_reader = None
    if args.validation_manifest:
        validation_manifest = pd.read_csv(args.validation_manifest)
        assert_temporal_holdout(manifest, validation_manifest)
        validation_reader = SeviriZarrClipReader(args.zarr_root)
        validation_dataset = SeviriVideoMAEDataset(validation_manifest, validation_reader, global_mean=mean, global_std=std)
        val_args = copy(args)
        val_args.batch_size = min(args.batch_size, 8)
        val_args.loader_batch_size = val_args.batch_size
        val_args.num_workers = min(args.num_workers, 4)
        from torch.utils.data import SequentialSampler
        # Evaluation index is manifest order, independent of worker scheduling.
        validation_loader = torch.utils.data.DataLoader(validation_dataset,
            batch_size=val_args.batch_size, sampler=SequentialSampler(validation_dataset),
            num_workers=val_args.num_workers, pin_memory=loader.pin_memory,
            generator=torch.Generator().manual_seed(args.validation_seed),
            **(dict(multiprocessing_context='spawn', persistent_workers=True,
                    prefetch_factor=2, worker_init_fn=initialize_loader_worker) if val_args.num_workers else {}))

    model = build_model(args).to(device)
    if model.config.num_channels != 7:
        raise RuntimeError(f"Model has {model.config.num_channels} channels, expected 7.")
    if reconstruction_values_per_tubelet(model.config) != 3584:
        raise RuntimeError("VideoMAE decoder target width is not 3584.")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    if args.warm_start:
        from transformers import VideoMAEForPreTraining
        legacy_stats = json.loads((args.warm_start / "channel_stats.json").read_text())
        if (legacy_stats.get("manifest_sha256") != hashlib.sha256(args.manifest.read_bytes()).hexdigest()
                or legacy_stats["global_mean_k"] != mean or legacy_stats["global_std_k"] != std):
            raise ValueError("Legacy warm-start manifest or normalization differs.")
        legacy_model = VideoMAEForPreTraining.from_pretrained(args.warm_start / "model")
        model.load_state_dict(legacy_model.state_dict())
        del legacy_model
        optimizer.load_state_dict(torch.load(args.warm_start / "optimizer.pt", map_location="cpu", weights_only=False))
    identity = dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    zarr_root=str(args.zarr_root.resolve()), sample_root=str(args.sample_root.resolve()) if args.sample_root else None,
                    mean=mean, std=std, config=model.config.to_dict(), batch_size=args.batch_size,
                    loader_batch_size=loader.batch_size, num_workers=args.num_workers,
                    mask_ratio=args.mask_ratio, seed=args.seed, learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay, precision=args.precision)
    if args.gradient_accumulation != 1:
        identity['gradient_accumulation'] = args.gradient_accumulation
    if args.validation_manifest:
        identity.update(validation_sha256=hashlib.sha256(args.validation_manifest.read_bytes()).hexdigest(),
                        validation_mask_ratio=args.validation_mask_ratio, validation_seed=args.validation_seed)
    start_epoch = 0
    elapsed_before = 0.0

    losses: list[float] = []
    epoch_metrics: list[dict] = []
    if args.warm_start:
        legacy = json.loads((args.warm_start / "smoke_summary.json").read_text())
        start_epoch = legacy["epochs"]
        losses = legacy["losses"]
        epoch_metrics = legacy["epoch_metrics"]
        elapsed_before = legacy["elapsed_seconds"]
        if args.epochs <= start_epoch:
            raise ValueError("Target epochs must exceed the legacy completed epoch count.")
    if args.resume:
        progress = restore_checkpoint(args.resume, model, optimizer, scheduler,
                                      {"loader": loader.generator, "sampler": loader.sampler.generator}, identity)
        start_epoch = progress["completed_epochs"]
        losses = progress["losses"]
        epoch_metrics = progress["epoch_metrics"]
        elapsed_before = progress["elapsed_seconds"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step_log = args.output_dir / "loss_steps.csv"
    epoch_log = args.output_dir / "loss_epochs.csv"
    if args.warm_start:
        for path in [step_log, epoch_log]:
            shutil.copyfile(args.warm_start / path.name, path)
    elif not args.resume:
        pd.DataFrame(columns=["epoch", "step", "samples", "loss", "data_wait_seconds",
                          "train_step_seconds", "wall_started", "wall_finished"]).to_csv(step_log, index=False)
        pd.DataFrame(columns=["epoch", "mean_loss", "elapsed_seconds", "clips_per_second",
                          "data_wait_seconds", "train_step_seconds", "wall_started", "wall_finished"] +
                          (['validation_mse', 'validation_rmse_k', 'validation_seconds', 'validation_own_mse'] if validation_loader else [])).to_csv(epoch_log, index=False)
    else:
        for path in [step_log, epoch_log]:
            frame = pd.read_csv(path)
            if path == epoch_log and validation_loader and 'validation_own_mse' not in frame:
                frame['validation_own_mse'] = np.nan
            frame[frame.epoch <= start_epoch].to_csv(path, index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    (args.output_dir / "run_config.json").write_text(json.dumps(
        dict(arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
             identity=identity, git_commit=commit, trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             model_sources_sha256={name:hashlib.sha256((ROOT/'src/emma_gpm'/name).read_bytes()).hexdigest()
                for name in (['videomae.py','videomae_v2.py'] if args.model_version=='v2' else ['videomae.py'])}), indent=2)+"\n")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    best_validation = min((m.get('validation_mse', float('inf')) for m in epoch_metrics), default=float('inf'))
    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        epoch_wall_started = time.time()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        data_wait_total = 0.0
        train_step_total = 0.0
        previous_step_finished = epoch_started
        optimizer.zero_grad(set_to_none=True)
        micro_steps = (len(dataset) + args.batch_size - 1) // args.batch_size
        for step, batch in enumerate(iter_training_batches(loader, args.batch_size), start=1):
            step_started = time.perf_counter()
            step_wall_started = time.time()
            data_wait = step_started - previous_step_finished
            pixel_values = batch["pixel_values"].to(device, non_blocking=loader.pin_memory)
            mask = make_tube_mask(
                model.config,
                pixel_values.shape[0],
                args.mask_ratio,
                device,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                decode_mask = None
                if args.model_version == 'v2':
                    from emma_gpm.videomae_v2 import running_cell_mask
                    decode_mask = running_cell_mask(model.config, len(pixel_values), args.decoder_mask_ratio, device=device)
                output = forward_seviri_mae(model, pixel_values, mask, decode_mask)
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch + 1} step={step}")
            group_start = ((step - 1) // args.gradient_accumulation) * args.gradient_accumulation * args.batch_size
            group_samples = min(args.batch_size * args.gradient_accumulation, len(dataset) - group_start)
            (loss * (len(pixel_values) / group_samples)).backward()
            if step % args.gradient_accumulation == 0 or step == micro_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            value = float(loss.detach().cpu())
            train_step_seconds = time.perf_counter() - step_started
            data_wait_total += data_wait
            train_step_total += train_step_seconds
            pd.DataFrame([dict(epoch=epoch + 1, step=step,
                               samples=int(pixel_values.shape[0]), loss=value,
                               data_wait_seconds=data_wait, train_step_seconds=train_step_seconds,
                               wall_started=step_wall_started, wall_finished=time.time())]).to_csv(
                step_log, mode="a", header=False, index=False)
            losses.append(value)
            epoch_samples += pixel_values.shape[0]
            epoch_loss_sum += value * pixel_values.shape[0]
            print(
                f"epoch={epoch + 1}/{args.epochs} step={step}/{(len(dataset) + args.batch_size - 1) // args.batch_size} "
                f"loss={value:.6f}",
                flush=True,
            )
            previous_step_finished = time.perf_counter()
        epoch_elapsed = time.perf_counter() - epoch_started
        metrics = {
            "epoch": epoch + 1,
            "mean_loss": epoch_loss_sum / epoch_samples,
            "elapsed_seconds": epoch_elapsed,
            "clips_per_second": epoch_samples / epoch_elapsed,
            "data_wait_seconds": data_wait_total,
            "train_step_seconds": train_step_total,
            "wall_started": epoch_wall_started,
            "wall_finished": time.time(),
        }
        improved = False
        if validation_loader:
            validation_started = time.perf_counter()
            validation = evaluate_reconstruction(model, validation_loader, device, std,
                ratio=args.validation_mask_ratio, seed=args.validation_seed, precision=args.precision)
            own_validation = validation if args.mask_ratio == args.validation_mask_ratio else evaluate_reconstruction(
                model, validation_loader, device, std, ratio=args.mask_ratio,
                seed=args.validation_seed, precision=args.precision)
            metrics.update(validation_mse=validation['mse'], validation_rmse_k=validation['rmse_k'],
                           validation_seconds=time.perf_counter() - validation_started,
                           validation_own_mse=own_validation['mse'])
            improved = validation['mse'] < best_validation
            if improved:
                best_validation = validation['mse']
            print('validation=' + json.dumps(validation), flush=True)
        epoch_metrics.append(metrics)
        pd.DataFrame([metrics]).to_csv(epoch_log, mode="a", header=False, index=False)
        print(json.dumps(metrics), flush=True)
        scheduler.step()
        progress = dict(completed_epochs=epoch + 1, losses=losses, epoch_metrics=epoch_metrics,
                        elapsed_seconds=elapsed_before + time.perf_counter() - started)
        if improved:
            save_checkpoint(args.output_dir / 'checkpoint_best.pt', model, optimizer,
                            scheduler, {"loader": loader.generator, "sampler": loader.sampler.generator}, progress, identity)
            temporary = args.output_dir / 'best_validation.json.part'
            temporary.write_text(json.dumps(dict(epoch=epoch + 1, **validation), indent=2)+'\n')
            temporary.replace(args.output_dir / 'best_validation.json')
        if args.checkpoint_every and ((epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs):
            save_checkpoint(args.output_dir / "checkpoint_latest.pt", model, optimizer,
                            scheduler, {"loader": loader.generator, "sampler": loader.sampler.generator}, progress, identity)
        (args.output_dir / "progress.json").write_text(json.dumps(dict(
            status="running", completed_epochs=epoch + 1, requested_epochs=args.epochs,
            last_epoch=metrics, checkpoint=str(args.output_dir / "checkpoint_latest.pt")), indent=2)+"\n")
        save_epoch_loss_plot(epoch_metrics, args.output_dir)

    elapsed = elapsed_before + time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir / "model")
    torch.save(optimizer.state_dict(), args.output_dir / "optimizer.pt")
    stats = {
        "channels": list(THERMAL_CHANNELS),
        "normalization": "single_global_affine_transform",
        "global_mean_k": mean,
        "global_std_k": std,
        "scope": "all pixels, channels, and clips in the training split",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "samples": len(manifest),
        "pixel_count": len(manifest) * 16 * 7 * 224 * 224,
        "zarr_root": str(args.zarr_root.resolve()),
    }
    (args.output_dir / "channel_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "completed",
        "device": str(device),
        "epochs": args.epochs,
        "samples": len(dataset),
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "precision": args.precision,
        "initialization": args.initialization,
        "cpu_smoke_decoder": args.cpu_smoke_decoder,
        "mask_ratio_requested": args.mask_ratio,
        "model_version": args.model_version,
        "decoder_mask_ratio": args.decoder_mask_ratio if args.model_version == 'v2' else 0.,
        "losses": losses,
        "final_loss": losses[-1],
        "elapsed_seconds": elapsed,
        "clips_per_second": len(dataset) * args.epochs / elapsed,
        "epoch_metrics": epoch_metrics,
        "num_workers": args.num_workers,
        "loader_batch_size": loader.batch_size,
        "pin_memory": loader.pin_memory,
        "prefetch_factor": args.prefetch_factor if args.num_workers else None,
        "persistent_workers": bool(args.num_workers),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None,
        "model_input_tchw": [16, 7, 224, 224],
        "reconstruction_values_per_tubelet": reconstruction_values_per_tubelet(
            model.config
        ),
        "manifest": str(args.manifest.resolve()),
        "zarr_root": str(args.zarr_root.resolve()),
        "sample_root": str(args.sample_root.resolve()) if args.sample_root else None,
    }
    (args.output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    save_epoch_loss_plot(epoch_metrics, args.output_dir)
    (args.output_dir / "progress.json").write_text(json.dumps(dict(
        status="completed", completed_epochs=args.epochs, requested_epochs=args.epochs), indent=2)+"\n")
    reader.close()
    if validation_reader:
        validation_reader.close()
    print(
        f"completed samples={len(dataset)} final_loss={losses[-1]:.6f} "
        f"elapsed_seconds={elapsed:.1f} output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
