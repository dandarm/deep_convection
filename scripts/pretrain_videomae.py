#!/usr/bin/env python3
"""Self-supervised seven-channel VideoMAE pretraining on SEVIRI clips."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.seviri import THERMAL_CHANNELS  # noqa: E402
from emma_gpm.seviri_clips import (  # noqa: E402
    SeviriVideoMAEDataset,
    SeviriZarrClipReader,
)
from emma_gpm.videomae import (  # noqa: E402
    build_videomae_7ch_from_rgb_checkpoint,
    build_videomae_small_7ch_from_scratch,
    forward_seviri_mae,
    reconstruction_values_per_tubelet,
)


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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--cpu-threads", type=int, default=0)
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


def build_model(args: argparse.Namespace):
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
    return build_videomae_small_7ch_from_scratch(**overrides)


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size must be positive")
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise SystemExit(
            "Training dependencies are missing. Install requirements-training.txt."
        ) from error

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    device = choose_device(args.device)
    manifest = pd.read_csv(args.manifest)
    if manifest.empty:
        raise SystemExit(f"Empty clip manifest: {args.manifest}")

    reader = SeviriZarrClipReader(args.zarr_root)
    mean, std = estimate_global_stats(manifest, reader)
    dataset = SeviriVideoMAEDataset(
        manifest,
        reader,
        global_mean=mean,
        global_std=std,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    model = build_model(args).to(device)
    if model.config.num_channels != 7:
        raise RuntimeError(f"Model has {model.config.num_channels} channels, expected 7.")
    if reconstruction_values_per_tubelet(model.config) != 3584:
        raise RuntimeError("VideoMAE decoder target width is not 3584.")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            mask = make_tube_mask(
                model.config,
                pixel_values.shape[0],
                args.mask_ratio,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            output = forward_seviri_mae(model, pixel_values, mask)
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch + 1} step={step}")
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            print(
                f"epoch={epoch + 1}/{args.epochs} step={step}/{len(loader)} "
                f"loss={value:.6f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir / "model")
    torch.save(optimizer.state_dict(), args.output_dir / "optimizer.pt")
    stats = {
        "channels": list(THERMAL_CHANNELS),
        "normalization": "single_global_affine_transform",
        "global_mean_k": mean,
        "global_std_k": std,
        "scope": "all pixels, channels, and clips in the training split",
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
        "initialization": args.initialization,
        "cpu_smoke_decoder": args.cpu_smoke_decoder,
        "mask_ratio_requested": args.mask_ratio,
        "losses": losses,
        "final_loss": losses[-1],
        "elapsed_seconds": elapsed,
        "model_input_tchw": [16, 7, 224, 224],
        "reconstruction_values_per_tubelet": reconstruction_values_per_tubelet(
            model.config
        ),
        "manifest": str(args.manifest.resolve()),
        "zarr_root": str(args.zarr_root.resolve()),
    }
    (args.output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    reader.close()
    print(
        f"completed samples={len(dataset)} final_loss={losses[-1]:.6f} "
        f"elapsed_seconds={elapsed:.1f} output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
