"""Temporal holdout and reproducible reconstruction validation."""
from pathlib import Path

import numpy as np
import pandas as pd


def split_temporal_starts(starts, validation_days=3, embargo_hours=24):
    starts = pd.DatetimeIndex(starts).sort_values()
    days = starts.normalize().unique()
    if validation_days < 1 or len(days) <= validation_days or embargo_hours < 0:
        raise ValueError('Insufficient days or invalid split parameters')
    boundary = days[-validation_days]
    validation = starts[starts >= boundary]
    # Entire last training clip must precede the holdout by the embargo.
    training = starts[starts + pd.Timedelta(minutes=75) + pd.Timedelta(hours=embargo_hours) < boundary]
    if training.empty or validation.empty:
        raise ValueError('Empty temporal split')
    return training, validation


def assert_temporal_holdout(training, validation, embargo_hours=24):
    """Reject overlapping/nearby intervals without expanding every frame."""
    train = pd.DatetimeIndex(pd.to_datetime(training.start_time, utc=True)).tz_localize(None)
    val = pd.DatetimeIndex(pd.to_datetime(validation.start_time, utc=True)).tz_localize(None)
    if train.empty or val.empty:
        raise ValueError('Empty training or validation manifest')
    if train.max() + pd.Timedelta(minutes=75) + pd.Timedelta(hours=embargo_hours) >= val.min():
        raise ValueError('Temporal leakage: training must end before validation and embargo')


def fixed_tube_mask(config, indices, ratio, seed, device):
    """One CPU-generated mask per manifest index, independent of batch/RNG."""
    import torch
    spatial_tokens = (int(config.image_size) // int(config.patch_size)) ** 2
    count = min(max(round(spatial_tokens * ratio), 1), spatial_tokens - 1)
    rows = []
    for index in indices:
        generator = torch.Generator().manual_seed(seed + int(index))
        mask = torch.zeros(spatial_tokens, dtype=torch.bool)
        mask[torch.randperm(spatial_tokens, generator=generator)[:count]] = True
        rows.append(mask.repeat(int(config.num_frames) // int(config.tubelet_size)))
    return torch.stack(rows).to(device)


def evaluate_reconstruction(model, loader, device, std_k, *, ratio=.9, seed=314159, precision='fp32'):
    import torch
    from .videomae import forward_seviri_mae, patchify_videomae_pixels
    training_mode = model.training
    model.eval()
    sums = torch.zeros(7, dtype=torch.float64, device=device)
    baseline = torch.zeros_like(sums)
    count = 0
    index = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                pixels = batch['pixel_values'].to(device, non_blocking=loader.pin_memory)
                size = len(pixels)
                mask = fixed_tube_mask(model.config, range(index, index + size), ratio, seed, device)
                decode_mask = None
                if getattr(model.config, 'videomae_version', 'v1') == 'v2':
                    from .videomae_v2 import fixed_running_cell_mask
                    decode_mask = fixed_running_cell_mask(model.config, range(index, index + size),
                        model.config.decoder_mask_ratio, seed + 1000000, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
                    output = (forward_seviri_mae(model, pixels, mask) if decode_mask is None else
                              forward_seviri_mae(model, pixels, mask, decode_mask))
                prediction_mask = getattr(output, 'prediction_mask', mask)
                target = patchify_videomae_pixels(pixels, model.config)[prediction_mask].reshape_as(output.logits)
                logits = output.logits.float()
                if decode_mask is not None:
                    target, logits = target[output.loss_mask], logits[output.loss_mask]
                target = target.reshape(-1, 7)
                error = logits.reshape(-1, 7) - target
                sums += error.square().sum(0, dtype=torch.float64)
                baseline += target.square().sum(0, dtype=torch.float64)
                count += len(target)
                index += size
        per_channel = (sums / count).cpu().numpy()
        mse = float(per_channel.mean())
        if not np.isfinite(mse):
            raise RuntimeError('Non-finite validation loss')
        return dict(samples=index, mask_ratio=ratio, mask_seed=seed, mse=mse,
                    decoder_mask_ratio=getattr(model.config, 'decoder_mask_ratio', 0.),
                    rmse_k=float(np.sqrt(mse) * std_k),
                    channel_rmse_k=(np.sqrt(per_channel) * std_k).tolist(),
                    mean_baseline_rmse_k=float((baseline.sum().item() / (7 * count)) ** .5 * std_k))
    finally:
        model.train(training_mode)
