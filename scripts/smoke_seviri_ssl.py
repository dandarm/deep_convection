#!/usr/bin/env python3
"""Full-size CUDA/FP32-vs-bf16 engineering preflight on synthetic fields."""
import argparse
import gc
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from emma_gpm.advection import advection_forecast
from emma_gpm.forecasting import ForecastWindow
from emma_gpm.ssl_masking import make_ssl_mask
from emma_gpm.ssl_model import SeviriSSL, history_config, ssl_loss
from emma_gpm.seviri_clips import SeviriZarrClipReader
from emma_gpm.videomae_v2 import running_cell_mask
from scripts.pretrain_seviri_ssl import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zarr-root", type=Path)
    parser.add_argument("--start-time")
    parser.add_argument("--origin-y", type=int, default=224)
    parser.add_argument("--origin-x", type=int, default=672)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    if args.output.exists():
        raise ValueError("Choose a new output report")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This preflight requires CUDA and bf16 support")
    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    sample = None
    if args.zarr_root:
        if not args.start_time:
            raise ValueError("Real-data smoke requires an exact --start-time")
        with SeviriZarrClipReader(args.zarr_root) as reader:
            raw = reader.load(args.start_time, origin_y=args.origin_y, origin_x=args.origin_x)
        values = torch.from_numpy(raw).double()
        mean, std = float(values.mean()), float(values.std(correction=0))
        if std <= 0 or raw.min() < 100 or raw.max() > 400:
            raise ValueError("Invalid Kelvin radiometry for smoke test")
        clip = ((values.float() - mean) / std)[None].cuda()
        sample = dict(start_time=args.start_time, origin_y=args.origin_y, origin_x=args.origin_x,
                      global_mean_k=mean, global_std_k=std, minimum_k=float(raw.min()), maximum_k=float(raw.max()),
                      statistics_scope="one integration clip only; never reuse for scientific training")
    else:
        base = torch.randn(1, 7, 224, 224, device="cuda")
        clip = torch.stack([torch.roll(base, index, -1) - .01 * index for index in range(16)], dim=1)
    clip = clip.repeat(args.batch_size, 1, 1, 1, 1)
    window = ForecastWindow()
    history, target = window.split(clip)
    adv, _, _ = advection_forecast(history, window.steps)
    records = []
    for objective in ("future", "mvm", "joint", "residual"):
        torch.manual_seed(20261024)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        config = history_config(window)
        model = SeviriSSL(config, objective=objective).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
        mask = make_ssl_mask(config, args.batch_size, .9, device="cuda")
        decode = running_cell_mask(config, args.batch_size, .5, device="cuda")
        with torch.no_grad():
            reference, _ = ssl_loss(model(history, mask=mask, decode_mask=decode, advection=adv), target)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = ssl_loss(model(history, mask=mask, decode_mask=decode, advection=adv), target)
        relative_error = abs(float(loss.detach()) - float(reference)) / max(abs(float(reference)), 1e-8)
        if not torch.isfinite(loss) or relative_error > .02:
            raise RuntimeError(f"{objective}: bf16/FP32 relative loss difference {relative_error}")
        parameter = next(model.encoder.parameters())
        before = parameter.detach().clone()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        if torch.equal(before, parameter):
            raise RuntimeError(f"{objective}: optimizer did not update the encoder")
        records.append(dict(objective=objective, fp32_loss=float(reference), bf16_loss=float(loss.detach()),
            relative_difference=relative_error, backward_finite=True, encoder_updated=True,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30))
        print(records[-1], flush=True)
        del model, optimizer, parameter, before, loss, reference
        gc.collect()
    write_json(args.output, dict(kind="one-clip observed SEVIRI engineering preflight" if sample else "synthetic engineering preflight, no observed SEVIRI",
        sample=sample,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, input_shape=list(history.shape),
        target_shape=list(target.shape), records=records))


if __name__ == "__main__":
    main()
