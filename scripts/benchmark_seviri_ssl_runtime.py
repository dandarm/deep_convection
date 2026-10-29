#!/usr/bin/env python3
"""Paired, real-input CUDA runtime benchmarks; separate from scientific runs."""
import argparse
from contextlib import nullcontext
import json
import shutil
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from emma_gpm.forecasting import ForecastWindow
from emma_gpm.seviri_frame_dataset import FrameFileDataset
from emma_gpm.seviri_clip_files import ClipBatchLoader
from emma_gpm.ssl_runtime import CudaRuntime, configure_runtime, make_optimizer, compile_model_regions
from emma_gpm.ssl_model import SeviriSSL, history_config, ssl_loss
from emma_gpm.ssl_masking import make_ssl_mask
from emma_gpm.videomae_v2 import running_cell_mask
from scripts.pretrain_videomae import seed_everything
from scripts.pretrain_seviri_ssl import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--frame-cache', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-size', choices=['small','base'], default='base')
    p.add_argument('--objective', choices=['future','joint'], default='future')
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--warmup', type=int, default=6)
    p.add_argument('--steps', type=int, default=48)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--trace', action='store_true')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT/'src/emma_gpm/ssl_runtime.py']
    source_sha256 = {str(f.relative_to(ROOT)):digest(f) for f in sources}
    for f in sources:
        target=args.output_dir/'source_archive'/f.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(f,target)
    torch.set_num_threads(4)
    runtime = CudaRuntime(**json.loads(args.runtime.read_text()))
    configure_runtime(runtime, 'cuda')
    seed_everything(20261024)
    stats = json.loads((args.dataset/'channel_stats.json').read_text())
    data = FrameFileDataset(args.frame_cache, args.dataset/'train.csv', zarr_root=stats['zarr_root'])
    loader = ClipBatchLoader(data, batch_size=args.batch_size, workers=4, prefetch=4,
        mean=stats['global_mean_k'], std=stats['global_std_k'], device='cuda', seed=20261024, shuffle=True)
    config = history_config(ForecastWindow(), model_size=args.model_size)
    model = SeviriSSL(config, objective=args.objective).cuda().train()
    optimizer = make_optimizer(model, runtime, lr=.00015, weight_decay=.05)
    compiled = compile_model_regions(model, runtime)
    rng = torch.Generator().manual_seed(20261025)
    iterator = iter(loader)
    step_log = (args.output_dir/'loss_steps.csv').open('w')
    step_log.write('step,loss,mvm,future\n')
    def step(index, audit=False):
        t = time.perf_counter()
        pixels = next(iterator)['pixel_values']
        wait = time.perf_counter()-t
        history, target = ForecastWindow().split(pixels)
        mask = decode = None
        if model.mae is not None:
            mask = make_ssl_mask(config, len(history), .9, 'tube', generator=rng, device='cuda')
            decode = running_cell_mask(config, len(history), .5, device='cpu', generator=rng).cuda()
        if runtime.compile_mode != 'off':
            torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(history, mask=mask, decode_mask=decode)
            loss, terms = ssl_loss(output, target)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        if audit:
            # Deterministic samples spread across every parameter tensor.
            gradients = torch.cat([p.grad.flatten()[::max(1,p.numel()//128)].float() for p in model.parameters() if p.grad is not None])
            initial = dict(loss=loss.detach().cpu(), prediction=output['prediction'][:2].detach().float().cpu(),
                           gradient_sample=gradients.detach().cpu(), gradient_norm=norm.detach().cpu())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        values = {name:float(v.detach()) for name,v in terms.items()}
        value = float(loss.detach())
        step_log.write(f'{index},{value},{values.get("mvm",0)},{values.get("future",0)}\n')
        step_log.flush()
        if audit:
            initial['parameter_sample'] = torch.cat([p.detach().flatten()[::max(1,p.numel()//128)].float().cpu() for p in model.parameters()])
            torch.save(initial, args.output_dir/'first_update.pt')
        return wait
    start = time.perf_counter()
    try:
        for i in range(args.warmup):
            step(i, audit=i==0)
            print(f'warmup {i+1}/{args.warmup}', flush=True)
        torch.cuda.synchronize()
        warmup = time.perf_counter()-start
        results = dict(runtime=runtime.to_dict(), compiled_regions=compiled, model_size=args.model_size,
            objective=args.objective, batch_size=args.batch_size, warmup_seconds=warmup,
            steps=args.steps, torch_version=torch.__version__, device=torch.cuda.get_device_name(),
            manifest_sha256=digest(args.dataset/'train.csv'), frame_metadata_sha256=data.metadata_sha256,
            source_sha256=source_sha256,
            repeats=[])
        for repeat in range(args.repeats):
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            waits=0.
            for i in range(args.steps): waits += step(args.warmup+repeat*args.steps+i)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            result=dict(seconds_per_update=elapsed/args.steps,clips_per_second=args.steps*args.batch_size/elapsed,
                loader_wait_seconds_per_update=waits/args.steps, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            results['repeats'].append(result)
            write_json(args.output_dir/'result.json',results)
            print(json.dumps(result),flush=True)
        if args.trace:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function('MEASURED_TRAINING_WINDOW'):
                    for i in range(3): step(1000+i)
                    torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.output_dir/'trace.json'))
            (args.output_dir/'operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
    finally:
        iterator.close()
        step_log.close()
        data.close()


if __name__=='__main__': main()
