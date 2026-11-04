#!/usr/bin/env python3
"""Evaluate an existing DDP checkpoint on both holdouts without weight updates."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import pandas as pd
import torch
import torch.distributed as dist
from transformers import VideoMAEConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from emma_gpm.forecasting import ForecastWindow
from emma_gpm.seviri_frame_dataset import FrameFileDataset
from emma_gpm.ssl_distributed import GlobalBatchSampler, initialize
from emma_gpm.ssl_evaluation import evaluate_ssl
from emma_gpm.ssl_model import SeviriSSL
from emma_gpm.ssl_runtime import CudaRuntime, configure_runtime, compile_model_regions
from emma_gpm.storage_roots import storage_root
from scripts.pretrain_seviri_ssl import digest, write_json, validate_manifests, source_hashes
from scripts.pretrain_seviri_ssl_ddp import expected_digest, indexed_loader


def evaluate(args, rank, world, device):
    torch.set_num_threads(4)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        if args.checkpoint_sha256 and digest(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError('Parent checkpoint checksum differs')
    dist.barrier()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    identity = saved['identity']
    if identity.get('synthetic') or identity['stats'].get('synthetic'):
        raise ValueError('This evaluator requires a scientific checkpoint')
    for name, key in [('train.csv', 'train_sha256'), ('validation.csv', 'validation_sha256')]:
        if digest(args.dataset/name) != identity[key]:
            raise ValueError('Checkpoint training/validation manifest differs')
    stats = json.loads((args.dataset/'channel_stats.json').read_text())
    if stats != identity['stats']:
        raise ValueError('Checkpoint normalization differs')
    window = ForecastWindow(**identity['window'])
    config = VideoMAEConfig.from_dict(identity['model_config'])
    model = SeviriSSL(config, len(window.steps), identity['objective'], identity['head_width'],
                      identity['frozen_encoder'])
    model.load_state_dict(saved['model'], strict=True)
    epoch = saved['progress']['completed_epochs']
    del saved
    model = model.to(device).eval()
    runtime = CudaRuntime(**identity['cuda_runtime'])
    configure_runtime(runtime, device)
    compile_model_regions(model, runtime)
    protocol = json.loads((args.extended_dataset/'protocol.json').read_text())
    ready = json.loads((args.extended_dataset/'dataset_ready.json').read_text())
    if ready['status'] != 'completed':
        raise ValueError('Extended holdout is incomplete')
    for name in ('validation.csv', 'protocol.json'):
        if digest(args.extended_dataset/name) != expected_digest(ready, name):
            raise ValueError('Extended holdout identity differs')
    if digest(args.extended_dataset/'validation.csv') != identity['extended_validation']['manifest_sha256']:
        raise ValueError('Checkpoint extended holdout differs')
    if digest(args.extended_dataset/'protocol.json') != identity['extended_validation']['protocol_sha256']:
        raise ValueError('Checkpoint extended protocol differs')
    if digest(storage_root(protocol['frame_cache'])/'metadata.json') != expected_digest(ready, 'metadata.json'):
        raise ValueError('Extended cache metadata differs')
    train = pd.read_csv(args.dataset/'train.csv')
    timings = {}
    def progress(**details):
        if rank == 0:
            write_json(args.output_dir/'progress.json', dict(updated_unix=time.time(), **details))
    for label, manifest, cache, raw in (
        ('legacy', args.dataset/'validation.csv', args.frame_cache, args.zarr_root),
        ('extended', args.extended_dataset/'validation.csv', protocol['frame_cache'], protocol['zarr_root'])):
        frame = pd.read_csv(manifest)
        validate_manifests(train, frame)
        progress(status='preparing', validation_set=label)
        start = time.perf_counter()
        data = FrameFileDataset(cache, manifest, zarr_root=raw)
        dist.barrier()
        preparation = time.perf_counter()-start
        sampler = GlobalBatchSampler(len(data), args.validation_batch_size, rank, world, 314159, False)
        loader = indexed_loader(data, sampler, args, device, stats, args.validation_batch_size)
        def heartbeat(count):
            progress(status='validating', validation_set=label, samples_per_rank=count)
        start = time.perf_counter()
        report = evaluate_ssl(model, loader, window, device, stats['global_mean_k'], stats['global_std_k'],
            precision=identity['precision'], distributed=True, sample_days=frame.day.tolist(),
            progress_callback=heartbeat)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-start
        times = torch.tensor([preparation, elapsed], device=device)
        dist.all_reduce(times, op=dist.ReduceOp.MAX)
        timings[label] = dict(preparation_seconds=float(times[0]),evaluation_seconds=float(times[1]),samples=len(data))
        if rank == 0:
            write_json(args.output_dir/('validation_'+label+'.json'), report)
            write_json(args.output_dir/'timings.json', timings)
            print(json.dumps(dict(validation_set=label, **timings[label])), flush=True)
        data.close()
    if rank == 0:
        write_json(args.output_dir/'evaluation_config.json',dict(checkpoint=str(args.checkpoint),
            checkpoint_sha256=args.checkpoint_sha256, checkpoint_epoch=epoch, optimizer_updates=0,
            world_size=world, sources=source_hashes(), evaluator_sha256=digest(Path(__file__)),
            window=asdict(window), timings=timings))
    progress(status='completed', checkpoint_epoch=epoch, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','dataset','extended-dataset','frame-cache','zarr-root','output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', help='Optional explicitly requested checkpoint integrity audit')
    parser.add_argument('--validation-batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prefetch', type=int, default=2)
    args = parser.parse_args();args.seed = 314159
    rank, world, device = initialize('cuda')
    try:
        evaluate(args, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
