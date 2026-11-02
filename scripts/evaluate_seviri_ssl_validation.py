#!/usr/bin/env python3
"""Evaluate a trusted SSL checkpoint on a frozen additional holdout."""
import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd
import torch
from transformers import VideoMAEConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.forecasting import ForecastWindow
from emma_gpm.seviri_clip_files import ClipBatchLoader
from emma_gpm.seviri_frame_dataset import FrameFileDataset
from emma_gpm.ssl_dataset import file_digest
from emma_gpm.ssl_evaluation import evaluate_ssl
from emma_gpm.ssl_model import SeviriSSL
from emma_gpm.ssl_runtime import CudaRuntime, configure_runtime, compile_model_regions
from emma_gpm.storage_roots import storage_root
from scripts.pretrain_seviri_ssl import validate_manifests, write_json, source_hashes


def expected_digest(ready, filename):
    matches = [digest for path,digest in ready['files'].items() if Path(path).name == filename]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one frozen identity for {filename}')
    return matches[0]


def evaluate(args):
    started = time.monotonic()
    sources = {**source_hashes(), 'scripts/evaluate_seviri_ssl_validation.py':file_digest(Path(__file__))}
    output = args.output_dir
    if any(p.name != '.lock' for p in output.iterdir()):
        raise ValueError('Evaluation output must be a new directory')
    protocol = json.loads((args.dataset/'protocol.json').read_text())
    ready = json.loads((args.dataset/'dataset_ready.json').read_text())
    if ready.get('status') != 'completed':
        raise ValueError('A completed validation dataset is required')
    for name in ('validation.csv', 'protocol.json'):
        path = args.dataset/name
        if file_digest(path) != expected_digest(ready, name):
            raise ValueError(f'Validation identity differs: {name}')
    if file_digest(storage_root(protocol['frame_cache'])/'metadata.json') != expected_digest(ready, 'metadata.json'):
        raise ValueError('Frame metadata differs from the frozen validation dataset')
    manifest = args.dataset/'validation.csv'
    frame = pd.read_csv(manifest)
    if len(frame) != protocol['samples']:
        raise ValueError('Validation manifest length differs from the frozen protocol')
    run = json.loads((args.run_dir/'run_config.json').read_text())
    checkpoint_path = args.run_dir/f'checkpoint_{args.checkpoint}.pt'
    checkpoint_hash = file_digest(checkpoint_path)
    # Checkpoints are trusted, local experiment artifacts.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if file_digest(checkpoint_path) != checkpoint_hash:
        raise ValueError('Checkpoint changed while loading; retry with a stable checkpoint')
    identity = checkpoint['identity']
    if json.loads(json.dumps(identity)) != run['identity']:
        raise ValueError('Checkpoint and run identity differ')
    training_path = args.training_manifest or Path(run['arguments']['manifest'])
    if file_digest(training_path) != identity['train_sha256']:
        raise ValueError('Historical training manifest changed')
    validate_manifests(pd.read_csv(training_path), frame)
    if args.smoke_samples:
        if args.smoke_samples < 1 or args.smoke_samples >= len(frame):
            raise ValueError('Smoke sample count must be smaller than the validation set')
        frame = frame.iloc[:args.smoke_samples].copy()
        manifest = output/'smoke_manifest.csv'
        frame.to_csv(manifest, index=False)
    if identity['objective'] in ('mvm', 'joint'):
        # MVM loss is a batch mean; complete days keep its daily attribution exact.
        counts = frame.groupby('day', sort=False).size().to_numpy()
        if any(n % args.batch_size for n in counts[:-1]):
            raise ValueError('For per-day MVM metrics, batch size must divide each full day sample count')
    window = ForecastWindow(**identity['window'])
    model = SeviriSSL(VideoMAEConfig.from_dict(identity['model_config']), len(window.steps),
                      identity['objective'], identity['head_width'], identity['frozen_encoder'])
    model.load_state_dict(checkpoint['model'], strict=True)
    checkpoint_progress = checkpoint['progress']
    del checkpoint
    device = torch.device(args.device)
    if args.precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('bf16 requires a compatible CUDA device')
    runtime = CudaRuntime(tf32=True, cudnn_benchmark=True, static_positions=True,
                          compile_mode=args.compile_mode) if device.type == 'cuda' else CudaRuntime()
    configure_runtime(runtime, device)
    model = model.to(device).eval()
    compile_model_regions(model, runtime)
    stats = identity['stats']
    write_json(output/'progress.json', dict(status='running', phase='verifying_frames', total_samples=len(frame)))
    data = FrameFileDataset(protocol['frame_cache'], manifest, zarr_root=protocol['zarr_root'])
    loader = ClipBatchLoader(data, batch_size=args.batch_size, workers=args.workers, prefetch=2,
        mean=stats['global_mean_k'], std=stats['global_std_k'], device=device, seed=314159, shuffle=False)
    last_update = 0.
    def heartbeat(samples):
        nonlocal last_update
        now = time.monotonic()
        if now-last_update >= 5 or samples == len(frame):
            write_json(output/'progress.json', dict(status='running', phase='evaluating', samples=samples,
                       total_samples=len(frame), elapsed_seconds=now-started))
            last_update = now
    try:
        report = evaluate_ssl(model, loader, window, device, stats['global_mean_k'], stats['global_std_k'],
            mask_ratio=identity['mask_ratio'], mask_strategy=identity['mask_strategy'],
            precision=args.precision, max_shift=identity['max_motion_pixels'],
            sample_days=frame.day.astype(str).tolist(), progress_callback=heartbeat)
    finally:
        data.close()
    for metrics in [report, *report.get('by_day', {}).values()]:
        if metrics['mvm_mse'] is not None:
            metrics['mvm_rmse_k'] = metrics['mvm_mse']**.5 * stats['global_std_k']
    for name, checksum in sources.items():
        if file_digest(ROOT/name) != checksum:
            raise ValueError(f'Evaluation source changed during execution: {name}')
    result = dict(status='completed', scope='smoke' if args.smoke_samples else 'full_validation',
        run=str(args.run_dir), checkpoint=args.checkpoint, checkpoint_sha256=checkpoint_hash,
        checkpoint_progress=checkpoint_progress, dataset=str(args.dataset), samples=len(frame),
        validation_manifest_sha256=file_digest(manifest), protocol_sha256=file_digest(args.dataset/'protocol.json'),
        training_normalization=stats, runtime=runtime.to_dict(), precision=args.precision,
        batch_size=args.batch_size, elapsed_seconds=time.monotonic()-started, evaluation=report,
        sources=sources, torch_version=torch.__version__)
    write_json(output/'result.json', result)
    write_json(output/'progress.json', dict(status='completed', scope=result['scope'], samples=len(frame),
                                          elapsed_seconds=result['elapsed_seconds']))
    print(f"Evaluated {len(frame)} clips in {result['elapsed_seconds']:.0f} s ({result['scope']})", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','run-dir','output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--checkpoint', choices=['best','latest'], default='best')
    p.add_argument('--training-manifest', type=Path,
                   help='Relocated training manifest; its SHA256 must match the checkpoint')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--precision', choices=['fp32','bf16'], default='bf16')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--compile-mode', choices=['off','default'], default='off')
    p.add_argument('--smoke-samples', type=int, help='Explicit partial engineering check; output is marked as smoke')
    args = p.parse_args()
    for name,value in vars(args).items():
        if isinstance(value, Path): setattr(args, name, value.resolve())
    if min(args.batch_size,args.workers,args.cpu_threads) < 1:
        p.error('Positive batch size and worker/thread counts required')
    torch.set_num_threads(args.cpu_threads); torch.set_num_interop_threads(1)
    with exclusive_file_lock(args.output_dir/'.lock'):
        try:
            evaluate(args)
        except Exception as exc:
            write_json(args.output_dir/'progress.json', dict(status='failed', error=str(exc)))
            raise


if __name__ == '__main__': main()
