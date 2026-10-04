#!/usr/bin/env python3
"""Sequential, resumable SEVIRI masking sweep and conditional crop expansion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))


def write_json(path, value):
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run_command(command, log):
    print('COMMAND ' + ' '.join(map(str, command)), flush=True)
    with log.open('a') as handle:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def prepare_split(output, zarr_root):
    import pandas as pd
    from build_random_seviri_clip_manifest import inventory, complete_start_times
    from emma_gpm.pretraining_validation import split_temporal_starts
    times, height, width = inventory(zarr_root, 224)
    starts = complete_start_times(times, num_frames=16, cadence_minutes=5)
    train, val = split_temporal_starts(starts)
    split = dict(training_starts=[t.isoformat() for t in train], validation_starts=[t.isoformat() for t in val],
                 validation_days=3, embargo_hours=24, frame_count=len(times), grid=[height, width],
                 limitation='Additional crops from the same archive, not new weather episodes. Old 150-epoch model saw holdout dates; all new runs start from scratch.')
    path = output / 'temporal_split.json'
    if path.exists() and json.loads(path.read_text()) != split:
        raise ValueError('Archive or split changed; choose a new experiment directory')
    write_json(path, split)
    for name, values in [('train', train), ('validation', val)]:
        pd.DataFrame({'start_time': [t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in values]}).to_csv(output / f'{name}_starts.csv', index=False)
    print(f'SPLIT train_starts={len(train)} validation_starts={len(val)}', flush=True)


def prepare_manifest(output, zarr_root, name, count, stats=True):
    manifest = output / f'{name}.csv'
    stats_path = output / f'{name}_stats.json'
    if manifest.exists() and (not stats or stats_path.exists()):
        import pandas as pd
        from pretrain_videomae import load_training_stats
        if len(pd.read_csv(manifest)) != count:
            raise ValueError('Unexpected existing manifest length')
        if stats:
            load_training_stats(stats_path, manifest, zarr_root)
        return
    split_name = 'validation' if name == 'validation' else 'train'
    command = [sys.executable, ROOT / 'scripts/build_random_seviri_clip_manifest.py',
        '--zarr-root', zarr_root, '--output', manifest, '--count', count, '--max-attempts', count * 20,
        '--seed', 20261002 if split_name == 'validation' else 20261001, '--num-workers', 16,
        '--starts-file', output / f'{split_name}_starts.csv']
    if stats:
        command += ['--stats-output', stats_path]
    run_command(command, output / f'prepare_{name}.log')


def preflight(output, zarr_root):
    """Real-data forward comparison, backward and optimizer with 50% masking."""
    import torch
    import pandas as pd
    from emma_gpm.seviri_clips import SeviriZarrClipReader
    from emma_gpm.videomae import build_videomae_small_7ch_from_scratch, forward_seviri_mae
    from pretrain_videomae import make_tube_mask, seed_everything
    torch.set_num_threads(4)
    seed_everything(20261001)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA bf16 unavailable')
    stats = json.loads((output / 'train4096_stats.json').read_text())
    row = pd.read_csv(output / 'train4096.csv').iloc[0]
    with SeviriZarrClipReader(zarr_root) as reader:
        clip = reader.load(row.start_time, origin_y=int(row.origin_y), origin_x=int(row.origin_x),
                           global_mean=stats['global_mean_k'], global_std=stats['global_std_k'])
    pixels = torch.from_numpy(clip.copy()).unsqueeze(0).cuda()
    model = build_videomae_small_7ch_from_scratch().cuda().eval()
    mask = make_tube_mask(model.config, 1, .5, 'cuda')
    with torch.no_grad():
        fp32 = float(forward_seviri_mae(model, pixels, mask).loss)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            bf16 = float(forward_seviri_mae(model, pixels, mask).loss)
    if abs(fp32 - bf16) / fp32 > .02:
        raise RuntimeError('bf16 preflight loss differs by more than 2% from fp32')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    batch = pixels.repeat(16, 1, 1, 1, 1)
    optimizer.zero_grad(set_to_none=True)
    for _ in range(2):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = forward_seviri_mae(model, batch, make_tube_mask(model.config, 16, .5, 'cuda')).loss
        (loss / 2).backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError('Non-finite preflight loss or gradients')
    optimizer.step()
    peak = torch.cuda.max_memory_reserved() / 2**30
    if peak > 21:
        raise RuntimeError(f'Preflight reserves {peak:.2f} GiB; reduce batch before launch')
    write_json(output / 'preflight.json', dict(fp32_loss=fp32, bf16_loss=bf16,
        relative_loss_difference=abs(fp32-bf16)/fp32, peak_reserved_gib=peak,
        gpu=torch.cuda.get_device_name(), micro_batch=16, gradient_accumulation=2, effective_batch=32,
        backward_finite=True, optimizer_step=True))
    del model, optimizer, batch, pixels, loss
    torch.cuda.empty_cache()


def train(output, zarr_root, name, manifest_name, ratio, epochs):
    destination = output / name
    destination.mkdir(exist_ok=True)
    config = destination / 'run_config.json'
    if (destination / 'smoke_summary.json').exists():
        summary = json.loads((destination / 'smoke_summary.json').read_text())
        arguments = json.loads(config.read_text())['arguments']
        if summary['epochs'] != epochs or summary['mask_ratio_requested'] != ratio or Path(arguments['manifest']) != output / f'{manifest_name}.csv':
            raise ValueError('Completed run differs from requested experiment')
        print(f'SKIP completed {name}', flush=True)
        return
    command = [sys.executable, ROOT / 'scripts/pretrain_videomae.py',
        '--zarr-root', zarr_root, '--manifest', output / f'{manifest_name}.csv',
        '--global-stats', output / f'{manifest_name}_stats.json',
        '--validation-manifest', output / 'validation.csv', '--validation-mask-ratio', .9,
        '--output-dir', destination, '--device', 'cuda', '--epochs', epochs,
        '--batch-size', 16, '--gradient-accumulation', 2, '--precision', 'bf16',
        '--num-workers', 16, '--loader-batch-size', 2, '--prefetch-factor', 2,
        '--pin-memory', '--cpu-threads', 4, '--initialization', 'scratch',
        '--no-cpu-smoke-decoder', '--checkpoint-every', 1, '--mask-ratio', ratio]
    checkpoint = destination / 'checkpoint_latest.pt'
    if checkpoint.exists():
        command += ['--resume', checkpoint]
    run_command(command, destination / 'training.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zarr-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--large-epochs', type=int, default=150)
    parser.add_argument('--adopt-training-pid', type=int,
        help='Wait for an already running trainer without restarting it (supervisor replacement).')
    parser.add_argument('--adopt-run', choices=['mask50', 'mask75', 'mask90'])
    parser.add_argument('--minimum-improvement', type=float, default=.05,
        help='Run 40000 crops if best mask reduces common-mask validation RMSE by less than this fraction vs 90%%.')
    args = parser.parse_args()
    if bool(args.adopt_training_pid) != bool(args.adopt_run):
        parser.error('adopt-training-pid and adopt-run must be supplied together')
    if args.epochs < 1 or args.large_epochs < 1 or not 0 <= args.minimum_improvement <= 1:
        parser.error('Invalid epoch count or improvement threshold')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = dict(zarr_root=str(args.zarr_root.resolve()), epochs=args.epochs, large_epochs=args.large_epochs,
                    masks=[.5, .75, .9], training_samples=4096, validation_samples=512,
                    large_samples=40000, minimum_rmse_improvement=args.minimum_improvement,
                    validation_mask=.9, validation_seed=314159, seed=20261001,
                    precision='bf16', micro_batch=16, effective_batch=32)
        if (output / 'plan.json').exists() and json.loads((output / 'plan.json').read_text()) != plan:
            raise ValueError('Existing queue plan differs; choose a new output directory')
        write_json(output / 'plan.json', plan)
        def status(stage, **extra):
            write_json(output / 'queue_status.json', dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **extra))
        try:
            status('preparing')
            prepare_split(output, args.zarr_root)
            prepare_manifest(output, args.zarr_root, 'train4096', 4096)
            prepare_manifest(output, args.zarr_root, 'validation', 512, stats=False)
            status('preflight')
            if not (output / 'preflight.json').exists():
                # Isolate CUDA preflight so all its allocations die before training.
                run_command([sys.executable, '-c',
                    'import sys;sys.path.insert(0,"scripts");from pathlib import Path;from run_seviri_masking_experiments import preflight;preflight(Path(sys.argv[1]),Path(sys.argv[2]))',
                    output, args.zarr_root], output / 'preflight.log')
            results = []
            for ratio in plan['masks']:
                name = f'mask{round(ratio * 100)}'
                status('training', run=name)
                if name == args.adopt_run:
                    process_path = Path(f'/proc/{args.adopt_training_pid}')
                    if process_path.exists():
                        command_line = (process_path / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
                        if 'pretrain_videomae.py' not in command_line or str(output / name) not in command_line:
                            raise ValueError('Adopted PID is not the expected trainer')
                    print(f'ADOPT trainer PID={args.adopt_training_pid}; preserving active epoch and workers', flush=True)
                    while process_path.exists():
                        try:
                            state = (process_path / 'stat').read_text().rsplit(')', 1)[1].strip().split()[0]
                        except FileNotFoundError:
                            break
                        if state == 'Z':
                            break
                        time.sleep(5)
                    if not (output / name / 'smoke_summary.json').exists():
                        raise RuntimeError('Adopted trainer stopped without completing; rerun queue to resume checkpoint')
                train(output, args.zarr_root, name, 'train4096', ratio, args.epochs)
                result = json.loads((output / name / 'best_validation.json').read_text())
                results.append(dict(run=name, training_mask=ratio, **result))
                write_json(output / 'masking_comparison.json', results)
            best = min(results, key=lambda row: row['rmse_k'])
            baseline = next(row for row in results if row['training_mask'] == .9)
            improvement = 1 - best['rmse_k'] / baseline['rmse_k']
            expand = improvement < args.minimum_improvement
            policy_path = output / 'expansion_policy.json'
            policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
            if policy.get('mode') == 'prepare_only':
                expand = False
            write_json(output / 'selection.json', dict(best=best, baseline=baseline,
                relative_rmse_improvement=improvement, expand_to_40000=expand,
                rule='Expand if improvement vs freshly trained 90% baseline is below threshold',
                threshold=args.minimum_improvement, expansion_policy=policy))
            if expand:
                status('preparing40000', best_mask=best['training_mask'])
                prepare_manifest(output, args.zarr_root, 'train40000', 40000)
                import pandas as pd
                small = pd.read_csv(output / 'train4096.csv')
                large = pd.read_csv(output / 'train40000.csv')
                if not small.equals(large.iloc[:len(small)].reset_index(drop=True)):
                    raise ValueError('Large manifest does not contain the small manifest as a prefix')
                status('training40000', best_mask=best['training_mask'])
                train(output, args.zarr_root, 'large40000', 'train40000', best['training_mask'], args.large_epochs)
                large_result = json.loads((output / 'large40000/best_validation.json').read_text())
                write_json(output / 'dataset_comparison.json', dict(small=best, large=large_result,
                    relative_rmse_improvement=1-large_result['rmse_k']/best['rmse_k'],
                    limitation='150 epochs on 40000 crops entails approximately 10x the updates, not a compute-matched comparison.'))
            status('completed', selected_mask=best['training_mask'], expanded=expand)
        except BaseException as error:
            status('failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
