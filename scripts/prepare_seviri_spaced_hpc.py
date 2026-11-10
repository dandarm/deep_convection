#!/usr/bin/env python3
"""Freeze an overlap-constrained experiment using precomputed clips in RAM."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def prepare(args):
    data = args.dataset.resolve()
    ready = json.loads((data/'dataset_ready.json').read_text())
    sampling = json.loads((data/'sampling_ready.json').read_text())
    if ready['status'] != 'completed' or sampling['max_iou_active'] != .2:
        raise ValueError('Completed strict IoU 0.20 dataset required')
    if sampling['measured_maximum_active_iou'] > .2 + 1e-12:
        raise ValueError('The sampling overlap constraint failed')
    if ready['train_samples'] % 4:
        raise ValueError('Training sample count must divide evenly across four ranks')
    for name, expected in ready['files'].items():
        if digest(name) != expected:
            raise ValueError('Completed dataset metadata changed: ' + name)
    # Leonardo's /dev/shm is approximately 252 GiB. The measured compressed
    # clips must fit with headroom; CPU/pinned buffers use the rest of 480 GiB.
    if ready['cache_gib'] > 235:
        raise ValueError('Precomputed clips exceed the planned shared-RAM budget')
    plan = json.loads(args.experiment.read_text())
    if (plan['initialization'] != 'scratch' or plan['global_batch_size'] != 608
            or plan['batch_size_per_gpu'] != 152 or plan['world_size'] != 4):
        raise ValueError('Preserve the approved Base Future experiment settings')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Frozen experiment output must be empty')
    archive = args.output/'source_archive'
    for directory in ('src', 'scripts', 'configs'):
        shutil.copytree(ROOT/directory, archive/directory, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    write(args.output/'runtime_files.json', {str(path.relative_to(archive)):digest(path)
        for path in archive.rglob('*') if path.is_file()})
    plan.update(training_clips=ready['train_samples'], primary_validation_clips=512,
        extended_validation_clips=10240, pipeline_benchmark_steps=32,
        updates_per_epoch=(ready['train_samples']+607)//608,
        planned_total_updates=((ready['train_samples']+607)//608)*plan['epochs'],
        dataset_plan_sha256=digest(data/'preparation_plan.json'),
        repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_archive='runtime_files.json', remote_output=str(args.remote_run/'training'),
        max_iou_active=.2, cache_gib=ready['cache_gib'],
        comparison_scope='Exploratory sampling comparison; calendar holdouts are fixed and optimizer-update budgets are reported explicitly.')
    write(args.output/'full_training.json', plan)
    write(args.output/'data_locations.json', dict(training=str(args.remote_dataset/'training'),
        extended=str(args.remote_dataset/'extended'), canonical_clip_root=str(data/'clips'),
        canonical_zarr_root=str(data/'sources'), ram_backend='tmpfs'))
    write(args.output/'storage_roots.json', dict(version=1, roots={
        str(data/'clips'):str(args.remote_dataset/'clips'), str(data/'sources'):str(args.remote_dataset/'sources')}))
    (args.output/'slurm_logs').mkdir()
    write(args.output/'submission.json', dict(status='prepared', remote_run=str(args.remote_run),
        remote_dataset=str(args.remote_dataset), command=['sbatch', '--parsable',
            '--job-name=seviri_iou020_base_future',
            '--export=ALL,SEVIRI_DDP_STAGE=full,SEVIRI_DDP_ROOT='+str(args.remote_run),
            '--output='+str(args.remote_run/'slurm_logs/%j.out'),
            '--error='+str(args.remote_run/'slurm_logs/%j.err'),
            str(args.remote_run/'source_archive/scripts/hpc/seviri_ssl_ddp.sbatch')]))
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'experiment', 'output', 'remote-dataset', 'remote-run'):
        p.add_argument('--'+name, type=Path, required=True)
    prepare(p.parse_args())
