#!/usr/bin/env python3
"""Freeze a from-scratch calendar experiment and its Leonardo storage mappings."""
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
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(args):
    data = args.dataset.resolve()
    ready = json.loads((data/'dataset_ready.json').read_text())
    if ready['status'] != 'completed':
        raise ValueError('Dataset preparation must finish before freezing the run')
    for path, expected in ready['files'].items():
        if digest(path) != expected:
            raise ValueError('Prepared dataset identity differs: '+path)
    metadata = json.loads((data/'frames/metadata.json').read_text())
    plan = json.loads(args.experiment.read_text())
    if (plan['initialization'] != 'scratch' or plan['global_batch_size'] != plan['batch_size_per_gpu']*plan['world_size']
            or plan['world_size'] != 4 or plan['epochs'] < 1 or plan['learning_rate'] <= 0):
        raise ValueError('Inconsistent from-scratch experiment settings')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Frozen run output must be an empty directory')
    remote_data, remote_run = args.remote_dataset, args.remote_run
    frame_files = {entry['file']:dict(bytes=entry['bytes'], sha256=entry['sha256'])
                   for entry in metadata['stores'].values()}
    frame_files['metadata.json'] = dict(bytes=(data/'frames/metadata.json').stat().st_size,
                                      sha256=digest(data/'frames/metadata.json'))
    small_files = {}
    for directory in ('sources', 'training/dataset', 'extended/dataset', 'preparation_sources', 'preparation_history'):
        for path in (data/directory).rglob('*'):
            if path.is_file():
                small_files[str(path.relative_to(data))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    for name in ('preparation_plan.json', 'dataset_ready.json'):
        small_files[name] = dict(bytes=(data/name).stat().st_size, sha256=digest(data/name))
    transfer = dict(version=1, canonical_frame_root=str(data/'frames'), canonical_zarr_root=str(data/'sources'),
        frames=frame_files, payload=small_files)
    write(data/'transfer_plan.json', transfer)
    for role in ('training', 'extended'):
        write(data/role/'transfer_plan.json', transfer)
        link = data/role/'frames'
        if not link.exists():
            link.symlink_to('../frames', target_is_directory=True)
    archive = args.output/'source_archive'
    for directory in ('src', 'scripts', 'configs'):
        shutil.copytree(ROOT/directory, archive/directory, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    write(args.output/'runtime_files.json', {str(path.relative_to(archive)):digest(path)
        for path in archive.rglob('*') if path.is_file()})
    plan.update(training_clips=ready['train_samples'], primary_validation_clips=ready['primary_validation_samples'],
        extended_validation_clips=ready['extended_validation_samples'],
        planned_total_updates=((ready['train_samples']+plan['global_batch_size']-1)//plan['global_batch_size'])*plan['epochs'],
        pipeline_benchmark_steps=32, dataset_plan_sha256=digest(data/'preparation_plan.json'),
        repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_archive='runtime_files.json', remote_output=str(remote_run/'training'))
    write(args.output/'full_training.json', plan)
    write(args.output/'data_locations.json', dict(training=str(remote_data/'training'), extended=str(remote_data/'extended'),
        canonical_frame_root=str(data/'frames'), canonical_zarr_root=str(data/'sources'), ram_backend='memfd'))
    write(args.output/'storage_roots.json', dict(version=1, roots={
        str(data/'frames'):str(remote_data/'frames'), str(data/'sources'):str(remote_data/'sources')}))
    (args.output/'slurm_logs').mkdir()
    write(args.output/'submission.json', dict(status='prepared', remote_run=str(remote_run),
        remote_dataset=str(remote_data), expected_epochs=plan['epochs'], expected_learning_rate=plan['learning_rate'],
        command=['sbatch', '--parsable', '--job-name=seviri_cal_base_future',
            '--export=ALL,SEVIRI_DDP_STAGE=full,SEVIRI_DDP_ROOT='+str(remote_run),
            '--output='+str(remote_run/'slurm_logs/%j.out'), '--error='+str(remote_run/'slurm_logs/%j.err'),
            str(remote_run/'source_archive/scripts/hpc/seviri_ssl_ddp.sbatch')]))
    print(json.dumps(dict(epochs=plan['epochs'], learning_rate=plan['learning_rate'],
        global_batch=plan['global_batch_size'], updates=plan['planned_total_updates'], frame_gib=ready['frame_gib']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'experiment', 'output', 'remote-dataset', 'remote-run'):
        p.add_argument('--'+name, type=Path, required=True)
    prepare(p.parse_args())
