#!/usr/bin/env python3
"""Run a frozen matrix of encoder sizes and datasets with Future and Joint."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.seviri_frame_dataset import FrameFileDataset
from emma_gpm.ssl_dataset import verify_snapshot
from emma_gpm.ssl_scaling import publish_summary
from scripts.build_seviri_ssl_extension import keys
from scripts.pretrain_seviri_ssl import digest, write_json, source_hashes, validate_manifests
from scripts.run_seviri_ssl_selected import selected_stages
from scripts.run_seviri_ssl_scaling import immutable_json, execute, stage_command, stage_completed


DATA_FILES = ('train.csv', 'validation.csv', 'channel_stats.json', 'source_snapshot.json', 'protocol.json', 'inventory.json')


def make_plans(output, config):
    cases, local_plans, sets, validation_hashes = [], {}, {}, set()
    for entry in config['cases']:
        name, model = entry['name'], entry['model_size']
        if Path(name).name != name or model not in ('small', 'base') or name in local_plans:
            raise ValueError('Unique case names and supported encoder sizes are required')
        source = Path(entry['dataset']).resolve()
        dataset = output / name / 'dataset'
        dataset.mkdir(parents=True, exist_ok=True)
        files = {}
        for filename in DATA_FILES:
            expected = digest(source / filename)
            target = dataset / filename
            if target.exists():
                if digest(target) != expected:
                    raise ValueError(f'Prepared case differs from its source: {target}')
            else:
                shutil.copyfile(source / filename, target)
            files[filename] = expected
        training = pd.read_csv(dataset / 'train.csv')
        validation = pd.read_csv(dataset / 'validation.csv')
        validate_manifests(training, validation)
        if len(training) != entry['train_samples']:
            raise ValueError('Manifest length differs from the requested dataset size')
        stats = json.loads((dataset / 'channel_stats.json').read_text())
        raw_root = Path(entry['zarr_root']).resolve()
        if Path(stats['zarr_root']).resolve() != raw_root or stats['manifest_sha256'] != files['train.csv']:
            raise ValueError('Normalization is bound to a different manifest or source archive')
        clip_set = set(keys(training))
        size = len(training)
        if size in sets and (sets[size] != clip_set or
                next(c['files']['train.csv'] for c in cases if c['train_samples'] == size) != files['train.csv']):
            raise ValueError('Encoder comparisons require identical ordered training manifests')
        if size in sets and next(c['files']['channel_stats.json'] for c in cases if c['train_samples'] == size) != files['channel_stats.json']:
            raise ValueError('Encoder comparisons require identical normalization')
        sets[size] = clip_set
        validation_hashes.add(files['validation.csv'])
        case = dict(name=name, model_size=model, train_samples=size, validation_samples=len(validation),
            zarr_root=str(raw_root), frame_cache=str(Path(entry['frame_cache']).resolve()),
            files=files, source_dataset=str(source),
            optimizer_updates_per_epoch=(size + config['batch_size']-1)//config['batch_size'])
        case['planned_optimizer_updates'] = case['optimizer_updates_per_epoch']*config['epochs']
        case['frame_metadata_sha256'] = digest(Path(case['frame_cache']) / 'metadata.json')
        arguments = argparse.Namespace(**{k:config[k] for k in ('epochs','batch_size','workers','seed')},
            zarr_root=raw_root, frame_cache=Path(case['frame_cache']), clip_root=None, model_size=model,
            gradient_accumulation=1, reference_series=output,
            variability_reference_manifest=Path(config['variability_reference_manifest']).resolve())
        local = dict(stages=selected_stages(arguments, dataset, output/name), dataset=files,
            sources=source_hashes(), seed=config['seed'], model_size=model,
            interpretation='Fresh initialization; Future and Joint; common validation; encoder and dataset size recorded separately')
        local_plans[name] = local
        cases.append(case)
    if len(validation_hashes) != 1:
        raise ValueError('Every experiment must use the identical validation manifest')
    sizes = sorted(sets)
    if any(not sets[a] < sets[b] for a, b in zip(sizes, sizes[1:])):
        raise ValueError('Increasing training sizes must be strict nested supersets')
    sources = {**source_hashes(), **{name:digest(ROOT/name) for name in (
        'scripts/run_seviri_ssl_matrix.py', 'scripts/run_seviri_ssl_selected.py',
        'scripts/run_seviri_ssl_scaling.py', 'scripts/run_seviri_ssl_comparison.py',
        'scripts/build_seviri_ssl_new_dates.py', 'scripts/build_seviri_ssl_extension.py',
        'scripts/build_seviri_ssl_scaling_dataset.py', 'scripts/build_seviri_ssl_pilot_dataset.py',
        'scripts/extend_seviri_frame_cache.py', 'src/emma_gpm/ssl_scaling.py',
        'src/emma_gpm/ssl_dataset.py', 'src/emma_gpm/ssl_sampling.py',
        'src/emma_gpm/seviri.py', 'src/emma_gpm/pretraining_validation.py',
        'src/emma_gpm/file_lock.py')}}
    plan = dict(cases=cases, order=[dict(case=c['name'], stage=s['name'])
        for c in cases for s in local_plans[c['name']]['stages']], sources=sources,
        epochs=config['epochs'], batch_size=config['batch_size'], effective_batch_size=config['batch_size'],
        gradient_accumulation=1, workers=config['workers'], seed=config['seed'], probe_epochs=0,
        variability_reference_manifest=str(Path(config['variability_reference_manifest']).resolve()),
        variability_reference_sha256=digest(Path(config['variability_reference_manifest'])),
        comparison='Same optimizer settings and validation; compare both equal epochs and equal optimizer updates',
        initialization='Random weights from the same seed within each encoder architecture',
        strict_nesting=True, validation_identical=True)
    return plan, local_plans


def verify(output, plan, *, source_data=False):
    for filename, expected in plan['sources'].items():
        if digest(ROOT/filename) != expected:
            raise ValueError(f'Frozen execution source changed: {filename}')
    if digest(Path(plan['variability_reference_manifest'])) != plan['variability_reference_sha256']:
        raise ValueError('Common variability reference changed')
    checked = set()
    for case in plan['cases']:
        dataset = output / case['name'] / 'dataset'
        for filename, expected in case['files'].items():
            if digest(dataset/filename) != expected:
                raise ValueError(f'Frozen case dataset changed: {dataset/filename}')
        if digest(Path(case['frame_cache'])/'metadata.json') != case['frame_metadata_sha256']:
            raise ValueError('Frame cache metadata changed')
        key = (case['zarr_root'], case['files']['source_snapshot.json'])
        if source_data and key not in checked:
            verify_snapshot(Path(case['zarr_root']), json.loads((dataset/'source_snapshot.json').read_text()))
            checked.add(key)


def run(args):
    config = json.loads(args.config.read_text())
    if min(config['epochs'], config['batch_size'], config['workers']) < 1 or config['batch_size'] % 2:
        raise ValueError('Positive epochs/workers and an even batch size are required')
    plan, plans = make_plans(args.output_dir, config)
    immutable_json(args.output_dir/'matrix_config.json', config)
    for case in plan['cases']:
        immutable_json(args.output_dir/case['name']/'comparison_plan.json', plans[case['name']])
    immutable_json(args.output_dir/'dataset_ready.json', dict(status='completed', cases=plan['cases'],
        strict_nesting=True, validation_identical=True))
    plan['dataset_ready_sha256'] = digest(args.output_dir/'dataset_ready.json')
    immutable_json(args.output_dir/'scaling_plan.json', plan)
    for filename, expected in plan['sources'].items():
        target = args.output_dir/'source_archive'/filename
        if target.exists():
            if digest(target) != expected: raise ValueError(f'Source archive changed: {target}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/filename, target)
    verify(args.output_dir, plan, source_data=True)
    # Full array digests, source stamps and manifest coverage are also checked by
    # every trainer. Here each distinct cache/manifest pair is checked pre-launch.
    checked = set()
    for case in plan['cases']:
        for manifest in ('train.csv', 'validation.csv'):
            key = (case['frame_cache'], case['files'][manifest])
            if key not in checked:
                dataset = args.output_dir/case['name']/'dataset'/manifest
                data = FrameFileDataset(case['frame_cache'], dataset, zarr_root=case['zarr_root'])
                data.close()
                checked.add(key)
    publish_summary(args.output_dir)
    print('Verified frozen plans:', len(plan['order']), 'experiments', flush=True)
    if args.prepare_only: return
    write_json(args.output_dir/'launcher.json', dict(pid=os.getpid(), command=list(sys.orig_argv),
        log=str(args.output_dir/'queue.log'), started_at_utc=datetime.now(timezone.utc).isoformat()))
    with ExitStack() as locks:
        for case in plan['cases']:
            locks.enter_context(exclusive_file_lock(args.output_dir/case['name']/'.comparison.lock'))
        for item in plan['order']:
            verify(args.output_dir, plan)
            queue = args.output_dir/item['case']
            stage = next(s for s in plans[item['case']]['stages'] if s['name'] == item['stage'])
            if stage_completed(queue, stage):
                print('Already completed', item['case'], stage['name'], flush=True)
                continue
            execute(args, stage_command(queue, stage), item['case'], stage['name'], queue/(stage['name']+'.log'))
            if not stage_completed(queue, stage):
                raise RuntimeError('Trainer exited before completing the requested epochs')
            if all(stage_completed(queue, s) for s in plans[item['case']]['stages']):
                write_json(queue/'queue_status.json', dict(status='completed', training_completed=True, stages=2))
        verify(args.output_dir, plan, source_data=True)
        publish_summary(args.output_dir)
        write_json(args.output_dir/'queue_status.json', dict(status='completed', training_completed=True,
            stages=len(plan['order']), completed_at_utc=datetime.now(timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output_dir, args.config = args.output_dir.resolve(), args.config.resolve()
    with exclusive_file_lock(args.output_dir/'.scaling.lock'):
        try:
            run(args)
        except Exception as exc:
            status_path = args.output_dir/'queue_status.json'
            previous = json.loads(status_path.read_text()) if status_path.exists() else {}
            write_json(status_path, dict(previous, status='failed', error=str(exc)))
            raise


if __name__ == '__main__':
    main()
