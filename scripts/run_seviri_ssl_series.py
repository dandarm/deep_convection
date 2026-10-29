#!/usr/bin/env python3
"""Continue the frozen SSL protocol through an explicitly recorded I/O revision."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.ssl_dataset import verify_snapshot
from emma_gpm.ssl_scaling import publish_summary
from scripts.pretrain_seviri_ssl import digest, write_json, source_hashes
from scripts.run_seviri_ssl_scaling import execute, stage_completed, stage_command


def revised_stage(stage, revision, revision_path):
    flags = list(stage['flags'])
    for key, value in revision['runtime'].items():
        flag = '--' + key.replace('_', '-')
        if flag in flags:
            flags[flags.index(flag) + 1] = str(value)
        else:
            flags += [flag, str(value)]
    flags += ['--clip-root', revision['clip_root'], '--execution-revision', str(revision_path)]
    return {**stage, 'flags': flags}


def read_protocol(root):
    plan = json.loads((root / 'scaling_plan.json').read_text())
    plans = {case['name']: json.loads((root / case['name'] / 'comparison_plan.json').read_text())
             for case in plan['cases']}
    return plan, plans


def prepare_revision(args, plan, plans):
    path = args.revision
    if path.exists():
        return json.loads(path.read_text())
    if args.clip_root is None:
        raise ValueError('Creating an execution revision requires --clip-root')
    for name, checksum in plan['sources'].items():
        if digest(args.output_dir / 'source_archive' / name) != checksum:
            raise ValueError(f'Original source archive differs: {name}')
        if name != 'scripts/pretrain_seviri_ssl.py' and digest(ROOT / name) != checksum:
            raise ValueError(f'Scientific implementation changed: {name}')
    parent_sources = next(iter(plans.values()))['sources']
    if any(local['sources'] != parent_sources for local in plans.values()):
        raise ValueError('Original experiment cases have different source identities')
    sources = {**plan['sources'], **source_hashes()}
    for name in ('scripts/run_seviri_ssl_series.py', 'scripts/materialize_ssl_clip_files.py'):
        sources[name] = digest(ROOT / name)
    parents = {}
    for item in plan['order']:
        directory = args.output_dir / item['case'] / item['stage']
        for checkpoint in directory.glob('checkpoint_*.pt'):
            parents[digest(checkpoint)] = str(checkpoint)
    value = dict(version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
        original_plan_sha256=digest(args.output_dir / 'scaling_plan.json'),
        original_case_plans={name: digest(args.output_dir / name / 'comparison_plan.json') for name in plans},
        parent_sources=parent_sources, parent_checkpoints=parents, sources=sources,
        clip_root=str(args.clip_root.resolve()),
        clip_metadata_sha256=digest(args.clip_root / 'metadata.json'),
        runtime=dict(batch_size=32, loader_batch_size=32, gradient_accumulation=1,
                     num_workers=4, prefetch_factor=4),
        reason='Lossless per-clip files, threaded pinned batches and global GPU normalization',
        numerical_note='Effective batch remains 32; floating-point reduction and stochastic mask grouping may differ')
    for name, checksum in sources.items():
        archive = path.parent / 'source_archive' / name
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, archive)
        if digest(archive) != checksum:
            raise ValueError(f'New source archive differs: {name}')
    write_json(path, value)
    return value


def verify_revision(args, revision, plans, *, datasets=False):
    if digest(args.output_dir / 'scaling_plan.json') != revision['original_plan_sha256']:
        raise ValueError('Original scientific protocol changed')
    for case, expected in revision['original_case_plans'].items():
        if digest(args.output_dir / case / 'comparison_plan.json') != expected:
            raise ValueError('Original case protocol changed')
    for name, checksum in revision['sources'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f'Execution source changed: {name}')
    if digest(Path(revision['clip_root']) / 'metadata.json') != revision['clip_metadata_sha256']:
        raise ValueError('Clip-file collection changed')
    if datasets:
        for case, local in plans.items():
            dataset = args.output_dir / case / 'dataset'
            for name, checksum in local['dataset'].items():
                if digest(dataset / name) != checksum:
                    raise ValueError(f'Prepared dataset changed: {dataset / name}')
            verify_snapshot(args.zarr_root, json.loads((dataset / 'source_snapshot.json').read_text()))


def run(args):
    plan, plans = read_protocol(args.output_dir)
    args.zarr_root = Path(plan['zarr_root'])
    revision = prepare_revision(args, plan, plans)
    verify_revision(args, revision, plans, datasets=True)
    if args.prepare_only:
        print('Prepared execution revision', args.revision, flush=True)
        return
    with ExitStack() as locks:
        for case in plans:
            locks.enter_context(exclusive_file_lock(args.output_dir / case / '.comparison.lock'))
        publish_summary(args.output_dir)
        for item in plan['order']:
            verify_revision(args, revision, plans)
            queue = args.output_dir / item['case']
            original = next(s for s in plans[item['case']]['stages'] if s['name'] == item['stage'])
            if stage_completed(queue, original):
                print('Already completed', item['case'], item['stage'], flush=True)
                continue
            if item['stage'].startswith('probe_') and item['stage'] != 'probe_random':
                parent = next(s for s in plans[item['case']]['stages'] if s['name'] == item['stage'][6:])
                if not stage_completed(queue, parent):
                    raise ValueError('Frozen probe requires a completed source encoder')
            stage = revised_stage(original, revision, args.revision)
            execute(args, stage_command(queue, stage), item['case'], item['stage'], queue / (item['stage'] + '.log'))
            if not stage_completed(queue, original):
                raise RuntimeError('Trainer exited before completing the requested stage')
        verify_revision(args, revision, plans, datasets=True)
        for case in plans:
            queue = args.output_dir / case
            execute(args, [sys.executable, '-u', str(ROOT / 'scripts/analyze_seviri_ssl_comparison.py'),
                '--queue-dir', str(queue), '--batch-size', '32', '--workers', '4'], case, 'analysis', queue / 'analysis.log')
            write_json(queue / 'queue_status.json', dict(status='completed', training_and_analysis=True))
        publish_summary(args.output_dir)
        write_json(args.output_dir / 'queue_status.json', dict(status='completed', stages=len(plan['order']),
            training_and_analysis=True, execution_revision=str(args.revision),
            completed_at_utc=datetime.now(timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--revision', type=Path, required=True)
    parser.add_argument('--clip-root', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.revision = args.revision.resolve()
    with exclusive_file_lock(args.output_dir / '.scaling.lock'):
        run(args)


if __name__ == '__main__':
    main()
