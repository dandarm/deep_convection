#!/usr/bin/env python3
"""Extend the frozen pilot with new observation dates and a common later holdout."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.ssl_dataset import (inventory_upload, available_candidates, balanced_candidate_order,
    source_files_for_manifests, file_digest, verify_snapshot)
from emma_gpm.ssl_sampling import stratified_indices
from scripts.build_seviri_ssl_pilot_dataset import init_worker, validate_candidate
from scripts.pretrain_seviri_ssl import write_json, validate_manifests

KEYS = ['start_time', 'origin_y', 'origin_x']
MANIFEST_COLUMNS = KEYS + ['day', 'time_bin', 'store', 'mean_abs_change_k',
                          'p90_abs_change_k', 'fraction_change_gt2k']


def expansion_partition(candidates, base, previous_validation):
    """Use later new days for additions; retain the latest whole day for evaluation."""
    boundary = pd.to_datetime(candidates.start_time, utc=True).max().normalize()
    stamps = pd.to_datetime(candidates.start_time, utc=True)
    previous_end = pd.to_datetime(previous_validation.start_time, utc=True).max().normalize()
    additional = candidates[(stamps.dt.normalize() > previous_end) &
                            (stamps + pd.Timedelta(hours=24, minutes=75) < boundary)].copy()
    reserved = pd.date_range('2020-03-03', '2020-03-09', tz='UTC').strftime('%Y-%m-%d')
    additional = additional[~additional.day.isin(reserved)]
    validation = candidates[stamps >= boundary].copy()
    if additional.empty or validation.empty:
        raise ValueError('New complete dates are insufficient for expansion and a later holdout separated by 24 hours')
    validate_manifests(base, validation)
    return additional.reset_index(drop=True), validation.reset_index(drop=True), boundary.isoformat()


def collect_cached(executor, candidates, count, output, name):
    """Commit validation records after each block so interrupted preparation can resume."""
    cache = output / f'{name}_checked.csv'
    records = pd.read_csv(cache).to_dict('records') if cache.exists() else []
    accepted = [row for row in records if row['valid']]
    done = {tuple(row[key] for key in KEYS) for row in records}
    pending = [row for row in candidates.to_dict('records')
               if tuple(row[key] for key in KEYS) not in done]
    for offset in range(0, len(pending), 128):
        if len(accepted) >= count:
            break
        block = list(executor.map(validate_candidate, pending[offset:offset + 128], chunksize=2))
        records.extend(block)
        accepted.extend(row for row in block if row['valid'])
        temporary = cache.with_suffix('.csv.part')
        pd.DataFrame(records).to_csv(temporary, index=False)
        temporary.replace(cache)
        write_json(output / 'preparation_progress.json', dict(status='validating', phase=name,
                   accepted=len(accepted), requested=count, rejected=len(records)-len(accepted)))
        print(f'{name}: {len(accepted)}/{count} valid, {len(records)-len(accepted)} rejected', flush=True)
    if len(accepted) < count:
        raise ValueError(f'Only {len(accepted)} valid clips in {name}; requested {count}')
    return pd.DataFrame(accepted[:count])


def global_moments(frame):
    mean = float(frame.mean_k.mean())
    variance = float(np.mean(frame.variance_k2 + (frame.mean_k - mean)**2))
    if not np.isfinite(mean) or not np.isfinite(variance) or variance <= 0:
        raise ValueError('Invalid pooled training statistics')
    return mean, float(np.sqrt(variance))


def build(args):
    out, previous = args.output_dir, args.base_dataset
    base = pd.read_csv(previous / 'train.csv')
    old_validation = pd.read_csv(previous / 'validation.csv')
    if args.count <= len(base) or args.candidates < args.count-len(base):
        raise ValueError('Expanded size must exceed baseline size and candidate pool must cover the addition')
    identity = dict(base_dataset=str(previous), zarr_root=str(args.zarr_root), count=args.count,
        candidates=args.candidates, validation_count=args.validation_count, stride=args.stride, seed=args.seed,
        previous={name:file_digest(previous/name) for name in
                  ('train.csv','validation.csv','channel_stats.json','source_snapshot.json','candidate_variability.csv')},
        sources={name:file_digest(ROOT/name) for name in ('scripts/build_seviri_ssl_scaling_dataset.py',
            'scripts/build_seviri_ssl_pilot_dataset.py','src/emma_gpm/ssl_dataset.py','src/emma_gpm/ssl_sampling.py')})
    identity_path = out / 'preparation_plan.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('Frozen preparation plan differs')
    else:
        write_json(identity_path, identity)
    if (out / 'dataset_ready.json').exists():
        ready = json.loads((out/'dataset_ready.json').read_text())
        for case in ready['cases']:
            dataset = out/case['name']/'dataset'
            for name, expected in case['files'].items():
                if file_digest(dataset/name) != expected:
                    raise ValueError(f'Prepared dataset changed: {case["name"]}/{name}')
            verify_snapshot(args.zarr_root, json.loads((dataset/'source_snapshot.json').read_text()))
        print('Prepared datasets verified', flush=True)
        return
    verify_snapshot(args.zarr_root, json.loads((previous/'source_snapshot.json').read_text()))
    inventory_path = out/'inventory.json'
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
    else:
        inventory = inventory_upload(args.zarr_root, args.stable_seconds)
        write_json(inventory_path, inventory)
    candidates = available_candidates(inventory, args.stride)
    additions, holdout, boundary = expansion_partition(candidates, base, old_validation)
    additions = balanced_candidate_order(additions, args.seed)
    holdout = balanced_candidate_order(holdout, args.seed+1)
    paths = source_files_for_manifests(inventory, [base, additions, holdout])
    snapshot_path = out/'preparation_source_snapshot.json'
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        if set(snapshot) != set(paths):
            raise ValueError('Frozen candidate source list differs')
        verify_snapshot(args.zarr_root, snapshot)
    else:
        snapshot = {name:file_digest(args.zarr_root/name) for name in paths}
        write_json(snapshot_path, snapshot)
    print(f'Frozen {len(paths)} files; additional candidates={len(additions)}, validation candidates={len(holdout)}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=init_worker, initargs=(args.zarr_root,)) as executor:
        valid = collect_cached(executor, additions, args.candidates, out, 'additional_pool')
        validation = collect_cached(executor, holdout, args.validation_count, out, 'validation_pool')
    indices, sampling = stratified_indices(valid.mean_abs_change_k, args.count-len(base), seed=args.seed)
    extra = valid.iloc[indices].reset_index(drop=True)
    expanded = pd.concat([base, extra[MANIFEST_COLUMNS]], ignore_index=True)
    validate_manifests(expanded, validation)
    baseline_moments = pd.read_csv(previous/'candidate_variability.csv')
    baseline_moments = base[KEYS].merge(baseline_moments, on=KEYS, how='left', validate='one_to_one')
    if baseline_moments[['mean_k','variance_k2']].isna().any().any():
        raise ValueError('Baseline clip moments are missing')
    pooled = pd.concat([baseline_moments[['mean_k','variance_k2']], extra[['mean_k','variance_k2']]], ignore_index=True)
    verify_snapshot(args.zarr_root, snapshot)
    valid.assign(selected=valid.index.isin(indices)).to_csv(out/'additional_candidate_variability.csv', index=False)
    cases = []
    for training, moments in ((base, baseline_moments), (expanded, pooled)):
        name = f'n{len(training)}'
        dataset = out/name/'dataset'
        dataset.mkdir(parents=True, exist_ok=True)
        if len(training) == len(base):
            shutil.copyfile(previous/'train.csv', dataset/'train.csv')
            shutil.copyfile(previous/'channel_stats.json', dataset/'channel_stats.json')
        else:
            training[MANIFEST_COLUMNS].to_csv(dataset/'train.csv', index=False)
            mean, std = global_moments(moments)
            write_json(dataset/'channel_stats.json', dict(global_mean_k=mean, global_std_k=std,
                samples=len(training), pixel_count=len(training)*16*7*224**2,
                normalization='single_global_affine_transform', channels=list(THERMAL_CHANNELS),
                manifest_sha256=file_digest(dataset/'train.csv'), zarr_root=str(args.zarr_root),
                scope='all pixels/channels/16 frames of selected training clips'))
        validation[MANIFEST_COLUMNS].to_csv(dataset/'validation.csv', index=False)
        source_files = source_files_for_manifests(inventory, [training, validation])
        write_json(dataset/'source_snapshot.json', {path:snapshot[path] for path in source_files})
        write_json(dataset/'inventory.json', inventory)
        protocol = dict(status='completed', kind='nested dataset-size comparison', train_samples=len(training),
            validation_samples=len(validation), train_days=training.groupby('day').size().to_dict(),
            validation_days=validation.groupby('day').size().to_dict(),
            train_last_frame=(pd.to_datetime(training.start_time, utc=True).max()+pd.Timedelta(minutes=75)).isoformat(),
            validation_first_frame=pd.to_datetime(validation.start_time, utc=True).min().isoformat(),
            validation_boundary=boundary, temporal_separation_hours=24, spatial_stride=args.stride,
            baseline_samples=len(base), new_samples=len(training)-len(base), baseline_dataset=str(previous),
            additional_sampling=sampling if len(training)>len(base) else None,
            previous_validation_reused_as_training=False, teacher_labels_used=False,
            manifests={file:file_digest(dataset/file) for file in ('train.csv','validation.csv')},
            source_files=len(source_files), seed=args.seed)
        write_json(dataset/'protocol.json', protocol)
        write_json(dataset/'progress.json', protocol)
        cases.append(dict(name=name, train_samples=len(training), validation_samples=len(validation),
            files={file:file_digest(dataset/file) for file in
                ('train.csv','validation.csv','channel_stats.json','source_snapshot.json','protocol.json','inventory.json')}))
    if cases[0]['files']['validation.csv'] != cases[1]['files']['validation.csv']:
        raise RuntimeError('Validation manifests differ')
    ready = dict(status='completed', created_at_utc=datetime.now(timezone.utc).isoformat(), cases=cases,
        strict_nesting=True, validation_identical=True, additional_days=extra.groupby('day').size().to_dict(),
        validation_days=validation.groupby('day').size().to_dict(), additional_sampling=sampling,
        source_snapshot_verified=True, previous_validation_retained_in_historical_pilot=str(previous/'validation.csv'))
    write_json(out/'dataset_ready.json', ready)
    write_json(out/'preparation_progress.json', ready)
    print(json.dumps(ready, indent=2), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dataset',type=Path,required=True)
    parser.add_argument('--zarr-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--count',type=int,default=8192)
    parser.add_argument('--candidates',type=int,default=8192)
    parser.add_argument('--validation-count',type=int,default=512)
    parser.add_argument('--stride',type=int,default=112)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--stable-seconds',type=float,default=10)
    parser.add_argument('--seed',type=int,default=20261024)
    args=parser.parse_args()
    for name in ('base_dataset','zarr_root','output_dir'):
        setattr(args,name,getattr(args,name).resolve())
    if min(args.count,args.candidates,args.validation_count,args.stride,args.workers)<1 or args.stable_seconds<0:
        parser.error('Positive counts, stride and workers and nonnegative stability interval required')
    with exclusive_file_lock(args.output_dir/'.preparation.lock'):
        build(args)


if __name__=='__main__':
    main()
