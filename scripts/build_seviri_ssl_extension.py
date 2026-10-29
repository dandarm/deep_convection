#!/usr/bin/env python3
"""Extend an existing training manifest within its dates, preserving validation."""
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
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.seviri_clip_files import clip_key
from emma_gpm.ssl_dataset import (available_candidates, balanced_candidate_order,
    source_files_for_manifests, file_digest, verify_snapshot)
from emma_gpm.ssl_sampling import stratified_indices
from scripts.build_seviri_ssl_pilot_dataset import init_worker
from scripts.build_seviri_ssl_scaling_dataset import collect_cached, global_moments, MANIFEST_COLUMNS
from scripts.pretrain_seviri_ssl import write_json, validate_manifests
from scripts.run_seviri_ssl_scaling import immutable_json


def keys(frame):
    return [clip_key(r.start_time, r.origin_y, r.origin_x) for r in frame.itertuples(index=False)]


def unused_candidates(candidates, existing):
    selected = set(keys(existing))
    return candidates.loc[[key not in selected for key in keys(candidates)]].reset_index(drop=True)


def assemble(base, pool, count, seed):
    """Retain every base clip and stratify only the disjoint addition pool."""
    if len(set(keys(base))) != len(base) or len(set(keys(pool))) != len(pool):
        raise ValueError('Duplicate clips in the base or addition pool')
    if set(keys(base)) & set(keys(pool)):
        raise ValueError('The addition pool overlaps the existing training manifest')
    if count <= len(base) or count - len(base) > len(pool):
        raise ValueError('Requested extension exceeds the validated addition pool')
    indices, sampling = stratified_indices(pool.mean_abs_change_k, count-len(base), seed=seed)
    extra = pool.iloc[indices].reset_index(drop=True)
    result = pd.concat([base[MANIFEST_COLUMNS], extra[MANIFEST_COLUMNS]], ignore_index=True)
    return result, extra, indices, sampling


def merge_candidate_caches(paths):
    """Reuse identical validated moments across overlapping historical caches."""
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = frame.loc[frame.valid.eq(True)].copy()
    frame['_key'] = keys(frame)
    columns = ['mean_k', 'variance_k2', 'mean_abs_change_k', 'p90_abs_change_k', 'fraction_change_gt2k']
    if not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError('Historical candidate moments must be finite')
    for _, group in frame.loc[frame._key.duplicated(keep=False)].groupby('_key'):
        if not np.allclose(group[columns], group[columns].iloc[0], rtol=1e-12, atol=1e-12):
            raise ValueError('Conflicting historical candidate moments')
    return frame.drop_duplicates('_key').drop(columns='_key').reset_index(drop=True)


def init_candidate_worker(root, frame_cache):
    init_worker(root)
    if frame_cache:
        from emma_gpm.seviri_frame_cache import SeviriFrameCacheReader
        from scripts import build_seviri_ssl_pilot_dataset as pilot
        pilot.READER.close()
        pilot.READER = SeviriFrameCacheReader(root, frame_cache)


def build(args):
    out = args.output_dir
    base_root = args.base_series
    base_data = args.base_dataset or base_root / 'n8192/dataset'
    base = pd.read_csv(base_data / 'train.csv')
    validation = pd.read_csv(base_data / 'validation.csv')
    old_protocol = json.loads((base_data / 'protocol.json').read_text())
    original = Path(old_protocol['baseline_dataset'])
    pool_paths = args.candidate_cache or [original / 'candidate_variability.csv', base_root / 'additional_candidate_variability.csv']
    inventory_path = base_data/'inventory.json' if (base_data/'inventory.json').exists() else base_root/'inventory.json'
    inventory = json.loads(inventory_path.read_text())
    existing = merge_candidate_caches(pool_paths)
    base_lookup = existing.assign(_key=keys(existing)).set_index('_key')
    base_moments = base_lookup.loc[keys(base)]
    identity = dict(base_dataset=str(base_data), frame_cache=str(args.frame_cache) if args.frame_cache else None,
        frame_metadata_sha256=file_digest(args.frame_cache/'metadata.json') if args.frame_cache else None, base_series=str(base_root), zarr_root=str(args.zarr_root), count=args.count,
        candidates=args.candidates, seed=args.seed, stride=112, training_days=sorted(base.day.unique()),
        files={str(path): file_digest(path) for path in [*pool_paths, base_data/'train.csv',
            base_data/'validation.csv', base_data/'channel_stats.json', inventory_path]},
        sources={name: file_digest(ROOT/name) for name in ('scripts/build_seviri_ssl_extension.py',
            'scripts/build_seviri_ssl_pilot_dataset.py','scripts/build_seviri_ssl_scaling_dataset.py',
            'src/emma_gpm/ssl_dataset.py','src/emma_gpm/ssl_sampling.py','src/emma_gpm/seviri_clips.py',
            'src/emma_gpm/seviri_frame_cache.py')})
    immutable_json(out / 'preparation_plan.json', identity)
    ready_path = out / 'dataset_ready.json'
    if ready_path.exists():
        for case in json.loads(ready_path.read_text())['cases']:
            dataset = out / case['name'] / 'dataset'
            for name, expected in case['files'].items():
                if file_digest(dataset/name) != expected:
                    raise ValueError(f'Prepared dataset changed: {name}')
            verify_snapshot(args.zarr_root, json.loads((dataset/'source_snapshot.json').read_text()))
        print('Existing extension verified', flush=True)
        return
    training_stores = set(base.store)
    train_inventory = {**inventory, 'stores': [r for r in inventory['stores'] if r['store'] in training_stores]}
    candidates = unused_candidates(available_candidates(train_inventory, 112), base)
    candidates = balanced_candidate_order(candidates, args.seed)
    validate_manifests(candidates, validation)
    source_paths = source_files_for_manifests(inventory, [base, candidates, validation])
    snapshot_path = out / 'preparation_source_snapshot.json'
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        if set(snapshot) != set(source_paths):
            raise ValueError('Candidate source coverage changed')
        verify_snapshot(args.zarr_root, snapshot)
    else:
        verify_snapshot(args.zarr_root, json.loads((base_data/'source_snapshot.json').read_text()))
        snapshot = {name: file_digest(args.zarr_root/name) for name in source_paths}
        write_json(snapshot_path, snapshot)
    print(f'Frozen {len(snapshot)} source files; {len(candidates)} candidate windows on existing training dates', flush=True)
    cache_path = out / 'additional_pool_checked.csv'
    if not cache_path.exists():
        prior = unused_candidates(existing, base)
        if not set(keys(prior)).issubset(set(keys(candidates))):
            raise ValueError('Previously validated candidates leave the allowed training partition')
        prior.to_csv(cache_path, index=False)
        print(f'Reusing {len(prior)} previously validated unused clips', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=init_candidate_worker, initargs=(args.zarr_root,args.frame_cache)) as executor:
        pool = collect_cached(executor, candidates, args.candidates, out, 'additional_pool')
    expanded, extra, indices, sampling = assemble(base, pool, args.count, args.seed)
    validate_manifests(expanded, validation)
    moments = pd.concat([base_moments[['mean_k','variance_k2']], extra[['mean_k','variance_k2']]])
    mean, std = global_moments(moments)
    verify_snapshot(args.zarr_root, snapshot)
    pool.assign(selected=pool.index.isin(indices)).to_csv(out/'additional_candidate_variability.csv', index=False)
    dataset = out / f'n{len(expanded)}' / 'dataset'
    dataset.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(dataset/'train.csv', index=False)
    shutil.copyfile(base_data/'validation.csv', dataset/'validation.csv')
    write_json(dataset/'channel_stats.json', dict(global_mean_k=mean, global_std_k=std,
        samples=len(expanded), pixel_count=len(expanded)*16*7*224**2,
        normalization='single_global_affine_transform', channels=list(THERMAL_CHANNELS),
        manifest_sha256=file_digest(dataset/'train.csv'), zarr_root=str(args.zarr_root),
        scope='all pixels/channels/16 frames of selected training clips'))
    selected_paths = source_files_for_manifests(inventory, [expanded, validation])
    write_json(dataset/'source_snapshot.json', {name:snapshot[name] for name in selected_paths})
    write_json(dataset/'inventory.json', inventory)
    protocol = dict(status='completed',kind='nested extension on existing training dates',
        train_samples=len(expanded),validation_samples=len(validation),
        train_days=expanded.groupby('day').size().to_dict(),validation_days=validation.groupby('day').size().to_dict(),
        train_last_frame=(pd.to_datetime(expanded.start_time,utc=True).max()+pd.Timedelta(minutes=75)).isoformat(),
        validation_first_frame=pd.to_datetime(validation.start_time,utc=True).min().isoformat(),
        temporal_separation_hours=24,spatial_stride=112,baseline_samples=len(base),
        baseline_dataset=str(base_data),new_samples=len(extra),additional_sampling=sampling,
        new_observation_dates=False,previous_validation_reused_as_training=False,teacher_labels_used=False,
        source_files=len(selected_paths),seed=args.seed)
    write_json(dataset/'protocol.json', protocol)
    write_json(dataset/'progress.json', protocol)
    case=dict(name=f'n{len(expanded)}',train_samples=len(expanded),validation_samples=len(validation),
        files={name:file_digest(dataset/name) for name in ('train.csv','validation.csv','channel_stats.json',
                                                        'source_snapshot.json','protocol.json','inventory.json')})
    ready=dict(status='completed',created_at_utc=datetime.now(timezone.utc).isoformat(),cases=[case],
        strict_nesting=True,validation_identical=file_digest(dataset/'validation.csv')==file_digest(base_data/'validation.csv'),
        source_snapshot_verified=True,additional_sampling=sampling,base_series=str(base_root))
    write_json(ready_path, ready)
    write_json(out/'preparation_progress.json', ready)
    print(f'Ready: {len(expanded)} train, {len(validation)} validation; mean {mean:.1f} K, std {std:.1f} K', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-series',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--zarr-root',type=Path,required=True)
    parser.add_argument('--base-dataset',type=Path)
    parser.add_argument('--candidate-cache',type=Path,nargs='+')
    parser.add_argument('--frame-cache',type=Path)
    parser.add_argument('--count',type=int,default=16384)
    parser.add_argument('--candidates',type=int,default=16384)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20261024)
    args=parser.parse_args()
    for name in ('base_series','output_dir','zarr_root'):
        setattr(args,name,getattr(args,name).resolve())
    if args.base_dataset: args.base_dataset=args.base_dataset.resolve()
    if args.frame_cache: args.frame_cache=args.frame_cache.resolve()
    if args.candidate_cache: args.candidate_cache=[p.resolve() for p in args.candidate_cache]
    if args.workers<1 or args.count<=8192 or args.candidates<args.count-8192:
        parser.error('Positive workers and an adequate extension candidate pool are required')
    with exclusive_file_lock(args.output_dir/'.preparation.lock'):
        build(args)


if __name__=='__main__':
    main()
