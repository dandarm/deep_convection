#!/usr/bin/env python3
"""Extend a fixed training set using verified clips from additional dates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import zarr
from numcodecs import blosc

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.seviri_frame_cache import source_stamp, SeviriFrameCacheReader
from emma_gpm.seviri_clips import SeviriZarrClipReader
from emma_gpm.ssl_dataset import (available_candidates, balanced_candidate_order,
    source_files_for_manifests, file_digest, verify_snapshot)
from scripts.build_seviri_ssl_extension import assemble, keys, merge_candidate_caches, init_candidate_worker
from scripts.build_seviri_ssl_scaling_dataset import collect_cached, global_moments
from scripts.extend_seviri_frame_cache import extend
from scripts.pretrain_seviri_ssl import write_json, validate_manifests
from scripts.run_seviri_ssl_scaling import immutable_json


def freeze_store(upload, frozen, record):
    """Copy only inventoried files; require stable content before publication."""
    key = record['store']
    stamp = source_stamp(upload / key)
    names = sorted(set(record['headers']) | {f'{key}/seviri/{n}' for n in record['present']})
    hashes = {}
    for name in names:
        source, target = upload / name, frozen / name
        expected = file_digest(source)
        if name in record['headers'] and expected != record['headers'][name]:
            raise ValueError(f'Inventoried metadata changed: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + '.part')
            shutil.copy2(source, temporary)
            if file_digest(temporary) != expected or file_digest(source) != expected:
                raise ValueError(f'Source changed during copy: {source}')
            temporary.replace(target)
        elif file_digest(target) != expected:
            raise ValueError(f'Frozen destination differs: {target}')
        hashes[name] = expected
    if source_stamp(upload / key) != stamp:
        raise ValueError(f'Upload changed while freezing: {key}')
    print(f'Frozen {key}: {len(hashes)} verified physical files', flush=True)
    return hashes


def rebind_existing_cache(reuse, destination, frozen):
    """Share immutable arrays only when the new root resolves to the same raw stores."""
    previous = json.loads((reuse / 'metadata.json').read_text())
    original = Path(previous['zarr_root'])
    destination.mkdir(parents=True, exist_ok=True)
    for key, entry in previous['stores'].items():
        if (original / key).resolve() != (frozen / key).resolve():
            raise ValueError('Rebinding requires the identical historical source directory')
        if source_stamp(frozen / key) != entry['source_stamp']:
            raise ValueError(f'Historical source changed: {key}')
        source, target = reuse / entry['file'], destination / entry['file']
        if file_digest(source) != entry['sha256']:
            raise ValueError(f'Historical frame array changed: {source}')
        if target.exists():
            if not os.path.samefile(source, target):
                raise ValueError(f'Existing array is not the shared source: {target}')
        else:
            os.link(source, target)
    immutable_json(destination / 'metadata.json', dict(previous, zarr_root=str(frozen),
        rebound_from=str(reuse), original_zarr_root=str(original),
        rebind_verification='identical resolved raw stores, source stamps and array SHA256'))


def check_cache_samples(root, cache, frame):
    records = []
    with SeviriZarrClipReader(root) as raw, SeviriFrameCacheReader(root, cache) as cached:
        for _, row in frame.groupby('day', sort=True).first().reset_index().iterrows():
            kwargs = dict(origin_y=int(row.origin_y), origin_x=int(row.origin_x))
            first = raw.load(row.start_time, **kwargs)
            second = cached.load(row.start_time, **kwargs)
            if not np.array_equal(first, second) or not np.isfinite(first).all():
                raise ValueError(f'Cache verification failed: {row.day}')
            records.append(dict(day=row.day, shape=list(first.shape), exact_equality=True,
                                minimum_k=float(first.min()), maximum_k=float(first.max())))
    return records


def build(args):
    out, frozen = args.output_dir, args.frozen_root
    base = pd.read_csv(args.base_dataset / 'train.csv')
    validation = pd.read_csv(args.base_dataset / 'validation.csv')
    old_stats = json.loads((args.base_dataset / 'channel_stats.json').read_text())
    original = Path(old_stats['zarr_root'])
    inventory = json.loads(args.inventory.read_text())
    selected = {f'{day[:4]}/{day}.zarr' for day in args.days}
    records = [r for r in inventory['stores'] if r['store'] in selected]
    if {r['store'] for r in records} != selected or selected & set(base.store):
        raise ValueError('New dates must be inventoried and disjoint from existing training')
    new_inventory = dict(inventory, stores=records, skipped=[])
    candidates = balanced_candidate_order(available_candidates(new_inventory, 112), args.seed)
    validate_manifests(pd.concat([base, candidates]), validation)
    if args.count <= len(base) or not args.count-len(base) <= args.candidates <= len(candidates):
        raise ValueError('Insufficient physical candidate windows for the requested extension')
    identity = dict(base_dataset=str(args.base_dataset), days=args.days, count=args.count,
        candidates=args.candidates, seed=args.seed, frozen_root=str(frozen), frame_cache=str(args.frame_cache),
        inventory_sha256=file_digest(args.inventory),
        base_files={p.name:file_digest(p) for p in args.base_dataset.iterdir() if p.is_file()},
        candidate_caches={str(p):file_digest(p) for p in args.candidate_cache},
        source_sha256=file_digest(Path(__file__)))
    immutable_json(out / 'preparation_plan.json', identity)
    if (out / 'dataset_ready.json').exists():
        ready = json.loads((out / 'dataset_ready.json').read_text())
        for case in ready['cases']:
            dataset = out / case['name'] / 'dataset'
            for name, checksum in case['files'].items():
                if file_digest(dataset / name) != checksum:
                    raise ValueError(f'Completed dataset changed: {name}')
            verify_snapshot(frozen, json.loads((dataset / 'source_snapshot.json').read_text()))
        print('Completed new-date extension verified', flush=True)
        return
    frozen.mkdir(parents=True, exist_ok=True)
    # Historical stores stay in their original directories. New uploads are copied.
    for year in {Path(store).parts[0] for store in set(base.store) | set(validation.store)}:
        target = frozen / year
        if target.exists():
            if target.resolve() != (original / year).resolve():
                raise ValueError('Historical source alias changed')
        else:
            target.symlink_to(original / year, target_is_directory=True)
    snapshot = json.loads((args.base_dataset / 'source_snapshot.json').read_text())
    verify_snapshot(frozen, snapshot)
    for record in records:
        snapshot.update(freeze_store(args.upload_root, frozen, record))
    write_json(out / 'preparation_source_snapshot.json', snapshot)
    prior_inventory = json.loads((args.base_dataset / 'inventory.json').read_text())
    old_stores = set(base.store) | set(validation.store)
    combined = dict(stores=[r for r in prior_inventory['stores'] if r['store'] in old_stores] + records,
                    skipped=[], stable_seconds=inventory['stable_seconds'])
    write_json(out / 'inventory.json', combined)
    reuse = args.frame_cache / 'historical_arrays'
    rebind_existing_cache(args.reuse_cache, reuse, frozen)
    days = sorted(set(args.days) | {Path(s).stem for s in old_stores})
    extend(frozen, args.frame_cache, reuse, days)
    print(f'Validating up to {args.candidates} clips from {len(candidates)} physical candidates', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
            initializer=init_candidate_worker, initargs=(frozen, args.frame_cache)) as executor:
        pool = collect_cached(executor, candidates, args.candidates, out, 'additional_pool')
    expanded, extra, indices, sampling = assemble(base, pool, args.count, args.seed)
    validate_manifests(expanded, validation)
    historical = merge_candidate_caches(args.candidate_cache)
    historical = historical.assign(_key=keys(historical)).set_index('_key').loc[keys(base)]
    mean, std = global_moments(pd.concat([historical[['mean_k','variance_k2']], extra[['mean_k','variance_k2']]]))
    parity = check_cache_samples(frozen, args.frame_cache, pd.concat([expanded, validation]))
    verify_snapshot(frozen, snapshot)
    pool.assign(selected=pool.index.isin(indices)).to_csv(out / 'additional_candidate_variability.csv', index=False)
    dataset = out / f'n{len(expanded)}' / 'dataset'
    dataset.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(dataset / 'train.csv', index=False)
    shutil.copyfile(args.base_dataset / 'validation.csv', dataset / 'validation.csv')
    write_json(dataset / 'channel_stats.json', dict(global_mean_k=mean, global_std_k=std,
        samples=len(expanded), pixel_count=len(expanded)*16*7*224**2,
        normalization='single_global_affine_transform', channels=list(THERMAL_CHANNELS),
        manifest_sha256=file_digest(dataset / 'train.csv'), zarr_root=str(frozen),
        scope='all pixels/channels/16 frames of selected training clips'))
    paths = source_files_for_manifests(combined, [expanded, validation])
    write_json(dataset / 'source_snapshot.json', {name:snapshot[name] for name in paths})
    write_json(dataset / 'inventory.json', combined)
    protocol = dict(status='completed', kind='nested extension with additional observation dates',
        train_samples=len(expanded), validation_samples=len(validation),
        train_days=expanded.groupby('day').size().to_dict(), validation_days=validation.groupby('day').size().to_dict(),
        baseline_dataset=str(args.base_dataset), baseline_samples=len(base), new_samples=len(extra),
        new_observation_dates=True, additional_dates=args.days, additional_sampling=sampling,
        previous_validation_reused_as_training=False, teacher_labels_used=False,
        temporal_separation_hours=24, spatial_stride=112, source_files=len(paths), seed=args.seed,
        cache_sample_verification=parity,
        interpretation='Increasing clip count and observed date diversity together; validation remains 2020-09-17')
    write_json(dataset / 'protocol.json', protocol)
    write_json(dataset / 'progress.json', protocol)
    case = dict(name=f'n{len(expanded)}', train_samples=len(expanded), validation_samples=len(validation),
        files={name:file_digest(dataset / name) for name in ('train.csv','validation.csv','channel_stats.json',
                                                           'source_snapshot.json','protocol.json','inventory.json')})
    write_json(out / 'dataset_ready.json', dict(status='completed', cases=[case], strict_nesting=True,
        validation_identical=True, source_snapshot_verified=True, frame_cache=str(args.frame_cache), zarr_root=str(frozen)))
    write_json(out / 'preparation_progress.json', dict(status='completed', train_samples=len(expanded)))
    print(f'Dataset ready: {len(expanded)} training clips, {len(validation)} validation clips; mean {mean:.2f} K, std {std:.2f} K', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output-dir','base-dataset','inventory','upload-root','frozen-root','frame-cache','reuse-cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--candidate-cache', type=Path, nargs='+', required=True)
    parser.add_argument('--days', nargs='+', required=True)
    parser.add_argument('--count', type=int, default=65536)
    parser.add_argument('--candidates', type=int, default=49152)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261024)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path): setattr(args, name, value.resolve())
    args.candidate_cache = [p.resolve() for p in args.candidate_cache]
    zarr.config.set({'async.concurrency':8, 'threading.max_workers':4})
    blosc.set_nthreads(1)
    with exclusive_file_lock(args.output_dir / '.prepare.lock'):
        build(args)


if __name__ == '__main__':
    main()
