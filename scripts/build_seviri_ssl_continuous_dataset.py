#!/usr/bin/env python3
"""Sample complete SEVIRI windows across daily stores, retaining fixed clips."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.seviri_clips import VIDEO_MAE_CHANNELS
from emma_gpm.seviri_frame_cache import file_digest, source_stamp
from emma_gpm.spatial_sampling import CropEvent, verify_overlap
from emma_gpm.ssl_continuous_sampling import (continuous_windows, event_array,
    preparation_phase, temporal_context, training_time_mask)
from emma_gpm.ssl_incremental_sampling import augment_events, audit_added_pairs, balanced_extension
from emma_gpm.ssl_random_sampling import geographic_origin_mask
from emma_gpm.ssl_seasonal_sampling import frame_pixel_validity, valid_crop_origins_from_pixels
from emma_gpm.ssl_spaced_sampling import window_crop_moments, balanced_subset
from scripts.build_seviri_ssl_calendar_dataset import physical_support, write_json
from scripts.build_seviri_ssl_spaced_dataset import SOURCES as SHARED_SOURCES, save_clip, finalize

SOURCES = (*SHARED_SOURCES, 'src/emma_gpm/ssl_incremental_sampling.py',
    'src/emma_gpm/ssl_continuous_sampling.py', 'src/emma_gpm/ssl_validation_set.py',
    'scripts/build_seviri_ssl_continuous_dataset.py')


def load_observations(record, config, *, prefix=False):
    """Read one daily store or its first 75 minutes, retaining physical gaps."""
    source = Path(config['archive'])/record['store']
    before = source_stamp(source)
    for name, expected in record['headers'].items():
        original = Path(config['archive'])/name
        if file_digest(original) != expected:
            raise ValueError('Frozen observation header changed: '+name)
        destination = Path(config['output'])/'sources'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Neighbor tasks can request identical headers; replace atomically.
        temporary = destination.with_name(destination.name+f'.{os.getpid()}.part')
        temporary.write_bytes(original.read_bytes())
        temporary.replace(destination)
    with xr.open_zarr(source, consolidated=True, chunks=None) as ds:
        if tuple(ds.channel.values) != VIDEO_MAE_CHANNELS or ds.seviri.dtype != np.float16:
            raise ValueError('Expected seven native float16 Kelvin channels')
        times = pd.DatetimeIndex(pd.to_datetime(record['times'], utc=True)).as_unit('ns')
        actual = pd.DatetimeIndex(pd.to_datetime(ds.time.values, utc=True)).as_unit('ns')
        if (not times.equals(actual) or list(ds.seviri.shape) != record['shape']
                or not times.is_unique or not times.is_monotonic_increasing):
            raise ValueError('Native dimensions or observations differ from the inventory')
        day = pd.Timestamp(Path(record['store']).stem, tz='UTC')
        if len(times) and not (times.normalize() == day).all():
            raise ValueError('The daily source contains observations outside its named UTC date')
        indices = np.flatnonzero(times < day+pd.Timedelta(minutes=75)) if prefix else np.arange(len(times))
        values = ds.seviri.isel(time=indices).transpose('time', 'channel', 'y', 'x').values
        grid = (ds.x.values, ds.y.values, ds.attrs['projection_wkt'])
    valid = frame_pixel_validity(values)
    present = physical_support(record)[indices]
    cy, cx = record['chunks'][2:]
    h, w = values.shape[-2:]
    for i in range(len(indices)):
        if not present[i].all():
            valid[i] &= np.repeat(np.repeat(present[i], cy, 0), cx, 1)[:h, :w]
    if source_stamp(source) != before:
        raise ValueError('Observation source changed while reading')
    return values, valid, times[indices], grid, before


def prepare_day(task):
    import zarr
    from numcodecs import blosc
    zarr.config.set({'async.concurrency': 8, 'threading.max_workers': 4})
    blosc.set_nthreads(1)
    record, following, config, context_rows, hashes = task
    output = Path(config['output'])
    day = pd.Timestamp(Path(record['store']).stem).strftime('%Y-%m-%d')
    identity = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    committed = output/'daily'/f'{day}.json'
    if committed.exists():
        result = json.loads(committed.read_text())
        if result['identity'] != identity:
            raise ValueError('Frozen continuous preparation changed: '+day)
        return result
    started = time.monotonic()
    context = pd.DataFrame(context_rows, columns=['start_time', 'origin_y', 'origin_x'])
    fixed = event_array(context)
    rows, files = [], {}
    parent = Path(config['parent_dataset']) if config.get('parent_dataset') else None
    if parent and (parent/'daily'/f'{day}.json').exists():
        old = json.loads((parent/'daily'/f'{day}.json').read_text())
        rows = old['clips']
        files = old['clip_files'].copy()
        for name in files:
            original, destination = parent/'candidate_clips'/name, output/'candidate_clips'/name
            if not destination.exists():
                os.link(original, destination)
            if not os.path.samefile(original, destination):
                raise ValueError('Reuse must retain the exact existing candidate file')
    values, valid, times, grid, before = load_observations(record, config)
    stamps = {record['store']: before}
    x, y, projection = grid
    if following:
        tail, tail_valid, tail_times, tail_grid, tail_stamp = load_observations(following, config, prefix=True)
        if (not np.array_equal(x, tail_grid[0]) or not np.array_equal(y, tail_grid[1])
                or projection != tail_grid[2]):
            raise ValueError('A continuous window must use identical native coordinate grids')
        values = np.concatenate([values, tail])
        valid = np.concatenate([valid, tail_valid])
        times = times.append(tail_times)
        stamps[following['store']] = tail_stamp
        del tail, tail_valid
    geographic = geographic_origin_mask(x, y, projection, config['domain'])
    windows = continuous_windows(times, day, allowed=training_time_mask(
        times, config['reserved_days'], config['temporal_separation_hours']))
    masks, origins = {}, {}
    for slot, indices in windows.items():
        pixels = valid[indices].all(axis=0)
        key = np.packbits(pixels).tobytes()
        if key not in masks:
            masks[key] = np.flatnonzero(valid_crop_origins_from_pixels(pixels, geographic))
        if len(masks[key]):
            origins[slot] = masks[key]
    if origins:
        added, blocks, reason = augment_events(fixed, origins, geographic.shape[1],
            seed=np.random.SeedSequence([config['seed'], pd.Timestamp(day).toordinal(), 3]),
            max_iou=config['max_iou'], max_shared_fraction=config['max_shared_fraction'],
            block_size=config['block_size'], minimum_acceptance=config['minimum_acceptance'],
            patience=config['patience'], max_proposals=config['max_proposals'])
    else:
        added, blocks, reason = np.empty((0, 3), np.int64), [], 'no_complete_windows'
    audit = audit_added_pairs(np.concatenate([fixed, added]), len(fixed))
    if (audit['maximum_active_iou'] > config['max_iou']+1e-12 or
            audit['maximum_new_pair_shared_fraction'] > config['max_shared_fraction']+1e-12):
        raise ValueError('Independent neighboring-day pair audit failed')
    write_json(output/'sampling_daily'/f'{day}.json', dict(day=day,
        added_samples=len(added), blocks=blocks, stopping_reason=reason, **audit))
    added = added[np.lexsort((added[:, 2], added[:, 1], added[:, 0]))]
    added_rows, moments_windows = [], []
    for slot in np.unique(added[:, 0]):
        positions = added[added[:, 0] == slot, 1:]
        indices = windows[int(slot)]
        stamp = times[indices[0]]
        moments_windows.append((indices, positions))
        for oy, ox in positions:
            name, entry = save_clip(values, indices, CropEvent(stamp, int(oy), int(ox)),
                                    x, y, output/'candidate_clips')
            files[name] = entry
            added_rows.append(dict(start_time=stamp.isoformat().replace('+00:00', 'Z'),
                origin_y=int(oy), origin_x=int(ox), day=day, time_bin=stamp.hour//6,
                store=record['store'], clip_id=Path(name).stem, file=name))
    offset = 0
    for means, variances in window_crop_moments(values, moments_windows):
        for mean, variance in zip(means, variances, strict=True):
            added_rows[offset].update(mean_k=float(mean), variance_k2=float(variance))
            offset += 1
    rows = rows+added_rows
    for store, stamp in stamps.items():
        if source_stamp(Path(config['archive'])/store) != stamp:
            raise ValueError('Observations changed during continuous preparation')
    result = dict(identity=identity, day=day, store=record['store'], shape=record['shape'],
        source_stamp=before, source_stamps=stamps, clips=rows, clip_files=files,
        fixed_samples=len(rows)-len(added), added_samples=len(added), blocks=blocks,
        stopping_reason=reason, newly_materialized_samples=len(added),
        complete_windows=len(origins), complete_cross_midnight_windows=sum(
            times[windows[t][0]].normalize() != times[windows[t][-1]].normalize() for t in origins),
        elapsed_seconds=time.monotonic()-started,
        maximum_concurrent_iou=audit['maximum_active_iou'], **audit)
    write_json(committed, result)
    return result


def build(args):
    config = json.loads(args.config.read_text())
    output = Path(config['output']).resolve()
    parent = Path(config['parent_dataset']).resolve() if config.get('parent_dataset') else None
    if parent and (output == parent or parent in output.parents):
        raise ValueError('Use a separate sibling dataset directory')
    if config['sampling_mode'] != 'continuous_uniform':
        raise ValueError('Continuous sampling mode required')
    inventory = json.loads(Path(config['inventory']).read_text())
    reference = Path(config['reference_dataset'])
    protocol = json.loads((reference/'training/dataset/protocol.json').read_text())
    if config['domain'] != protocol['domain_lonlat']:
        raise ValueError('Preserve the established geographic domain')
    columns = ['start_time', 'origin_y', 'origin_x', 'day', 'time_bin', 'store', 'clip_id',
               'file', 'mean_k', 'variance_k2']
    fixed = pd.read_csv(parent/'candidates.csv') if parent else pd.DataFrame(columns=columns)
    preserved = pd.read_csv(parent/'training_crop_moments.csv') if parent else fixed
    if not set(preserved.clip_id) <= set(fixed.clip_id):
        raise ValueError('The candidate pool must contain every preserved clip')
    config.update(output=str(output), archive=inventory['root'],
        reserved_days=protocol['reserved_days'], temporal_separation_hours=24,
        parent_samples=len(preserved), preserved_candidates=len(fixed),
        parent_manifest_sha256=file_digest(parent/'training/dataset/train.csv') if parent else None,
        parent_candidates_sha256=file_digest(parent/'candidates.csv') if parent else None)
    records = {}
    for record in inventory['stores']:
        day = pd.Timestamp(Path(record['store']).stem).strftime('%Y-%m-%d')
        if config.get('training_days') is not None and day not in config['training_days']:
            continue
        if not training_time_mask(record['times'], config['reserved_days'], 24).any():
            continue
        if day in records:
            raise ValueError('Each training date must have one inventoried native store')
        records[day] = record
    if not set(fixed.day) <= set(records):
        raise ValueError('Every preserved candidate date must remain eligible')
    hashes = {name:file_digest(ROOT/name) for name in SOURCES}
    plan = dict(config=config, sources=hashes, records=records)
    path = output/'preparation_plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Frozen continuous preparation plan changed')
    write_json(path, plan)
    for name in SOURCES:
        target = output/'preparation_sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    for name in ('candidate_clips', 'daily', 'sources'):
        (output/name).mkdir(exist_ok=True)
    results = []
    # Days in the same phase cannot share observations (clip duration < 24h).
    # Phase 1 sees every phase-0 addition and the entire frozen candidate pool.
    # Task order and accepted sets therefore do not depend on worker scheduling.
    for phase in (0, 1):
        available = pd.concat([fixed, pd.DataFrame([row for result in results
            for row in result['clips']])], ignore_index=True).drop_duplicates('clip_id')
        tasks = []
        for day, record in sorted(records.items()):
            if preparation_phase(day) != phase:
                continue
            next_day = (pd.Timestamp(day)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            context = temporal_context(available, day)
            tasks.append((record, records.get(next_day), config,
                context[['start_time', 'origin_y', 'origin_x']].to_dict('records'), hashes))
        if args.limit_days:
            tasks = tasks[:max(0, args.limit_days-len(results))]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
            for result in pool.map(prepare_day, tasks, chunksize=1):
                results.append(result)
                write_json(output/'progress.json', dict(status='continuous_sampling', phase=phase,
                    completed_days=len(results), total_days=len(records), last_day=result['day'],
                    added_candidates=sum(r['added_samples'] for r in results), updated_unix=time.time()))
                print(f"{len(results)}/{len(records)} {result['day']}: {result['added_samples']} additions; "
                    f"{result['complete_cross_midnight_windows']} cross-midnight windows; "
                    f"{result['elapsed_seconds']:.0f} s", flush=True)
        if args.limit_days and len(results) >= args.limit_days and len(results) < len(records):
            return
    results.sort(key=lambda r:r['day'])
    candidates = pd.DataFrame([row for r in results for row in r['clips']])
    selected = (balanced_extension(preserved, candidates, seed=config['seed']+3,
                                  target=config['target_samples']) if len(preserved)
        else balanced_subset(candidates, config['target_samples']//4*4, config['seed']+3))
    if not set(preserved.clip_id) <= set(selected.clip_id):
        raise ValueError('Every previous selected clip must be retained')
    verify_overlap([CropEvent(pd.Timestamp(r.start_time), r.origin_y, r.origin_x)
        for r in candidates.itertuples()], max_iou=config['max_iou'])
    if parent and (file_digest(parent/'candidates.csv') != config['parent_candidates_sha256']
            or file_digest(parent/'training/dataset/train.csv') != config['parent_manifest_sha256']):
        raise ValueError('Parent manifests changed during preparation')
    finalize(config, results, hashes, selected=selected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit-days', type=int, help='Prepare an initial subset; resume with the same frozen config')
    args = parser.parse_args()
    output = Path(json.loads(args.config.read_text())['output'])
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(output/'.preparation.lock'):
        build(args)
