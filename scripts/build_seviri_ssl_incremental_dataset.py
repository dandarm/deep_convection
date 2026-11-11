#!/usr/bin/env python3
"""Extend a frozen SEVIRI clip dataset until random proposal acceptance is low."""
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
from emma_gpm.ssl_incremental_sampling import augment_events, audit_added_pairs, balanced_extension
from emma_gpm.ssl_random_sampling import geographic_origin_mask
from emma_gpm.ssl_seasonal_sampling import frame_pixel_validity, valid_crop_origins_from_pixels
from emma_gpm.ssl_spaced_sampling import window_crop_moments
from scripts.build_seviri_ssl_calendar_dataset import physical_support, write_json
from scripts.build_seviri_ssl_spaced_dataset import SOURCES as SHARED_SOURCES, save_clip, finalize

SOURCES = (*SHARED_SOURCES, 'src/emma_gpm/ssl_incremental_sampling.py',
           'scripts/build_seviri_ssl_incremental_dataset.py')


def prepare_day(task):
    import zarr
    from numcodecs import blosc
    zarr.config.set({'async.concurrency': 8, 'threading.max_workers': 4})
    blosc.set_nthreads(1)
    record, config, fixed_rows, hashes = task
    parent, output = Path(config['parent_dataset']), Path(config['output'])
    day = pd.Timestamp(Path(record['store']).stem).strftime('%Y-%m-%d')
    identity = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    committed = output/'daily'/f'{day}.json'
    if committed.exists():
        result = json.loads(committed.read_text())
        if result['identity'] != identity:
            raise ValueError('Frozen incremental sampling parameters changed: '+day)
        return result
    started = time.monotonic()
    source = Path(config['archive'])/record['store']
    before = source_stamp(source)
    for name, expected in record['headers'].items():
        if file_digest(Path(config['archive'])/name) != expected:
            raise ValueError('Frozen observation header changed: '+name)
    parent_day = json.loads((parent/'daily'/f'{day}.json').read_text())
    if before != parent_day['source_stamp']:
        raise ValueError('Parent observation metadata changed: '+day)
    reuse = Path(config['reuse_candidates_from']) if config.get('reuse_candidates_from') else None
    reused_day = json.loads((reuse/'daily'/f'{day}.json').read_text()) if reuse else None
    if reused_day and reused_day['source_stamp'] != before:
        raise ValueError('Reused candidates belong to different observation metadata')
    files, rows = {}, []
    for row in fixed_rows:
        name = row['file']
        original, destination = parent/'clips'/name, output/'candidate_clips'/name
        if not destination.exists():
            os.link(original, destination)
        if not os.path.samefile(original, destination):
            raise ValueError('Parent clips must retain their exact existing files')
        files[name] = parent_day['clip_files'][name]
        rows.append(dict(row))
    midnight = pd.Timestamp(day, tz='UTC')
    fixed = np.array([[int((pd.Timestamp(r['start_time'])-midnight).total_seconds()/300),
        r['origin_y'], r['origin_x']] for r in rows], dtype=np.int64)
    with xr.open_zarr(source, consolidated=True, chunks=None) as ds:
        if tuple(ds.channel.values) != VIDEO_MAE_CHANNELS or ds.seviri.dtype != np.float16:
            raise ValueError('Expected seven native float16 Kelvin channels')
        x, y = ds.x.values, ds.y.values
        geographic = geographic_origin_mask(x, y, ds.attrs['projection_wkt'], config['domain'])
        times = pd.DatetimeIndex(pd.to_datetime(record['times'], utc=True)).as_unit('ns')
        actual = pd.DatetimeIndex(pd.to_datetime(ds.time.values, utc=True)).as_unit('ns')
        if not times.equals(actual) or not times.is_unique or not times.is_monotonic_increasing:
            raise ValueError('Observation times differ from the fixed inventory')
        values = ds.seviri.transpose('time', 'channel', 'y', 'x').values
        if list(values.shape) != record['shape']:
            raise ValueError('Native observation dimensions changed')
        frame_valid = frame_pixel_validity(values)
        present = physical_support(record)
        cy, cx = record['chunks'][2:]
        h, w = values.shape[-2:]
        for i in range(len(times)):
            if not present[i].all():
                frame_valid[i] &= np.repeat(np.repeat(present[i], cy, 0), cx, 1)[:h, :w]
        origins_by_start, windows_by_start, masks = {}, {}, {}
        for i in range(len(times)-15):
            indices = np.arange(i, i+16)
            stamps = times[indices]
            if (not np.all(np.diff(stamps.asi8) == 300_000_000_000)
                    or stamps[0].normalize() != midnight or stamps[-1].normalize() != midnight):
                continue
            slot = int((stamps[0]-midnight).total_seconds()/300)
            pixels = frame_valid[indices].all(axis=0)
            key = np.packbits(pixels).tobytes()
            if key not in masks:
                masks[key] = np.flatnonzero(valid_crop_origins_from_pixels(pixels, geographic))
            if len(masks[key]):
                origins_by_start[slot] = masks[key]
                windows_by_start[slot] = indices
        additions, blocks, reason = augment_events(fixed, origins_by_start, geographic.shape[1],
            seed=np.random.SeedSequence([config['seed'], int(Path(record['store']).stem), 2]),
            max_iou=config['max_iou'], max_shared_fraction=config['max_shared_fraction'],
            block_size=config['block_size'], minimum_acceptance=config['minimum_acceptance'],
            patience=config['patience'], max_proposals=config['max_proposals'])
        reusable = {}
        if reused_day:
            if blocks[:len(reused_day['blocks'])] != reused_day['blocks']:
                raise ValueError('Extended search must reproduce every original proposal block')
            for row in reused_day['clips']:
                slot = int((pd.Timestamp(row['start_time'])-midnight).total_seconds()/300)
                reusable[(slot, row['origin_y'], row['origin_x'])] = row
            if not set(reusable) <= set(map(tuple, np.concatenate([fixed, additions]))):
                raise ValueError('Extended search must retain every previously accepted candidate')
        audit = audit_added_pairs(np.concatenate([fixed, additions]), len(fixed))
        if (audit['maximum_active_iou'] > config['max_iou']+1e-12 or
                audit['maximum_new_pair_shared_fraction'] > config['max_shared_fraction']+1e-12):
            raise ValueError('Independent pair audit failed')
        # Persist the proposal diagnostics before the more expensive clip writes.
        reuse_count = len(reusable) if reused_day else len(fixed)
        new_count = len(fixed)+len(additions)-reuse_count
        write_json(output/'sampling_daily'/f'{day}.json', dict(day=day, fixed_samples=len(fixed),
            added_samples=len(additions), reused_candidate_samples=reuse_count,
            newly_materialized_samples=new_count, stopping_reason=reason, blocks=blocks, **audit))
        additions = additions[np.lexsort((additions[:, 2], additions[:, 1], additions[:, 0]))]
        new_events = []
        for event in additions:
            previous = reusable.get(tuple(event))
            if previous is None:
                new_events.append(event)
                continue
            name = previous['file']
            original, destination = reuse/'candidate_clips'/name, output/'candidate_clips'/name
            if not destination.exists():
                os.link(original, destination)
            if not os.path.samefile(original, destination):
                raise ValueError('Reuse must preserve the exact existing candidate file')
            files[name] = reused_day['clip_files'][name]
            rows.append(dict(previous))
        new_events = np.asarray(new_events, dtype=np.int64).reshape(-1, 3)
        added_rows, windows = [], []
        for slot in np.unique(new_events[:, 0]):
            origins = new_events[new_events[:, 0] == slot, 1:]
            indices = windows_by_start[int(slot)]
            stamp = midnight+pd.Timedelta(minutes=int(slot)*5)
            windows.append((indices, origins))
            for oy, ox in origins:
                event = CropEvent(stamp, int(oy), int(ox))
                name, entry = save_clip(values, indices, event, x, y, output/'candidate_clips')
                files[name] = entry
                added_rows.append(dict(start_time=stamp.isoformat().replace('+00:00', 'Z'),
                    origin_y=int(oy), origin_x=int(ox), day=day, time_bin=stamp.hour//6,
                    store=record['store'], clip_id=Path(name).stem, file=name))
        offset = 0
        for means, variances in window_crop_moments(values, windows):
            for mean, variance in zip(means, variances, strict=True):
                added_rows[offset].update(mean_k=float(mean), variance_k2=float(variance))
                offset += 1
        rows.extend(added_rows)
        del values, frame_valid, masks, origins_by_start
    chronological_audit = verify_overlap([CropEvent(pd.Timestamp(r['start_time']),r['origin_y'],r['origin_x'])
        for r in rows], max_iou=config['max_iou'])
    if source_stamp(source) != before:
        raise ValueError('Observations changed during incremental preparation')
    result = dict(identity=identity, day=day, store=record['store'], shape=record['shape'],
        source_stamp=before, clips=rows, clip_files=files, fixed_samples=len(fixed),
        added_samples=len(additions), blocks=blocks, stopping_reason=reason,
        reused_candidate_samples=reuse_count, newly_materialized_samples=new_count,
        original_proposal_blocks_reproduced=bool(reused_day),
        complete_windows=len(windows_by_start), elapsed_seconds=time.monotonic()-started,
        **chronological_audit, **audit)
    write_json(committed, result)
    return result


def build(args):
    config = json.loads(args.config.read_text())
    parent, output = Path(config['parent_dataset']).resolve(), Path(config['output']).resolve()
    if output == parent or parent in output.parents:
        raise ValueError('The extension must use a separate sibling dataset directory')
    if config['sampling_mode'] != 'incremental_uniform':
        raise ValueError('Incremental sampling mode required')
    fixed = pd.read_csv(parent/'training_crop_moments.csv')
    parent_manifest = parent/'training/dataset/train.csv'
    if set(fixed.clip_id) != set(pd.read_csv(parent_manifest).clip_id):
        raise ValueError('Parent moments and fixed manifest differ')
    inventory = json.loads(Path(config['inventory']).read_text())
    config.update(output=str(output), archive=inventory['root'], parent_dataset=str(parent),
        parent_samples=len(fixed), parent_manifest_sha256=file_digest(parent_manifest))
    preserved = fixed
    preserve_manifest = None
    if config.get('preserve_selected_from'):
        previous = Path(config['preserve_selected_from'])
        preserved = pd.read_csv(previous/'training_crop_moments.csv')
        preserve_manifest = previous/'training/dataset/train.csv'
        if (set(preserved.clip_id) != set(pd.read_csv(preserve_manifest).clip_id)
                or not set(fixed.clip_id) <= set(preserved.clip_id)):
            raise ValueError('The preserved selection must include every original parent clip')
        config.update(preserved_samples=len(preserved), preserved_manifest_sha256=file_digest(preserve_manifest))
    protocol = json.loads((parent/'training/dataset/protocol.json').read_text())
    if config['domain'] != protocol['domain_lonlat']:
        raise ValueError('Preserve the parent geographic domain')
    days = set(fixed.day)
    records = [record for record in inventory['stores']
        if pd.Timestamp(Path(record['store']).stem).strftime('%Y-%m-%d') in days]
    if len(records) != len(days):
        raise ValueError('All original observation dates must be present')
    hashes = {name: file_digest(ROOT/name) for name in SOURCES}
    plan = dict(config=config, sources=hashes, dates=[r['store'] for r in records],
        reference_manifest_sha256=config['parent_manifest_sha256'])
    path = output/'preparation_plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Frozen preparation plan changed')
    write_json(path, plan)
    for name in SOURCES:
        destination = output/'preparation_sources'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    for name in ('candidate_clips', 'daily'):
        (output/name).mkdir(exist_ok=True)
    if not (output/'sources').exists():
        (output/'sources').symlink_to(parent/'sources', target_is_directory=True)
    tasks = [(r, config, fixed[fixed.day == pd.Timestamp(Path(r['store']).stem).strftime('%Y-%m-%d')].to_dict('records'), hashes)
             for r in records]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        for result in pool.map(prepare_day, tasks[:args.limit_days] if args.limit_days else tasks, chunksize=1):
            results.append(result)
            write_json(output/'progress.json', dict(status='incremental_sampling',
                completed_days=len(results), total_days=len(records), parent_samples=len(fixed),
                added_candidates=sum(r['added_samples'] for r in results),
                newly_materialized_samples=sum(r['newly_materialized_samples'] for r in results),
                last_day=result['day'], updated_unix=time.time()))
            print(f"{len(results)}/{len(records)} {result['day']}: {result['fixed_samples']} fixed + "
                  f"{result['added_samples']} accepted additions ({result['newly_materialized_samples']} new files); "
                  f"{result['stopping_reason']}; {result['elapsed_seconds']:.0f} s", flush=True)
    if args.limit_days and args.limit_days < len(tasks):
        return
    candidates = pd.DataFrame([r for result in results for r in result['clips']])
    selected = balanced_extension(preserved, candidates, seed=config['seed']+3)
    if not set(preserved.clip_id) <= set(selected.clip_id) or len(selected) > config['target_samples']:
        raise ValueError('Extension must preserve its parent and respect the sample ceiling')
    if file_digest(parent_manifest) != config['parent_manifest_sha256']:
        raise ValueError('Parent manifest changed during preparation')
    if preserve_manifest and file_digest(preserve_manifest) != config['preserved_manifest_sha256']:
        raise ValueError('The preserved selection changed during preparation')
    finalize(config, results, hashes, selected=selected)
    write_json(output/'incremental_sampling_summary.json', dict(parent_samples=len(fixed),
        total_candidates=len(candidates), selected_samples=len(selected), added_selected=len(selected)-len(fixed),
        preserved_samples=len(preserved), new_selected_since_previous=len(selected)-len(preserved),
        newly_materialized_samples=sum(r['newly_materialized_samples'] for r in results),
        parent_manifest_sha256=config['parent_manifest_sha256'], parent_clips_preserved=True,
        stopping_days=pd.Series([r['stopping_reason'] for r in results]).value_counts().to_dict(),
        added_pair_limit=config['max_shared_fraction']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--limit-days', type=int, help='Prepare an initial subset; resume all days with the same config')
    args = parser.parse_args()
    destination = Path(json.loads(args.config.read_text())['output'])
    destination.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(destination/'.preparation.lock'):
        build(args)
