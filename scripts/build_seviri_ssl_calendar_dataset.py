#!/usr/bin/env python3
"""Build calendar-balanced random crops and retain their native observations."""
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
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.seviri_clips import VIDEO_MAE_CHANNELS
from emma_gpm.seviri_frame_cache import file_digest, source_stamp
from emma_gpm.ssl_random_sampling import geographic_origin_mask, origin_rectangles
from emma_gpm.ssl_seasonal_sampling import (calendar_quotas, crop_moments,
    equal_quotas, selected_time_indices, valid_crop_origins, frame_pixel_validity,
    valid_crop_origins_from_pixels)
from emma_gpm.ssl_validation_set import assert_day_separation


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def physical_support(record):
    t, _, h, w = record['shape']
    cy, cx = record['chunks'][2:]
    present = np.zeros((t, (h+cy-1)//cy, (w+cx-1)//cx), bool)
    for name in record['present']:
        it, channel, y, x = map(int, name.split('.'))
        if channel == 0:
            present[it, y, x] = True
    return present


def eligible_windows(record, geographic):
    """Enumerate complete timestamps with at least one geographically valid crop."""
    times = pd.DatetimeIndex(pd.to_datetime(record['times'], utc=True))
    present = physical_support(record)
    cy, cx = record['chunks'][2:]
    h, w = record['shape'][2:]
    memo, groups = {}, {i: [] for i in range(4)}
    for i, start in enumerate(times):
        stamps = pd.date_range(start, periods=16, freq='5min')
        ids = times.get_indexer(stamps)
        if (ids < 0).any() or stamps[-1].normalize() != start.normalize():
            continue
        complete = present[ids].all(axis=0)
        key = complete.tobytes()
        if key not in memo:
            rect = origin_rectangles(complete, h, w, cy, cx)
            memo[key] = any(geographic[y0:y1, x0:x1].any() for y0, y1, x0, x1 in rect)
        if memo[key]:
            groups[start.hour//6].append(i)
    return groups


def prepare_day(task):
    import zarr
    from numcodecs import blosc
    zarr.config.set({'async.concurrency': 8, 'threading.max_workers': 4})
    blosc.set_nthreads(1)
    record, quotas, options = task
    store = record['store']
    day = pd.Timestamp(Path(store).stem).strftime('%Y-%m-%d')
    output = Path(options['output'])
    committed = output / 'daily' / (day + '.json')
    identity = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    if committed.exists():
        saved = json.loads(committed.read_text())
        if saved['identity'] != identity:
            raise ValueError('Preparation parameters changed for ' + day)
        entry = saved['entry']
        if (output/'frames'/entry['file']).stat().st_size != entry['bytes']:
            raise ValueError('Incomplete committed frame file for ' + day)
        return saved
    started = time.monotonic()
    source = Path(options['archive']) / store
    before = source_stamp(source)
    frozen = output / 'sources'
    # Coordinate/header copies suffice for map views. Training reads the native arrays.
    for name, expected in record['headers'].items():
        contents = (Path(options['archive']) / name).read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError('Inventoried header changed: ' + name)
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
    with xr.open_zarr(source, consolidated=True, chunks=None) as ds:
        if tuple(ds.channel.values) != VIDEO_MAE_CHANNELS or ds.seviri.dtype != np.float16:
            raise ValueError('Expected the seven native float16 Kelvin channels')
        geographic = geographic_origin_mask(ds.x.values, ds.y.values, ds.attrs['projection_wkt'], options['domain'])
        groups = eligible_windows(record, geographic)
        times = pd.DatetimeIndex(pd.to_datetime(record['times'], utc=True))
        existing = options.get('existing_frames')
        if existing:
            metadata = json.loads((Path(existing)/'metadata.json').read_text())
            entry = metadata['stores'][store]
            if (entry['time'] != [t.tz_localize(None).isoformat() for t in times]
                    or entry['x'] != ds.x.values.tolist() or entry['y'] != ds.y.values.tolist()):
                raise ValueError('Existing frame coordinates differ from the inventory')
            values = np.load(Path(existing)/entry['file'], allow_pickle=False)
        else:
            values = ds.seviri.transpose('time', 'channel', 'y', 'x').values
        if values.shape != tuple(record['shape']) or values.dtype != np.float16:
            raise ValueError('Native array shape or dtype differs')
        frame_valid = frame_pixel_validity(values) if existing else None
        present = physical_support(record)
        cy, cx = record['chunks'][2:]
        h, w = record['shape'][2:]
        rng = np.random.default_rng(np.random.SeedSequence([options['seed'], int(Path(store).stem)]))
        records, selected_starts, window_counts = [], [], {}
        for time_bin in range(4):
            selected = []
            for index in rng.permutation(groups[time_bin]):
                indices = times.get_indexer(pd.date_range(times[index], periods=16, freq='5min'))
                physical = np.repeat(np.repeat(present[indices].all(axis=0), cy, 0), cx, 1)[:h, :w]
                valid = (valid_crop_origins_from_pixels(frame_valid[indices].all(axis=0), geographic, physical)
                    if existing else valid_crop_origins(values[indices], geographic, physical))
                locations = np.flatnonzero(valid)
                if len(locations):
                    selected.append((int(index), indices, locations))
                if len(selected) == options['starts_per_bin']:
                    break
            if not selected:
                raise ValueError(f'No radiometrically valid window in {day}, six-hour bin {time_bin}')
            window_counts[time_bin] = len(selected)
            shares = equal_quotas(range(len(selected)), quotas[time_bin], rng)
            for j, (index, indices, locations) in enumerate(selected):
                if shares[j] > len(locations):
                    raise ValueError('Insufficient unique crop origins in selected window')
                chosen = rng.choice(locations, shares[j], replace=False)
                origins = np.column_stack(np.unravel_index(chosen, geographic.shape))
                moments = None if existing else crop_moments(values[indices], origins)
                stamp = times[index].isoformat().replace('+00:00', 'Z')
                selected_starts.append(stamp)
                for slot, (y, x) in enumerate(origins):
                    row = dict(start_time=stamp, origin_y=int(y), origin_x=int(x), day=day, time_bin=time_bin, store=store)
                    if moments is not None:
                        row.update(mean_k=float(moments[0][slot]), variance_k2=float(moments[1][slot]))
                    records.append(row)
        if source_stamp(source) != before:
            raise ValueError('Source changed during preparation: ' + store)
        retained = selected_time_indices(times, selected_starts)
        target = output/'frames'/f'{Path(store).stem}.npy'
        target.parent.mkdir(parents=True, exist_ok=True)
        if existing:
            # Share the full immutable validation-day files, also used by the historical holdout.
            upstream = Path(existing)/entry['file']
            if not target.exists():
                os.link(upstream, target)
            shape, retained_times = entry['shape'], entry['time']
            checksum = entry['sha256']
        else:
            temporary = target.with_suffix('.npy.part')
            shape = [len(retained), *values.shape[1:]]
            cached = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=tuple(shape))
            for slot, index in enumerate(retained):
                cached[slot] = values[index]
            cached.flush()
            del cached
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            reread = np.load(temporary, mmap_mode='r', allow_pickle=False)
            for slot, index in enumerate(retained):
                if not np.array_equal(reread[slot].view(np.uint16), values[index].view(np.uint16)):
                    raise ValueError('Saved native observations differ from the source')
            del reread
            checksum = file_digest(temporary)
            temporary.replace(target)
            retained_times = [times[i].tz_localize(None).isoformat() for i in retained]
        entry = dict(file=target.name, bytes=target.stat().st_size, sha256=checksum,
            source_stamp=source_stamp(frozen/store), shape=shape, dtype='float16', time=retained_times,
            channel=list(VIDEO_MAE_CHANNELS), y=ds.y.values.tolist(), x=ds.x.values.tolist(),
            original_source=str(source), original_source_stamp=before,
            selected_window_count=len(selected_starts), retained_native_frames=len(retained_times))
    result = dict(identity=identity, store=store, day=day, entry=entry, clips=records,
        window_counts=window_counts, elapsed_seconds=time.monotonic()-started)
    write_json(committed, result)
    return result


def build(args):
    config = json.loads(args.config.read_text())
    output = Path(config['output']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(Path(config['inventory']).read_text())
    source_root = Path(inventory['root'])
    validation = json.loads((ROOT/'configs/ssl_validation_extended.json').read_text())['days']
    reserved = set(validation)
    for name in config['probe_reservations']:
        for days in json.loads((ROOT/name).read_text())['splits'].values():
            reserved.update(days)
    training_records, validation_records, excluded = [], [], []
    for record in inventory['stores']:
        day = pd.Timestamp(Path(record['store']).stem)
        date = day.strftime('%Y-%m-%d')
        if day.year != config['year'] or day.month not in config['months']:
            continue
        if date in validation:
            validation_records.append(record)
            continue
        # Reserve complete dates on each side; the final interval check includes 75 minutes.
        if any(abs((day-pd.Timestamp(d)).days) <= 1 for d in reserved):
            excluded.append(dict(day=date, reason='reserved validation/probe period and temporal gap'))
            continue
        with xr.open_zarr(source_root/record['store'], consolidated=True, chunks=None) as ds:
            geographic = geographic_origin_mask(ds.x.values, ds.y.values, ds.attrs['projection_wkt'], config['domain'])
        groups = eligible_windows(record, geographic)
        if any(len(v) < config['starts_per_bin'] for v in groups.values()):
            excluded.append(dict(day=date, reason='insufficient physical windows in a six-hour bin'))
            continue
        training_records.append(record)
    if len(validation_records) != len(validation):
        raise ValueError('The reserved validation observations must all be available')
    options = dict(output=str(output), archive=str(source_root), domain=config['domain'],
                   seed=config['seed'], starts_per_bin=config['starts_per_bin'])
    tasks, train_days = [], [pd.Timestamp(Path(r['store']).stem).strftime('%Y-%m-%d') for r in training_records]
    train_quotas = calendar_quotas(train_days, config['train_samples'], config['seed'])
    val_quotas = calendar_quotas(validation, config['validation_samples'], config['seed']+1)
    for records, quotas, opts in ((training_records, train_quotas, options),
        (validation_records, val_quotas, dict(options, existing_frames=config['validation_frames'],
            seed=config['seed']+1, starts_per_bin=config.get('validation_starts_per_bin', config['starts_per_bin'])))):
        for record in records:
            date = pd.Timestamp(Path(record['store']).stem).strftime('%Y-%m-%d')
            tasks.append((record, quotas[date], opts))
    plan = dict(config=config, excluded=excluded, train_quotas=train_quotas, validation_quotas=val_quotas,
        sources={name:file_digest(ROOT/name) for name in ('scripts/build_seviri_ssl_calendar_dataset.py',
            'src/emma_gpm/ssl_seasonal_sampling.py', 'src/emma_gpm/ssl_random_sampling.py')})
    plan_path = output/'preparation_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != json.loads(json.dumps(plan)):
        raise ValueError('Frozen preparation plan changed')
    write_json(plan_path, plan)
    print(f'Preparing {len(train_days)} training dates and {len(validation)} validation dates', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        for result in pool.map(prepare_day, tasks, chunksize=1):
            results.append(result)
            progress = dict(status='preparing', completed_days=len(results), total_days=len(tasks),
                last_day=result['day'], samples=sum(len(r['clips']) for r in results), updated_unix=time.time())
            write_json(output/'progress.json', progress)
            print(f"{len(results)}/{len(tasks)} {result['day']}: {len(result['clips'])} clips, "
                  f"{result['entry']['bytes']/2**30:.1f} GiB, {result['elapsed_seconds']:.0f} s", flush=True)
    train = pd.DataFrame([clip for result in results if result['day'] in train_days for clip in result['clips']])
    extended = pd.DataFrame([clip for result in results if result['day'] in validation for clip in result['clips']])
    primary_quotas = calendar_quotas(validation, config['primary_validation_samples'], config['seed']+2)
    rng = np.random.default_rng(config['seed']+2)
    primary = pd.concat([group.iloc[rng.choice(len(group), primary_quotas[day][int(hour)], replace=False)]
        for (day, hour), group in extended.groupby(['day', 'time_bin'])])
    for frame in (train, primary, extended):
        frame.sort_values(['day', 'start_time', 'origin_y', 'origin_x'], inplace=True)
        if frame.duplicated(['start_time', 'origin_y', 'origin_x']).any():
            raise ValueError('Duplicate selected clips')
    assert_day_separation(train, reserved, 24)
    from scripts.pretrain_seviri_ssl import validate_manifests
    validate_manifests(train, extended)
    entries = {r['store']:r['entry'] for r in results}
    metadata = dict(version=1, units='K', layout='native time,channel,y,x', lossless=True,
        zarr_root=str(output/'sources'), stores=entries, total_bytes=sum(e['bytes'] for e in entries.values()),
        temporal_storage='union of selected complete RSS windows; native timestamps preserved')
    if metadata['total_bytes'] > config['maximum_frame_gib']*2**30:
        raise ValueError('Prepared observations exceed the planned RAM budget')
    write_json(output/'frames/metadata.json', metadata)
    dataset = output/'training/dataset'
    dataset.mkdir(parents=True, exist_ok=True)
    columns = ['start_time', 'origin_y', 'origin_x', 'day', 'time_bin', 'store']
    train[columns].to_csv(dataset/'train.csv', index=False)
    primary[columns].to_csv(dataset/'validation.csv', index=False)
    mean = float(train.mean_k.mean())
    std = float(np.sqrt(np.mean(train.variance_k2 + np.square(train.mean_k-mean))))
    stats = dict(global_mean_k=mean, global_std_k=std, samples=len(train), pixel_count=len(train)*16*7*224**2,
        normalization='single_global_affine_transform', channels=list(VIDEO_MAE_CHANNELS),
        manifest_sha256=file_digest(dataset/'train.csv'), zarr_root=str(output/'sources'),
        scope='all pixels, all seven channels, all 16 frames of selected training clips')
    write_json(dataset/'channel_stats.json', stats)
    protocol = dict(status='completed', seed=config['seed'], train_samples=len(train), validation_samples=len(primary),
        train_days=train.groupby('day').size().to_dict(), validation_days=primary.groupby('day').size().to_dict(),
        domain_lonlat=config['domain'], geographic_check_passed=True, teacher_labels_used=False,
        temporal_separation_hours=24, reserved_days=sorted(reserved),
        sampling='equal month totals, equal dates within month, equal six-hour bins; uniform valid start times and pixel origins',
        selected_starts_per_six_hour_bin=config['starts_per_bin'],
        validation_starts_per_six_hour_bin=config.get('validation_starts_per_bin', config['starts_per_bin']),
        radiometric_selection='all selected pixels finite and between 100 and 400 K',
        geographic_constraint='every pixel centre in the full crop',
        zarr_root=str(output/'sources'), frame_cache=str(output/'frames'),
        training_unique_start_times=train.start_time.nunique(), validation_unique_start_times=extended.start_time.nunique(),
        historical_validation='results/ssl_validation_10k_20261102 remains a separate fixed reference')
    write_json(dataset/'protocol.json', protocol)
    train.to_csv(output/'training_crop_moments.csv', index=False)
    ext = output/'extended/dataset'
    ext.mkdir(parents=True, exist_ok=True)
    extended[columns].to_csv(ext/'validation.csv', index=False)
    write_json(ext/'protocol.json', dict(protocol, samples=len(extended),
        name='calendar_balanced_validation_10240_20261110', days=validation,
        normalization='global training statistics of the evaluated checkpoint'))
    write_json(ext/'dataset_ready.json', dict(status='completed', files={str(p):file_digest(p)
        for p in (ext/'validation.csv', ext/'protocol.json', output/'frames/metadata.json')}))
    for name, expected in plan['sources'].items():
        if file_digest(ROOT/name) != expected:
            raise ValueError('Dataset preparation implementation changed during execution')
    ready = dict(status='completed', train_samples=len(train), primary_validation_samples=len(primary),
        extended_validation_samples=len(extended), frame_gib=metadata['total_bytes']/2**30,
        files={str(p):file_digest(p) for p in (dataset/'train.csv', dataset/'validation.csv',
            dataset/'channel_stats.json', dataset/'protocol.json', output/'frames/metadata.json')})
    write_json(output/'dataset_ready.json', ready)
    write_json(output/'progress.json', dict(ready, updated_unix=time.time()))
    print(f'Ready: {len(train)} training clips, mean {mean:.1f} K, std {std:.1f} K, '
          f"{ready['frame_gib']:.1f} GiB", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    arguments = parser.parse_args()
    destination = Path(json.loads(arguments.config.read_text())['output'])
    destination.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(destination/'.preparation.lock'):
        build(arguments)
