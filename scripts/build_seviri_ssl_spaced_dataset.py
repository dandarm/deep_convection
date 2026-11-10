#!/usr/bin/env python3
"""Build calendar-balanced SEVIRI clips with a strict concurrent IoU limit."""
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
from emma_gpm.seviri_clip_files import codec, clip_key, digest_bytes, SHAPE
from emma_gpm.seviri_frame_cache import file_digest, source_stamp
from emma_gpm.spatial_sampling import ActiveCropSampler, verify_overlap
from emma_gpm.ssl_random_sampling import geographic_origin_mask
from emma_gpm.ssl_seasonal_sampling import frame_pixel_validity, valid_crop_origins_from_pixels
from emma_gpm.ssl_spaced_sampling import asynchronous_start_counts, balanced_subset, window_crop_moments
from emma_gpm.ssl_validation_set import assert_day_separation
from scripts.build_seviri_ssl_calendar_dataset import physical_support, write_json


SOURCES = ('scripts/build_seviri_ssl_spaced_dataset.py', 'src/emma_gpm/spatial_sampling.py',
    'src/emma_gpm/ssl_spaced_sampling.py', 'src/emma_gpm/ssl_seasonal_sampling.py',
    'src/emma_gpm/ssl_random_sampling.py', 'src/emma_gpm/seviri_clip_files.py',
    'scripts/build_seviri_ssl_calendar_dataset.py')


def save_clip(values, indices, event, x, y, output):
    raw = values[indices, :, event.y:event.y+224, event.x:event.x+224]
    if y[0] < y[-1]:
        raw = raw[..., ::-1, :]
    if x[0] > x[-1]:
        raw = raw[..., ::-1]
    raw = np.ascontiguousarray(raw)
    if raw.shape != SHAPE or raw.dtype != np.float16:
        raise ValueError('Unexpected materialized clip shape or dtype')
    key = clip_key(event.start, event.y, event.x)
    name = hashlib.sha256(key.encode()).hexdigest()[:24] + '.blosc'
    encoded = codec().encode(raw)
    target = output / name
    temporary = target.with_suffix('.part')
    temporary.write_bytes(encoded)
    reread = temporary.read_bytes()
    decoded = np.empty_like(raw)
    codec().decode(reread, out=decoded)
    if reread != encoded or not np.array_equal(raw.view(np.uint16), decoded.view(np.uint16)):
        raise ValueError('Saved clip differs from its native float16 observations')
    temporary.replace(target)
    return name, dict(key=key, bytes=len(encoded), sha256=digest_bytes(encoded))


def prepare_day(task):
    import zarr
    from numcodecs import blosc
    zarr.config.set({'async.concurrency': 8, 'threading.max_workers': 4})
    blosc.set_nthreads(1)
    record, config, hashes = task
    output, archive = Path(config['output']), Path(config['archive'])
    day = pd.Timestamp(Path(record['store']).stem).strftime('%Y-%m-%d')
    committed = output / 'daily' / (day + '.json')
    identity = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    if committed.exists():
        saved = json.loads(committed.read_text())
        if saved['identity'] != identity:
            raise ValueError('Preparation identity changed: ' + day)
        for name, entry in saved['clip_files'].items():
            if (output/'candidate_clips'/name).stat().st_size != entry['bytes']:
                raise ValueError('Incomplete committed clip: ' + name)
        return saved
    started = time.monotonic()
    source = archive / record['store']
    before = source_stamp(source)
    for name, expected in record['headers'].items():
        contents = (archive/name).read_bytes()
        if digest_bytes(contents) != expected:
            raise ValueError('Inventoried header changed: ' + name)
        target = output/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
    with xr.open_zarr(source, consolidated=True, chunks=None) as ds:
        if tuple(ds.channel.values) != VIDEO_MAE_CHANNELS or ds.seviri.dtype != np.float16:
            raise ValueError('Seven native float16 Kelvin channels are required')
        x, y = ds.x.values, ds.y.values
        geographic = geographic_origin_mask(x, y, ds.attrs['projection_wkt'], config['domain'])
        times = pd.DatetimeIndex(pd.to_datetime(record['times'], utc=True)).as_unit('ns')
        actual = pd.DatetimeIndex(pd.to_datetime(ds.time.values, utc=True)).as_unit('ns')
        if not times.equals(actual) or not times.is_unique or not times.is_monotonic_increasing:
            raise ValueError('Native timestamps differ from the frozen inventory')
        values = ds.seviri.transpose('time', 'channel', 'y', 'x').values
        if list(values.shape) != record['shape']:
            raise ValueError('Native array dimensions changed')
        frame_valid = frame_pixel_validity(values)
        present = physical_support(record)
        cy, cx = record['chunks'][2:]
        h, w = values.shape[-2:]
        # Physical chunk presence remains separate from radiometric validity.
        for i in range(len(times)):
            if not present[i].all():
                frame_valid[i] &= np.repeat(np.repeat(present[i], cy, 0), cx, 1)[:h, :w]
        asynchronous = config.get('sampling_mode') == 'uniform_asynchronous'
        sampler = ActiveCropSampler((h, w), seed=np.random.SeedSequence(
            [config['seed'], int(Path(record['store']).stem)]), max_iou=config['max_iou'],
            attempts=config['sampling_attempts'], proposals_per_start=config.get('proposals_per_start',1),
            coverage_weight=0. if asynchronous else 1., border_boost=0. if asynchronous else .25)
        schedule = None
        if asynchronous:
            counts = asynchronous_start_counts(288, seed=np.random.SeedSequence(
                [config['seed'], int(Path(record['store']).stem), 1]),
                streams=config['temporal_streams'], min_gap_frames=config['min_gap_frames'],
                max_gap_frames=config['max_gap_frames'])
            schedule = dict(zip(pd.date_range(day,periods=len(counts),freq='5min',tz='UTC'), counts))
        rows, windows, files, mask_cache = [], [], {}, {}
        complete_windows = 0
        for index in range(len(times)-15):
            indices = np.arange(index, index+16)
            stamps = times[indices]
            if (not np.all(np.diff(stamps.asi8) == 300_000_000_000)
                    or stamps[0].normalize() != stamps[-1].normalize()):
                continue
            complete_windows += 1
            proposals = int(schedule.get(stamps[0],0)) if schedule is not None else None
            if proposals == 0:
                continue
            pixels = frame_valid[indices].all(axis=0)
            key = np.packbits(pixels).tobytes()
            if key not in mask_cache:
                mask_cache[key] = valid_crop_origins_from_pixels(pixels, geographic)
            selected = sampler.sample_window(stamps[0], mask_cache[key],proposals=proposals)
            if not selected:
                continue
            windows.append((indices, [(event.y, event.x) for event in selected]))
            for event in selected:
                name, entry = save_clip(values, indices, event, x, y, output/'candidate_clips')
                files[name] = entry
                rows.append(dict(start_time=event.start.isoformat().replace('+00:00', 'Z'),
                    origin_y=event.y, origin_x=event.x, day=day, time_bin=event.start.hour//6,
                    store=record['store'], clip_id=Path(name).stem, file=name))
        offset = 0
        for (indices, origins), (means, variances) in zip(windows,
                window_crop_moments(values, windows), strict=True):
            for mean, variance in zip(means, variances, strict=True):
                rows[offset].update(mean_k=float(mean), variance_k2=float(variance))
                offset += 1
        del values, frame_valid
        audit = verify_overlap(sampler.events, max_iou=config['max_iou'])
    if source_stamp(source) != before:
        raise ValueError('Source changed during preparation: ' + record['store'])
    result = dict(identity=identity, day=day, store=record['store'], shape=record['shape'],
        source_stamp=before, complete_windows=complete_windows, clips=rows, clip_files=files,
        **audit, exhausted_windows=sampler.exhausted_windows,
        sampling_mode=config.get('sampling_mode','coverage_dense'),
        planned_temporal_proposals=int(sum(schedule.values())) if schedule is not None else None,
        elapsed_seconds=time.monotonic()-started)
    write_json(committed, result)
    return result


def finalize(config, results, hashes, *, selected=None):
    output, reference = Path(config['output']), Path(config['reference_dataset'])
    candidates = pd.DataFrame([row for result in results for row in result['clips']])
    candidates.to_csv(output/'candidates.csv', index=False)
    if selected is None:
        selected = balanced_subset(candidates, config['target_samples'], config['seed']+1)
    protocol = json.loads((reference/'training/dataset/protocol.json').read_text())
    assert_day_separation(selected, protocol['reserved_days'], 24)
    from scripts.pretrain_seviri_ssl import validate_manifests
    extended = pd.read_csv(reference/'extended/dataset/validation.csv')
    validate_manifests(selected, extended)
    dataset = output/'training/dataset'
    dataset.mkdir(parents=True, exist_ok=True)
    columns = ['start_time', 'origin_y', 'origin_x', 'day', 'time_bin', 'store', 'clip_id']
    selected[columns].to_csv(dataset/'train.csv', index=False)
    selected.to_csv(output/'training_crop_moments.csv', index=False)
    for relative in ('training/dataset/validation.csv', 'extended/dataset/validation.csv'):
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((reference/relative).read_bytes())
    mean = float(selected.mean_k.mean())
    std = float(np.sqrt(np.mean(selected.variance_k2 + (selected.mean_k-mean)**2)))
    write_json(dataset/'channel_stats.json', dict(global_mean_k=mean, global_std_k=std,
        samples=len(selected), pixel_count=len(selected)*np.prod(SHAPE).item(),
        normalization='single_global_affine_transform', channels=list(VIDEO_MAE_CHANNELS),
        manifest_sha256=file_digest(dataset/'train.csv'), zarr_root=str(output/'sources'),
        scope='all pixels, all seven channels, all 16 frames of selected training clips'))
    records = {name: entry for result in results for name, entry in result['clip_files'].items()}
    clip_root = output/'clips'
    clip_root.mkdir(exist_ok=True)
    clips = {}
    for name in selected.file:
        if not (clip_root/name).exists():
            os.link(output/'candidate_clips'/name, clip_root/name)
        clips[name] = records[name]
    metadata = dict(version=1, shape=list(SHAPE), dtype='float16', units='K',
        channels=list(VIDEO_MAE_CHANNELS), orientation='north-up west-left', cadence_minutes=5,
        lossless=True, codec=codec().get_config(), zarr_root=str(output/'sources'), clips=clips,
        manifests={file_digest(dataset/'train.csv'):dict(path=str(dataset/'train.csv'),
            files=selected.file.tolist(), count=len(selected))},
        total_bytes=sum(entry['bytes'] for entry in clips.values()),
        uncompressed_bytes=len(clips)*np.prod(SHAPE).item()*2)
    write_json(clip_root/'metadata.json', metadata)
    from emma_gpm.ssl_archive_coverage import manifest_frame_counts
    frame_counts = manifest_frame_counts(selected)
    total_frames = int(frame_counts.sum())
    shapes = {result['day']:result['shape'] for result in results}
    native_bytes = sum(int(count)*7*np.prod(shapes[day.strftime('%Y-%m-%d')][-2:]).item()*2
        for day, count in frame_counts.items())
    summary = dict(status='sampling_and_training_clips_completed', train_samples=len(selected),
        candidate_samples=len(candidates), target_samples=config['target_samples'],
        target_reached=len(selected)==config['target_samples'],
        candidate_counts_per_month=candidates.groupby(candidates.day.str[:7]).size().to_dict(),
        training_counts_per_month=selected.groupby(selected.day.str[:7]).size().to_dict(),
        training_dates=selected.day.nunique(), unique_start_times=selected.start_time.nunique(),
        distinct_native_frames=total_frames, max_iou_active=config['max_iou'],
        measured_maximum_active_iou=max(result['maximum_concurrent_iou'] for result in results),
        compressed_training_gib=metadata['total_bytes']/2**30,
        uncompressed_training_gib=metadata['uncompressed_bytes']/2**30,
        full_native_frame_training_gib=native_bytes/2**30,
        global_mean_k=mean, global_std_k=std, validation_reference=str(reference),
        validation_clips_materialized=False, training_submitted=False,
        capacity_interpretation='Realized accepted proposals under the configured randomized search; no claim of optimal packing.',
        source_sha256=hashes, manifest_sha256=file_digest(dataset/'train.csv'))
    asynchronous = config.get('sampling_mode') == 'uniform_asynchronous'
    description = ('Independent asynchronous streams; uniform feasible spatial proposals; strict active IoU; calendar-balanced subset'
        if asynchronous else 'Every available complete five-minute start; strict active IoU and cumulative spatial coverage preference; calendar-balanced subset')
    protocol.update(train_samples=len(selected), train_days=selected.groupby('day').size().to_dict(),
        sampling=description, sampling_mode=config.get('sampling_mode','coverage_dense'),
        coverage_weight=0. if asynchronous else 1., border_boost=0. if asynchronous else .25,
        max_iou_active=config['max_iou'], sampling_attempts=config['sampling_attempts'],
        proposals_per_start=config.get('proposals_per_start'),
        candidate_samples=len(candidates), training_unique_start_times=selected.start_time.nunique(),
        zarr_root=str(output/'sources'), clip_root=str(clip_root),
        storage='Lossless precomputed north-up float16 clips; candidate files retained separately',
        validation_reference=str(reference))
    if asynchronous:
        protocol.update(temporal_streams=config['temporal_streams'],
            initial_start_jitter_frames=[0,15], min_gap_frames=config['min_gap_frames'],
            max_gap_frames=config['max_gap_frames'], temporal_schedule='UTC day; independent stream schedules and spatial RNG')
        summary.update(sampling_mode='uniform_asynchronous',temporal_streams=config['temporal_streams'],
            coverage_weight=0.,border_boost=0.)
    if config.get('sampling_mode') == 'incremental_uniform':
        for name in ('temporal_streams','initial_start_jitter_frames','min_gap_frames','max_gap_frames',
                     'surplus_clips_removed_for_equal_rank_sizes'):
            protocol.pop(name, None)
        incremental = dict(parent_dataset=config['parent_dataset'],
            parent_samples=config['parent_samples'], parent_manifest_sha256=config['parent_manifest_sha256'],
            added_samples=len(selected)-config['parent_samples'],
            max_shared_observation_fraction_new=config['max_shared_fraction'],
            shared_fraction_scope='Pairs containing at least one added clip; all parent clips are retained',
            measured_maximum_new_pair_shared_fraction=max(r['maximum_new_pair_shared_fraction'] for r in results),
            sampling_mode='incremental_uniform', coverage_weight=0., border_boost=0.,
            proposal_block_size=config['block_size'], minimum_acceptance_rate=config['minimum_acceptance'],
            stopping_patience=config['patience'], maximum_proposals_per_day=config['max_proposals'])
        protocol.update(incremental, sampling='Fixed parent clips with uniform random complete start times and admissible origins; adaptive rejection stopping',
            temporal_schedule='Uniform proposals over complete five-minute starts; parent events fixed', ddp_world_size=4)
        summary.update(incremental)
    protocol.pop('frame_cache', None)
    protocol.pop('selected_starts_per_six_hour_bin', None)
    write_json(dataset/'protocol.json', protocol)
    for name, expected in hashes.items():
        if file_digest(ROOT/name) != expected:
            raise ValueError('Preparation code changed during execution: ' + name)
    write_json(output/'sampling_ready.json', summary)
    write_json(output/'progress.json', dict(summary, updated_unix=time.time()))
    print(json.dumps(summary, indent=2), flush=True)


def build(args):
    config = json.loads(args.config.read_text())
    if config.get('sampling_mode','coverage_dense') not in ('coverage_dense','uniform_asynchronous'):
        raise ValueError('Unknown spatial/temporal sampling mode')
    if config.get('sampling_mode') == 'uniform_asynchronous':
        asynchronous_start_counts(288,seed=config['seed'],streams=config['temporal_streams'],
            min_gap_frames=config['min_gap_frames'],max_gap_frames=config['max_gap_frames'])
    output = Path(config['output']).resolve()
    config['output'] = str(output)
    inventory = json.loads(Path(config['inventory']).read_text())
    config['archive'] = inventory['root']
    reference = Path(config['reference_dataset'])
    reference_protocol = json.loads((reference/'training/dataset/protocol.json').read_text())
    dates = set(reference_protocol['train_days'])
    records = [record for record in inventory['stores']
        if pd.Timestamp(Path(record['store']).stem).strftime('%Y-%m-%d') in dates]
    if len(records) != len(dates):
        raise ValueError('All reference training dates must be inventoried')
    if config['domain'] != reference_protocol['domain_lonlat']:
        raise ValueError('Keep the reference geographic domain')
    if args.limit_days:
        records = records[:args.limit_days]
    hashes = {name:file_digest(ROOT/name) for name in SOURCES}
    plan = dict(config=config, sources=hashes, dates=[record['store'] for record in records],
        reference_manifest_sha256=file_digest(reference/'training/dataset/train.csv'))
    destination = output/'preparation_plan.json'
    if destination.exists() and json.loads(destination.read_text()) != plan:
        raise ValueError('Frozen preparation plan changed')
    write_json(destination, plan)
    for name in SOURCES:
        target = output/'preparation_sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    (output/'candidate_clips').mkdir(exist_ok=True)
    results = []
    tasks = [(record, config, hashes) for record in records]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        for result in pool.map(prepare_day, tasks, chunksize=1):
            results.append(result)
            progress = dict(status='sampling', completed_days=len(results), total_days=len(records),
                candidate_samples=sum(len(day['clips']) for day in results), last_day=result['day'],
                updated_unix=time.time())
            write_json(output/'progress.json', progress)
            print(f"{len(results)}/{len(records)} {result['day']}: {len(result['clips'])} clips; "
                f"{result['elapsed_seconds']:.0f} s", flush=True)
    finalize(config, results, hashes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit-days', type=int, help='Separate diagnostic output only')
    args = parser.parse_args()
    destination = Path(json.loads(args.config.read_text())['output'])
    destination.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(destination/'.preparation.lock'):
        build(args)
