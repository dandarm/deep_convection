#!/usr/bin/env python3
"""Materialize the unchanged calendar holdouts for a precomputed-clip dataset."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.seviri_frame_cache import file_digest
from emma_gpm.spatial_sampling import CropEvent
from scripts.build_seviri_ssl_spaced_dataset import save_clip
from scripts.build_seviri_ssl_calendar_dataset import write_json


def align_training_ranks(output, world_size=4):
    """Remove at most three surplus clips, preserving balanced monthly totals."""
    from emma_gpm.ssl_archive_coverage import manifest_frame_counts
    summary_path = output/'sampling_ready.json'
    summary = json.loads(summary_path.read_text())
    frame = pd.read_csv(output/'training_crop_moments.csv')
    remove = len(frame) % world_size
    if not remove:
        return
    initial_manifest = output/'training/dataset/train.csv'
    columns = pd.read_csv(initial_manifest, nrows=0).columns.tolist()
    old_hash = file_digest(initial_manifest)
    rng = np.random.default_rng(20261110)
    removed = []
    for _ in range(remove):
        month = frame.groupby(frame.day.str[:7]).size().sort_values(ascending=False).index[0]
        monthly = frame[frame.day.str[:7] == month]
        day = monthly.groupby('day').size().sort_values(ascending=False).index[0]
        daily = monthly[monthly.day == day]
        time_bin = daily.groupby('time_bin').size().sort_values(ascending=False).index[0]
        index = int(rng.choice(daily[daily.time_bin == time_bin].index.to_numpy()))
        removed.append(frame.loc[index, 'clip_id'])
        frame = frame.drop(index)
    metadata_path = output/'clips/metadata.json'
    metadata = json.loads(metadata_path.read_text())
    archive = output/'preparation_history/ddp_sample_alignment'
    archive.mkdir(parents=True, exist_ok=True)
    for path in (summary_path, initial_manifest, metadata_path, output/'training/dataset/channel_stats.json'):
        shutil.copyfile(path, archive/path.name)
    frame[columns].to_csv(initial_manifest, index=False)
    frame.to_csv(output/'training_crop_moments.csv', index=False)
    metadata['manifests'].pop(old_hash)
    metadata['manifests'][file_digest(initial_manifest)] = dict(path=str(initial_manifest),
        files=frame.file.tolist(), count=len(frame))
    metadata['clips'] = {name:metadata['clips'][name] for name in frame.file}
    metadata['total_bytes'] = sum(record['bytes'] for record in metadata['clips'].values())
    metadata['uncompressed_bytes'] = len(frame)*16*7*224*224*2
    write_json(metadata_path, metadata)
    mean = float(frame.mean_k.mean())
    std = float(np.sqrt(np.mean(frame.variance_k2 + (frame.mean_k-mean)**2)))
    stats_path = output/'training/dataset/channel_stats.json'
    stats = json.loads(stats_path.read_text())
    stats.update(samples=len(frame), pixel_count=len(frame)*16*7*224*224,
        manifest_sha256=file_digest(initial_manifest), global_mean_k=mean, global_std_k=std)
    write_json(stats_path, stats)
    protocol_path = output/'training/dataset/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol.update(train_samples=len(frame), train_days=frame.groupby('day').size().to_dict(),
        training_unique_start_times=frame.start_time.nunique(), ddp_world_size=world_size,
        surplus_clips_removed_for_equal_rank_sizes=removed)
    write_json(protocol_path, protocol)
    counts = manifest_frame_counts(frame)
    native_bytes = 0
    for day, count in counts.items():
        shape = json.loads((output/'daily'/f'{day:%Y-%m-%d}.json').read_text())['shape']
        native_bytes += int(count)*7*np.prod(shape[-2:]).item()*2
    summary.update(train_samples=len(frame), unique_start_times=frame.start_time.nunique(),
        training_counts_per_month=frame.groupby(frame.day.str[:7]).size().to_dict(),
        distinct_native_frames=int(counts.sum()), full_native_frame_training_gib=native_bytes/2**30,
        compressed_training_gib=metadata['total_bytes']/2**30,
        uncompressed_training_gib=metadata['uncompressed_bytes']/2**30,
        global_mean_k=mean, global_std_k=std, manifest_sha256=file_digest(initial_manifest),
        ddp_world_size=world_size, surplus_clips_removed_for_equal_rank_sizes=removed)
    write_json(summary_path, summary)
    write_json(archive/'selection.json', dict(world_size=world_size, removed_clip_ids=removed,
        final_samples=len(frame), selection='Largest month, day and six-hour groups; seeded random surplus removal'))


def prepare_day(task):
    from numcodecs import blosc
    blosc.set_nthreads(1)
    source_root, output_root, store, entry, rows = task
    reference, output = Path(source_root), Path(output_root)
    day = rows[0]['day']
    record_path = output/'validation_daily'/f'{day}.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record['source_frame_sha256'] != entry['sha256']:
            raise ValueError('Validation source identity changed')
        for name, info in record['clips'].items():
            if (output/'validation_clips'/name).stat().st_size != info['bytes']:
                raise ValueError('Incomplete validation clip')
        return record
    before = (reference/'frames'/entry['file']).stat()
    values = np.load(reference/'frames'/entry['file'], allow_pickle=False)
    if list(values.shape) != entry['shape'] or values.dtype != np.float16:
        raise ValueError('Unexpected fixed validation frames')
    times = pd.DatetimeIndex(pd.to_datetime(entry['time'], utc=True))
    clips = {}
    for row in rows:
        start = pd.Timestamp(row['start_time'])
        indices = times.get_indexer(pd.date_range(start, periods=16, freq='5min'))
        if (indices < 0).any():
            raise ValueError('Missing fixed validation observation')
        name, info = save_clip(values, indices, CropEvent(start, row['origin_y'], row['origin_x']),
            np.asarray(entry['x']), np.asarray(entry['y']), output/'validation_clips')
        clips[name] = info
    after = (reference/'frames'/entry['file']).stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('Immutable validation frames changed during materialization')
    for path in (reference/'sources'/store).rglob('*'):
        if path.is_file():
            target = output/'sources'/path.relative_to(reference/'sources')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    result = dict(day=day, clips=clips, source_frame_sha256=entry['sha256'])
    write_json(record_path, result)
    return result


def finalize(output, reference, results):
    import hashlib
    from emma_gpm.seviri_clip_files import clip_key
    align_training_ranks(output)
    summary = json.loads((output/'sampling_ready.json').read_text())
    metadata = json.loads((output/'clips/metadata.json').read_text())
    for result in results:
        for name, info in result['clips'].items():
            if not (output/'clips'/name).exists():
                os.link(output/'validation_clips'/name, output/'clips'/name)
            metadata['clips'][name] = info
    paths = [output/'training/dataset/train.csv']
    for role in ('training', 'extended'):
        path = output/role/'dataset/validation.csv'
        if path.read_bytes() != (reference/role/'dataset/validation.csv').read_bytes():
            raise ValueError('The fixed validation manifest changed')
        frame = pd.read_csv(path)
        names = [hashlib.sha256(clip_key(row.start_time, row.origin_y, row.origin_x).encode()).hexdigest()[:24]+'.blosc'
            for row in frame.itertuples(index=False)]
        if any(name not in metadata['clips'] for name in names):
            raise ValueError('Missing fixed validation clip file')
        metadata['manifests'][file_digest(path)] = dict(path=str(path), files=names, count=len(frame))
        paths.append(path)
    metadata['total_bytes'] = sum(record['bytes'] for record in metadata['clips'].values())
    metadata['uncompressed_bytes'] = len(metadata['clips'])*16*7*224*224*2
    write_json(output/'clips/metadata.json', metadata)
    summary.update(validation_clips_materialized=True, total_cache_gib=metadata['total_bytes']/2**30)
    write_json(output/'sampling_ready.json', summary)
    ext = output/'extended/dataset'
    original_protocol = json.loads((reference/'extended/dataset/protocol.json').read_text())
    protocol = {key:original_protocol[key] for key in ('name', 'samples', 'days',
        'domain_lonlat', 'geographic_check_passed', 'geographic_constraint',
        'radiometric_selection', 'temporal_separation_hours') if key in original_protocol}
    protocol.update(clip_root=str(output/'clips'), zarr_root=str(output/'sources'),
        storage='Immutable precomputed lossless Kelvin clips',
        sampling='Fixed calendar-balanced validation manifest; original clip identities preserved',
        reference_manifest_sha256=file_digest(reference/'extended/dataset/validation.csv'),
        reference_protocol_sha256=file_digest(reference/'extended/dataset/protocol.json'),
        normalization='Global training statistics of the evaluated checkpoint')
    write_json(ext/'protocol.json', protocol)
    write_json(ext/'dataset_ready.json', dict(status='completed', files={str(path):file_digest(path)
        for path in (ext/'validation.csv', ext/'protocol.json', output/'clips/metadata.json')}))
    paths.extend([output/'training/dataset/channel_stats.json', output/'training/dataset/protocol.json',
        ext/'protocol.json', output/'clips/metadata.json'])
    ready = dict(status='completed', train_samples=summary['train_samples'], primary_validation_samples=512,
        extended_validation_samples=10240, cache_gib=metadata['total_bytes']/2**30,
        cache_backend='immutable_lossless_clip_files', files={str(path):file_digest(path) for path in paths},
        validation_manifest_bytes_preserved=True, training_submitted=False,
        finalization_source_sha256=file_digest(Path(__file__)))
    finalizer = output/'preparation_sources/scripts'/Path(__file__).name
    finalizer.parent.mkdir(parents=True, exist_ok=True)
    finalizer.write_bytes(Path(__file__).read_bytes())
    write_json(output/'dataset_ready.json', ready)
    clip_files = {name:dict(bytes=record['bytes'], sha256=record['sha256']) for name, record in metadata['clips'].items()}
    clip_files['metadata.json'] = dict(bytes=(output/'clips/metadata.json').stat().st_size,
        sha256=file_digest(output/'clips/metadata.json'))
    payload = {}
    preparation = json.loads((output/'preparation_plan.json').read_text())
    payload_mode = preparation.get('config',{}).get('transfer_payload','full_provenance')
    if payload_mode not in ('training_only','full_provenance'):
        raise ValueError('Unknown transfer payload mode')
    directories = (('training/dataset','extended/dataset') if payload_mode == 'training_only' else
        ('sources','training/dataset','extended/dataset','preparation_sources','preparation_history'))
    for directory in directories:
        for path in (output/directory).rglob('*'):
            if path.is_file():
                payload[str(path.relative_to(output))] = dict(bytes=path.stat().st_size, sha256=file_digest(path))
    for name in ('dataset_ready.json', 'preparation_plan.json', 'sampling_ready.json'):
        path = output/name
        payload[name] = dict(bytes=path.stat().st_size, sha256=file_digest(path))
    transfer = dict(version=1, canonical_clip_root=str(output/'clips'), canonical_zarr_root=str(output/'sources'),
        clips=clip_files, payload=payload, payload_mode=payload_mode)
    for role in ('training', 'extended'):
        write_json(output/role/'transfer_plan.json', transfer)
        link = output/role/'clips'
        if not link.exists():
            link.symlink_to('../clips', target_is_directory=True)
    write_json(output/'transfer_plan.json', transfer)
    write_json(output/'validation_progress.json', dict(ready, updated_unix=time.time()))
    write_json(output/'progress.json', dict(ready, updated_unix=time.time()))
    print(json.dumps(ready, indent=2), flush=True)


def reuse_validation(output, reference, source):
    """Reuse the same immutable holdout clips without decoding or rehashing data."""
    import hashlib
    from emma_gpm.seviri_clip_files import clip_key
    source = Path(source)
    ready = json.loads((source/'dataset_ready.json').read_text())
    if ready['status'] != 'completed':
        raise ValueError('Validation reuse requires a completed materialized dataset')
    metadata = json.loads((source/'clips/metadata.json').read_text())
    for role in ('training','extended'):
        if (source/role/'dataset/validation.csv').read_bytes() != (reference/role/'dataset/validation.csv').read_bytes():
            raise ValueError('Reused validation must match the fixed holdout exactly')
    frame = pd.read_csv(reference/'extended/dataset/validation.csv')
    results = []
    for store, rows in frame.groupby('store'):
        clips = {}
        for row in rows.itertuples(index=False):
            key = clip_key(row.start_time,row.origin_y,row.origin_x)
            name = hashlib.sha256(key.encode()).hexdigest()[:24]+'.blosc'
            record = metadata['clips'][name]
            original = source/'clips'/name
            if record['key'] != key or original.stat().st_size != record['bytes']:
                raise ValueError('Reused clip identity or size differs')
            destination = output/'validation_clips'/name
            if not destination.exists():os.link(original,destination)
            if not os.path.samefile(original,destination):
                raise ValueError('Validation reuse must preserve the original immutable file')
            clips[name] = record
        # Coordinate headers stay local for plots; each file is shared by a
        # hard link. They are excluded from training_only HPC payloads.
        for original in (source/'sources'/store).rglob('*'):
            if original.is_file():
                destination = output/'sources'/original.relative_to(source/'sources')
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():os.link(original,destination)
        result = dict(day=str(rows.day.iloc[0]),clips=clips,reused_from=str(source))
        write_json(output/'validation_daily'/f"{result['day']}.json",result)
        results.append(result)
    return results


def build(args):
    config = json.loads(args.config.read_text())
    output, reference = Path(config['output']), Path(config['reference_dataset'])
    (output/'validation_clips').mkdir(parents=True, exist_ok=True)
    if config.get('reuse_validation_clips_from'):
        results = reuse_validation(output,reference,config['reuse_validation_clips_from'])
        write_json(output/'validation_progress.json',dict(status='reused_existing_files',
            source=config['reuse_validation_clips_from'],days=len(results),updated_unix=time.time()))
    else:
        manifest = pd.read_csv(reference/'extended/dataset/validation.csv')
        metadata = json.loads((reference/'frames/metadata.json').read_text())
        tasks = [(str(reference), str(output), store, metadata['stores'][store], rows.to_dict('records'))
            for store, rows in manifest.groupby('store')]
        results = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
            for result in pool.map(prepare_day, tasks, chunksize=1):
                results.append(result)
                write_json(output/'validation_progress.json', dict(status='materializing',
                    completed_days=len(results), total_days=len(tasks), updated_unix=time.time()))
                print(f'{len(results)}/{len(tasks)} validation days', flush=True)
    if args.wait_for_training:
        while not (output/'sampling_ready.json').exists():
            time.sleep(10)
    if (output/'sampling_ready.json').exists():
        finalize(output, reference, results)
    else:
        print('Validation clips ready; rerun after training sampling has completed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--wait-for-training', action='store_true')
    args = parser.parse_args()
    output = Path(json.loads(args.config.read_text())['output'])
    with exclusive_file_lock(output/'.validation_preparation.lock'):
        build(args)
