#!/usr/bin/env python3
"""Freeze an additional multi-day validation set, preserving existing manifests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from numcodecs import blosc

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.seviri_clips import IncompleteSeviriClipError
from emma_gpm.seviri_frame_cache import build_frame_cache, SeviriFrameCacheReader, source_stamp
from emma_gpm.ssl_dataset import file_digest, verify_snapshot
from emma_gpm.ssl_random_sampling import geographic_origin_mask, random_candidates
from emma_gpm.ssl_validation_set import assert_day_separation, select_balanced_validation
from scripts.build_seviri_ssl_new_dates import freeze_store, check_cache_samples
from scripts.pretrain_seviri_ssl import validate_manifests, write_json
from scripts.run_seviri_ssl_scaling import immutable_json


def assert_final_frame_hashes(entries, cache, hashes):
    for entry in entries.values():
        path = str(cache/entry['file'])
        if hashes.get(path) != entry['sha256']:
            raise ValueError(f'Frame array changed after its radiometric checks: {path}')


def check_clip(reader, row):
    try:
        raw = reader.load(row['start_time'], origin_y=row['origin_y'], origin_x=row['origin_x'])
        if raw.min() < 100 or raw.max() > 400:
            raise ValueError('Radiometry outside [100,400] K')
        change = np.abs(np.diff(raw[:10, 4], axis=0))
        return dict(row, valid=True, mean_abs_change_k=float(change.mean(dtype=np.float64)),
                    p90_abs_change_k=float(np.quantile(change, .9)),
                    fraction_change_gt2k=float(np.mean(change > 2)))
    except (IncompleteSeviriClipError, ValueError, OSError) as exc:
        return dict(row, valid=False, rejection=str(exc))


def build(args):
    out = args.output_dir
    config = json.loads(args.config.read_text())
    days = config['days']
    count = config['samples_per_day']
    sources = {name:file_digest(ROOT/name) for name in (
        'scripts/build_seviri_ssl_validation.py', 'src/emma_gpm/ssl_validation_set.py',
        'src/emma_gpm/ssl_random_sampling.py', 'src/emma_gpm/seviri_clips.py',
        'src/emma_gpm/seviri_frame_cache.py', 'src/emma_gpm/ssl_dataset.py',
        'scripts/build_seviri_ssl_new_dates.py')}
    inputs = {str(p):file_digest(p) for p in [args.config, args.inventory, *args.training_manifests,
                                             args.legacy_validation, args.current_validation]}
    plan = dict(config=config, inputs=inputs, sources=sources,
                upload_root=str(args.upload_root), frozen_root=str(args.frozen_root),
                frame_cache=str(args.frame_cache))
    immutable_json(out/'preparation_plan.json', plan)
    if (out/'dataset_ready.json').exists():
        ready = json.loads((out/'dataset_ready.json').read_text())
        for path, checksum in ready['files'].items():
            if file_digest(Path(path)) != checksum:
                raise ValueError(f'Completed validation dataset changed: {path}')
        verify_snapshot(args.frozen_root, json.loads((out/'source_snapshot.json').read_text()))
        print('Completed validation set verified', flush=True)
        return
    for path in args.training_manifests:
        assert_day_separation(pd.read_csv(path), days, config['temporal_separation_hours'])
    for label, path in [('legacy_512', args.legacy_validation), ('current_128k_512', args.current_validation)]:
        target = out/f'{label}.csv'
        if target.exists() and file_digest(target) != inputs[str(path)]:
            raise ValueError(f'Preserved reference changed: {label}')
        shutil.copyfile(path, target)
    inventory = json.loads(args.inventory.read_text())
    lookup = {r['store']:r for r in inventory['stores']}
    records = [lookup[f'{day[:4]}/{day.replace("-", "")}.zarr'] for day in days]
    write_json(out/'inventory.json', dict(stores=records, skipped=[], stable_seconds=inventory['stable_seconds']))
    args.frame_cache.mkdir(parents=True, exist_ok=True)
    args.frozen_root.mkdir(parents=True, exist_ok=True)
    entries, snapshot, selected, parity = {}, {}, [], []
    metadata = None
    for index, (day, record) in enumerate(zip(days, records, strict=True)):
        write_json(out/'progress.json', dict(status='building', phase='freezing_sources', day=day,
                   completed_days=index, total_days=len(days)))
        day_out = out/'daily'/day
        day_out.mkdir(parents=True, exist_ok=True)
        snapshot.update(freeze_store(args.upload_root, args.frozen_root, record))
        cache = args.frame_cache/'daily_builds'/day
        if not (cache/'metadata.json').exists():
            # A partial day is unpublished. Require explicit cleanup after an interrupted cache build.
            metadata = build_frame_cache(args.frozen_root, cache, [day.replace('-', '')])
        else:
            metadata = json.loads((cache/'metadata.json').read_text())
        entry = metadata['stores'][record['store']]
        if (file_digest(cache/entry['file']) != entry['sha256']
                or source_stamp(args.frozen_root/record['store']) != entry['source_stamp']):
            raise ValueError(f'Cached observations changed: {day}')
        target = args.frame_cache/entry['file']
        if target.exists():
            if not os.path.samefile(target, cache/entry['file']):
                raise ValueError('Existing frame file differs from its daily source')
        else:
            os.link(cache/entry['file'], target)
        entries[record['store']] = entry
        with xr.open_zarr(args.frozen_root/record['store'], consolidated=True, chunks=None) as ds:
            mask = geographic_origin_mask(ds.x.values, ds.y.values, ds.attrs['projection_wkt'], config['domain_lonlat'])
        proposals = random_candidates(dict(stores=[record]), count*2, config['seed']+index,
                                      origin_masks={record['store']:mask})
        proposals.to_csv(day_out/'proposals.csv', index=False)
        write_json(out/'progress.json', dict(status='building', phase='checking_radiometry', day=day,
                   completed_days=index, total_days=len(days)))
        with SeviriFrameCacheReader(args.frozen_root, cache) as reader:
            reader._open_store(args.frozen_root/record['store'])
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                checked = pd.DataFrame(executor.map(lambda row:check_clip(reader, row), proposals.to_dict('records')))
        checked.to_csv(day_out/'checked.csv', index=False)
        valid = checked.loc[checked.valid].drop(columns=['valid', 'rejection'], errors='ignore')
        chosen = select_balanced_validation(valid, [day], count, config['seed']+index+1000)
        chosen.to_csv(day_out/'validation.csv', index=False)
        selected.append(chosen)
        parity.extend(check_cache_samples(args.frozen_root, cache, chosen))
        print(f'Validation {day}: {len(chosen)} selected; {len(checked)-len(valid)} invalid proposals', flush=True)
    validation = pd.concat(selected, ignore_index=True)
    for path in args.training_manifests:
        validate_manifests(pd.read_csv(path), validation)
    if len(validation) != len(days)*count:
        raise ValueError('Unexpected validation sample count')
    metadata.update(stores=entries, total_bytes=sum(e['bytes'] for e in entries.values()))
    metadata.pop('elapsed_seconds', None)
    write_json(args.frame_cache/'metadata.json', metadata)
    validation.to_csv(out/'validation.csv', index=False)
    verify_snapshot(args.frozen_root, snapshot)
    write_json(out/'source_snapshot.json', snapshot)
    for path, checksum in inputs.items():
        if file_digest(Path(path)) != checksum:
            raise ValueError(f'Input changed during preparation: {path}')
    for path, checksum in sources.items():
        if file_digest(ROOT/path) != checksum:
            raise ValueError(f'Builder source changed: {path}')
    protocol = dict(config, status='completed', samples=len(validation),
        samples_by_day=validation.groupby('day').size().to_dict(),
        normalization='use the global training mean and standard deviation from each evaluated checkpoint',
        source_files=len(snapshot), zarr_root=str(args.frozen_root), frame_cache=str(args.frame_cache),
        references={label:inputs[str(path)] for label,path in (
            ('legacy_512',args.legacy_validation), ('current_128k_512',args.current_validation))},
        training_manifests={str(p):inputs[str(p)] for p in args.training_manifests},
        cache_sample_verification=parity, teacher_labels_used=False,
        interpretation='Equal weighting of selected UTC days and six-hour bins in January-June 2022')
    write_json(out/'protocol.json', protocol)
    paths = [out/name for name in ('validation.csv','legacy_512.csv','current_128k_512.csv',
                                   'source_snapshot.json','inventory.json','protocol.json','preparation_plan.json')]
    paths += [args.frame_cache/'metadata.json', *[args.frame_cache/e['file'] for e in entries.values()]]
    files = {str(p):file_digest(p) for p in paths}
    assert_final_frame_hashes(entries, args.frame_cache, files)
    write_json(out/'dataset_ready.json', dict(status='completed', samples=len(validation), days=len(days), files=files))
    write_json(out/'progress.json', dict(status='completed', samples=len(validation), days=len(days)))
    print(f'Frozen validation ready: {len(validation)} clips across {len(days)} days', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','inventory','upload-root','frozen-root','frame-cache','output-dir',
                 'legacy-validation','current-validation'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--training-manifests', type=Path, nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value, Path): setattr(args, name, value.resolve())
    args.training_manifests = [p.resolve() for p in args.training_manifests]
    if args.workers < 1: parser.error('Positive worker count required')
    zarr.config.set({'async.concurrency':8, 'threading.max_workers':4}); blosc.set_nthreads(1)
    with exclusive_file_lock(args.output_dir/'.prepare.lock'):
        try:
            build(args)
        except Exception as exc:
            write_json(args.output_dir/'progress.json', dict(status='failed', error=str(exc)))
            raise


if __name__ == '__main__': main()
