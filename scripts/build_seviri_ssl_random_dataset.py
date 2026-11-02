#!/usr/bin/env python3
"""Resample all SSL crops at pixel resolution within an explicit geographic domain."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from numcodecs import blosc

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.seviri_frame_cache import source_stamp
from emma_gpm.ssl_dataset import file_digest,verify_snapshot
from emma_gpm.ssl_random_sampling import geographic_origin_mask,random_candidates
from emma_gpm.ssl_sampling import stratified_indices
from scripts.build_seviri_ssl_extension import init_candidate_worker,keys
from scripts.build_seviri_ssl_new_dates import check_cache_samples
from scripts.build_seviri_ssl_pilot_dataset import validate_candidate
from scripts.build_seviri_ssl_scaling_dataset import global_moments,MANIFEST_COLUMNS
from scripts.pretrain_seviri_ssl import write_json,validate_manifests
from scripts.run_seviri_ssl_scaling import immutable_json


def collect(executor,candidates,output,name):
    """Check the entire proposal pool, retaining day/time locality for disk reads."""
    path=output/f'{name}_checked.csv'
    records=pd.read_csv(path).to_dict('records') if path.exists() else []
    identity=['start_time','origin_y','origin_x']
    done={tuple(row[k] for k in identity) for row in records}
    pending=[row for row in candidates.to_dict('records') if tuple(row[k] for k in identity) not in done]
    for offset in range(0,len(pending),512):
        records.extend(executor.map(validate_candidate,pending[offset:offset+512],chunksize=8))
        temporary=path.with_suffix('.csv.part')
        pd.DataFrame(records).to_csv(temporary,index=False)
        temporary.replace(path)
        accepted=sum(bool(row['valid']) for row in records)
        write_json(output/'preparation_progress.json',dict(status='validating',phase=name,
            processed=len(records),total=len(candidates),accepted=accepted,rejected=len(records)-accepted))
        print(f'{name}: {len(records)}/{len(candidates)} checked; {accepted} valid',flush=True)
    if len(records)!=len(candidates):
        raise ValueError('Cached validation records differ from the frozen proposal pool')
    return pd.DataFrame([row for row in records if row['valid']])


def build(args):
    out=args.output_dir
    inventory=json.loads(args.inventory.read_text())
    previous=pd.read_csv(args.previous_dataset/'train.csv')
    old_validation=pd.read_csv(args.previous_dataset/'validation.csv')
    validation_stores=set(old_validation.store)
    training_inventory=dict(inventory,stores=[r for r in inventory['stores'] if r['store'] not in validation_stores])
    validation_inventory=dict(inventory,stores=[r for r in inventory['stores'] if r['store'] in validation_stores])
    if not validation_inventory['stores']:
        raise ValueError('Previous validation observation dates must remain available')
    sources={name:file_digest(ROOT/name) for name in (
        'scripts/build_seviri_ssl_random_dataset.py','src/emma_gpm/ssl_random_sampling.py',
        'scripts/build_seviri_ssl_pilot_dataset.py','scripts/build_seviri_ssl_scaling_dataset.py',
        'scripts/build_seviri_ssl_extension.py','scripts/build_seviri_ssl_new_dates.py',
        'src/emma_gpm/ssl_sampling.py','src/emma_gpm/seviri_clips.py','src/emma_gpm/seviri_frame_cache.py')}
    plan=dict(count=args.count,candidates=args.candidates,validation_count=args.validation_count,
        domain_lonlat=args.domain,seed=args.seed,zarr_root=str(args.zarr_root),frame_cache=str(args.frame_cache),
        frame_metadata_sha256=file_digest(args.frame_cache/'metadata.json'),
        inventory_sha256=file_digest(args.inventory),previous_dataset=str(args.previous_dataset),
        previous_train_sha256=file_digest(args.previous_dataset/'train.csv'),
        previous_validation_sha256=file_digest(args.previous_dataset/'validation.csv'),sources=sources)
    immutable_json(out/'preparation_plan.json',plan)
    for name in sources:
        target=out/'source_archive'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    masks={}
    for record in inventory['stores']:
        with xr.open_zarr(args.zarr_root/record['store'],consolidated=True,chunks=None) as ds:
            masks[record['store']]=geographic_origin_mask(ds.x.values,ds.y.values,
                ds.attrs['projection_wkt'],args.domain)
    candidates=random_candidates(training_inventory,args.candidates,args.seed,origin_masks=masks)
    val_candidates=random_candidates(validation_inventory,args.validation_count*2,args.seed+1,origin_masks=masks)
    validate_manifests(candidates,val_candidates)
    candidates.to_csv(out/'training_candidates.csv',index=False)
    val_candidates.to_csv(out/'validation_candidates.csv',index=False)
    snapshot_path=out/'preparation_source_snapshot.json'
    if snapshot_path.exists():
        snapshot=json.loads(snapshot_path.read_text());verify_snapshot(args.zarr_root,snapshot)
    else:
        snapshot={}
        for record in inventory['stores']:
            store=record['store'];stamp=source_stamp(args.zarr_root/store)
            paths=set(record['headers'])|{f'{store}/seviri/{name}' for name in record['present']}
            for name in sorted(paths):
                snapshot[name]=file_digest(args.zarr_root/name)
                if name in record['headers'] and snapshot[name]!=record['headers'][name]:
                    raise ValueError(f'Inventoried coordinates changed: {name}')
            if source_stamp(args.zarr_root/store)!=stamp:
                raise ValueError(f'Sources changed during capture: {store}')
            print(f'Frozen source hashes: {store}',flush=True)
        write_json(snapshot_path,snapshot)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
            initializer=init_candidate_worker,initargs=(args.zarr_root,args.frame_cache)) as executor:
        pool=collect(executor,candidates,out,'training_pool')
        validation_pool=collect(executor,val_candidates,out,'validation_pool')
    if len(pool)<args.count or len(validation_pool)<args.validation_count:
        raise ValueError('Insufficient radiometrically valid proposals; use a new versioned larger pool')
    indices,sampling=stratified_indices(pool.mean_abs_change_k,args.count,seed=args.seed)
    selected=pool.iloc[indices].reset_index(drop=True)
    chosen=np.random.default_rng(args.seed+1).choice(len(validation_pool),args.validation_count,replace=False)
    validation=validation_pool.iloc[chosen].reset_index(drop=True)
    validate_manifests(selected,validation)
    for frame in (selected,validation):
        for store,group in frame.groupby('store'):
            if not masks[store][group.origin_y,group.origin_x].all():
                raise ValueError('A crop extends beyond the agreed geographic domain')
    mean,std=global_moments(selected)
    parity=check_cache_samples(args.zarr_root,args.frame_cache,pd.concat([selected,validation]))
    verify_snapshot(args.zarr_root,snapshot)
    for name,checksum in sources.items():
        if file_digest(ROOT/name)!=checksum:
            raise ValueError(f'Preparation source changed: {name}')
    dataset=out/f'n{args.count}'/'dataset';dataset.mkdir(parents=True,exist_ok=True)
    selected[MANIFEST_COLUMNS].to_csv(dataset/'train.csv',index=False)
    validation[MANIFEST_COLUMNS].to_csv(dataset/'validation.csv',index=False)
    pool.assign(selected=pool.index.isin(indices)).to_csv(out/'candidate_variability.csv',index=False)
    write_json(dataset/'channel_stats.json',dict(global_mean_k=mean,global_std_k=std,samples=len(selected),
        pixel_count=len(selected)*16*7*224**2,normalization='single_global_affine_transform',
        channels=list(THERMAL_CHANNELS),manifest_sha256=file_digest(dataset/'train.csv'),
        zarr_root=str(args.zarr_root),scope='all pixels/channels/16 frames of selected training clips'))
    write_json(dataset/'source_snapshot.json',snapshot)
    write_json(dataset/'inventory.json',inventory)
    protocol=dict(status='completed',kind='fully resampled pixel-resolution crops in geographic domain',
        train_samples=len(selected),validation_samples=len(validation),
        train_days=selected.groupby('day').size().to_dict(),validation_days=validation.groupby('day').size().to_dict(),
        previous_dataset=str(args.previous_dataset),strict_nesting=False,
        shared_previous_training_clips=len(set(keys(selected))&set(keys(previous))),
        validation_identical=False,validation_dates_preserved=True,domain_lonlat=args.domain,
        geographic_constraint='every pixel centre in the entire 224 by 224 crop, in both splits',
        geographic_check_passed=True,spatial_resolution_pixels=1,spatial_stride=None,
        spatial_sampling='uniform proposals over physically supported time and pixel-origin tuples within UTC date and six-hour bin',
        training_sampling=sampling,validation_sampling='uniform selection from valid pixel-resolution proposals',
        temporal_separation_hours=24,overlap_within_training_allowed=True,teacher_labels_used=False,
        train_unique_spatial_origins=len(selected[['origin_y','origin_x']].drop_duplicates()),
        validation_unique_spatial_origins=len(validation[['origin_y','origin_x']].drop_duplicates()),
        cache_sample_verification=parity,source_files=len(snapshot),seed=args.seed,
        source_scope='all inventoried physical files in the selected training and validation stores',
        interpretation='Dataset size, date coverage, spatial sampling and geographic restriction change together; compare architectures on this identical dataset')
    write_json(dataset/'protocol.json',protocol);write_json(dataset/'progress.json',protocol)
    case=dict(name=f'n{len(selected)}',train_samples=len(selected),validation_samples=len(validation),
        files={name:file_digest(dataset/name) for name in ('train.csv','validation.csv','channel_stats.json',
            'source_snapshot.json','protocol.json','inventory.json')})
    write_json(out/'dataset_ready.json',dict(status='completed',cases=[case],strict_nesting=False,
        validation_identical=False,validation_dates_preserved=True,geographic_check_passed=True,
        source_snapshot_verified=True,frame_cache=str(args.frame_cache),zarr_root=str(args.zarr_root)))
    write_json(out/'preparation_progress.json',dict(status='completed',train_samples=len(selected),validation_samples=len(validation)))
    print(f'Dataset ready: {len(selected)} train, {len(validation)} validation; mean {mean:.1f} K, std {std:.1f} K',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output-dir','previous-dataset','inventory','zarr-root','frame-cache'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--count',type=int,default=131072)
    p.add_argument('--candidates',type=int,default=196608)
    p.add_argument('--validation-count',type=int,default=512)
    p.add_argument('--domain',type=float,nargs=4,default=[-20,40,30,70],metavar=('WEST','EAST','SOUTH','NORTH'))
    p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    if min(args.count,args.validation_count,args.workers)<1 or args.candidates<args.count:
        p.error('Positive counts and workers, with a sufficient proposal pool required')
    zarr.config.set({'async.concurrency':8,'threading.max_workers':4});blosc.set_nthreads(1)
    with exclusive_file_lock(args.output_dir/'.prepare.lock'):
        build(args)


if __name__=='__main__':main()
