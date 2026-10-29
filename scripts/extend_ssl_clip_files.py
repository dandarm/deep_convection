#!/usr/bin/env python3
"""Materialize added clips and hard-link shared immutable clips into a new collection."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.seviri_clips import VIDEO_MAE_CHANNELS
from emma_gpm.seviri_clip_files import ClipFileDataset, clip_key, codec, digest_bytes, SHAPE
from scripts.materialize_ssl_clip_files import initialize, materialize
from scripts.pretrain_seviri_ssl import write_json, digest
from scripts.run_seviri_ssl_scaling import immutable_json


def collection_plan(manifests, parent):
    by_key={record['key']:(name,record) for name,record in parent['clips'].items()}
    if len(by_key)!=len(parent['clips']):
        raise ValueError('Parent collection contains duplicate clip keys')
    used={};shared={};tasks=[];bindings={}
    next_index=len(parent['clips'])
    for path in manifests:
        names=[]
        for row in pd.read_csv(path).itertuples(index=False):
            key=clip_key(row.start_time,row.origin_y,row.origin_x)
            if key not in used:
                if key in by_key:
                    name,record=by_key[key];shared[name]=record
                else:
                    name=f'extension_{next_index:06d}.blosc';next_index+=1
                    if name in parent['clips']:
                        raise ValueError('New clip filename collides with parent collection')
                    tasks.append((name,key,row.start_time,int(row.origin_y),int(row.origin_x)))
                used[key]=name
            names.append(used[key])
        bindings[digest(path)]=dict(path=str(path.resolve()),files=names,count=len(names))
    return shared,tasks,bindings


def build(args):
    parent=json.loads((args.reuse_root/'metadata.json').read_text())
    if Path(parent['zarr_root']).resolve()!=args.zarr_root:
        raise ValueError('Parent clip collection belongs to another archive')
    expected=dict(version=1,shape=list(SHAPE),dtype='float16',units='K',channels=list(VIDEO_MAE_CHANNELS),
                  orientation='north-up west-left',cadence_minutes=5,lossless=True,codec=codec().get_config())
    if any(parent.get(key)!=value for key,value in expected.items()):
        raise ValueError('Unexpected parent clip collection format')
    identity=dict(parent_root=str(args.reuse_root),parent_metadata_sha256=digest(args.reuse_root/'metadata.json'),
        zarr_root=str(args.zarr_root),manifests={str(path):digest(path) for path in args.manifest})
    immutable_json(args.output_dir/'materialization_plan.json',identity)
    if (args.output_dir/'metadata.json').exists():
        for manifest in args.manifest:
            ClipFileDataset(args.output_dir,manifest,zarr_root=args.zarr_root)
        print('Existing clip collection verified',flush=True)
        return
    shared,tasks,bindings=collection_plan(args.manifest,parent)
    clips=dict(shared)
    for name,record in shared.items():
        source=args.reuse_root/name;target=args.output_dir/name
        if source.stat().st_size!=record['bytes']:
            raise ValueError(f'Truncated parent clip: {name}')
        if target.exists():
            if not os.path.samefile(source,target):
                raise ValueError(f'Existing shared clip is not the expected hard link: {name}')
        else:
            os.link(source,target)
    journal=args.output_dir/'completed_clips.json'
    if journal.exists():
        for name,record in json.loads(journal.read_text()).items():
            if name in shared:
                continue
            path=args.output_dir/name
            if path.is_file() and path.stat().st_size==record['bytes'] and digest(path)==record['sha256']:
                clips[name]=record
    tasks=[task for task in tasks if task[0] not in clips]
    started=time.perf_counter()
    write_json(journal,clips)
    print(f'Reusing {len(shared)} hard-linked clips; materializing {len(tasks)} additions',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                             initializer=initialize,initargs=(str(args.zarr_root),str(args.output_dir))) as pool:
        for index,(name,record) in enumerate(pool.map(materialize,tasks,chunksize=1),1):
            clips[name]=record
            if index%256==0 or index==len(tasks):
                write_json(journal,clips)
                write_json(args.output_dir/'progress.json',dict(status='materializing',
                    linked_clips=len(shared),new_clips_completed=len(clips)-len(shared),
                    total_clips=len(clips)+len(tasks)-index))
                print(f'{index}/{len(tasks)} new clips; elapsed {time.perf_counter()-started:.0f} s',flush=True)
    metadata=dict(**expected,zarr_root=str(args.zarr_root),manifests=bindings,clips=clips,
        elapsed_seconds=time.perf_counter()-started,total_bytes=sum(r['bytes'] for r in clips.values()),
        uncompressed_bytes=len(clips)*int(np.prod(SHAPE))*2,
        linked_clips=len(shared),parent_root=str(args.reuse_root),parent_metadata_sha256=identity['parent_metadata_sha256'])
    write_json(args.output_dir/'metadata.json',metadata)
    for manifest in args.manifest:
        ClipFileDataset(args.output_dir,manifest,zarr_root=args.zarr_root)
    write_json(args.output_dir/'progress.json',dict(status='completed',total_clips=len(clips),linked_clips=len(shared)))
    print(f'Ready: {len(clips)} clips, {metadata["total_bytes"]/2**30:.1f} GiB logical size',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,action='append',required=True)
    parser.add_argument('--zarr-root',type=Path,required=True)
    parser.add_argument('--reuse-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    for name in ('zarr_root','reuse_root','output_dir'):
        setattr(args,name,getattr(args,name).resolve())
    args.manifest=[p.resolve() for p in args.manifest]
    if args.workers<1 or args.output_dir==args.reuse_root:
        parser.error('Use positive workers and a new output collection')
    with exclusive_file_lock(args.output_dir/'.materialization.lock'):
        build(args)


if __name__=='__main__':
    main()
