#!/usr/bin/env python3
"""Materialize the unique clips of several manifests into exact LZ4 files."""
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
sys.path.insert(0,str(ROOT/'src'))
from emma_gpm.seviri_clips import SeviriZarrClipReader, VIDEO_MAE_CHANNELS
from emma_gpm.seviri_clip_files import codec,clip_key,digest_bytes,SHAPE


def initialize(root,output):
    global reader,destination
    import zarr
    from numcodecs import blosc
    zarr.config.set({'async.concurrency':8,'threading.max_workers':4});blosc.set_nthreads(1)
    reader=SeviriZarrClipReader(root);destination=Path(output)


def materialize(task):
    name,key,start,y,x=task
    original=reader.load(start,origin_y=y,origin_x=x)
    raw=np.ascontiguousarray(original,dtype=np.float16)
    if not np.array_equal(raw.astype(np.float32),original):
        raise ValueError('Source cannot be represented exactly as float16')
    encoded=codec().encode(raw)
    restored=np.empty_like(raw);codec().decode(encoded,out=restored)
    if not np.array_equal(raw.view(np.uint16),restored.view(np.uint16)):
        raise ValueError('Lossless encoding changed source values')
    path=destination/name;temp=path.with_suffix('.part')
    with temp.open('wb') as stream:
        stream.write(encoded);stream.flush();os.fsync(stream.fileno())
    reread=temp.read_bytes()
    if reread!=encoded:raise ValueError('Saved clip differs from encoded data')
    temp.replace(path)
    return name,dict(key=key,bytes=len(encoded),sha256=digest_bytes(encoded))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,action='append',required=True)
    parser.add_argument('--zarr-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Output directory must be empty')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    unique={};manifests={};tasks=[]
    for path in args.manifest:
        frame=pd.read_csv(path);names=[]
        for row in frame.itertuples(index=False):
            key=clip_key(row.start_time,row.origin_y,row.origin_x)
            if key not in unique:
                name=f'clip_{len(unique):06d}.blosc';unique[key]=name
                tasks.append((name,key,row.start_time,int(row.origin_y),int(row.origin_x)))
            names.append(unique[key])
        manifests[digest_bytes(path.read_bytes())]=dict(path=str(path.resolve()),files=names,count=len(names))
    started=time.perf_counter();clips={}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(args.zarr_root),str(args.output_dir))) as pool:
        for name,record in pool.map(materialize,tasks,chunksize=1):
            clips[name]=record
            if len(clips)%256==0:print(f'{len(clips)}/{len(tasks)} clips; {sum(c["bytes"] for c in clips.values())/2**30:.1f} GiB',flush=True)
    metadata=dict(version=1,shape=list(SHAPE),dtype='float16',units='K',channels=list(VIDEO_MAE_CHANNELS),
        orientation='north-up west-left',cadence_minutes=5,lossless=True,codec=codec().get_config(),
        zarr_root=str(args.zarr_root.resolve()),manifests=manifests,clips=clips,
        elapsed_seconds=time.perf_counter()-started,total_bytes=sum(c['bytes'] for c in clips.values()),
        uncompressed_bytes=len(clips)*np.prod(SHAPE).item()*2)
    temp=args.output_dir/'metadata.json.part';temp.write_text(json.dumps(metadata,indent=2)+'\n');temp.replace(args.output_dir/'metadata.json')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('clips','manifests')},indent=2),flush=True)


if __name__=='__main__':main()
