#!/usr/bin/env python3
"""Resumable RDT-CW-inspired diagnostic scan. Never a pretraining sampler."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import shutil
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from pyproj import CRS,Transformer
from emma_gpm.rdt_teacher import RDTConfig,label_rdt_clip,projected_pixel_areas
from emma_gpm.rdt_preview import archive_inventory
from emma_gpm.rdt_dataset import diagnostic_manifest,occupancy_metrics,track_metrics
from emma_gpm.rdt_refined import RefinedRDTConfig,label_refined_clip
from emma_gpm.rdt_context import context_window,crop_teacher_result
from emma_gpm.seviri_clips import SeviriZarrClipReader,IncompleteSeviriClipError


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.part');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def file_hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def init_worker(root,output,warmup,geometry,area_path,teacher_version="legacy",halo=0,baseline_dir=None):
    global reader,out,context,grid,areas,transformer,version,margin,baseline
    import zarr
    from numcodecs import blosc
    zarr.config.set({'async.concurrency':4,'threading.max_workers':2});blosc.set_nthreads(1)
    version=teacher_version;margin=halo;baseline=Path(baseline_dir) if baseline_dir else None
    size=min(224+2*halo,len(geometry[0]),len(geometry[1]))
    reader=SeviriZarrClipReader(root,num_frames=16+warmup,crop_size=size);out=Path(output);context=warmup;grid=geometry
    areas=np.load(area_path,mmap_mode='r')
    transformer=Transformer.from_crs(CRS.from_user_input(grid[2]),'EPSG:4326',always_xy=True)


def evaluate(row):
    key=row['clip_id'];record=out/'records'/f'{key}.json';mask_path=out/'masks'/f'{key}.npz'
    if record.exists():
        cached=json.loads(record.read_text())
        if cached['status']=='rejected' or (mask_path.exists() and cached['mask_sha256']==file_hash(mask_path)
             and (out/'objects'/f'{key}.csv.gz').exists() and (out/'links'/f'{key}.csv.gz').exists()):return cached
    started=time.monotonic();y,x=int(row['origin_y']),int(row['origin_x'])
    try:
        yy,xx,size,roi=context_window(y,x,len(grid[1]),len(grid[0]),halo=margin,
            north_reversed=grid[1][0]<grid[1][-1],west_reversed=grid[0][0]>grid[0][-1])
        clip=reader.load(row['context_start'],origin_y=yy,origin_x=xx)
        area=np.asarray(areas[yy:yy+size,xx:xx+size])
        if grid[1][0]<grid[1][-1]:area=area[::-1]
        if grid[0][0]>grid[0][-1]:area=area[:,::-1]
        if not np.isfinite(area).all():raise ValueError('Pixel corners outside visible Earth')
        times=pd.date_range(row['context_start'],periods=len(clip),freq='5min')
        result=label_refined_clip(clip,times,area) if version=='refined_v2' else label_rdt_clip(clip,times,area)
        area=area[roi]
        result=crop_teacher_result(result,roi,context,area)
        row.update(teacher_version=version,analysis_halo=margin)
        if 'phase_uncertain' in result:row['phase_unknown_pixels']=int(result['phase_uncertain'].sum())
        row.update(occupancy_metrics(result,area))
        row.update(track_metrics(result['objects']))
        if baseline and (baseline/'objects'/f'{key}.csv.gz').exists():
            old_objects=pd.read_csv(baseline/'objects'/f'{key}.csv.gz')
            row.update({'baseline_'+k:v for k,v in track_metrics(old_objects).items()})
        lon,lat=transformer.transform(float(grid[0][x:x+224].mean()),float(grid[1][y:y+224].mean()))
        row.update(status='ok',longitude=lon,latitude=lat)
        tmp=mask_path.with_suffix('.npz.part')
        with tmp.open('wb') as handle:
            np.savez_compressed(handle,**{field:result[field] for field in ['classes','tracks','uncertain','phase_uncertain'] if field in result})
        tmp.replace(mask_path)
        for field in ['objects','links']:
            path=out/field/f'{key}.csv.gz';tmp=path.with_suffix('.gz.part')
            result[field].to_csv(tmp,index=False,compression='gzip');tmp.replace(path)
        row['mask_sha256']=file_hash(mask_path)
    except (IncompleteSeviriClipError,ValueError) as error:
        row.update(status='rejected',reason=str(error),teacher_version=version,analysis_halo=margin)
    finally:
        # Bound open daily store metadata in each persistent worker.
        reader.close()
    row['processing_seconds']=time.monotonic()-started
    row={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()}
    atomic_json(record,row)
    return row


def publish(output,records,requested,started,status):
    frame=pd.DataFrame(records);temp=output/'catalog.csv.part';frame.to_csv(temp,index=False);temp.replace(output/'catalog.csv')
    good=frame[frame.status=='ok']
    months=[]
    if len(good):
        for month,group in frame.groupby('month'):
            valid=group[group.status=='ok'];n=len(valid)
            months.append(dict(month=int(month),processed=len(group),valid=n,rejected=len(group)-n,
                convective_clips=int((valid.convective_pixels>0).sum()) if n else 0,
                ot_clips=int((valid.ot_pixels>0).sum()) if n else 0))
    summary=dict(status=status,processed=len(frame),requested=requested,valid=len(good),rejected=len(frame)-len(good),
        convective_clips=int((good.convective_pixels>0).sum()) if len(good) else 0,
        ot_clips=int((good.ot_pixels>0).sum()) if len(good) else 0,
        elapsed_seconds=time.monotonic()-started,months=months,
        interpretation='month-balanced diagnostic pool; conditional on valid clips; not atmospheric prevalence')
    if len(good):
        summary.update(convective_tracks=int(good.convective_tracks.sum()),ot_tracks=int(good.ot_tracks.sum()),
            max_convective_fraction=float(good.convective_fraction.max()),max_convective_peak_fraction=float(good.convective_peak_fraction.max()),
            max_ot_peak_fraction=float(good.ot_peak_fraction.max()),mean_unknown_fraction=float(good.unknown_fraction.mean()))
    atomic_json(output/'progress.json',summary)
    if status=='complete':atomic_json(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['months','interpretation']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zarr-root',type=Path,default=Path('/media/fenrir/disk1/Datasets/MSG-SEVIRI'))
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--teacher',choices=['legacy','refined_v2'],default='refined_v2')
    parser.add_argument('--halo',type=int,default=64)
    parser.add_argument('--manifest-from',type=Path,default=ROOT/'results/rdt_preview/annual8192')
    parser.add_argument('--count',type=int,default=8192);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20261012);parser.add_argument('--warmup',type=int,default=3)
    args=parser.parse_args()
    if args.count<1 or args.workers<1 or args.warmup<0 or args.halo<0:parser.error('Invalid count/workers/context')
    output=args.output_dir or ROOT/'results/rdt_preview'/('annual8192_refined' if args.teacher=='refined_v2' else 'annual8192_legacy_rerun')
    halo=args.halo if args.teacher=='refined_v2' else 0
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'scan.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (output/'protocol.json').exists():atomic_json(output/'progress.json',dict(status='inventory',processed=0,requested=args.count))
    inventory,times,grid=archive_inventory(args.zarr_root)
    if args.manifest_from and (args.manifest_from/'manifest.csv').exists():
        original=args.manifest_from/'manifest.csv'
        manifest=pd.read_csv(original).head(args.count)
        prior=json.loads((args.manifest_from/'protocol.json').read_text())
        if prior['root']!=str(args.zarr_root.resolve()) or prior['design']['warmup_frames']!=args.warmup:
            raise ValueError('Manifest root/context differs')
        geometry_hash=hashlib.sha256(grid[0].tobytes()+grid[1].tobytes()+grid[2].encode()).hexdigest()
        if geometry_hash!=prior['geometry_sha256']:raise ValueError('Manifest geometry differs')
        design=dict(prior['design'],requested=args.count,selected=len(manifest),
                    reused_manifest_sha256=file_hash(original),source_manifest=str(original.resolve()))
    else:
        manifest,design=diagnostic_manifest(times,len(grid[1]),len(grid[0]),count=args.count,seed=args.seed,warmup=args.warmup)
    sources=['src/emma_gpm/rdt_teacher.py','src/emma_gpm/rdt_preview.py','src/emma_gpm/rdt_dataset.py','src/emma_gpm/seviri_clips.py','scripts/build_seviri_rdt_dataset.py']
    sources+=['src/emma_gpm/rdt_refined.py','src/emma_gpm/rdt_context.py','src/emma_gpm/rdt_comparison.py']
    identity=dict(root=str(args.zarr_root.resolve()),config=asdict(RefinedRDTConfig() if args.teacher=='refined_v2' else RDTConfig()),teacher_version=args.teacher,analysis_halo=halo,design=design,
        manifest_sha256=hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest(),
        inventory_sha256=hashlib.sha256(inventory.to_csv(index=False).encode()).hexdigest(),
        sources={p:file_hash(ROOT/p) for p in sources},
        geometry_sha256=hashlib.sha256(grid[0].tobytes()+grid[1].tobytes()+grid[2].encode()).hexdigest())
    identity=json.loads(json.dumps(identity));path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('Incompatible scan snapshot or code; use a new output directory')
    atomic_json(path,identity);manifest.to_csv(output/'manifest.csv',index=False);inventory.to_csv(output/'inventory.csv',index=False)
    area_path=output/'pixel_area_native.npy'
    cached_area=args.manifest_from/'pixel_area_native.npy' if args.manifest_from else None
    if not area_path.exists() and cached_area and cached_area.exists():shutil.copyfile(cached_area,area_path)
    if not area_path.exists():
        atomic_json(output/'progress.json',dict(status='geometry',processed=0,requested=len(manifest)))
        physical=projected_pixel_areas(*grid,allow_invalid=True)
        tmp=area_path.with_suffix('.npy.part')
        with tmp.open('wb') as handle:np.save(handle,physical)
        tmp.replace(area_path)
    for directory in ['records','masks','objects','links']:(output/directory).mkdir(exist_ok=True)
    started=time.monotonic();records=[]
    stream=iter(manifest.to_dict('records'))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
            initializer=init_worker,initargs=(str(args.zarr_root),str(output),args.warmup,grid,str(area_path),args.teacher,halo,str(args.manifest_from) if args.manifest_from else None)) as pool:
        pending=deque()
        for _ in range(2*args.workers):
            row=next(stream,None)
            if row is not None:pending.append(pool.submit(evaluate,row))
        while pending:
            records.append(pending.popleft().result())
            if len(records)%128==0:publish(output,records,len(manifest),started,'running')
            row=next(stream,None)
            if row is not None:pending.append(pool.submit(evaluate,row))
    publish(output,records,len(manifest),started,'complete')
    if args.teacher=='refined_v2' and args.manifest_from and (args.manifest_from/'catalog.csv').exists():
        from emma_gpm.rdt_comparison import compare_catalogs
        compare_catalogs(output,args.manifest_from)


if __name__=='__main__':main()
