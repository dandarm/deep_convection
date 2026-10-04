#!/usr/bin/env python3
"""Prepare strict spatially separated SEVIRI clips and a nested small baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from build_random_seviri_clip_manifest import inventory, complete_start_times
from emma_gpm.seviri_clips import SeviriZarrClipReader, IncompleteSeviriClipError, VIDEO_MAE_CHANNELS
from emma_gpm.spatial_sampling import CropEvent, choose_crop, verify_overlap, shared_frames
from emma_gpm.pretraining_validation import assert_temporal_holdout


def write_stats(path, manifest, rows, moments, zarr_root):
    total, squared, count = moments
    mean=total/count
    std=float(np.sqrt(max(0.,squared/count-mean*mean)))
    if not np.isfinite(std) or std<=0:
        raise ValueError('Invalid global training moments')
    data=dict(channels=list(VIDEO_MAE_CHANNELS),normalization='single_global_affine_transform',
              global_mean_k=mean,global_std_k=std,scope='all pixels, channels, and clips in the training split',
              manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),samples=len(rows),
              pixel_count=int(count),zarr_root=str(zarr_root.resolve()))
    path.write_text(json.dumps(data,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zarr-root',type=Path,required=True)
    parser.add_argument('--starts-file',type=Path,required=True)
    parser.add_argument('--validation-manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261002)
    parser.add_argument('--max-count',type=int,default=40000)
    parser.add_argument('--max-iou-active',type=float,default=0.)
    parser.add_argument('--sampling-attempts',type=int,default=2048)
    parser.add_argument('--crops-per-start',type=int,default=26)
    parser.add_argument('--coverage-weight',type=float,default=1.)
    parser.add_argument('--border-boost',type=float,default=.25)
    parser.add_argument('--min-gap-pixels',type=int,default=0)
    parser.add_argument('--center-bounds',type=float,nargs=4,metavar=('LON_MIN','LON_MAX','LAT_MIN','LAT_MAX'))
    args=parser.parse_args()
    if args.max_count<2 or not 0<=args.max_iou_active<1 or args.sampling_attempts<1 or args.crops_per_start<1 or args.min_gap_pixels<0:
        parser.error('Invalid sampling parameters')
    if args.center_bounds and not (args.center_bounds[0]<args.center_bounds[1] and args.center_bounds[2]<args.center_bounds[3]):
        parser.error('Invalid geographic center bounds')
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    if (output/'dataset_report.json').exists():
        raise SystemExit('Completed dataset exists; select another output directory')
    times,height,width=inventory(args.zarr_root,224)
    complete=complete_start_times(times,num_frames=16,cadence_minutes=5)
    starts=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(args.starts_file).start_time,utc=True)).tz_localize(None).sort_values()
    if not starts.is_unique or not starts.isin(complete).all():
        raise ValueError('Duplicate/incomplete requested starts')
    assert_temporal_holdout(pd.DataFrame({'start_time':starts}),pd.read_csv(args.validation_manifest))
    center_allowed=None
    if args.center_bounds:
        from pyproj import Transformer
        from emma_gpm.seviri import RSS_CRS
        first=starts[0]
        with xr.open_zarr(args.zarr_root/f'{first:%Y}'/f'{first:%Y%m%d}.zarr',consolidated=True,chunks=None) as ds:
            x=ds.x.values.copy(); y=ds.y.values.copy()
        transform=Transformer.from_crs(RSS_CRS,'EPSG:4326',always_xy=True)
        lonmin,lonmax,latmin,latmax=args.center_bounds
        def center_allowed(y0,x0):
            lon,lat=transform.transform(float(x[x0+112]),float(y[y0+112]))
            return lonmin<=lon<=lonmax and latmin<=lat<=latmax
    rng=np.random.default_rng(args.seed)
    coverage=np.zeros((height,width),dtype=np.uint32)
    active=[];events=[];rows=[];moments=[]
    rejected_invalid=0; exhausted=0
    import zarr
    from numcodecs import blosc
    zarr.config.set({'async.concurrency':8,'threading.max_workers':4})
    blosc.set_nthreads(1)
    with SeviriZarrClipReader(args.zarr_root) as reader:
        for start in starts:
            active=[event for event in active if shared_frames(CropEvent(start,0,0),event)]
            for _ in range(args.crops_per_start):
                chosen=None
                # Retry finite-pixel failures; do not repair or count rejected crops.
                for _retry in range(16):
                    candidate=choose_crop(rng,start,active,coverage,224,
                        attempts=args.sampling_attempts,max_iou=args.max_iou_active,
                        coverage_weight=args.coverage_weight,border_boost=args.border_boost,
                        min_gap_pixels=args.min_gap_pixels,center_allowed=center_allowed)
                    if candidate is None:
                        break
                    try:
                        pixels=reader.load(start,origin_y=candidate.y,origin_x=candidate.x)
                    except IncompleteSeviriClipError:
                        rejected_invalid+=1
                        # Coverage bias discourages repeating a rejected region.
                        # This search map is not used for radiometric statistics.
                        coverage[candidate.y:candidate.y+224,candidate.x:candidate.x+224]+=1
                        continue
                    chosen=candidate
                    break
                if chosen is None:
                    exhausted+=1
                    break
                values=pixels.astype(np.float64)
                moments.append((float(values.sum()),float(np.square(values).sum()),values.size))
                active.append(chosen);events.append(chosen)
                coverage[chosen.y:chosen.y+224,chosen.x:chosen.x+224]+=1
                rows.append(dict(clip_id=f'spaced_{len(rows):06d}',start_time=start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    origin_y=chosen.y,origin_x=chosen.x,selection='random_strict_active_overlap_and_coverage',seed=args.seed))
                if len(rows)>=args.max_count:
                    break
            print(f'start={start} accepted={len(rows)} active={len(active)} invalid={rejected_invalid}',flush=True)
            if len(rows)>=args.max_count:
                break
    if len(rows)<2:
        raise ValueError('Fewer than two valid clips; cannot prepare nested comparison')
    verification=verify_overlap(events,max_iou=args.max_iou_active)
    frame=pd.DataFrame(rows)
    manifest=output/'train_spaced_all.csv'
    frame.to_csv(manifest,index=False)
    totals=np.asarray(moments,dtype=np.float64).sum(0)
    write_stats(output/'train_spaced_all_stats.json',manifest,rows,totals,args.zarr_root)
    # A roughly tenfold expansion; stratify across chronological manifest.
    small_count=min(len(rows)-1,max(8,len(rows)//10))
    indices=np.sort(rng.choice(len(rows),small_count,replace=False))
    # Guarantee broad temporal coverage when the subset permits it.
    days=pd.to_datetime(frame.start_time,utc=True).dt.normalize()
    if small_count>=days.nunique():
        representatives=[int(rng.choice(group.index)) for _,group in frame.groupby(days)]
        remaining=np.setdiff1d(np.arange(len(rows)),representatives)
        indices=np.sort(np.r_[representatives,rng.choice(remaining,small_count-len(representatives),replace=False)])
    small=frame.iloc[indices].reset_index(drop=True)
    small_path=output/'train_spaced_small.csv'
    small.to_csv(small_path,index=False)
    write_stats(output/'train_spaced_small_stats.json',small_path,small,np.asarray(moments,dtype=np.float64)[indices].sum(0),args.zarr_root)
    # Occupancy of real frame/pixel voxels: verify zero reused voxels for strict IoU=0.
    frame_counts={}
    for event in events:
        for timestamp in pd.date_range(event.start,periods=16,freq='5min'):
            frame_counts[timestamp]=frame_counts.get(timestamp,0)+1
    report=dict(grid_height=height,grid_width=width,pixels_per_frame_per_channel=height*width,
        values_per_frame_all_channels=height*width*7,grid_nonoverlapping_tiles=(height//224)*(width//224),
        area_upper_bound_per_frame=(height*width)//(224*224),training_starts=len(starts),
        requested_cap=args.max_count,accepted_samples=len(frame),small_samples=len(small),
        growth_ratio=len(frame)/len(small),maximum_active_crops=max(frame_counts.values()),
        rejected_nonfinite_or_incomplete=rejected_invalid,exhausted_start_searches=exhausted,
        sampling=dict(seed=args.seed,max_iou_active=args.max_iou_active,sampling_attempts=args.sampling_attempts,
            crops_per_start=args.crops_per_start,coverage_weight=args.coverage_weight,border_boost=args.border_boost,
            min_gap_pixels=args.min_gap_pixels,center_bounds=args.center_bounds),
        **verification,
        source_notebook='https://github.com/dandarm/Demetra/blob/main/notebooks/random_tiles.ipynb',
        source_implementation='https://github.com/dandarm/Demetra/blob/main/moduli/videomae/dataset/build_dataset.py',
        limitations=['No claim of globally optimal packing: greedy random search can leave usable space.',
                    'Reuse of a region is permitted only after the previous clip last frame.',
                    'Sample expansion uses the same downloaded archive; no new meteorological episodes.'])
    (output/'dataset_report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    accepted_coverage=np.zeros_like(coverage)
    for event in events:
        accepted_coverage[event.y:event.y+224,event.x:event.x+224]+=1
    image=axes[0].imshow(accepted_coverage,origin='upper',cmap='viridis')
    axes[0].set_title('Copertura cumulativa dei crop accettati (indici Zarr nativi)')
    fig.colorbar(image,ax=axes[0],label='Numero di clip che coprono il pixel')
    first_start=events[0].start
    axes[1].set_xlim(0,width);axes[1].set_ylim(height,0);axes[1].set_aspect('equal')
    for event in events:
        if event.start==first_start:
            axes[1].add_patch(Rectangle((event.x,event.y),224,224,fill=False,edgecolor='tab:blue'))
    axes[1].set_title(f'Crop simultanei, primo istante: {first_start}')
    for axis in axes:
        axis.set(xlabel='Indice x nativo',ylabel='Indice y nativo')
    fig.tight_layout();fig.savefig(output/'sampling_diagnostics.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
