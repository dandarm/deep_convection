#!/usr/bin/env python3
"""Count label-based candidate strata at every permitted spatial crop origin.

Temporal windows follow the dataset plan. Optional input checks verify complete
seven-channel radiometry over all sixteen frames. Diagnostics never publish a
training dataset or alter its strata definitions.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from emma_gpm.three_head_data import valid_origins, _crop_sums, teacher_period_directory
from emma_gpm.rdt.storage import atomic_json


def audit(plan,split,output,reader=None):
    folder=teacher_period_directory(plan,split)
    report=json.loads((folder/'complete.json').read_text())
    frames=report['frames'];size=plan['crop_size'];length=plan['clip_frames']
    with np.load(folder/'geometry.npz') as geometry:
        origins=valid_origins(geometry['latitude'],geometry['longitude'],size,plan['geographic_bounds'])
    totals=dict(A=0,B=0,C=0,unassigned=0,empty=0,input_rejected=0)
    diagnostic_thresholds=[.001,.005,.01]
    low_convective_counts=dict.fromkeys(map(str,diagnostic_thresholds),0)
    centred_cold_counts=0
    examples=[];windows=0
    for start in range(3,len(frames)-length+1,plan['temporal_stride_frames']):
        times=pd.DatetimeIndex([row['time'] for row in frames[start:start+length]])
        if not (np.diff(times.asi8)==pd.Timedelta(minutes=5).value).all():
            continue
        arrays={key:[] for key in ('ct','rdt','valid')}
        for row in frames[start:start+length]:
            with np.load(folder/row['file']) as archive:
                for key in arrays:arrays[key].append(archive[key])
        ct,rdt,valid=(np.stack(arrays[key]) for key in ('ct','rdt','valid'))
        counts={key:_crop_sums(values.sum(0),origins,size) for key,values in dict(
            valid=valid,conv=(rdt>=2)&(rdt<=4)&valid,
            cold=(rdt==1)&valid,high=np.isin(ct,[3,4])&valid).items()}
        input_complete=np.ones(len(origins),dtype=bool)
        if reader is not None:
            complete_pixels=np.ones_like(valid[0])
            for stamp in times:
                frame=reader.load(stamp)
                complete_pixels &= (np.isfinite(frame).all(0)
                                    & (frame>=100).all(0) & (frame<=400).all(0))
            input_complete=_crop_sums(complete_pixels,origins,size)==size*size
        nonempty=counts['valid']>0
        observed=nonempty&input_complete
        denominator=np.maximum(counts['valid'],1)
        high=counts['high']/denominator;cold=counts['cold']/denominator
        rich=observed&(high>=plan['bucket_b_min_high_fraction'])&(cold>=plan['bucket_b_min_cold_fraction'])
        cy,cx=(origins+size//2).T
        cold_centre=(rdt[-1,cy,cx]==1)&np.isin(ct[-1,cy,cx],[3,4])
        centred_cold_counts+=int((rich&cold_centre).sum())
        a=observed&(counts['conv']>0);b=rich&~a;c=observed&~a&~b&(high<.5)
        for key,mask in [('A',a),('B',b),('C',c),('unassigned',observed&~a&~b&~c),
                         ('empty',input_complete&~nonempty),('input_rejected',~input_complete)]:
            totals[key]+=int(mask.sum())
        for threshold in diagnostic_thresholds:
            low_convective_counts[str(threshold)]+=int((rich&(counts['conv']/denominator<=threshold)).sum())
        for index in np.flatnonzero(b)[:max(0,20-len(examples))]:
            y,x=origins[index]
            examples.append(dict(start_time=frames[start]['time'],y=int(y),x=int(x),high_fraction=float(high[index]),cold_fraction=float(cold[index])))
        windows+=1
        state=dict(status='running',split=split,windows=windows,spatial_origins=len(origins),counts=totals,
                   scope='All permitted pixel origins at planned temporal stride',
                   complete_seven_channel_inputs_checked=reader is not None,
                   high_cloud_candidates_below_convective_fraction=low_convective_counts,
                   high_cloud_candidates_with_cold_centre=centred_cold_counts,
                   diagnostic_thresholds_applied_to_training=False,strict_bucket_b_examples=examples)
        atomic_json(output,state)
        print(windows,totals,flush=True)
    if not windows:
        raise ValueError('The prepared period has no complete planned temporal window')
    state['status']='completed';atomic_json(output,state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--split',choices=['train','validation','test'],default='train')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check-inputs',action='store_true',help='Check actual seven-channel radiometry in every temporal window')
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    from emma_gpm.rdt.storage import DomainFrameReader
    reader=DomainFrameReader(plan['source_root']) if args.check_inputs else None
    try:
        audit(plan,args.split,args.output,reader=reader)
    finally:
        if reader is not None:reader.close()


if __name__=='__main__':main()
