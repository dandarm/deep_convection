#!/usr/bin/env python3
"""Resume a matched four-objective pilot and five common frozen-encoder probes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.ssl_dataset import verify_snapshot
from scripts.pretrain_seviri_ssl import digest, write_json, source_hashes


def stages(args):
    base=['--manifest',str(args.dataset/'train.csv'),
        '--validation-manifest',str(args.dataset/'validation.csv'),
        '--global-stats',str(args.dataset/'channel_stats.json'),
        '--zarr-root',str(args.zarr_root),'--device','cuda','--precision','bf16',
        '--batch-size',str(args.batch_size),'--gradient-accumulation','2',
        '--loader-batch-size','2','--num-workers',str(args.workers),'--prefetch-factor','2',
        '--cpu-threads','4','--seed',str(args.seed),'--mask-ratio','0.9','--mask-strategy','tube',
        '--learning-rate','0.00015','--head-width','192']
    result=[]
    for objective in ('future','mvm','joint','residual'):
        result.append(dict(name=objective,kind='pretraining',epochs=args.epochs,
            flags=base+['--objective',objective,'--epochs',str(args.epochs)]))
    for encoder in ('random','mvm','future','joint','residual'):
        flags=base+['--objective','future','--freeze-encoder','--epochs',str(args.probe_epochs)]
        if encoder!='random':
            flags+=['--encoder-from',str(args.output_dir/encoder/'encoder')]
        result.append(dict(name='probe_'+encoder,kind='frozen_common_head',epochs=args.probe_epochs,flags=flags))
    return result


def run(args):
    if not (args.dataset/'protocol.json').exists():
        raise ValueError('Dataset preparation has not completed')
    plan=dict(stages=stages(args),dataset={name:digest(args.dataset/name) for name in
        ('train.csv','validation.csv','channel_stats.json','source_snapshot.json')},
        sources=source_hashes(),seed=args.seed,
        interpretation='engineering pilot; overlapping clips and a single seed; equal samples/updates, unequal compute')
    plan_path=args.output_dir/'comparison_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Frozen comparison plan differs; use a new output directory')
    write_json(plan_path,plan)
    verify_snapshot(args.zarr_root,json.loads((args.dataset/'source_snapshot.json').read_text()))
    for stage in plan['stages']:
        output=args.output_dir/stage['name']
        progress_path=output/'progress.json'
        progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
        if progress.get('status')=='completed' and progress.get('completed_epochs')==stage['epochs'] and (output/'encoder/config.json').exists():
            print('Already completed:',stage['name'],flush=True)
            continue
        command=[sys.executable,'-u',str(ROOT/'scripts/pretrain_seviri_ssl.py'),*stage['flags'],'--output-dir',str(output)]
        if (output/'checkpoint_latest.pt').exists():
            command+=['--resume',str(output/'checkpoint_latest.pt')]
        log_path=args.output_dir/(stage['name']+'.log')
        started=time.time()
        with log_path.open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            write_json(args.output_dir/'queue_status.json',dict(status='running',stage=stage['name'],
                pid=process.pid,supervisor_pid=os.getpid(),command=command,started_at_utc=datetime.now(timezone.utc).isoformat()))
            print('Started',stage['name'],'PID',process.pid,flush=True)
            code=process.wait()
        if code:
            write_json(args.output_dir/'queue_status.json',dict(status='failed',stage=stage['name'],returncode=code,log=str(log_path)))
            raise RuntimeError(f"{stage['name']} exited {code}; see {log_path}")
        print('Completed',stage['name'],'seconds',round(time.time()-started,2),flush=True)
    verify_snapshot(args.zarr_root,json.loads((args.dataset/'source_snapshot.json').read_text()))
    write_json(args.output_dir/'queue_status.json',dict(status='completed',stages=len(plan['stages']),
        source_snapshot_verified=True,completed_at_utc=datetime.now(timezone.utc).isoformat()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--zarr-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--probe-epochs',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20261024)
    args=parser.parse_args()
    for name in ('dataset','zarr_root','output_dir'):
        setattr(args,name,getattr(args,name).resolve())
    if min(args.epochs,args.probe_epochs,args.workers,args.batch_size)<1 or args.batch_size%2:
        parser.error('Positive epochs/workers and even batch size required')
    with exclusive_file_lock(args.output_dir/'.comparison.lock'):
        run(args)


if __name__=='__main__':
    main()
