#!/usr/bin/env python3
"""Repeat the frozen V2 size comparison at encoder90/decoder50, from scratch."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from run_seviri_v2_experiments import prepare,train
from run_seviri_masking_experiments import write_json


def comparison(output,results):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(results)
    frame.to_csv(output/'mask90_scaling_comparison.csv',index=False)
    write_json(output/'mask90_scaling_comparison.json',dict(results=results,
        criterion='Best common validation encoder90/decoder50, identical holdout',
        limitation='Equal epochs; larger datasets entail more optimizer updates'))
    fig,ax=plt.subplots(figsize=(7,4))
    for ratio,group in frame.groupby('training_mask'):
        ax.plot(group.samples,group.rmse_k**2,'o-',label=f'Encoder training {ratio:.0%}')
    ax.set(xlabel='Clip di training',ylabel='Validation MSE (K²)',
           title='VideoMAE V2 Small — validation encoder90%, decoder50%')
    ax.set_xticks([2000,4000,8000]);ax.grid(alpha=.3);ax.legend();fig.tight_layout()
    fig.savefig(output/'mask90_loss_vs_samples.png',dpi=180);plt.close(fig)


def execute(output,zarr_root,epochs,status):
    # Reuse verified manifests/statistics; do not include newly downloaded days.
    prepare(output,zarr_root,16)
    if not (output/'preflight.json').exists():
        raise ValueError('The original CUDA preflight is required')
    results=[]
    for name,count in [('mask75',2000),('large4000',4000),('large8000',8000),('mask90',2000)]:
        summary=json.loads((output/name/'smoke_summary.json').read_text())
        expected_ratio=.9 if name=='mask90' else .75
        if (summary['epochs'],summary['samples'],summary.get('model_version'),summary['mask_ratio_requested'])!=(epochs,count,'v2',expected_ratio):
            raise ValueError('Existing baseline does not match requested comparison')
        best=json.loads((output/name/'best_validation.json').read_text())
        results.append(dict(run=name,samples=count,training_mask=expected_ratio,
                            epoch=best['epoch'],rmse_k=best['rmse_k']))
    comparison(output,results)
    for count in (4000,8000):
        name=f'large{count}_mask90'
        status('training',run=name,encoder_mask=.9,decoder_mask=.5)
        train(output,zarr_root,name,count,.9,epochs)
        best=json.loads((output/name/'best_validation.json').read_text())
        results.append(dict(run=name,samples=count,training_mask=.9,
                            epoch=best['epoch'],rmse_k=best['rmse_k']))
        comparison(output,results)
    status('completed',encoder_mask=.9,decoder_mask=.5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-dir',type=Path,required=True)
    parser.add_argument('--zarr-root',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=150)
    args=parser.parse_args()
    if args.epochs<1:
        parser.error('Positive epoch count required')
    output=args.queue_dir.resolve()
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=dict(epochs=args.epochs,sizes=[4000,8000],encoder_mask=.9,decoder_mask=.5,
                  initialization='scratch',source_queue=str(output),zarr_root=str(args.zarr_root.resolve()))
        path=output/'mask90_scaling_plan.json'
        if path.exists() and json.loads(path.read_text())!=plan:
            raise ValueError('Existing mask90 scaling plan differs')
        write_json(path,plan)
        def status(stage,**extra):
            write_json(output/'mask90_scaling_status.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**extra))
        monitor=None
        try:
            status('checking')
            with (output/'mask90_gpu_telemetry.csv').open('a') as handle:
                monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw',
                    '--format=csv','--loop-ms=1000'],stdout=handle,stderr=subprocess.DEVNULL)
            execute(output,args.zarr_root,args.epochs,status)
        except BaseException as error:
            status('failed',error=repr(error));raise
        finally:
            if monitor is not None:
                monitor.terminate();monitor.wait(timeout=10)


if __name__=='__main__':
    main()
