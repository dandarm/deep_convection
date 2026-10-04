#!/usr/bin/env python3
"""Resumable V2-Small masking sweep and nested 2000/4000/8000 learning curve."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from run_seviri_masking_experiments import write_json,run_command


def spaced_starts(starts):
    """Select exact complete windows with no common frames (80-minute stride)."""
    import pandas as pd
    chosen=[]
    for start in starts.sort_values():
        if not chosen or start-chosen[-1]>=pd.Timedelta(minutes=80):
            chosen.append(start)
    return pd.DatetimeIndex(chosen)


def prepare(output,zarr_root,workers):
    import numpy as np
    import pandas as pd
    from build_random_seviri_clip_manifest import inventory,complete_start_times,parallel_candidates,utc_iso
    from build_spaced_seviri_manifest import write_stats
    from emma_gpm.pretraining_validation import split_temporal_starts,assert_temporal_holdout
    from emma_gpm.spatial_sampling import CropEvent,verify_overlap
    if (output/'dataset_ready.json').exists():
        from pretrain_videomae import load_training_stats
        ready=json.loads((output/'dataset_ready.json').read_text())
        for name,digest in ready['manifest_sha256'].items():
            if hashlib.sha256((output/f'{name}.csv').read_bytes()).hexdigest()!=digest:
                raise ValueError('Prepared manifest changed')
        for count in (2000,4000,8000):
            load_training_stats(output/f'train{count}_stats.json',output/f'train{count}.csv',zarr_root)
        return
    snapshot_path=output/'temporal_split.json'
    if snapshot_path.exists():
        snapshot=json.loads(snapshot_path.read_text())
        height,width=snapshot['grid']
        train=pd.DatetimeIndex(snapshot['training_starts'])
        val=pd.DatetimeIndex(snapshot['validation_starts'])
    else:
        # Only atomically published and verified acquisition days. Freeze this
        # snapshot so the ongoing download cannot change any experiment split.
        markers=sorted((zarr_root/'_acquisition/2020/completed_days').glob('*.json'))
        days={pd.Timestamp(json.loads(p.read_text())['day']) for p in markers}
        times,height,width=inventory(zarr_root,224)
        times=times[times.normalize().isin(days)]
        starts=complete_start_times(times,num_frames=16,cadence_minutes=5)
        train,val=split_temporal_starts(starts,validation_days=3,embargo_hours=24)
        snapshot=dict(grid=[height,width],verified_days=len(days),frames=len(times),
            training_starts=[t.isoformat() for t in train],validation_starts=[t.isoformat() for t in val],
            embargo_hours=24,validation_days=3,created_unix=time.time())
        write_json(snapshot_path,snapshot)
    rng=np.random.default_rng(20261003)
    reports={}; digests={}
    for name,starts,count,collect_moments in [('train8000',train,8000,True),('validation',val,256,False)]:
        candidates=[]
        # Randomly offset a full nonoverlapping lattice at each window. This
        # avoids the low packing density of greedy random origins, while each
        # crop's geographic origin changes between windows.
        # Shuffle the candidate pool before validation for broad date coverage
        # in each prefix, rather than filling the first days chronologically.
        for start in spaced_starts(starts):
            rows,cols=height//224,width//224
            offset_y=int(rng.integers(height-rows*224+1))
            offset_x=int(rng.integers(width-cols*224+1))
            for y in range(rows):
                for x in range(cols):
                    candidates.append((utc_iso(start),offset_y+y*224,offset_x+x*224))
        rng.shuffle(candidates)
        print(f'PREPARE {name}: candidates={len(candidates)} windows={len(spaced_starts(starts))}',flush=True)
        accepted=[]; moments=[]; events=[]; rejected=0
        stream=parallel_candidates(iter(candidates),workers,(zarr_root,16,5,224,collect_moments))
        try:
            for result in stream:
                if result is None:
                    rejected+=1;continue
                (stamp,y,x),moment=result
                accepted.append(dict(clip_id=f'{name}_{len(accepted):06d}',start_time=stamp,
                    origin_y=y,origin_x=x,selection='random_offset_lattice_nonoverlapping_space_time',seed=20261003))
                events.append(CropEvent(pd.Timestamp(stamp),y,x));moments.append(moment)
                if len(accepted)%256==0:
                    print(f'{name} accepted={len(accepted)}/{count} rejected={rejected}',flush=True)
                if len(accepted)==count:
                    break
        finally:
            stream.close()
        if len(accepted)!=count:
            raise ValueError(f'Only {len(accepted)}/{count} valid nonoverlapping samples; expand verified archive')
        frame=pd.DataFrame(accepted)
        reports[name]=dict(samples=len(frame),days=pd.to_datetime(frame.start_time).dt.date.nunique(),
            temporal_windows=frame.start_time.nunique(),rejected=rejected,**verify_overlap(events))
        counts=(2000,4000,8000) if collect_moments else (256,)
        for size in counts:
            key=f'train{size}' if collect_moments else 'validation'
            path=output/f'{key}.csv'
            subset=frame.iloc[:size].copy()
            subset.to_csv(path,index=False)
            digests[key]=hashlib.sha256(path.read_bytes()).hexdigest()
            if collect_moments:
                write_stats(output/f'{key}_stats.json',path,subset,
                            np.asarray(moments[:size],dtype=np.float64).sum(0),zarr_root)
            reports[key]=dict(samples=size,days=pd.to_datetime(subset.start_time).dt.date.nunique(),
                temporal_windows=subset.start_time.nunique(),maximum_concurrent_iou=0.)
        if collect_moments:
            reports['train8000'].update(verify_overlap(events))
        reports[name].update(rejected_candidates=rejected,candidate_pool=len(candidates),
                             sampling='random-offset nonoverlapping lattice and 80-minute windows')
    assert_temporal_holdout(pd.read_csv(output/'train8000.csv'),pd.read_csv(output/'validation.csv'))
    write_json(output/'dataset_ready.json',dict(manifest_sha256=digests,reports=reports,
        snapshot='Frozen verified acquisition days; independent of subsequent downloads',
        nesting='train2000 is prefix of train4000, which is prefix of train8000',
        normalization='One scalar mean/std from all pixels of each training manifest; compare sizes in Kelvin'))


def preflight(output,zarr_root):
    import pandas as pd
    import torch
    from emma_gpm.seviri_clips import SeviriZarrClipReader
    from emma_gpm.videomae import forward_seviri_mae
    from emma_gpm.videomae_v2 import build_videomae_v2_small_7ch_from_scratch,running_cell_mask
    from pretrain_videomae import make_tube_mask,seed_everything
    torch.set_num_threads(4);seed_everything(20261003)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA bf16 unavailable')
    stats=json.loads((output/'train2000_stats.json').read_text())
    row=pd.read_csv(output/'train2000.csv').iloc[0]
    with SeviriZarrClipReader(zarr_root) as reader:
        clip=reader.load(row.start_time,origin_y=int(row.origin_y),origin_x=int(row.origin_x),
            global_mean=stats['global_mean_k'],global_std=stats['global_std_k'])
    pixels=torch.from_numpy(clip.copy()).unsqueeze(0).cuda()
    model=build_videomae_v2_small_7ch_from_scratch().cuda().eval()
    mask=make_tube_mask(model.config,1,.5,'cuda');decode=running_cell_mask(model.config,1,.5,device='cuda')
    with torch.no_grad():
        fp32=float(forward_seviri_mae(model,pixels,mask,decode).loss)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            bf16=float(forward_seviri_mae(model,pixels,mask,decode).loss)
    if abs(fp32-bf16)/fp32>.02:
        raise RuntimeError('bf16/FP32 preflight discrepancy >2%')
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=1.5e-4)
    torch.cuda.reset_peak_memory_stats()
    batch=pixels.repeat(16,1,1,1,1)
    for _ in range(2):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=forward_seviri_mae(model,batch,make_tube_mask(model.config,16,.5,'cuda'),
                running_cell_mask(model.config,16,.5,device='cuda')).loss
        (loss/2).backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError('Nonfinite loss/gradients')
    optimizer.step()
    peak=torch.cuda.max_memory_reserved()/2**30
    if peak>21:
        raise RuntimeError('Preflight exceeds 21 GiB reserved')
    write_json(output/'preflight.json',dict(model='VideoMAE V2 Small 7ch',fp32_loss=fp32,bf16_loss=bf16,
        relative_difference=abs(fp32-bf16)/fp32,peak_reserved_gib=peak,microbatch=16,effective_batch=32,
        encoder_mask=.5,decoder_mask=.5,backward_finite=True,optimizer_step=True))


def train(output,zarr_root,name,count,ratio,epochs):
    destination=output/name;destination.mkdir(exist_ok=True)
    summary_path=destination/'smoke_summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        if (summary['epochs'],summary['samples'],summary.get('model_version'),summary['mask_ratio_requested'])!=(epochs,count,'v2',ratio):
            raise ValueError('Existing completed run differs from plan')
        return
    command=[sys.executable,ROOT/'scripts/pretrain_videomae.py','--zarr-root',zarr_root,
        '--manifest',output/f'train{count}.csv','--global-stats',output/f'train{count}_stats.json',
        '--validation-manifest',output/'validation.csv','--validation-mask-ratio',.9,
        '--output-dir',destination,'--device','cuda','--epochs',epochs,
        '--model-version','v2','--decoder-mask-ratio',.5,'--mask-ratio',ratio,
        '--batch-size',16,'--gradient-accumulation',2,'--precision','bf16',
        '--num-workers',16,'--loader-batch-size',2,'--prefetch-factor',2,'--pin-memory',
        '--cpu-threads',4,'--seed',20261003,'--initialization','scratch','--no-cpu-smoke-decoder',
        '--checkpoint-every',1]
    checkpoint=destination/'checkpoint_latest.pt'
    if checkpoint.exists():
        command+=['--resume',checkpoint]
    run_command(command,destination/'training.log')


def update_size_comparison(output,results):
    """Compare identical holdout/masks in Kelvin; normalized scales differ by size."""
    import numpy as np
    import pandas as pd
    frame=pd.DataFrame(results)
    frame.to_csv(output/'dataset_comparison.csv',index=False)
    write_json(output/'dataset_comparison.json',dict(results=results,
        selection='Best checkpoint by common encoder90/decoder50 validation MSE within each run',
        limitation='Equal epochs, different update counts; validation reused for selection, no independent test yet'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(7,4))
    axis.plot(frame.samples,np.square(frame.rmse_k),'-o',label='Validation comune: encoder90%, decoder50%')
    axis.set(xlabel='Numero di clip di training',ylabel='MSE di validation (K²)',title='VideoMAE V2 Small — scala del dataset')
    axis.set_xticks(frame.samples);axis.grid(alpha=.3);axis.legend();fig.tight_layout()
    fig.savefig(output/'loss_vs_samples.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zarr-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--prepare-workers',type=int,default=16)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.epochs<1 or args.prepare_workers<1:
        parser.error('Positive epochs/workers required')
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=dict(model='VideoMAE V2 Small 7ch',epochs=args.epochs,sizes=[2000,4000,8000],
            encoder_masks=[.5,.75,.9],decoder_mask=.5,validation_samples=256,
            common_validation_encoder_mask=.9,seed=20261003,zarr_root=str(args.zarr_root.resolve()),
            initialization='scratch for all five runs',selection='lowest common-mask validation MSE; tie order50/75/90')
        if (output/'plan.json').exists() and json.loads((output/'plan.json').read_text())!=plan:
            raise ValueError('Plan changed; select a new output directory')
        write_json(output/'plan.json',plan)
        def status(stage,**extra):
            write_json(output/'queue_status.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**extra))
        monitor=None
        try:
            status('preparing');prepare(output,args.zarr_root,args.prepare_workers)
            if args.prepare_only:
                status('prepared');return
            status('preflight')
            if not (output/'preflight.json').exists():
                run_command([sys.executable,'-c',
                    'import sys;sys.path.insert(0,"scripts");from pathlib import Path;from run_seviri_v2_experiments import preflight;preflight(Path(sys.argv[1]),Path(sys.argv[2]))',
                    output,args.zarr_root],output/'preflight.log')
            with (output/'gpu_telemetry.csv').open('a') as telemetry:
                monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw',
                    '--format=csv','--loop-ms=1000'],stdout=telemetry,stderr=subprocess.DEVNULL)
            results=[]
            for ratio in (.5,.75,.9):
                name=f'mask{round(ratio*100)}'
                status('training',run=name)
                train(output,args.zarr_root,name,2000,ratio,args.epochs)
                result=json.loads((output/name/'best_validation.json').read_text())
                results.append(dict(run=name,training_mask=ratio,**result))
                write_json(output/'masking_comparison.json',results)
            best=min(results,key=lambda row:row['mse'])
            write_json(output/'selection.json',dict(best=best,criterion=plan['selection']))
            learning_curve=[dict(run=best['run'],samples=2000,training_mask=best['training_mask'],
                epoch=best['epoch'],rmse_k=best['rmse_k'])]
            update_size_comparison(output,learning_curve)
            for count in (4000,8000):
                name=f'large{count}'
                status('training',run=name,selected_mask=best['training_mask'])
                train(output,args.zarr_root,name,count,best['training_mask'],args.epochs)
                result=json.loads((output/name/'best_validation.json').read_text())
                learning_curve.append(dict(run=name,samples=count,training_mask=best['training_mask'],
                    epoch=result['epoch'],rmse_k=result['rmse_k']))
                update_size_comparison(output,learning_curve)
            status('completed',selected_mask=best['training_mask'])
        except BaseException as error:
            status('failed',error=repr(error));raise
        finally:
            if monitor is not None:
                monitor.terminate();monitor.wait(timeout=10)


if __name__=='__main__':
    main()
