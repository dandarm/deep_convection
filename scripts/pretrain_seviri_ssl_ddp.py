#!/usr/bin/env python3
"""Single-node DDP SSL with global batch accounting and two fixed holdouts."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from emma_gpm.advection import advection_forecast
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.forecasting import ForecastWindow
from emma_gpm.seviri_clip_files import ClipBatchLoader
from emma_gpm.seviri_frame_dataset import FrameFileDataset
from emma_gpm.ssl_distributed import GlobalBatchSampler,initialize,save_distributed,restore_distributed
from emma_gpm.ssl_diagnostics import encoder_digest
from emma_gpm.ssl_evaluation import evaluate_ssl
from emma_gpm.ssl_masking import fixed_ssl_mask
from emma_gpm.ssl_model import SeviriSSL,history_config,ssl_loss
from emma_gpm.ssl_runtime import CudaRuntime,configure_runtime,compile_model_regions,make_optimizer,apply_optimizer_runtime
from emma_gpm.storage_roots import storage_mapping_digest,storage_root
from emma_gpm.videomae_v2 import fixed_running_cell_mask
from scripts.pretrain_seviri_ssl import SyntheticClips,validate_manifests,write_json,digest,source_hashes
from scripts.pretrain_videomae import load_training_stats,seed_everything
from scripts.evaluate_seviri_ssl_validation import expected_digest


def args_from_cli():
    early=argparse.ArgumentParser(add_help=False);early.add_argument('--config',type=Path)
    initial,_=early.parse_known_args()
    p=argparse.ArgumentParser(description=__doc__,parents=[early])
    for name in ('dataset','zarr-root','frame-cache','extended-dataset','output-dir','resume','stop-after-epoch-file','execution-revision'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--model-size',choices=['small','base'],default='base')
    p.add_argument('--objective',choices=['mvm','future','joint','residual'],default='future')
    p.add_argument('--batch-size-per-gpu',type=int,default=8)
    p.add_argument('--validation-batch-size',type=int,default=32)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--extended-every',type=int,default=10)
    p.add_argument('--learning-rate',type=float,default=1.5e-4)
    p.add_argument('--weight-decay',type=float,default=.05)
    p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--prefetch',type=int,default=2)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    p.add_argument('--compile-mode',choices=['off','default'],default='default')
    p.add_argument('--synthetic-smoke',action='store_true')
    p.add_argument('--capacity-probe',action='store_true')
    p.add_argument('--probe-steps',type=int,default=8)
    p.add_argument('--pipeline-benchmark-steps',type=int,default=0)
    p.add_argument('--benchmark-warmup',type=int,default=4)
    p.add_argument('--log-every',type=int,default=16)
    if initial.config:
        defaults=json.loads(initial.config.read_text());valid={a.dest for a in p._actions}
        if set(defaults)-valid:p.error('Unknown configuration keys: '+str(set(defaults)-valid))
        p.set_defaults(**defaults)
    args=p.parse_args()
    for name in ('config','dataset','zarr_root','frame_cache','extended_dataset','output_dir','resume','stop_after_epoch_file','execution_revision'):
        value=getattr(args,name)
        if value is not None:setattr(args,name,Path(value).resolve())
    if args.output_dir is None:p.error('--output-dir is required')
    if min(args.batch_size_per_gpu,args.validation_batch_size,args.epochs,args.workers,args.prefetch,
           args.cpu_threads,args.extended_every,args.log_every)<1:p.error('Positive counts required')
    if args.learning_rate<=0 or args.weight_decay<0 or args.probe_steps<4:p.error('Invalid optimization/probe parameters')
    if args.synthetic_smoke and args.capacity_probe:p.error('Select one engineering mode')
    if args.pipeline_benchmark_steps<0 or args.benchmark_warmup<1:p.error('Invalid benchmark counts')
    if args.pipeline_benchmark_steps and (args.capacity_probe or args.resume):p.error('Pipeline benchmarks start from fresh weights')
    return args


def indexed_loader(data,sampler,args,device,stats,batch_size):
    if isinstance(data,SyntheticClips):
        class Loader:
            pin_memory=False
            dataset=data
            def __len__(self):return len(sampler)
            def __iter__(self):
                for indices in sampler:
                    yield dict(pixel_values=torch.stack([data[i]['pixel_values'] for i in indices]),sample_indices=indices)
        return Loader()
    return ClipBatchLoader(data,batch_size=batch_size,workers=args.workers,prefetch=args.prefetch,
        mean=stats['global_mean_k'],std=stats['global_std_k'],device=device,seed=args.seed,
        shuffle=False,batch_sampler=sampler)


def step_loss(wrapped,model,clip,indices,epoch,size,window,args,device):
    history,target=window.split(clip)
    mask=decode=adv=None
    if model.mae is not None:
        keys=[epoch*size+int(i) for i in indices]
        mask=fixed_ssl_mask(model.config,keys,.9,'tube',args.seed+1,device)
        decode=fixed_running_cell_mask(model.config,keys,.5,args.seed+1000001,device)
    if args.objective=='residual':adv,_,_=advection_forecast(history,window.steps,12)
    if args.compile_mode!='off':torch.compiler.cudagraph_mark_step_begin()
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=args.precision=='bf16'):
        output=wrapped(history,mask=mask,decode_mask=decode,advection=adv)
        loss,terms=ssl_loss(output,target)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite distributed training loss')
    return loss,terms


def capacity_probe(args,model,wrapped,optimizer,device,rank,world,window):
    if device.type!='cuda':raise ValueError('VRAM capacity probing requires CUDA')
    shape=(args.batch_size_per_gpu,16,7,224,224)
    # Two live normalized clips and a raw transfer buffer cover batch transition peaks.
    clips=[torch.randn(shape,device=device) for _ in range(2)]
    raw_buffer=torch.empty(shape,dtype=torch.float16,device=device)
    elapsed=[]
    torch.cuda.reset_peak_memory_stats(device)
    for index in range(args.probe_steps):
        dist.barrier();torch.cuda.synchronize(device);start=time.perf_counter()
        loss,_=step_loss(wrapped,model,clips[index%2],list(range(shape[0])),index,
                         shape[0],window,args,device)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device);elapsed.append(time.perf_counter()-start)
    free,total=torch.cuda.mem_get_info(device)
    report=dict(rank=rank,batch_per_gpu=shape[0],global_batch=shape[0]*world,
        gpu=torch.cuda.get_device_name(device),total_gib=total/2**30,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
        process_device_used_gib=(total-free)/2**30,
        steady_seconds_per_update=float(np.median(elapsed[3:])),steps=args.probe_steps)
    reports=[None]*world;dist.all_gather_object(reports,report)
    if rank==0:write_json(args.output_dir/'capacity.json',dict(status='completed',scope='synthetic_full_model_capacity',
                    model_size=args.model_size,objective=args.objective,compile_mode=args.compile_mode,gpus=reports))
    del clips,raw_buffer


def pipeline_benchmark(args,model,wrapped,optimizer,loader,device,rank,world,window):
    """Measure real loading, forward, DDP backward and optimizer after warmup."""
    count=args.benchmark_warmup+args.pipeline_benchmark_steps
    if count>len(loader):raise ValueError('Benchmark exceeds one dataset traversal')
    def synchronize():
        if device.type=='cuda':torch.cuda.synchronize(device)
    def io_bytes():
        path=Path('/proc/self/io')
        return {k:int(v) for k,v in (line.split(':') for line in path.read_text().splitlines())} if path.exists() else {}
    iterator=iter(loader);wrapped.train();rows=[];elapsed=None;before_io={}
    try:
        for step in range(count):
            if step==args.benchmark_warmup:
                synchronize();dist.barrier();synchronize()
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                before_io=io_bytes();started=time.perf_counter()
            begin=time.perf_counter();batch=next(iterator)
            clip=batch['pixel_values'].to(device,non_blocking=loader.pin_memory)
            synchronize();loaded=time.perf_counter()
            loss,_=step_loss(wrapped,model,clip,batch['sample_indices'],0,len(loader.dataset),window,args,device)
            synchronize();forward=time.perf_counter()
            loss.backward();synchronize();backward=time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True);synchronize();end=time.perf_counter()
            if step>=args.benchmark_warmup:
                rows.append(dict(data_and_transfer_seconds=loaded-begin,forward_seconds=forward-loaded,
                    backward_and_ddp_seconds=backward-forward,optimizer_seconds=end-backward,total_seconds=end-begin))
            if rank==0:write_json(args.output_dir/'progress.json',dict(status='benchmarking',
                current_step=step+1,total_steps=count,global_batch=args.batch_size_per_gpu*world,updated_unix=time.time()))
        elapsed=time.perf_counter()-started
        after_io=io_bytes()
    finally:
        if hasattr(iterator,'close'):iterator.close()
    report=dict(rank=rank,seconds_per_update=elapsed/len(rows),timings=rows,
                mean_seconds={name:float(np.mean([row[name] for row in rows])) for name in rows[0]},
                read_bytes=after_io.get('read_bytes',0)-before_io.get('read_bytes',0))
    if device.type=='cuda':
        free,total=torch.cuda.mem_get_info(device)
        report.update(total_gib=total/2**30,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,process_device_used_gib=(total-free)/2**30)
    reports=[None]*world;dist.all_gather_object(reports,report)
    if rank==0:
        seconds=max(row['seconds_per_update'] for row in reports)
        write_json(args.output_dir/'pipeline_benchmark.json',dict(status='completed',
            scope='synthetic_pipeline_smoke' if args.synthetic_smoke else 'real_data_pipeline_benchmark',
            batch_per_gpu=args.batch_size_per_gpu,global_batch=args.batch_size_per_gpu*world,
            measured_steps=len(rows),warmup_steps=args.benchmark_warmup,seconds_per_update=seconds,
            clips_per_second=args.batch_size_per_gpu*world/seconds,ranks=reports))
        write_json(args.output_dir/'progress.json',dict(status='completed',scope='pipeline_benchmark',updated_unix=time.time()))


def run(args,rank,world,device):
    torch.set_num_threads(args.cpu_threads);seed_everything(args.seed)
    if rank==0:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        if any(p.name!='.lock' for p in args.output_dir.iterdir()) and not args.resume:
            raise ValueError('Output exists; use explicit resume or an empty directory')
    dist.barrier()
    window=ForecastWindow()
    runtime=(CudaRuntime(tf32=True,cudnn_benchmark=True,fused_adamw=True,static_positions=True,
                        compile_mode=args.compile_mode) if device.type=='cuda' else CudaRuntime())
    configure_runtime(runtime,device)
    if args.precision=='bf16' and (device.type!='cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('bf16 requires a compatible GPU')
    overrides=dict(image_size=32,patch_size=8,hidden_size=48,num_hidden_layers=1,num_attention_heads=3,
        intermediate_size=96,decoder_hidden_size=24,decoder_num_hidden_layers=1,decoder_num_attention_heads=3,
        decoder_intermediate_size=48) if args.synthetic_smoke else {}
    config=history_config(window,model_size=args.model_size,**overrides)
    width=12 if args.synthetic_smoke else 192
    model=SeviriSSL(config,len(window.steps),args.objective,width)
    initial_encoder_sha256=encoder_digest(model.encoder) if rank==0 else None
    model=model.to(device)
    compiled=compile_model_regions(model,runtime)
    wrapped=DistributedDataParallel(model,device_ids=[device.index] if device.type=='cuda' else None,
        forward_sync_buffers=False,gradient_as_bucket_view=True,static_graph=True,bucket_cap_mb=64)
    optimizer=make_optimizer(model,runtime,lr=args.learning_rate,weight_decay=args.weight_decay)
    if args.capacity_probe:
        capacity_probe(args,model,wrapped,optimizer,device,rank,world,window);return
    if args.synthetic_smoke:
        raw=SyntheticClips(16,args.seed)
        pixels=torch.stack([raw[i]['pixel_values'] for i in range(16)]).double()
        mean,std=float(pixels.mean()),float(pixels.std(correction=0))
        stats=dict(global_mean_k=mean,global_std_k=std,synthetic=True)
        training,validation=SyntheticClips(16,args.seed,mean,std),SyntheticClips(8,args.seed+10000,mean,std)
        train_hash=val_hash=None
    else:
        if not all([args.dataset,args.frame_cache,args.zarr_root]):raise ValueError('Real data roots are required')
        train_frame=pd.read_csv(args.dataset/'train.csv');val_frame=pd.read_csv(args.dataset/'validation.csv')
        validate_manifests(train_frame,val_frame)
        stats=load_training_stats(args.dataset/'channel_stats.json',args.dataset/'train.csv',args.zarr_root)
        if stats['samples']!=len(train_frame):raise ValueError('Training statistics sample count differs')
        if rank==0:write_json(args.output_dir/'progress.json',dict(status='preparing',phase='opening_training_frames',updated_unix=time.time()))
        training=FrameFileDataset(args.frame_cache,args.dataset/'train.csv',zarr_root=args.zarr_root)
        validation=FrameFileDataset(args.frame_cache,args.dataset/'validation.csv',zarr_root=args.zarr_root)
        train_hash,val_hash=(digest(args.dataset/name) for name in ('train.csv','validation.csv'))
    sampler=GlobalBatchSampler(len(training),args.batch_size_per_gpu,rank,world,args.seed)
    val_sampler=GlobalBatchSampler(len(validation),args.validation_batch_size,rank,world,314159,False)
    loader=indexed_loader(training,sampler,args,device,stats,args.batch_size_per_gpu)
    val_loader=indexed_loader(validation,val_sampler,args,device,stats,args.validation_batch_size)
    sources={**source_hashes(),**{name:digest(ROOT/name) for name in (
        'src/emma_gpm/ssl_distributed.py','scripts/pretrain_seviri_ssl_ddp.py')}}
    identity=dict(window=asdict(window),model_config=config.to_dict(),objective=args.objective,head_width=width,
        frozen_encoder=False,stats=stats,train_sha256=train_hash,validation_sha256=val_hash,seed=args.seed,
        world_size=world,batch_size=args.batch_size_per_gpu*world,batch_size_per_gpu=args.batch_size_per_gpu,
        gradient_accumulation=1,learning_rate=args.learning_rate,weight_decay=args.weight_decay,
        mask_ratio=.9,mask_strategy='tube',max_motion_pixels=12,precision=args.precision,
        cuda_runtime=runtime.to_dict(),sources=sources,storage_mapping_sha256=storage_mapping_digest(),
        sampler='global epoch permutation; no padding or duplicate samples',mask_rng='global sample index and epoch')
    extended=None;extended_config=None
    if args.extended_dataset:
        ready=json.loads((args.extended_dataset/'dataset_ready.json').read_text())
        extended_config=json.loads((args.extended_dataset/'protocol.json').read_text())
        if ready['status']!='completed':raise ValueError('Extended holdout is incomplete')
        for name in ('validation.csv','protocol.json'):
            if digest(args.extended_dataset/name)!=expected_digest(ready,name):raise ValueError('Extended identity differs')
        cache=storage_root(extended_config['frame_cache'])
        if digest(cache/'metadata.json')!=expected_digest(ready,'metadata.json'):raise ValueError('Extended frame metadata changed')
        extended_frame=pd.read_csv(args.extended_dataset/'validation.csv');validate_manifests(train_frame,extended_frame)
        identity['extended_validation']=dict(manifest_sha256=digest(args.extended_dataset/'validation.csv'),
            protocol_sha256=digest(args.extended_dataset/'protocol.json'),every=args.extended_every)
    progress=dict(completed_epochs=0,optimizer_steps=0,best_score=None,epochs=[])
    if args.resume:progress=restore_distributed(args.resume,model,optimizer,identity,device,args.execution_revision)
    if args.epochs<progress['completed_epochs']:raise ValueError('Requested epochs precede the checkpoint')
    apply_optimizer_runtime(optimizer,runtime)
    if rank==0:
        write_json(args.output_dir/'run_config.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            identity=identity,initial_encoder_sha256=initial_encoder_sha256,compiled_regions=compiled,
            torch_version=torch.__version__,data_kind='synthetic engineering' if args.synthetic_smoke else 'SEVIRI SSL'))
        write_json(args.output_dir/'channel_stats.json',stats)
    if args.pipeline_benchmark_steps:
        pipeline_benchmark(args,model,wrapped,optimizer,loader,device,rank,world,window)
        for data in (training,validation):
            if hasattr(data,'close'):data.close()
        dist.barrier();return
    def evaluate_extended(epoch_number):
        nonlocal extended
        started=time.perf_counter()
        if extended is None:
            if rank==0:write_json(args.output_dir/'progress.json',dict(status='preparing',validation_set='extended',
                phase='opening_extended_frames',updated_unix=time.time(),**progress))
            extended=FrameFileDataset(extended_config['frame_cache'],args.extended_dataset/'validation.csv',
                                      zarr_root=extended_config['zarr_root'])
        dist.barrier();prepared=time.perf_counter()
        ext_sampler=GlobalBatchSampler(len(extended),args.validation_batch_size,rank,world,314159,False)
        ext_loader=indexed_loader(extended,ext_sampler,args,device,stats,args.validation_batch_size)
        def extended_heartbeat(samples):
            if rank==0:write_json(args.output_dir/'progress.json',dict(status='validating',validation_set='extended',
                current_epoch=epoch_number,validation_samples_on_rank=samples,updated_unix=time.time(),**progress))
        ext_report=evaluate_ssl(model,ext_loader,window,device,stats['global_mean_k'],stats['global_std_k'],
            precision=args.precision,distributed=True,sample_days=extended.manifest.day.tolist(),progress_callback=extended_heartbeat)
        if rank==0:
            write_json(args.output_dir/f'validation_extended/epoch_{epoch_number:03d}.json',ext_report)
            write_json(args.output_dir/f'validation_extended/timing_{epoch_number:03d}.json',
                dict(preparation_seconds=prepared-started,evaluation_seconds=time.perf_counter()-prepared))
    for epoch in range(progress['completed_epochs'],args.epochs):
        sampler.epoch=epoch;wrapped.train();dist.barrier()
        if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter();totals=torch.zeros(4,dtype=torch.float64,device=device)
        wait_seconds=0.;previous=time.perf_counter();rows=[]
        for step,batch in enumerate(loader,1):
            wait_seconds+=time.perf_counter()-previous
            clip=batch['pixel_values'].to(device,non_blocking=loader.pin_memory)
            loss,terms=step_loss(wrapped,model,clip,batch['sample_indices'],epoch,len(training),window,args,device)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            values={name:float(value.detach()) for name,value in terms.items()}
            n=len(clip);totals+=torch.tensor([float(loss.detach())*n,values.get('mvm',0)*n,
                                            values.get('future',0)*n,n],device=device)
            progress['optimizer_steps']+=1
            if step%args.log_every==0 or step==len(loader):
                global_totals=totals.clone();dist.all_reduce(global_totals)
                if rank==0:
                    rows.append(dict(epoch=epoch+1,step=step,optimizer_steps=progress['optimizer_steps'],
                        samples=int(global_totals[3]),loss=float(global_totals[0]/global_totals[3])))
                    write_json(args.output_dir/'progress.json',dict(status='training',current_epoch=epoch+1,
                        current_step=step,steps_per_epoch=len(loader),global_batch=args.batch_size_per_gpu*world,
                        world_size=world,updated_unix=time.time(),**progress))
            previous=time.perf_counter()
        if device.type=='cuda':torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-started
        dist.all_reduce(totals)
        timings=torch.tensor([elapsed,wait_seconds],dtype=torch.float64,device=device);dist.all_reduce(timings,op=dist.ReduceOp.MAX)
        if int(totals[3])!=len(training):raise RuntimeError('Training epoch duplicated or omitted samples')
        validation_started=time.perf_counter()
        def heartbeat(samples):
            if rank==0:write_json(args.output_dir/'progress.json',dict(status='validating',current_epoch=epoch+1,
                validation_samples_on_rank=samples,updated_unix=time.time(),**progress))
        report=evaluate_ssl(model,val_loader,window,device,stats['global_mean_k'],stats['global_std_k'],
            precision=args.precision,distributed=True,progress_callback=heartbeat)
        score=report['mvm_mse'] if args.objective=='mvm' else report['neural']['mse']
        improved=progress['best_score'] is None or score<progress['best_score']
        if improved:progress['best_score']=score
        memory=dict(rank=rank,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30) if device.type=='cuda' else dict(rank=rank)
        memories=[None]*world;dist.all_gather_object(memories,memory)
        row=dict(epoch=epoch+1,optimizer_steps=progress['optimizer_steps'],train_loss=float(totals[0]/totals[3]),
            train_seconds=float(timings[0]),data_and_transfer_seconds_max_rank=float(timings[1]),
            seconds_per_optimizer_update=float(timings[0])/len(loader),clips_per_second=len(training)/float(timings[0]),
            train_batches=len(loader),batch_size=args.batch_size_per_gpu*world,batch_size_per_gpu=args.batch_size_per_gpu,
            world_size=world,validation_score=score,validation_rmse_k=report.get('neural',{}).get('rmse_k'),
            validation_mvm=report['mvm_mse'],validation_seconds=time.perf_counter()-validation_started,gpu_memory=memories)
        progress['completed_epochs']=epoch+1;progress['epochs'].append(row)
        save_distributed(args.output_dir/'checkpoint_latest.pt',model,optimizer,progress,identity,device)
        if improved:save_distributed(args.output_dir/'checkpoint_best.pt',model,optimizer,progress,identity,device)
        if rank==0:
            write_json(args.output_dir/'validation_latest.json',report)
            write_json(args.output_dir/f'validation_epochs/epoch_{epoch+1:03d}.json',report)
            if improved:write_json(args.output_dir/'validation_best.json',report)
            pd.DataFrame(progress['epochs']).to_csv(args.output_dir/'loss_epochs.csv',index=False)
            pd.DataFrame(rows).to_csv(args.output_dir/f'loss_steps_epoch_{epoch+1:03d}.csv',index=False)
            print(json.dumps(row),flush=True)
        stop=torch.tensor(int(bool(args.stop_after_epoch_file and args.stop_after_epoch_file.exists())),device=device)
        dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if stop.item() and epoch+1<args.epochs:
            if rank==0:write_json(args.output_dir/'progress.json',dict(status='paused',updated_unix=time.time(),**progress))
            break
        if extended_config and ((epoch+1)%args.extended_every==0 or epoch+1==args.epochs):
            evaluate_extended(epoch+1)
        if rank==0:write_json(args.output_dir/'progress.json',dict(status='running',updated_unix=time.time(),**progress))
    else:
        missing=torch.tensor(int(bool(rank==0 and extended_config and not
            (args.output_dir/f'validation_extended/epoch_{args.epochs:03d}.json').exists())),device=device)
        dist.broadcast(missing,src=0)
        if missing.item():evaluate_extended(args.epochs)
        if rank==0:
            model.encoder.save_pretrained(args.output_dir/'encoder')
            write_json(args.output_dir/'progress.json',dict(status='completed',updated_unix=time.time(),**progress))
    for data in (training,validation,extended):
        if hasattr(data,'close'):data.close()
    dist.barrier()


def main():
    args=args_from_cli();rank,world,device=initialize(args.device)
    try:
        from contextlib import nullcontext
        with exclusive_file_lock(args.output_dir/'.lock') if rank==0 else nullcontext():run(args,rank,world,device)
    except BaseException as exc:
        if rank==0:write_json(args.output_dir/'progress.json',dict(status='failed',error=str(exc),updated_unix=time.time()))
        traceback.print_exc();sys.stderr.flush();os._exit(1)
    dist.destroy_process_group()


if __name__=='__main__':main()
