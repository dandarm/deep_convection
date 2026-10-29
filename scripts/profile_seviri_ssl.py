#!/usr/bin/env python3
"""Isolated throughput and operation timings on frozen SEVIRI training inputs.

Runs in a separate directory and never updates scientific checkpoints.
CUDA events measure stream intervals; optional CUPTI traces measure kernel
activity intervals. Neither measurement is an SM occupancy counter.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
import gc
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import threading

import numpy as np
import pandas as pd
import psutil
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))

from emma_gpm.advection import advection_forecast
from emma_gpm.forecasting import ForecastWindow
from emma_gpm.seviri_clips import SeviriZarrClipReader,SeviriVideoMAEDataset,normalize_seviri_clip
from emma_gpm.seviri_frame_cache import SeviriFrameCacheReader
from emma_gpm.ssl_model import SeviriSSL,history_config,ssl_loss
from emma_gpm.ssl_masking import make_ssl_mask
from emma_gpm.videomae_v2 import running_cell_mask
from emma_gpm.training_checkpoint import save_checkpoint
from emma_gpm.ssl_evaluation import evaluate_ssl
from scripts.pretrain_videomae import initialize_loader_worker,seed_everything
from scripts.pretrain_seviri_ssl import write_json


def summary(values):
    return dict(mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),
                p90_ms=float(np.quantile(values,.9)),samples=len(values))


def union_duration(intervals):
    end=None
    total=0.
    for start,stop in sorted(intervals):
        if end is None or start>end:
            total+=stop-start
        elif stop>end:
            total+=stop-end
        end=max(stop,end) if end is not None else stop
    return total


def trace_activity(path):
    trace=json.loads(Path(path).read_text())['traceEvents']
    window=next(e for e in trace if e.get('name')=='MEASURED_TRAINING_WINDOW' and e.get('ph')=='X')
    low,high=window['ts'],window['ts']+window['dur']
    result=dict(window_ms=window['dur']/1000)
    for label,categories in (('kernel',{'kernel'}),('device_activity',{'kernel','gpu_memcpy','gpu_memset'})):
        intervals=[]
        for event in trace:
            if event.get('cat') in categories and event.get('ph')=='X':
                a,b=max(low,event['ts']),min(high,event['ts']+event['dur'])
                if b>a:intervals.append((a,b))
        occupied=union_duration(intervals)
        result[label+'_events']=len(intervals)
        result[label+'_active_ms']=occupied/1000
        result[label+'_active_fraction']=occupied/window['dur'] if intervals else None
    return result


def profile_worker(worker_id,concurrency=8,threads=4):
    initialize_loader_worker(worker_id)
    import zarr
    zarr.config.set({'async.concurrency':concurrency,'threading.max_workers':threads})


def dataset(args,split='train',limit=None):
    stats=json.loads((args.dataset/'channel_stats.json').read_text())
    frame=pd.read_csv(args.dataset/f'{split}.csv')
    if limit:
        selected=np.random.default_rng(20261025).permutation(len(frame))[:limit]
        frame=frame.iloc[selected].reset_index(drop=True)
    reader=(SeviriFrameCacheReader(stats['zarr_root'],args.frame_cache,require_finite=not args.finite_on_gpu) if args.frame_cache
            else SeviriZarrClipReader(stats['zarr_root']))
    data=(RawFrameDataset(frame,reader) if args.input_mode=='raw' else
          SeviriVideoMAEDataset(frame,reader,global_mean=stats['global_mean_k'],global_std=stats['global_std_k']))
    return data,stats


class RawFrameDataset:
    def __init__(self,frame,reader):
        self.manifest=frame
        self.reader=reader

    def __len__(self):return len(self.manifest)

    def __getitem__(self,index):
        row=self.manifest.iloc[index]
        values=self.reader.load_native(row.start_time,origin_y=row.origin_y,origin_x=row.origin_x)
        return {'pixel_values':torch.from_numpy(values)}


class ThreadedRawLoader:
    """Read native frames directly into pinned batches with bounded prefetch."""
    pin_memory=True

    def __init__(self,data,args):
        self.data=data;self.batch_size=args.io_batch;self.workers=args.workers;self.prefetch=args.thread_prefetch

    def __iter__(self):
        local=threading.local();readers=[];lock=threading.Lock()
        def prepare(first):
            if not hasattr(local,'reader'):
                local.reader=SeviriFrameCacheReader(self.data.reader.root,self.data.reader.cache_root,
                    require_finite=self.data.reader.require_finite)
                with lock:readers.append(local.reader)
            stop=min(first+self.batch_size,len(self.data))
            pixels=torch.empty((stop-first,16,7,224,224),dtype=torch.float16,pin_memory=True)
            destination=pixels.numpy()
            for slot,index in enumerate(range(first,stop)):
                row=self.data.manifest.iloc[index]
                local.reader.load_native_into(row.start_time,origin_y=row.origin_y,origin_x=row.origin_x,out=destination[slot])
            return {'pixel_values':pixels}
        pool=ThreadPoolExecutor(max_workers=self.workers)
        starts=iter(range(0,len(self.data),self.batch_size));pending=deque()
        try:
            for first in islice(starts,self.prefetch):pending.append(pool.submit(prepare,first))
            while pending:
                result=pending.popleft().result()
                first=next(starts,None)
                if first is not None:pending.append(pool.submit(prepare,first))
                yield result
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
            for reader in readers:reader.close()


def loader(data,args):
    if args.loader_kind=='thread':return ThreadedRawLoader(data,args)
    return torch.utils.data.DataLoader(data,batch_size=args.io_batch,shuffle=False,num_workers=args.workers,
        pin_memory=True,**(dict(multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=2,
            worker_init_fn=partial(profile_worker,concurrency=args.zarr_concurrency,threads=args.zarr_threads))
            if args.workers else {}))


class CachedInputs:
    """A fixed real-input pool for isolating IPC/copy costs, benchmark only."""
    def __init__(self,data,count=32):
        self.length=len(data)
        self.pixels=torch.stack([data[i]['pixel_values'] for i in range(min(count,len(data)))]).share_memory_()
        data.reader.close()

    def __len__(self):return self.length

    def __getitem__(self,index):return {'pixel_values':self.pixels[index%len(self.pixels)]}


def profiled_batches(data_loader,batch_size):
    iterator=iter(data_loader)
    while True:
        pieces=[]
        count=0
        waiting=0.
        try:
            while count<batch_size:
                started=time.perf_counter()
                part=next(iterator)['pixel_values']
                waiting+=time.perf_counter()-started
                pieces.append(part)
                count+=len(part)
        except StopIteration:
            if not pieces:return
        if count>batch_size:raise ValueError('I/O batch must divide device batch')
        started=time.perf_counter()
        if len(pieces)==1:
            pixels=pieces[0]
        else:
            pixels=torch.empty((count,*pieces[0].shape[1:]),dtype=pieces[0].dtype,pin_memory=True)
            torch.cat(pieces,out=pixels)
        yield pixels,dict(loader_wait_ms=waiting*1000,assembly_ms=(time.perf_counter()-started)*1000)


def cpu_snapshot():
    current=psutil.Process()
    processes=[current]+current.children(recursive=True)
    times=read_bytes=write_bytes=0.
    for process in processes:
        try:
            cpu=process.cpu_times();io=process.io_counters()
            times+=cpu.user+cpu.system
            read_bytes+=io.read_bytes;write_bytes+=io.write_bytes
        except psutil.NoSuchProcess:pass
    return dict(cpu_seconds=times,physical_read_bytes=read_bytes,physical_write_bytes=write_bytes)


class Timing:
    def __init__(self):
        self.cpu={};self.events=[]

    @contextmanager
    def section(self,name,cuda=True):
        start=time.perf_counter()
        if cuda:
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
        with torch.profiler.record_function(name):
            yield
        if cuda:
            b.record();self.events.append((name,a,b))
        self.cpu.setdefault(name,[]).append((time.perf_counter()-start)*1000)

    def report(self):
        torch.cuda.synchronize()
        gpu={}
        for name,a,b in self.events:gpu.setdefault(name,[]).append(a.elapsed_time(b))
        return dict(host_intervals={name:summary(values) for name,values in self.cpu.items()},
                    cuda_stream_intervals={name:summary(values) for name,values in gpu.items()})


@contextmanager
def external_cuda_capture(args):
    """Delimit an optional Nsight CUDA/NVTX capture without CPU-op tracing."""
    if args.nsys_range:
        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push('MEASURED_TRAINING_WINDOW')
    try:
        yield
    finally:
        if args.nsys_range:
            torch.cuda.nvtx.range_pop()
            torch.cuda.profiler.stop()


def model_and_optimizer(objective, model_size="small"):
    seed_everything(20261024)
    config=history_config(ForecastWindow(), model_size=model_size)
    model=SeviriSSL(config,objective='future' if objective=='probe' else objective,
                    head_width=192,freeze_encoder=objective=='probe').cuda().train()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.00015,weight_decay=.05)
    return model,optimizer


def training_step(model,optimizer,pixels,index,accumulation,timing,mask_rng,logging_path=None,normalization=None):
    if normalization is None:
        with timing.section('host_to_device_and_window'):
            history,target=ForecastWindow().split(pixels.cuda(non_blocking=True) if pixels.device.type=='cpu' else pixels)
    else:
        with timing.section('raw_host_to_device'):
            pixels=pixels.cuda(non_blocking=True) if pixels.device.type=='cpu' else pixels
        with timing.section('global_normalization'):
            pixels=(pixels.float()-normalization[0])/normalization[1]
        with timing.section('window_validation'):
            history,target=ForecastWindow().split(pixels)
    mask=decode=adv=None
    with timing.section('mask_and_advection'):
        if model.mae is not None:
            mask=make_ssl_mask(model.config,len(history),.9,'tube',generator=mask_rng,device='cuda')
            decode=running_cell_mask(model.config,len(history),.5,device='cpu',generator=mask_rng).cuda()
        if model.objective=='residual':adv,_,_=advection_forecast(history,ForecastWindow().steps,12)
    with timing.section('forward_and_loss'):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(history,mask=mask,decode_mask=decode,advection=adv)
            loss,terms=ssl_loss(output,target)
    with timing.section('finite_loss_check',cuda=False):
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite benchmark loss')
    with timing.section('backward'):
        (loss/accumulation).backward()
    if (index+1)%accumulation==0:
        with timing.section('gradient_norm_check'):
            torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
        with timing.section('optimizer_step'):
            optimizer.step();optimizer.zero_grad(set_to_none=True)
    with timing.section('scalar_metrics',cuda=False):
        values={'loss':float(loss.detach()),**{name:float(value.detach()) for name,value in terms.items()}}
    if logging_path:
        with timing.section('step_log_csv',cuda=False):
            pd.DataFrame([dict(step=index,**values)]).to_csv(logging_path,mode='a',header=not logging_path.exists(),index=False)
    return values


def io_profile(args):
    data,stats=dataset(args,limit=(args.steps+args.warmup+2*args.workers+8)*args.batch_size)
    timings=[]
    single=[]
    profile_worker(0,args.zarr_concurrency,args.zarr_threads)
    for row in data.manifest.iloc[:32].itertuples(index=False):
        start=time.perf_counter()
        raw=data.reader.load(row.start_time,origin_y=int(row.origin_y),origin_x=int(row.origin_x))
        read_end=time.perf_counter()
        normalized=normalize_seviri_clip(raw,stats['global_mean_k'],stats['global_std_k'])
        normalize_end=time.perf_counter()
        torch.from_numpy(np.ascontiguousarray(normalized))
        single.append(dict(read_validate_orient_ms=(read_end-start)*1000,
            normalize_ms=(normalize_end-read_end)*1000,tensor_view_ms=(time.perf_counter()-normalize_end)*1000))
    data.reader.close()
    torch.set_num_threads(args.cpu_threads)
    data_loader=loader(data,args)
    batches=profiled_batches(data_loader,args.batch_size)
    start=time.perf_counter()
    for _ in range(args.warmup):next(batches)
    warmup=time.perf_counter()-start
    before=cpu_snapshot();start=time.perf_counter()
    samples=0
    for _ in range(args.steps):
        pixels,times=next(batches);timings.append(times);samples+=len(pixels)
    elapsed=time.perf_counter()-start;after=cpu_snapshot()
    return dict(mode='io',samples=samples,wall_seconds=elapsed,clips_per_second=samples/elapsed,
        startup_and_warmup_seconds=warmup,io_batch=args.io_batch,workers=args.workers,
        zarr_concurrency=args.zarr_concurrency,zarr_threads=args.zarr_threads,
        batch_intervals={name:summary([r[name] for r in timings]) for name in timings[0]},
        single_reader_first_clip=single[0],
        single_reader_warm={name:summary([r[name] for r in single[1:]]) for name in single[0]},
        resources={name:after[name]-before[name] for name in before})


def pipeline_profile(args):
    # Keep unread inputs beyond the timed interval: otherwise the workers
    # finish early and short large-I/O-batch trials understate CPU contention.
    data,stats=dataset(args,limit=(args.steps+args.warmup+2*args.workers+8)*args.batch_size)
    if len(data)<(args.steps+args.warmup)*args.batch_size:
        raise ValueError('Manifest is too short for the requested measured batches and warmup')
    remaining_after_measurement=len(data)-(args.steps+args.warmup)*args.batch_size
    if args.cache_inputs:
        data=CachedInputs(data)
    data_loader=loader(data,args)
    batches=profiled_batches(data_loader,args.batch_size)
    model,optimizer=model_and_optimizer(args.objective, args.model_size)
    rng=torch.Generator().manual_seed(20261025)
    accumulation=args.accumulation
    normalization=tuple(float(np.float32(stats[k])) for k in ('global_mean_k','global_std_k')) if args.input_mode=='raw' else None
    optimizer.zero_grad(set_to_none=True)
    started=time.perf_counter()
    for i in range(args.warmup):
        pixels,_=next(batches)
        training_step(model,optimizer,pixels,i,accumulation,Timing(),rng,normalization=normalization)
    torch.cuda.synchronize()
    warmup=time.perf_counter()-started
    optimizer.zero_grad(set_to_none=True)
    timing=Timing();batch_times=[];waits=[];samples=0
    torch.cuda.reset_peak_memory_stats()
    measurement={}
    def measured():
        nonlocal samples
        before=cpu_snapshot();started=time.perf_counter()
        for i in range(args.steps):
            beginning=time.perf_counter()
            with torch.profiler.record_function('next_training_batch'):
                pixels,io=next(batches)
            waits.append(io);samples+=len(pixels)
            training_step(model,optimizer,pixels,i,accumulation,timing,rng,args.output_dir/'step_log.csv',normalization=normalization)
            batch_times.append((time.perf_counter()-beginning)*1000)
        torch.cuda.synchronize()
        measurement.update(elapsed=time.perf_counter()-started,before=before,after=cpu_snapshot())
    if args.trace:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            with torch.profiler.record_function('MEASURED_TRAINING_WINDOW'):measured()
        profile.export_chrome_trace(str(args.output_dir/'trace.json'))
        activities=trace_activity(args.output_dir/'trace.json')
    else:
        with external_cuda_capture(args):measured()
        activities=None
    elapsed=measurement['elapsed'];before=measurement['before'];after=measurement['after']
    result=dict(mode='pipeline',objective=args.objective,batch_size=args.batch_size,accumulation=accumulation,
        effective_batch=args.batch_size*accumulation,io_batch=args.io_batch,workers=args.workers,
        cpu_threads=args.cpu_threads,cached_input_pool=32 if args.cache_inputs else None,
        input_mode=args.input_mode,loader_kind=args.loader_kind,frame_cache=str(args.frame_cache) if args.frame_cache else None,
        zarr_concurrency=args.zarr_concurrency,zarr_threads=args.zarr_threads,
        unread_samples_after_measurement=remaining_after_measurement,
        samples=samples,wall_seconds=elapsed,clips_per_second=samples/elapsed,
        startup_and_warmup_seconds=warmup,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,batch_wall=summary(batch_times),
        batch_io={name:summary([r[name] for r in waits]) for name in waits[0]},
        resources={name:after[name]-before[name] for name in before},timings=timing.report(),trace=activities)
    result['batch_series']=[dict(batch=index+1,wall_ms=wall,**io)
        for index,(wall,io) in enumerate(zip(batch_times,waits))]
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    start=time.perf_counter()
    save_checkpoint(args.output_dir/'benchmark_checkpoint.pt',model,optimizer,scheduler,
        {'mask':rng},dict(benchmark=True),dict(benchmark=True,objective=args.objective))
    result['checkpoint_seconds']=time.perf_counter()-start
    result['checkpoint_bytes']=(args.output_dir/'benchmark_checkpoint.pt').stat().st_size
    return result


def gpu_profile(args):
    data,stats=dataset(args,limit=min(args.batch_size,32))
    profile_worker(0,args.zarr_concurrency,args.zarr_threads)
    values=torch.stack([data[i]['pixel_values'] for i in range(len(data))])
    values=values.repeat((args.batch_size+len(values)-1)//len(values),1,1,1,1)[:args.batch_size].cuda()
    data.reader.close()
    torch.set_num_threads(args.cpu_threads)
    model,optimizer=model_and_optimizer(args.objective, args.model_size)
    rng=torch.Generator().manual_seed(20261025)
    optimizer.zero_grad(set_to_none=True)
    normalization=tuple(float(np.float32(stats[k])) for k in ('global_mean_k','global_std_k')) if args.input_mode=='raw' else None
    for i in range(args.warmup):training_step(model,optimizer,values,i,args.accumulation,Timing(),rng,normalization=normalization)
    torch.cuda.synchronize();optimizer.zero_grad(set_to_none=True)
    timing=Timing();durations=[]
    torch.cuda.reset_peak_memory_stats()
    measured_time=[]
    def measured():
        started=time.perf_counter()
        for i in range(args.steps):
            start=time.perf_counter()
            training_step(model,optimizer,values,i,args.accumulation,timing,rng,normalization=normalization)
            torch.cuda.synchronize()
            durations.append((time.perf_counter()-start)*1000)
        measured_time.append(time.perf_counter()-started)
    activities=None
    if args.trace:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            with torch.profiler.record_function('MEASURED_TRAINING_WINDOW'):measured()
        profile.export_chrome_trace(str(args.output_dir/'trace.json'))
        activities=trace_activity(args.output_dir/'trace.json')
    else:
        with external_cuda_capture(args):measured()
    elapsed=measured_time[0]
    return dict(mode='gpu_resident',objective=args.objective,batch_size=args.batch_size,
        accumulation=args.accumulation,effective_batch=args.batch_size*args.accumulation,
        samples=args.steps*args.batch_size,wall_seconds=elapsed,clips_per_second=args.steps*args.batch_size/elapsed,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,batch_wall=summary(durations),timings=timing.report(),trace=activities)


def validation_profile(args):
    data,stats=dataset(args,split='validation',limit=args.steps*args.batch_size)
    args.io_batch=args.batch_size
    data_loader=loader(data,args)
    model,_=model_and_optimizer(args.objective, args.model_size)
    # Warm the complete evaluation path, including MVM, motion estimation,
    # metric accumulation and plotting, before timing the steady evaluation.
    class WarmupLoader:
        pin_memory=data_loader.pin_memory
        def __iter__(self):return islice(data_loader,1)
    evaluate_ssl(model,WarmupLoader(),ForecastWindow(),torch.device('cuda'),
        stats['global_mean_k'],stats['global_std_k'],precision='bf16',
        variability_edges=[.7451066689696711,1.4390741811047336],
        example_path=args.output_dir/'validation_warmup.png')
    torch.cuda.synchronize()
    timing=Timing()
    import emma_gpm.ssl_evaluation as evaluation
    original_adv=evaluation.advection_forecast
    original_plot=evaluation.plot_forecasts
    original_update=evaluation.ForecastMetrics.update
    original_variability=evaluation.variability_statistics
    original_forward=model.forward
    def wrapped(name,function):
        def invoke(*a,**kw):
            with timing.section(name):return function(*a,**kw)
        return invoke
    evaluation.advection_forecast=wrapped('validation_advection',original_adv)
    evaluation.plot_forecasts=wrapped('validation_plot',original_plot)
    evaluation.ForecastMetrics.update=wrapped('validation_metric_accumulation',original_update)
    evaluation.variability_statistics=wrapped('validation_variability',original_variability)
    model.forward=wrapped('validation_model_forward',original_forward)
    start=time.perf_counter()
    try:
        report=evaluate_ssl(model,data_loader,ForecastWindow(),torch.device('cuda'),
            stats['global_mean_k'],stats['global_std_k'],precision='bf16',variability_edges=[.7451066689696711,1.4390741811047336],
            example_path=args.output_dir/'validation_example.png')
        torch.cuda.synchronize()
    finally:
        evaluation.advection_forecast=original_adv;evaluation.plot_forecasts=original_plot
        evaluation.ForecastMetrics.update=original_update;evaluation.variability_statistics=original_variability
        model.forward=original_forward
    return dict(mode='validation',objective=args.objective,samples=len(data),
        wall_seconds=time.perf_counter()-start,timings=timing.report())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--mode',choices=('io','gpu','pipeline','validation'),required=True)
    parser.add_argument('--model-size',choices=('small','base'),default='small')
    parser.add_argument('--objective',choices=('mvm','future','joint','residual','probe'),default='future')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--accumulation',type=int,default=2)
    parser.add_argument('--io-batch',type=int,default=2)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--zarr-concurrency',type=int,default=8)
    parser.add_argument('--zarr-threads',type=int,default=4)
    parser.add_argument('--warmup',type=int,default=4)
    parser.add_argument('--steps',type=int,default=24)
    parser.add_argument('--trace',action='store_true')
    parser.add_argument('--cache-inputs',action='store_true',help='Isolate IPC/copy costs using 32 real normalized inputs in shared RAM')
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--frame-cache',type=Path,help='Lossless native-frame NPY cache')
    parser.add_argument('--input-mode',choices=('normalized','raw'),default='normalized')
    parser.add_argument('--loader-kind',choices=('process','thread'),default='process')
    parser.add_argument('--nsys-range',action='store_true',help='Delimit an external Nsight cudaProfilerApi capture')
    parser.add_argument('--finite-on-gpu',action='store_true',help='Validate complete raw clips on CUDA in ForecastWindow.split before model use')
    parser.add_argument('--thread-prefetch',type=int,default=2,help='Maximum pending native-frame batches in the thread loader')
    args=parser.parse_args()
    if args.nsys_range and args.trace:parser.error('Choose external Nsight or PyTorch trace collection')
    if args.finite_on_gpu and args.input_mode!='raw':parser.error('GPU finite validation requires raw inputs')
    if args.thread_prefetch<1:parser.error('Thread prefetch must be positive')
    if args.input_mode=='raw' and (not args.frame_cache or args.cache_inputs or args.mode=='validation'):
        parser.error('Raw input requires a frame cache; cached-pool and validation modes use normalized inputs')
    if args.loader_kind=='thread' and (args.input_mode!='raw' or args.workers<1):
        parser.error('Thread loader requires raw cached frames and at least one worker')
    if args.batch_size%args.io_batch or args.warmup%args.accumulation or args.steps%args.accumulation:
        parser.error('I/O size must divide device batch; warmup/measurement must contain complete update groups')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'result.json').exists():raise ValueError('Use a separate output directory')
    torch.set_num_threads(args.cpu_threads)
    info=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        torch_version=torch.__version__,cpu_count=os.cpu_count(),benchmark_only=True,
        profiling_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.output_dir/'configuration.json',info)
    try:
        result={'io':io_profile,'pipeline':pipeline_profile,'gpu':gpu_profile,'validation':validation_profile}[args.mode](args)
        result['status']='completed'
    except torch.cuda.OutOfMemoryError as exc:
        result=dict(status='out_of_memory',error=str(exc),objective=args.objective,batch_size=args.batch_size)
    write_json(args.output_dir/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
