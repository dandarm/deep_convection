#!/usr/bin/env python3
"""Measure four-GPU capacity, then compare three batch sizes over five epochs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def launch(root,name,options,allow_oom=False):
    output=root/name;log=root/(name.replace('/','_')+'.log')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
             str(ROOT/'scripts/pretrain_seviri_ssl_ddp.py'),'--output-dir',str(output),*options]
    started=time.time()
    write_json(root/'controller.json',dict(status='running',current=name,command=command,updated_unix=started))
    with log.open('ab') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        text=log.read_text(errors='replace')
        if allow_oom and ('CUDA out of memory' in text or 'torch.OutOfMemoryError' in text):
            return None
        raise RuntimeError(f'{name} failed with exit {result.returncode}; see {log}')
    return output


def capacity(root):
    common=['--synthetic-smoke','--batch-size-per-gpu','2','--epochs','1','--compile-mode','default']
    for objective in ('future','mvm','joint'):
        path=root/'smoke'/objective
        if not (path/'encoder/config.json').exists():
            launch(root,'smoke/'+objective,[*common,'--objective',objective])
    trials=[]
    def probe(batch,compile_mode):
        name=f'capacity/{compile_mode}_b{batch}'
        path=root/name
        if (path/'capacity.json').exists():
            result=json.loads((path/'capacity.json').read_text())
        else:
            log=root/(name.replace('/','_')+'.log')
            if path.exists() and log.exists() and ('CUDA out of memory' in log.read_text(errors='replace')
                                                    or 'torch.OutOfMemoryError' in log.read_text(errors='replace')):
                trials.append(dict(batch_per_gpu=batch,compile_mode=compile_mode,status='out_of_memory'))
                return False
            completed=launch(root,name,['--capacity-probe','--model-size','base','--objective','future',
                 '--batch-size-per-gpu',str(batch),'--compile-mode',compile_mode],True)
            result=json.loads((completed/'capacity.json').read_text()) if completed else None
        if result is None:
            trials.append(dict(batch_per_gpu=batch,compile_mode=compile_mode,status='out_of_memory'))
            write_json(root/'capacity_trials.json',trials);return False
        fraction=max(max(g['peak_reserved_gib'],g['process_device_used_gib'])/g['total_gib'] for g in result['gpus'])
        accepted=fraction<=.92
        trials.append(dict(batch_per_gpu=batch,compile_mode=compile_mode,status='accepted' if accepted else 'memory_margin',
            max_memory_fraction=fraction,slowest_gpu_seconds_per_update=max(g['steady_seconds_per_update'] for g in result['gpus'])))
        write_json(root/'capacity_trials.json',trials)
        return accepted
    lower,upper=0,8
    while probe(upper,'off'):
        lower=upper;upper*=2
        if upper>2048:raise RuntimeError('Capacity search exceeded its bounded range')
    while upper-lower>8:
        mid=((lower+upper)//16)*8
        if probe(mid,'off'):lower=mid
        else:upper=mid
    if lower<8:raise RuntimeError('Even the reference per-GPU batch does not fit')
    while not probe(lower,'default'):
        lower-=8
        if lower<8:raise RuntimeError('Compiled reference batch failed the memory margin')
    # Compilation may reduce activation memory; continue searching in the actual training mode.
    upper=((lower*5//4+7)//8)*8
    upper=max(upper,lower+8)
    while probe(upper,'default'):
        lower=upper;upper=max(lower+8,((lower*5//4+7)//8)*8)
        if upper>2048:raise RuntimeError('Compiled capacity search exceeded its bounded range')
    while upper-lower>8:
        mid=((lower+upper)//16)*8
        if probe(mid,'default'):lower=mid
        else:upper=mid
    write_json(root/'batch_choice.json',dict(status='completed',model_size='base',objective='future',
        world_size=4,large_batch_per_gpu=lower,large_global_batch=lower*4,memory_limit_fraction=.92,
        granularity=8,search='full-model eager bracket and compiled capacity refinement',trials=trials))
    write_json(root/'controller.json',dict(status='capacity_completed',updated_unix=time.time()))


def pilot_batches(baseline_global_batch,large_batch_per_gpu):
    if baseline_global_batch<4 or baseline_global_batch%4:
        raise ValueError('A positive global batch divisible by four is required')
    if large_batch_per_gpu<max(32,baseline_global_batch//4):
        raise ValueError('Measured capacity does not cover the requested 32 clips per GPU')
    plan=[]
    for label,batch in [('reference',baseline_global_batch//4),('per_gpu_reference',32),
                        ('large',large_batch_per_gpu)]:
        existing=next((row for row in plan if row['batch_per_gpu']==batch),None)
        if existing is not None:
            existing['comparisons'].append(label)
        else:
            plan.append(dict(label=label,batch_per_gpu=batch,global_batch=batch*4,comparisons=[label]))
    return plan


def pilots(root,baseline_global_batch):
    settings=json.loads((root/'data_locations.json').read_text())
    choice=json.loads((root/'batch_choice.json').read_text())
    if choice['status']!='completed':raise ValueError('Completed capacity search required')
    plan=pilot_batches(baseline_global_batch,choice['large_batch_per_gpu'])
    for name in ('training','extended'):
        directory=Path(settings[name]);verification=json.loads((directory/'verification.json').read_text())
        if verification['status']!='verified' or verification['transfer_plan_sha256']!=digest(directory/'transfer_plan.json'):
            raise ValueError(f'{name} transfer has not passed its checksum verification')
    os.environ['SEVIRI_STORAGE_ROOTS']=str(root/'storage_roots.json')
    common=['--dataset',str(Path(settings['training'])/'dataset'),
        '--zarr-root',settings['canonical_zarr_root'],'--frame-cache',settings['canonical_frame_root'],
        '--extended-dataset',str(Path(settings['extended'])/'dataset'),
        '--model-size','base','--objective','future','--epochs','5','--extended-every','10','--compile-mode','default']
    write_json(root/'pilots_plan.json',dict(world_size=4,epochs=5,model_size='base',objective='future',trials=plan))
    for trial in plan:
        label,batch=trial['label'],trial['batch_per_gpu']
        path=root/'pilots'/label
        progress=json.loads((path/'progress.json').read_text()) if (path/'progress.json').exists() else {}
        if progress.get('status')=='completed':continue
        options=[*common,'--batch-size-per-gpu',str(batch),'--stop-after-epoch-file',str(root/'stop_after_epoch')]
        if (path/'checkpoint_latest.pt').exists():options+=['--resume',str(path/'checkpoint_latest.pt')]
        launch(root,'pilots/'+label,options)
        progress=json.loads((path/'progress.json').read_text())
        if progress['status']=='paused':
            write_json(root/'controller.json',dict(status='paused',current=label,updated_unix=time.time()));return 75
    write_json(root/'controller.json',dict(status='completed',updated_unix=time.time()))
    return 0


def ram_pilots(root):
    from scripts.stage_seviri_ssl_ram import stage
    write_json(root/'controller.json',dict(status='staging_ram',updated_unix=time.time()))
    stage(root)
    os.environ['SEVIRI_STORAGE_ROOTS']=str(root/'ram_storage_roots.json')
    settings=json.loads((root/'data_locations.json').read_text())
    choice=json.loads((root/'batch_choice.json').read_text())
    common=['--dataset',str(Path(settings['training'])/'dataset'),
        '--zarr-root',settings['canonical_zarr_root'],'--frame-cache',settings['canonical_frame_root'],
        '--model-size','base','--objective','future','--epochs','5','--compile-mode','default']
    reports={}
    def benchmark(batch):
        name=f'benchmarks/ram_b{batch}';path=root/name
        if not (path/'pipeline_benchmark.json').exists():
            if path.exists():
                suffix='.incomplete_'+str(time.time_ns())
                path.rename(path.with_name(path.name+suffix))
                log=root/(name.replace('/','_')+'.log')
                if log.exists():log.rename(log.with_name(log.name+suffix))
            result=launch(root,name,[*common,'--batch-size-per-gpu',str(batch),
                '--pipeline-benchmark-steps','32','--benchmark-warmup','4'],allow_oom=True)
            if result is None:return False
        report=json.loads((path/'pipeline_benchmark.json').read_text())
        reports[str(batch)]=report
        write_json(root/'pipeline_comparison.json',reports)
        return max(max(g['peak_reserved_gib'],g['process_device_used_gib'])/g['total_gib'] for g in report['ranks'])<=.92
    for batch in (8,32):
        if not benchmark(batch):raise ValueError('Reference batch exceeded the memory margin')
    batch=choice['large_batch_per_gpu']
    while not benchmark(batch):
        batch-=8
        if batch<32:raise ValueError('The real pipeline does not fit the requested per-GPU reference batch')
    while benchmark(batch+8):
        batch+=8
        if batch>=2048:raise ValueError('Real pipeline capacity search exceeded its bounded range')
    plan=[row for row in pilot_batches(32,batch) if row['label']!='reference']
    plan.sort(key=lambda row:row['batch_per_gpu'],reverse=True)
    write_json(root/'pilots_plan.json',dict(world_size=4,epochs=5,model_size='base',objective='future',
        trials=plan,storage='verified shared RAM frames',previous_reference=settings.get('previous_reference'),
        large_batch_selection='largest measured pipeline batch below 92 percent VRAM'))
    common+=['--extended-dataset',str(Path(settings['extended'])/'dataset'),'--extended-every','10']
    for trial in plan:
        label,batch=trial['label'],trial['batch_per_gpu'];path=root/'pilots'/label
        progress=json.loads((path/'progress.json').read_text()) if (path/'progress.json').exists() else {}
        if progress.get('status')=='completed':continue
        options=[*common,'--batch-size-per-gpu',str(batch),'--stop-after-epoch-file',str(root/'stop_after_epoch')]
        if (path/'checkpoint_latest.pt').exists():options+=['--resume',str(path/'checkpoint_latest.pt')]
        launch(root,'pilots/'+label,options)
        progress=json.loads((path/'progress.json').read_text())
        if progress['status']=='paused':
            write_json(root/'controller.json',dict(status='paused',current=label,updated_unix=time.time()));return 75
    write_json(root/'controller.json',dict(status='completed',updated_unix=time.time()));return 0


def full_training(root):
    from scripts.stage_seviri_ssl_ram import stage
    stage(root)
    os.environ['SEVIRI_STORAGE_ROOTS']=str(root/'ram_storage_roots.json')
    locations=json.loads((root/'data_locations.json').read_text())
    plan=json.loads((root/'full_training.json').read_text())
    dataset=Path(locations['training'])/'dataset'
    extended=Path(locations['extended'])/'dataset'
    if plan.get('reference_checkpoint'):
        reference=root/'reference_validation'
        status=json.loads((reference/'progress.json').read_text()) if (reference/'progress.json').exists() else {}
        if status.get('status')!='completed':
            if reference.exists():reference.rename(reference.with_name('reference_validation.incomplete_'+str(time.time_ns())))
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                str(ROOT/'scripts/evaluate_seviri_ssl_checkpoint_ddp.py'),
                '--checkpoint',plan['reference_checkpoint'],'--dataset',str(dataset),
                '--extended-dataset',str(extended),'--frame-cache',locations['canonical_frame_root'],
                '--zarr-root',locations['canonical_zarr_root'],'--output-dir',str(reference)]
            write_json(root/'controller.json',dict(status='running',current='reference_validation',updated_unix=time.time()))
            with (root/'reference_validation.log').open('ab') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        previous=json.loads(Path(plan['reference_report']).read_text())
        current=json.loads((reference/'validation_legacy.json').read_text())
        for name in ('neural','persistence','advection'):
            if (current[name]['samples']!=previous[name]['samples'] or
                    abs(current[name]['rmse_k']-previous[name]['rmse_k'])>1e-4):
                raise ValueError('Reference evaluation differs after the loading correction: '+name)
    path=root/'training'
    progress=json.loads((path/'progress.json').read_text()) if (path/'progress.json').exists() else {}
    if progress.get('status')=='completed' and progress.get('completed_epochs')==plan['epochs']:return 0
    resume=path/'checkpoint_latest.pt'
    input_options = (['--clip-root', locations['canonical_clip_root']] if locations.get('canonical_clip_root')
                     else ['--frame-cache', locations['canonical_frame_root']])
    options=['--dataset',str(dataset),'--zarr-root',locations['canonical_zarr_root'],
        *input_options,'--extended-dataset',str(extended),
        '--model-size',plan['model_size'],'--objective',plan['objective'],
        '--epochs',str(plan['epochs']),'--batch-size-per-gpu',str(plan['batch_size_per_gpu']),
        '--learning-rate',str(plan.get('learning_rate',1.5e-4)),
        '--extended-every',str(plan.get('extended_every',10)),'--compile-mode','default',
        '--stop-after-epoch-file',str(root/'stop_after_epoch')]
    benchmark_steps=plan.get('pipeline_benchmark_steps',0)
    if benchmark_steps and not resume.exists() and not (root/'benchmark/pipeline_benchmark.json').exists():
        launch(root,'benchmark',[*options,'--pipeline-benchmark-steps',str(benchmark_steps),
            '--benchmark-warmup',str(plan.get('pipeline_benchmark_warmup',4))])
    if resume.exists():options+=['--resume',str(resume)]
    elif plan.get('parent_checkpoint'):
        options+=['--resume',plan['parent_checkpoint'],'--execution-revision',str(root/'execution_revision.json')]
    launch(root,'training',options)
    progress=json.loads((path/'progress.json').read_text())
    paused=progress['status']=='paused'
    write_json(root/'controller.json',dict(status='paused' if paused else 'completed',updated_unix=time.time()))
    return 75 if paused else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['capacity','pilots','ram-pilots','full'],required=True)
    p.add_argument('--baseline-global-batch',type=int,default=32)
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    plan=json.loads((args.root/'runtime_files.json').read_text())
    for name,expected in plan.items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen runtime source changed: '+name)
    try:
        if args.stage=='capacity':capacity(args.root);return 0
        if args.stage=='ram-pilots':return ram_pilots(args.root)
        if args.stage=='full':return full_training(args.root)
        return pilots(args.root,args.baseline_global_batch)
    except BaseException as exc:
        write_json(args.root/'controller.json',dict(status='failed',error=str(exc),updated_unix=time.time()));raise


if __name__=='__main__':sys.exit(main())
