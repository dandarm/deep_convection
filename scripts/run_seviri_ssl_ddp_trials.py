#!/usr/bin/env python3
"""Run a bounded four-GPU capacity search, then two matched five-epoch trials."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pretrain_seviri_ssl import digest,write_json


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


def pilots(root,baseline_global_batch):
    settings=json.loads((root/'data_locations.json').read_text())
    choice=json.loads((root/'batch_choice.json').read_text())
    if choice['status']!='completed' or baseline_global_batch%4:
        raise ValueError('Completed capacity search and global batch divisible by four required')
    for name in ('training','extended'):
        directory=Path(settings[name]);verification=json.loads((directory/'verification.json').read_text())
        if verification['status']!='verified' or verification['transfer_plan_sha256']!=digest(directory/'transfer_plan.json'):
            raise ValueError(f'{name} transfer has not passed its checksum verification')
    os.environ['SEVIRI_STORAGE_ROOTS']=str(root/'storage_roots.json')
    common=['--dataset',str(Path(settings['training'])/'dataset'),
        '--zarr-root',settings['canonical_zarr_root'],'--frame-cache',settings['canonical_frame_root'],
        '--extended-dataset',str(Path(settings['extended'])/'dataset'),
        '--model-size','base','--objective','future','--epochs','5','--extended-every','10','--compile-mode','default']
    for label,batch in [('reference',baseline_global_batch//4),('large',choice['large_batch_per_gpu'])]:
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['capacity','pilots'],required=True)
    p.add_argument('--baseline-global-batch',type=int,default=32)
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    plan=json.loads((args.root/'runtime_files.json').read_text())
    for name,expected in plan.items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen runtime source changed: '+name)
    try:
        if args.stage=='capacity':capacity(args.root);return 0
        return pilots(args.root,args.baseline_global_batch)
    except BaseException as exc:
        write_json(args.root/'controller.json',dict(status='failed',error=str(exc),updated_unix=time.time()));raise


if __name__=='__main__':sys.exit(main())
