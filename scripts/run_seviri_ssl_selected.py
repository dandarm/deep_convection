#!/usr/bin/env python3
"""Run the approved Future/Joint experiments with frozen data and source plans."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.ssl_dataset import verify_snapshot
from emma_gpm.ssl_scaling import publish_summary
from emma_gpm.seviri_clip_files import ClipFileDataset
from emma_gpm.seviri_frame_dataset import FrameFileDataset
from scripts.pretrain_seviri_ssl import digest, write_json, source_hashes
from scripts.run_seviri_ssl_comparison import stages
from scripts.run_seviri_ssl_scaling import immutable_json, execute, stage_command, stage_completed


def selected_stages(args, dataset, queue):
    local=argparse.Namespace(dataset=dataset,output_dir=queue,zarr_root=args.zarr_root,
        epochs=args.epochs,probe_epochs=args.epochs,batch_size=args.batch_size,workers=args.workers,seed=args.seed)
    selected=[s for s in stages(local) if s['name'] in ('future','joint')]
    runtime={'--gradient-accumulation':getattr(args,'gradient_accumulation',1),'--loader-batch-size':args.batch_size,'--prefetch-factor':4}
    for stage in selected:
        for name,value in runtime.items():
            stage['flags'][stage['flags'].index(name)+1]=str(value)
        cache = getattr(args,'frame_cache',None)
        reference = getattr(args,'variability_reference_manifest',None) or args.reference_series/'n4096/dataset/train.csv'
        stage['flags'] += ['--frame-cache' if cache else '--clip-root',str(cache or args.clip_root),
            '--model-size',getattr(args,'model_size','small'),'--variability-reference-manifest',str(reference)]
    return selected


def prepare(args):
    input_root = args.frame_cache or args.clip_root
    reference_dataset = args.reference_series/args.reference_case/'dataset'
    variability_reference = args.variability_reference_manifest or args.reference_series/'n4096/dataset/train.csv'
    reference_progress = {name: json.loads((args.reference_series/args.reference_case/name/'progress.json').read_text()) for name in ('future','joint')}
    reference_updates = {v['optimizer_steps'] for v in reference_progress.values()}
    if len(reference_updates) != 1: raise ValueError('Reference objectives have different completed update budgets')
    ready=json.loads((args.output_dir/'dataset_ready.json').read_text())
    if len(ready['cases'])!=1 or not ready['strict_nesting'] or not ready['validation_identical']:
        raise ValueError('Selected experiments require one prepared nested extension with shared validation')
    case=ready['cases'][0];name=case['name'];queue=args.output_dir/name;dataset=queue/'dataset'
    for filename,expected in case['files'].items():
        if digest(dataset/filename)!=expected:
            raise ValueError(f'Prepared dataset changed: {filename}')
    if digest(dataset/'validation.csv')!=digest(reference_dataset/'validation.csv'):
        raise ValueError('Validation differs from the comparison series')
    for manifest in ('train.csv','validation.csv'):
        data = (FrameFileDataset if args.frame_cache else ClipFileDataset)(input_root,dataset/manifest,zarr_root=args.zarr_root)
        if hasattr(data,'close'): data.close()
    sources={**source_hashes(),**{filename:digest(ROOT/filename) for filename in (
        'scripts/run_seviri_ssl_selected.py','scripts/run_seviri_ssl_scaling.py',
        'scripts/run_seviri_ssl_comparison.py','src/emma_gpm/ssl_scaling.py',
        'scripts/build_seviri_ssl_extension.py','scripts/extend_ssl_clip_files.py',
        'scripts/extend_seviri_frame_cache.py','scripts/build_seviri_frame_cache.py')}}
    local=dict(stages=selected_stages(args,dataset,queue),dataset=case['files'],sources=source_hashes(),seed=args.seed,
        interpretation='Future and Joint with shared validation; dataset size, model capacity and update budget are recorded independently')
    plan=dict(cases=[{key:case[key] for key in ('name','train_samples','validation_samples')}],
        order=[dict(case=name,stage=stage['name']) for stage in local['stages']],
        epochs=args.epochs,probe_epochs=0,seed=args.seed,batch_size=args.batch_size,
        model_size=args.model_size,effective_batch_size=args.batch_size*args.gradient_accumulation,
        gradient_accumulation=args.gradient_accumulation,workers=args.workers,zarr_root=str(args.zarr_root),sources=sources,
        dataset_ready_sha256=digest(args.output_dir/'dataset_ready.json'),clip_root=str(args.clip_root) if args.clip_root else None, frame_cache=str(args.frame_cache) if args.frame_cache else None,
        clip_metadata_sha256=digest(input_root/'metadata.json'),reference_series=str(args.reference_series),
        variability_reference_manifest=str(variability_reference),
        variability_reference_sha256=digest(variability_reference),
        comparison=dict(reference_case=args.reference_case,
            reference_clips=json.loads((reference_dataset/'protocol.json').read_text())['train_samples'],
            reference_optimizer_updates=next(iter(reference_updates)),
            optimizer_updates_per_epoch=(case['train_samples']+args.batch_size*args.gradient_accumulation-1)//(args.batch_size*args.gradient_accumulation),
            planned_optimizer_updates=args.epochs*((case['train_samples']+args.batch_size*args.gradient_accumulation-1)//(args.batch_size*args.gradient_accumulation))),
        analysis='Per-epoch channel/horizon, structural and variability metrics; two pretraining runs only')
    immutable_json(queue/'comparison_plan.json',local)
    immutable_json(args.output_dir/'scaling_plan.json',plan)
    for filename,expected in sources.items():
        archive=args.output_dir/'source_archive'/filename
        if archive.exists():
            if digest(archive)!=expected:raise ValueError(f'Archived source changed: {filename}')
        else:
            archive.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/filename,archive)
    verify(args,plan,local,dataset,source_data=True)
    publish_summary(args.output_dir)
    return plan,local,queue,dataset


def verify(args,plan,local,dataset,source_data=False):
    for filename,expected in plan['sources'].items():
        if digest(ROOT/filename)!=expected:raise ValueError(f'Execution source changed: {filename}')
    input_root = args.frame_cache or args.clip_root
    if digest(input_root/'metadata.json')!=plan['clip_metadata_sha256']:
        raise ValueError('Clip collection metadata changed')
    if digest(Path(plan['variability_reference_manifest']))!=plan['variability_reference_sha256']:
        raise ValueError('Shared variability reference changed')
    for filename,expected in local['dataset'].items():
        if digest(dataset/filename)!=expected:raise ValueError(f'Dataset changed: {filename}')
    if source_data:
        verify_snapshot(args.zarr_root,json.loads((dataset/'source_snapshot.json').read_text()))


def run(args):
    plan,local,queue,dataset=prepare(args)
    if args.prepare_only:
        print('Prepared frozen Future/Joint plans',flush=True)
        return
    write_json(args.output_dir/'launcher.json',dict(pid=os.getpid(),command=list(sys.orig_argv),
        log=str(args.output_dir/'queue.log'),started_at_utc=datetime.now(timezone.utc).isoformat()))
    with exclusive_file_lock(queue/'.comparison.lock'):
        for stage in local['stages']:
            verify(args,plan,local,dataset)
            if stage_completed(queue,stage):
                print('Already completed',stage['name'],flush=True);continue
            execute(args,stage_command(queue,stage),queue.name,stage['name'],queue/(stage['name']+'.log'))
            if not stage_completed(queue,stage):raise RuntimeError('Trainer exited before completing the requested epochs')
        verify(args,plan,local,dataset,source_data=True)
        done=dict(status='completed',stages=len(local['stages']),training_completed=True,
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(queue/'queue_status.json',done);write_json(args.output_dir/'queue_status.json',done)
        publish_summary(args.output_dir)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--reference-series',type=Path,required=True)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--clip-root',type=Path)
    inputs.add_argument('--frame-cache',type=Path)
    parser.add_argument('--model-size',choices=('small','base'),default='small')
    parser.add_argument('--gradient-accumulation',type=int,default=1)
    parser.add_argument('--reference-case',default='n8192')
    parser.add_argument('--variability-reference-manifest',type=Path)
    parser.add_argument('--zarr-root',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=75)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261024)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    for name in ('output_dir','reference_series','clip_root','frame_cache','zarr_root','variability_reference_manifest'):
        if getattr(args,name) is not None: setattr(args,name,getattr(args,name).resolve())
    if min(args.epochs,args.batch_size,args.workers,args.gradient_accumulation)<1 or args.batch_size%2:
        parser.error('Use positive epochs/workers and an even batch size')
    with exclusive_file_lock(args.output_dir/'.scaling.lock'):
        run(args)


if __name__=='__main__':
    main()
