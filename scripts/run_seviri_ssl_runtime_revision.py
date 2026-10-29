#!/usr/bin/env python3
"""Resume a frozen SEVIRI matrix with an audited CUDA execution revision."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.ssl_runtime import CudaRuntime
from emma_gpm.ssl_scaling import publish_summary
from scripts.pretrain_seviri_ssl import digest, source_hashes, write_json
from scripts.run_seviri_ssl_matrix import verify
from scripts.run_seviri_ssl_scaling import execute, stage_command, stage_completed

# This migration covers execution controls, with model/data/loss implementations
# still bound to their original source hashes.
ALLOWED_SOURCE_CHANGES = {'scripts/pretrain_seviri_ssl.py', 'src/emma_gpm/ssl_execution.py'}


def read_protocol(root):
    plan = json.loads((root/'scaling_plan.json').read_text())
    plans = {c['name']:json.loads((root/c['name']/'comparison_plan.json').read_text()) for c in plan['cases']}
    return plan, plans


def prepare(args, plan, plans):
    path = args.revision_dir/'revision.json'
    if path.exists():
        return json.loads(path.read_text())
    if args.cuda_runtime is None:
        raise ValueError('Creating a revision requires --cuda-runtime')
    runtime = CudaRuntime(**json.loads(args.cuda_runtime.read_text())).to_dict()
    for filename, checksum in plan['sources'].items():
        if digest(args.output_dir/'source_archive'/filename) != checksum:
            raise ValueError(f'Original source archive changed: {filename}')
        if filename not in ALLOWED_SOURCE_CHANGES and digest(ROOT/filename) != checksum:
            raise ValueError(f'Scientific source changed: {filename}')
    sources = {name:digest(ROOT/name) for name in plan['sources']}
    sources.update(source_hashes())
    sources[str(Path(__file__).relative_to(ROOT))] = digest(__file__)
    for filename, checksum in sources.items():
        target = args.revision_dir/'source_archive'/filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/filename, target)
        if digest(target) != checksum: raise ValueError(f'Source changed during capture: {filename}')
    runtime_path = args.revision_dir/'cuda_runtime.json'
    write_json(runtime_path,runtime)
    case_revisions={}
    for case in plan['cases']:
        name=case['name'];parents={}
        for stage in plans[name]['stages']:
            stage_dir=args.output_dir/name/stage['name']
            archive=args.revision_dir/'parents'/name/stage['name']
            archive.mkdir(parents=True,exist_ok=True)
            for checkpoint in stage_dir.glob('checkpoint_*.pt'):
                target=archive/checkpoint.name
                # Copy works across filesystems and keeps parent states immutable.
                shutil.copyfile(checkpoint,target)
                parents[digest(target)]=str(target)
            for filename in ('run_config.json','progress.json','loss_epochs.csv'):
                if (stage_dir/filename).exists(): shutil.copyfile(stage_dir/filename,archive/filename)
        local=dict(version=2,parent_sources=plans[name]['sources'],parent_checkpoints=parents,
            sources=sources,cuda_runtime=runtime,clip_metadata_sha256=case['frame_metadata_sha256'],
            runtime=dict(batch_size=plan['batch_size'],loader_batch_size=plan['batch_size'],
                         gradient_accumulation=plan['gradient_accumulation'],num_workers=plan['workers']))
        target=args.revision_dir/name/'execution_revision.json';write_json(target,local)
        case_revisions[name]=dict(path=str(target),sha256=digest(target))
    revision=dict(version=1,created_at_utc=datetime.now(timezone.utc).isoformat(),
        original_plan_sha256=digest(args.output_dir/'scaling_plan.json'),
        case_plans={name:digest(args.output_dir/name/'comparison_plan.json') for name in plans},
        sources=sources,cases=case_revisions,cuda_runtime=runtime,cuda_runtime_path=str(runtime_path),
        cuda_runtime_sha256=digest(runtime_path),
        reason='CUDA runtime optimizations adapted from Demetra; same data, model, loss and optimizer hyperparameters',
        numerical_note='Kernel selection and compilation can change floating-point rounding; execution boundary is recorded')
    write_json(path,revision)
    return revision


def verify_revision(args, plan, revision, *, source_data=False):
    if digest(args.output_dir/'scaling_plan.json') != revision['original_plan_sha256']:
        raise ValueError('Original matrix plan changed')
    for case,checksum in revision['case_plans'].items():
        if digest(args.output_dir/case/'comparison_plan.json') != checksum:
            raise ValueError(f'Case plan changed: {case}')
    for case,record in revision['cases'].items():
        if digest(record['path']) != record['sha256']:
            raise ValueError(f'Execution revision changed: {case}')
    if digest(revision['cuda_runtime_path']) != revision['cuda_runtime_sha256']:
        raise ValueError('CUDA runtime configuration changed')
    verify(args.output_dir,{**plan,'sources':revision['sources']},source_data=source_data)


def run(args):
    plan,plans=read_protocol(args.output_dir)
    with ExitStack() as locks:
        locks.enter_context(exclusive_file_lock(args.output_dir/'.scaling.lock'))
        for case in plan['cases']:
            locks.enter_context(exclusive_file_lock(args.output_dir/case['name']/'.comparison.lock'))
        if not args.prepare_only:
            write_json(args.output_dir/'launcher.json',dict(pid=os.getpid(),command=list(sys.orig_argv),
                log=str(args.output_dir/'queue.log'),started_at_utc=datetime.now(timezone.utc).isoformat(),
                execution_revision=str(args.revision_dir/'revision.json')))
            write_json(args.output_dir/'queue_status.json',dict(status='preparing',supervisor_pid=os.getpid(),
                pid=None,log=str(args.output_dir/'queue.log')))
        revision=prepare(args,plan,plans)
        verify_revision(args,plan,revision,source_data=True)
        if args.prepare_only: return
        write_json(args.output_dir/'launcher.json',dict(pid=os.getpid(),command=list(sys.orig_argv),
            log=str(args.output_dir/'queue.log'),started_at_utc=datetime.now(timezone.utc).isoformat(),
            execution_revision=str(args.revision_dir/'revision.json')))
        for item in plan['order']:
            verify_revision(args,plan,revision)
            queue=args.output_dir/item['case']
            stage=next(s for s in plans[item['case']]['stages'] if s['name']==item['stage'])
            if stage_completed(queue,stage):
                print('Already completed',item['case'],stage['name'],flush=True)
                continue
            revised={**stage,'flags':[*stage['flags'],'--cuda-runtime',revision['cuda_runtime_path'],
                '--execution-revision',revision['cases'][item['case']]['path']]}
            execute(args,stage_command(queue,revised),item['case'],stage['name'],queue/(stage['name']+'.log'))
            if not stage_completed(queue,stage): raise RuntimeError('Requested epochs remain incomplete')
            if all(stage_completed(queue,s) for s in plans[item['case']]['stages']):
                write_json(queue/'queue_status.json',dict(status='completed',training_completed=True,stages=2))
        verify_revision(args,plan,revision,source_data=True)
        publish_summary(args.output_dir)
        write_json(args.output_dir/'queue_status.json',dict(status='completed',training_completed=True,
            stages=len(plan['order']),completed_at_utc=datetime.now(timezone.utc).isoformat()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--revision-dir',type=Path,required=True)
    p.add_argument('--cuda-runtime',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    args.output_dir,args.revision_dir=args.output_dir.resolve(),args.revision_dir.resolve()
    run(args)


if __name__=='__main__': main()
