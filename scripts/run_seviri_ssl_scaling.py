#!/usr/bin/env python3
"""Run the four objectives and five frozen probes at both nested dataset sizes."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))

from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.ssl_dataset import verify_snapshot
from emma_gpm.ssl_scaling import publish_summary
from scripts.pretrain_seviri_ssl import digest,write_json,source_hashes
from scripts.run_seviri_ssl_comparison import stages


def immutable_json(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:
            raise ValueError(f'Frozen plan changed: {path}')
    else:
        write_json(path,value)


def make_plans(args,ready):
    cases=[]
    plans={}
    reference=min(ready['cases'],key=lambda case:case['train_samples'])['name']
    reference_manifest=args.output_dir/reference/'dataset/train.csv'
    for case in ready['cases']:
        queue=args.output_dir/case['name']
        local=argparse.Namespace(dataset=queue/'dataset',output_dir=queue,zarr_root=args.zarr_root,
            batch_size=args.batch_size,workers=args.workers,seed=args.seed,epochs=args.epochs,probe_epochs=args.probe_epochs)
        plan=dict(stages=stages(local),dataset={name:digest(local.dataset/name) for name in
            ('train.csv','validation.csv','channel_stats.json','source_snapshot.json')},sources=source_hashes(),
            seed=args.seed,interpretation='nested training sets; common later validation; one seed; equal epochs and matched-update diagnostics')
        for stage in plan['stages']:
            stage['flags']+=['--variability-reference-manifest',str(reference_manifest)]
        plans[case['name']]=plan
        cases.append(dict(name=case['name'],train_samples=case['train_samples'],validation_samples=case['validation_samples']))
    if len({value['dataset']['validation.csv'] for value in plans.values()})!=1:
        raise ValueError('Scaling comparisons require identical validation manifests')
    order=[dict(case=case['name'],stage=stage['name'])
           for stage in next(iter(plans.values()))['stages'] for case in cases]
    sources={**source_hashes(),**{name:digest(ROOT/name) for name in (
        'scripts/run_seviri_ssl_scaling.py','scripts/run_seviri_ssl_comparison.py',
        'scripts/analyze_seviri_ssl_comparison.py','src/emma_gpm/ssl_diagnostics.py','src/emma_gpm/ssl_scaling.py')}}
    plan=dict(cases=cases,order=order,epochs=args.epochs,probe_epochs=args.probe_epochs,
        dataset_ready_sha256=digest(args.output_dir/'dataset_ready.json'),
        zarr_root=str(args.zarr_root),seed=args.seed,batch_size=args.batch_size,workers=args.workers,
        gradient_accumulation=2,sources=sources,
        variability_reference_manifest=str(reference_manifest),
        comparison='same epochs and settings; doubled training clips double optimizer updates; both budgets are recorded',
        probe_data='common within each size; data-size treatment applies to both pretraining and probe fitting',
        analysis='temporal interventions, radiometric ridge probes, frozen-weight checks and prediction examples after training')
    return plan,plans


def stage_command(queue,stage):
    output=queue/stage['name']
    command=[sys.executable,'-u',str(ROOT/'scripts/pretrain_seviri_ssl.py'),*stage['flags'],'--output-dir',str(output)]
    if (output/'checkpoint_latest.pt').exists():
        command+=['--resume',str(output/'checkpoint_latest.pt')]
    return command


def stage_completed(queue,stage):
    output=queue/stage['name']
    path=output/'progress.json'
    if not path.exists():
        return False
    progress=json.loads(path.read_text())
    if progress.get('status')!='completed' or progress.get('completed_epochs')!=stage['epochs']:
        return False
    required=[output/'checkpoint_latest.pt',output/'validation_latest.json',output/'encoder/config.json']
    weights=[output/'encoder/model.safetensors',output/'encoder/pytorch_model.bin']
    if not all(p.is_file() for p in required) or not any(p.is_file() for p in weights):
        raise ValueError(f'Completed stage is missing required artifacts: {output}')
    return True


def execute(args,command,case,stage,log_path):
    started=time.time()
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    with log_path.open('a') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=environment)
        status=dict(status='running',case=case,stage=stage,pid=process.pid,supervisor_pid=os.getpid(),
            command=command,started_at_utc=datetime.now(timezone.utc).isoformat(),log=str(log_path))
        write_json(args.output_dir/'queue_status.json',status)
        write_json(args.output_dir/case/'queue_status.json',status)
        print('Started',case,stage,'PID',process.pid,flush=True)
        while True:
            try:
                code=process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                publish_summary(args.output_dir)
                write_json(args.output_dir/'queue_status.json',dict(**status,
                    heartbeat_utc=datetime.now(timezone.utc).isoformat()))
        publish_summary(args.output_dir)
        if code:
            failed=dict(status='failed',case=case,stage=stage,returncode=code,log=str(log_path))
            write_json(args.output_dir/'queue_status.json',failed)
            write_json(args.output_dir/case/'queue_status.json',failed)
            raise RuntimeError(f'{case}/{stage} exited {code}; see {log_path}')
    print('Completed',case,stage,'seconds',round(time.time()-started,1),flush=True)


def run(args):
    ready=json.loads((args.output_dir/'dataset_ready.json').read_text())
    plan,plans=make_plans(args,ready)
    with ExitStack() as locks:
        for case in plan['cases']:
            locks.enter_context(exclusive_file_lock(args.output_dir/case['name']/'.comparison.lock'))
        for case in plan['cases']:
            dataset=args.output_dir/case['name']/'dataset'
            expected=next(value for value in ready['cases'] if value['name']==case['name'])['files']
            for name,checksum in expected.items():
                if digest(dataset/name)!=checksum:
                    raise ValueError(f'Prepared dataset changed: {dataset/name}')
            verify_snapshot(args.zarr_root,json.loads((dataset/'source_snapshot.json').read_text()))
            immutable_json(args.output_dir/case['name']/'comparison_plan.json',plans[case['name']])
        immutable_json(args.output_dir/'scaling_plan.json',plan)
        for name,checksum in plan['sources'].items():
            archive=args.output_dir/'source_archive'/name
            if archive.exists():
                if digest(archive)!=checksum:
                    raise ValueError(f'Archived source differs: {name}')
            else:
                archive.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,archive)
        publish_summary(args.output_dir)
        for item in plan['order']:
            for name,expected in plan['sources'].items():
                if digest(ROOT/name)!=expected:
                    raise ValueError(f'Campaign source changed: {name}')
            queue=args.output_dir/item['case']
            stage=next(s for s in plans[item['case']]['stages'] if s['name']==item['stage'])
            if stage_completed(queue,stage):
                print('Already completed',item['case'],stage['name'],flush=True)
                continue
            if stage['name'].startswith('probe_') and stage['name']!='probe_random':
                parent=next(s for s in plans[item['case']]['stages'] if s['name']==stage['name'][6:])
                if not stage_completed(queue,parent):
                    raise ValueError('Probe requires completed 150-epoch source encoder')
            execute(args,stage_command(queue,stage),item['case'],stage['name'],queue/(stage['name']+'.log'))
            if not stage_completed(queue,stage):
                raise RuntimeError('Trainer exited without a completed stage')
        for case in plan['cases']:
            queue=args.output_dir/case['name']
            verify_snapshot(args.zarr_root,json.loads((queue/'dataset/source_snapshot.json').read_text()))
            write_json(queue/'queue_status.json',dict(status='completed',stages=len(plans[case['name']]['stages']),
                source_snapshot_verified=True,completed_at_utc=datetime.now(timezone.utc).isoformat()))
            execute(args,[sys.executable,'-u',str(ROOT/'scripts/analyze_seviri_ssl_comparison.py'),
                '--queue-dir',str(queue),'--batch-size',str(args.batch_size),'--workers',str(min(args.workers,4))],
                case['name'],'analysis',queue/'analysis.log')
            write_json(queue/'queue_status.json',dict(status='completed',training_and_analysis=True))
        publish_summary(args.output_dir)
        write_json(args.output_dir/'queue_status.json',dict(status='completed',stages=len(plan['order']),
            training_and_analysis=True,completed_at_utc=datetime.now(timezone.utc).isoformat()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--zarr-root',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--probe-epochs',type=int,default=150)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20261024)
    args=parser.parse_args()
    args.output_dir=args.output_dir.resolve()
    args.zarr_root=args.zarr_root.resolve()
    if min(args.epochs,args.probe_epochs,args.batch_size,args.workers)<1 or args.batch_size%2:
        parser.error('Positive epochs/workers and an even batch size are required')
    with exclusive_file_lock(args.output_dir/'.scaling.lock'):
        run(args)


if __name__=='__main__':
    main()
