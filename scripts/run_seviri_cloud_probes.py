#!/usr/bin/env python3
"""CPU probe queue: completed exports first, then wait for ongoing pretraining."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
# Probe execution is deliberately CPU-only, including when launched from a GPU shell.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from emma_gpm.rdt.storage import atomic_json, sha256
from emma_gpm.file_lock import exclusive_file_lock


def ready(path):
    progress = path/'progress.json'
    return (progress.exists() and json.loads(progress.read_text()).get('status') == 'completed'
            and (path/'encoder/model.safetensors').exists() and (path/'encoder/config.json').exists())


def random_key(run):
    config,stats = run['identity']['model_config'],run['identity']['stats']
    values = dict(config=config,mean=stats['global_mean_k'],std=stats['global_std_k'],seed=run['identity']['seed'])
    digest = hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()[:12]
    return f"random_{run['arguments'].get('model_size','small')}_{digest}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/cloud_mask_probe_20261031')
    parser.add_argument('--results-root',type=Path,default=ROOT/'results')
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/ssl_cloud_probe_series.json')
    parser.add_argument('--extend-from-series',type=Path,help='Continue completed probes from this series into a new output directory')
    parser.add_argument('--no-wait',action='store_true',help='Process available encoders and exit')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        os.environ[name] = str(plan['cpu_threads'])
    import torch
    torch.set_num_threads(plan['cpu_threads']); torch.set_num_interop_threads(1)
    from emma_gpm.cloud_probe import run_cloud_probe
    args.output_dir.mkdir(parents=True,exist_ok=True)
    sources = {name:sha256(ROOT/name) for name in ['scripts/run_seviri_cloud_probes.py',
        'src/emma_gpm/cloud_probe.py','src/emma_gpm/cloud_probe_data.py','src/emma_gpm/ssl_model.py','src/emma_gpm/seviri_clips.py']}
    identity = dict(plan=plan,dataset=str(args.dataset.resolve()),results_root=str(args.results_root.resolve()),sources=sources)
    if args.extend_from_series is not None:
        if args.extend_from_series.resolve() == args.output_dir.resolve():
            raise ValueError('Extension output must differ from the parent series')
        identity['extend_from_series'] = str(args.extend_from_series.resolve())
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    state = dict(pid=os.getpid(),status='starting',completed=[],failed={},pending=[],current=None)
    def publish(**fields):
        state.update(fields,updated_unix=time.time())
        atomic_json(args.output_dir/'progress.json',state)
    with exclusive_file_lock(args.output_dir/'queue.lock'):
        identity_path = args.output_dir/'queue_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('Queue identity changed; select a new output directory')
        atomic_json(identity_path,identity)
        (args.output_dir/'queue.pid').write_text(str(os.getpid())+'\n')
        try:
            while not (args.dataset/'manifest.json').exists():
                data_progress = args.dataset/'progress.json'
                status = json.loads(data_progress.read_text()) if data_progress.exists() else {}
                if status.get('status') == 'failed':
                    raise RuntimeError(f'Dataset construction failed: {status.get("error")}')
                publish(status='waiting_for_dataset',dataset_progress=status)
                time.sleep(10)
            while True:
                for name,digest in sources.items():
                    if sha256(ROOT/name) != digest:
                        raise RuntimeError(f'Queue source changed: {name}')
                jobs,seen_random,pending = [],set(),[]
                for relative in plan['run_directories']:
                    path = args.results_root/relative
                    if not ready(path):
                        pending.append(relative); continue
                    run = json.loads((path/'run_config.json').read_text())
                    control = random_key(run)
                    # Pair each first trained model with its matched random control.
                    jobs.append((relative.replace('/','__'),path,'pretrained',control))
                    if plan['include_random_controls'] and control not in seen_random:
                        jobs.append((control,path,'random',control)); seen_random.add(control)
                completed = [name for name,_,_,_ in jobs if (args.output_dir/name/'result.json').exists()]
                publish(status='running',completed=completed,pending=pending,available_jobs=len(jobs),current=None)
                for name,path,kind,control in jobs:
                    if name in completed or name in state['failed']:
                        continue
                    publish(current=name,phase='initializing',paired_random_control=control)
                    def heartbeat(progress):
                        publish(phase=progress['stage'],job_progress=progress)
                    print(f'Starting {name}',flush=True)
                    try:
                        parent = args.extend_from_series/name if args.extend_from_series is not None else None
                        if parent is not None and not parent.exists():
                            parent = None  # Newly completed encoders train the head from epoch 1.
                        result = run_cloud_probe(path,args.dataset,args.output_dir/name,kind=kind,
                            **{k:plan[k] for k in ['epochs','batch_size','learning_rate','weight_decay','seed','precision']},
                            heartbeat=heartbeat,extend_from=parent)
                        atomic_json(args.output_dir/name/'comparison.json',dict(paired_random_control=control,run_directory=str(path)))
                        completed.append(name)
                        print(f"Completed {name}: cloud IoU {result['test']['cloud_iou']:.3f}",flush=True)
                    except Exception as error:
                        traceback.print_exc()
                        state['failed'][name] = repr(error)
                        atomic_json(args.output_dir/name/'progress.json',dict(status='failed',pid=os.getpid(),updated_unix=time.time(),error=repr(error)))
                    publish(completed=completed,current=None)
                if not pending or args.no_wait:
                    publish(status='completed_with_errors' if state['failed'] else ('available_completed' if pending else 'completed'),current=None)
                    return 1 if state['failed'] else 0
                publish(status='waiting_for_encoders',current=None,phase='waiting')
                time.sleep(30)
        except BaseException as error:
            publish(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error))
            raise

if __name__ == '__main__':
    sys.exit(main())
