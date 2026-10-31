#!/usr/bin/env python3
"""Lightweight SSH status for the cloud-mask probe queue and its heartbeat."""
import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def active_output(root=ROOT):
    pointer = root/'results/cloud_probe_active_series.json'
    if pointer.exists():
        path = Path(json.loads(pointer.read_text())['output_dir'])
        return path if path.is_absolute() else root/path
    return root/'results/cloud_mask_probe_20261031'


def process_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid),0)
        proc = Path(f'/proc/{pid}/stat')
        return not proc.exists() or proc.read_text().split(') ',1)[1][0] != 'Z'
    except (ProcessLookupError,ValueError):
        return False
    except PermissionError:
        return True


def inspect(output,stale_seconds=180):
    path = Path(output)/'progress.json'
    if not path.exists():
        return dict(status='not_started',health='nessun processo registrato')
    result = json.loads(path.read_text())
    age = max(0.,time.time()-result.get('updated_unix',0))
    result['heartbeat_age_seconds'] = round(age,1)
    result['process_alive'] = process_alive(result.get('pid'))
    active = result['status'] in ['starting','running','waiting_for_dataset','waiting_for_encoders']
    if active and not result['process_alive']:
        health = 'ARRESTATO: il processo registrato è terminato'
    elif active and age > stale_seconds:
        health = 'POSSIBILE BLOCCO: avanzamento non aggiornato oltre la soglia'
    elif result['status'] in ['failed','completed_with_errors','interrupted']:
        health = 'ERRORE O INTERRUZIONE: consultare queue.log'
    else:
        health = 'regolare'
    result['health'] = health
    if result['status'] == 'waiting_for_dataset':
        identity = json.loads((Path(output)/'queue_identity.json').read_text())
        progress = Path(identity['dataset'])/'progress.json'
        if progress.exists():
            result['dataset_progress'] = json.loads(progress.read_text())
            p = result['dataset_progress']
            if p.get('status') == 'building' and (not process_alive(p.get('pid')) or time.time()-p.get('updated_unix',0)>stale_seconds):
                result['health'] = 'POSSIBILE BLOCCO: costruzione dataset ferma o senza aggiornamenti'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=None)
    parser.add_argument('--stale-seconds',type=float,default=180)
    parser.add_argument('--json',action='store_true')
    args = parser.parse_args()
    args.output_dir = args.output_dir or active_output()
    s = inspect(args.output_dir,args.stale_seconds)
    if args.json:
        print(json.dumps(s)); return
    print(f"Cloud-mask probe: {s['status']} — {s['health']}")
    print('Serie:',args.output_dir)
    if s['status'] == 'waiting_for_dataset':
        identity = json.loads((args.output_dir/'queue_identity.json').read_text())
        print(f"Encoder pianificati: {len(identity['plan']['run_directories'])}; in attesa del dataset prima di avviare i probe")
    else:
        print(f"Completati: {len(s.get('completed',[]))}/{s.get('available_jobs','?')} probe disponibili; encoder attesi: {len(s.get('pending',[]))}; errori: {len(s.get('failed',{}))}")
    print(f"PID: {s.get('pid','—')}; ultimo aggiornamento: {s.get('heartbeat_age_seconds','—')} s fa")
    if s.get('current'):
        print('Modello:',s['current'])
    p = s.get('job_progress',{})
    if s.get('current') and p.get('stage') == 'feature_extraction':
        print(f"Rappresentazioni: {p['completed']}/{p['total']} clip")
    elif s.get('current') and p.get('stage') == 'head_training':
        print(f"Testa: epoca {p['epoch']}/{p['epochs']}, batch {p['batch']}/{p['batches']}")
    if s['status'] == 'waiting_for_dataset':
        p = s.get('dataset_progress',{})
        print(f"Dataset: {p.get('day','—')}, frame {p.get('frame',0)}/{p.get('frames','?')}, clip {p.get('samples',0)}")
    for key,error in s.get('failed',{}).items():
        print(f'Errore {key}: {error}')

if __name__ == '__main__':
    main()
