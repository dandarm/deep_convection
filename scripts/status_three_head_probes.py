#!/usr/bin/env python3
"""Compact SSH-friendly progress and stale-heartbeat detection for the probe suite."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from emma_gpm.three_head_study import load_suite

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path('/data1/deep_convection_results/three_head_probes_20261109'))
parser.add_argument('--json',action='store_true')
args=parser.parse_args()
suite=load_suite(args.root)
rows=[]
for name,run in suite['runs'].items():
    progress=run['progress']
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(progress['heartbeat_utc'])).total_seconds() if progress.get('heartbeat_utc') else None
    status='completed' if run['result'] else progress.get('status','pending')
    rows.append(dict(experiment=name,status=status,epochs=progress.get('completed_epochs',0),
        planned_epochs=run['config']['epochs'],heartbeat_age_seconds=round(age) if age is not None else None,
        stalled_suspected=age is not None and age>600 and status not in ('completed','pending')))
if args.json:
    print(json.dumps(rows,indent=2))
else:
    preparation=args.root/'preparation_status.json'
    if preparation.exists():
        state=json.loads(preparation.read_text())
        if not state.get('training_started',False):
            print('Preparation: '+state['message'])
    print(f"Completed: {sum(r['status']=='completed' for r in rows)}/{len(rows)}")
    for row in rows:
        warning='  CHECK: heartbeat older than 10 minutes' if row['stalled_suspected'] else ''
        print(f"{row['experiment']:<32} {row['status']:<23} {row['epochs']:3}/{row['planned_epochs']}{warning}")
