#!/usr/bin/env python3
"""Run the three-head frozen probes and sequential transfer matrix in order."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from emma_gpm.rdt.storage import atomic_json
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.three_head_training import run_probe, pd_timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'configs/three_head_probe_experiments.json')
    parser.add_argument('--only', nargs='*', help='Explicit subset; parent runs must already exist')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = Path(plan['output']); output.mkdir(parents=True, exist_ok=True)
    runs = [{**plan['defaults'], **row} for row in plan['experiments']]
    if args.only:
        names = {row['name'] for row in runs}
        if not set(args.only) <= names:
            parser.error('Unknown experiment requested')
        runs = [row for row in runs if row['name'] in args.only]
    with exclusive_file_lock(output/'suite.lock'):
        atomic_json(output/'experiment_plan.json',plan)
        completed = []
        for run in runs:
            state = dict(status='running', current=run['name'], completed=completed, total=len(runs),
                         pid=os.getpid(), heartbeat_utc=pd_timestamp())
            atomic_json(output/'suite_status.json',state)
            print(f"Starting {run['name']}",flush=True)
            try:
                result=run_probe(run,plan['dataset'],output)
            except Exception as error:
                atomic_json(output/'suite_status.json',{**state,'status':'failed','error':str(error),
                                                       'traceback':traceback.format_exc()})
                raise
            completed.append(run['name'])
            print(f"Completed {run['name']}; selected epoch {result['best_epoch']}",flush=True)
        atomic_json(output/'suite_status.json',dict(status='completed',completed=completed,total=len(runs),
                                                   heartbeat_utc=pd_timestamp()))


if __name__ == '__main__':
    main()
