#!/usr/bin/env python3
"""Prepare full-domain aligned teachers, then materialize downstream clips."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from emma_gpm.three_head_data import prepare_teacher_period, materialize_dataset
from build_rdt_pure_gen_utls_week import _geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/three_head_probe_dataset_summer2022.json')
    parser.add_argument('--split', choices=['train', 'validation', 'test'])
    parser.add_argument('--materialize', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.materialize:
        result = materialize_dataset(plan)
        print(json.dumps(dict(clips=len(result['samples']), statistics=result['statistics']), indent=2), flush=True)
    elif args.split:
        if plan.get('teacher_periods'):
            parser.error('This plan reuses immutable teacher periods; use --materialize or the original generation plan')
        result = prepare_teacher_period(plan['source_root'], plan['teacher_output'], args.split, plan, _geometry)
        print(json.dumps(dict(split=args.split, frames=len(result['frames']),
                             quality_gate=result['quality_gate']), indent=2), flush=True)
    else:
        parser.error('Choose --split or --materialize')


if __name__ == '__main__':
    main()
