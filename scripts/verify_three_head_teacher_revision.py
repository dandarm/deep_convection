#!/usr/bin/env python3
"""Compare complete, timestamp-aligned teacher periods before and after revision.

This is a preparation audit. Training reads the resulting report and never
repeats this full-domain file comparison.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from emma_gpm.rdt.storage import atomic_json


def compare_period(legacy, corrected):
    before = json.loads((legacy / 'complete.json').read_text())
    after = json.loads((corrected / 'complete.json').read_text())
    before_frames = {row['time']: row for row in before['frames']}
    if set(before_frames) != {row['time'] for row in after['frames']}:
        raise ValueError('The paired periods must contain identical timestamps')
    changes = np.zeros((7, 7), dtype=np.int64)
    counts = dict(cold_pixels=0, legacy_valid_cold_pixels=0,
                  corrected_valid_cold_pixels=0, changed_native_class_pixels=0,
                  changed_detection_pixels=0, changed_ct_pixels=0,
                  corrected_cold_without_history=0)
    for row in after['frames']:
        old_row = before_frames[row['time']]
        with np.load(legacy / old_row['file']) as old, np.load(corrected / row['file']) as new:
            old_rdt, new_rdt = old['native_rdt'], new['native_rdt']
            if old_rdt.shape != new_rdt.shape or not np.isin(old_rdt, range(7)).all() or not np.isin(new_rdt, range(7)).all():
                raise ValueError('Unexpected native RDT class or domain shape')
            cold = new_rdt > 0
            counts['cold_pixels'] += int(cold.sum())
            counts['legacy_valid_cold_pixels'] += int((old['valid'] & (old_rdt > 0)).sum())
            counts['corrected_valid_cold_pixels'] += int((new['valid'] & cold).sum())
            counts['corrected_cold_without_history'] += int((~new['valid'] & cold).sum())
            counts['changed_detection_pixels'] += int(((old_rdt > 0) != cold).sum())
            counts['changed_native_class_pixels'] += int((old_rdt != new_rdt).sum())
            counts['changed_ct_pixels'] += int((old['native_ct'] != new['native_ct']).sum())
            changes += np.bincount(7 * old_rdt.astype(np.int64).ravel() + new_rdt.ravel(), minlength=49).reshape(7, 7)
            for head, classes in [('cma', 2), ('ct', 5), ('rdt', 5)]:
                values, valid = new[head], new['valid']
                if not np.isin(values, [*range(classes), 255]).all() or not np.array_equal(values != 255, valid):
                    raise ValueError(f'Inconsistent common target mask: {row["time"]} / {head}')
    if counts['changed_detection_pixels'] or counts['changed_ct_pixels']:
        raise ValueError('The revision unexpectedly changed cell geometry or the independent CT teacher')
    denominator = counts['cold_pixels']
    return dict(frames=len(after['frames']), counts=counts,
                target_valid_cold_fraction={
                    'legacy': counts['legacy_valid_cold_pixels'] / denominator if denominator else None,
                    'corrected': counts['corrected_valid_cold_pixels'] / denominator if denominator else None},
                native_class_transition_counts=changes.tolist(),
                legacy_quality_gate=before['quality_gate'], corrected_quality_gate=after['quality_gate'],
                corrected_eligible_target_gate=after['eligible_target_gate'],
                corrected_native_rdt_by_ct_confusion=after['native_rdt_by_ct_confusion'],
                common_target_mask_verified=True, identical_detection_and_ct_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--corrected', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for split in ('train', 'validation', 'test'):
        reports[split] = compare_period(args.legacy / split, args.corrected / split)
        print(split, reports[split]['target_valid_cold_fraction'], flush=True)
    atomic_json(args.output, dict(legacy=str(args.legacy), corrected=str(args.corrected),
                                 status='completed', splits=reports,
                                 all_periods_accepted=all(r['corrected_quality_gate']['passed'] for r in reports.values())))


if __name__ == '__main__':
    main()
