#!/usr/bin/env python3
"""Compare the accepted spaced manifest against the frozen calendar reference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from emma_gpm.ssl_overlap_audit import overlap_summary


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for label, dataset in [('calendar', args.reference), ('spaced', args.dataset)]:
        path = dataset/'training/dataset/train.csv'
        summary, daily, neighbors = overlap_summary(pd.read_csv(path))
        summary['manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output/f'{label}_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        daily.to_csv(args.output/f'{label}_daily.csv', index=False)
        neighbors.to_parquet(args.output/f'{label}_neighbors.parquet', index=False)
        reports[label] = summary
        print(label, json.dumps(summary), flush=True)
    labels = ['Calendar sampling', 'Active IoU ≤ 0.20']
    colors = ['#8c9ba6', '#087f8c']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    specs = [('clips', 1000, 'Training clips', 'Thousands of clips'),
        ('distinct_native_frames', 1000, 'Distinct observations', 'Thousands of SEVIRI frames'),
        ('unique_pixel_frames', 1e9, 'Unique observed positions', 'Billions of pixel/frame positions')]
    for ax, (key, scale, title, unit) in zip(axes, specs):
        values = [reports[name][key]/scale for name in ('calendar', 'spaced')]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, labels=[f'{value:.1f}' for value in values], padding=4)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values)*1.18)
        ax.tick_params(axis='x', labelrotation=12)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output/'sampling_comparison.png', dpi=170)
    plt.close(fig)
    (args.output/'comparison.json').write_text(json.dumps(reports, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
