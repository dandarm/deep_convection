#!/usr/bin/env python3
"""Paired day-block comparisons of cross-day descriptor agreement, no probe."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from emma_gpm.embedding_clustering import day_mean_interval


def compare(output, pooling='mean_slope', method='average'):
    if json.loads((output / 'progress.json').read_text())['stage'] != 'completed':
        raise ValueError('Complete the embedding analysis before comparing models')
    rows = []
    for split in ('train', 'validation', 'test_diagnostic'):
        destination = output / split
        schema = json.loads((destination / 'descriptor_schema.json').read_text())
        descriptors = list(pd.read_csv(destination / 'descriptors.csv').columns)
        days = pd.to_datetime(pd.read_csv(destination / 'manifest.csv').start_time, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
        gains = {}
        for run in ('mask50', 'mask75', 'mask90', 'random'):
            with np.load(destination / f'{run}_{pooling}_{method}_neighbors.npz', allow_pickle=False) as data:
                gains[run] = data['gain_by_clip'].copy()
            if gains[run].shape != (len(days), len(descriptors)):
                raise ValueError('Descriptor/manifest alignment mismatch')
        for reference in ('mask50', 'mask75', 'random'):
            # Subtract on the same clips BEFORE bootstrapping the same days.
            mean, low, high = day_mean_interval(gains['mask90'] - gains[reference], days)
            for index, name in enumerate(descriptors):
                rows.append(dict(split=split, pooling=pooling, method=method,
                    comparison=f'mask90_minus_{reference}', descriptor=name,
                    group=schema[name]['group'], difference=mean[index],
                    ci_low=low[index], ci_high=high[index], days=len(np.unique(days))))
    table = pd.DataFrame(rows)
    name = f'mask90_paired_{pooling}_{method}'
    table.to_csv(output / f'{name}.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = sorted(descriptors, key=lambda item: (schema[item]['group'], item))
    for split in table.split.unique():
        fig, axes = plt.subplots(1, 2, figsize=(13, 12), sharey=True)
        for axis, reference in zip(axes, ('mask50', 'mask75')):
            subset = table[(table.split == split) & (table.comparison == f'mask90_minus_{reference}')].set_index('descriptor').loc[order]
            for index, row in enumerate(subset.itertuples()):
                if np.isfinite([row.difference, row.ci_low, row.ci_high]).all():
                    axis.plot([row.ci_low, row.ci_high], [index, index], color='gray', lw=1)
                    axis.plot(row.difference, index, 'o', color='tab:blue', markersize=3)
            axis.axvline(0, color='black', linestyle='--', lw=1)
            axis.set(title=f'90% − {reference[4:]}%', xlabel='Differenza nel guadagno dei vicini')
            axis.grid(axis='x', alpha=.2)
        axes[0].set_yticks(range(len(order)), order, fontsize=7)
        axes[0].invert_yaxis()
        fig.suptitle(f'{split}: destra favorisce 90%, sinistra favorisce altro modello\nIntervalli descrittivi 95%: bootstrap appaiato per giornata, non test multipli di significatività')
        fig.tight_layout(); fig.savefig(output / f'{split}_mask90_differences.png', dpi=150); plt.close(fig)
    (output / f'{name}.json').write_text(json.dumps(dict(
        interpretation='Positive differences favor mask90 for this descriptor, geometry and pooling',
        uncertainty='Paired day-block percentile bootstrap; correlated days, one pretraining seed; no multiple-testing claims',
        samples='Same clips, same eligible cross-day pairs; equal-day mean',
        pooling=pooling, method=method, bootstrap_seed=20261005, bootstrap_repeats=400), indent=2)+'\n')
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/seviri_embedding_clustering')
    parser.add_argument('--pooling', choices=['mean', 'mean_slope'], default='mean_slope')
    parser.add_argument('--method', choices=['average', 'ward'], default='average')
    args = parser.parse_args()
    table = compare(args.output_dir, args.pooling, args.method)
    print(f'Saved {len(table)} paired descriptor comparisons')
