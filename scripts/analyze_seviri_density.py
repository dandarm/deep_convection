#!/usr/bin/env python3
"""Add HDBSCAN diagnostics from existing embeddings, without GPU extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from emma_gpm.density_clustering import density_partition, density_scores, density_stability


def analyze(root, fraction=.03, min_samples=5):
    metrics, physical, fingerprints = [], [], {}
    for split in ('train', 'validation', 'test_diagnostic'):
        directory = root / split
        frame = pd.read_csv(directory / 'manifest.csv')
        days = pd.to_datetime(frame.start_time, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
        descriptors = pd.read_csv(directory / 'descriptors.csv')
        schema = json.loads((directory / 'descriptor_schema.json').read_text())
        for run in ('mask50', 'mask75', 'mask90', 'random'):
            path = directory / f'{run}_embeddings.npz'
            fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
            cached = json.loads((directory / f'{run}_embeddings_identity.json').read_text())
            if cached['npz_sha256'] != fingerprint:
                raise ValueError('Embedding cache changed')
            fingerprints[f'{split}/{run}'] = fingerprint
            with np.load(path, allow_pickle=False) as arrays:
                for pooling in ('mean', 'mean_slope'):
                    values = arrays[pooling]
                    labels, probabilities, minimum = density_partition(values, fraction, min_samples)
                    scores, agreement, distances = density_scores(values, descriptors.to_numpy(), labels)
                    stability, assignment_jaccard = density_stability(values, days, labels, fraction, min_samples)
                    assigned = labels >= 0
                    scores.update(run=run, split=split, pooling=pooling, method='hdbscan',
                        stability_ari=stability, assignment_jaccard=assignment_jaccard,
                        min_cluster_size=minimum, min_samples=min_samples,
                        utc_day_adjusted_mutual_information=(float(adjusted_mutual_info_score(days[assigned], labels[assigned]))
                            if scores['clusters'] >= 2 else np.nan))
                    metrics.append(scores)
                    for column, name in enumerate(descriptors):
                        physical.append(dict(run=run, split=split, pooling=pooling, method='hdbscan',
                            descriptor=name, group=schema[name]['group'], adjusted_eta2=agreement[column]))
                    prefix = f'{run}_{pooling}_hdbscan'
                    assignments = frame.copy()
                    assignments['cluster'] = labels
                    assignments['membership_strength'] = probabilities
                    assignments.to_csv(directory / f'{prefix}_assignments.csv', index=False)
                    representatives = []
                    for label in np.unique(labels):
                        indices = np.flatnonzero(labels == label)
                        rng = np.random.default_rng(20261005 + int(label) + 1)
                        if label < 0:
                            # Noise is NOT a coherent cluster with a medoid.
                            choices = [(f'non assegnata {i+1}', int(index)) for i, index in enumerate(
                                rng.choice(indices, min(3, len(indices)), replace=False))]
                        else:
                            medoid = indices[np.argmin(distances[np.ix_(indices, indices)].mean(1))]
                            choices = [('medoid', medoid), ('edge', indices[np.argmax(distances[medoid, indices])]),
                                       ('random', int(rng.choice(indices)))]
                        for role, index in choices:
                            representatives.append(dict(cluster=int(label), role=role, index=int(index),
                                **frame.iloc[index].to_dict()))
                    pd.DataFrame(representatives).to_csv(directory / f'{prefix}_representatives.csv', index=False)
                    print(json.dumps(scores), flush=True)
    pd.DataFrame(metrics).to_csv(root / 'density_metrics.csv', index=False)
    pd.DataFrame(physical).to_csv(root / 'density_descriptor_agreement.csv', index=False)
    (root / 'density_protocol.json').write_text(json.dumps(dict(
        algorithm='sklearn HDBSCAN, EOM, Euclidean on unit embeddings; no PCA',
        min_cluster_fraction=fraction, min_cluster_floor=10, min_samples=min_samples,
        n_clusters='Inferred from density hierarchy; -1 means unassigned, never a cluster',
        physical_scores='Conditional on assigned clips; compare alongside coverage, not directly with full-sample scores',
        stability='Whole-day resampling; ARI on jointly assigned points; assignment Jaccard reported separately',
        embedding_sha256=fingerprints, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        module_sha256=hashlib.sha256((ROOT/'src/emma_gpm/density_clustering.py').read_bytes()).hexdigest()),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/seviri_embedding_clustering')
    parser.add_argument('--min-cluster-fraction', type=float, default=.03)
    parser.add_argument('--min-samples', type=int, default=5)
    args = parser.parse_args()
    analyze(args.output_dir, args.min_cluster_fraction, args.min_samples)
