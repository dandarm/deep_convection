"""Density partitions with explicit coverage and unassigned-clip handling."""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

from .embedding_clustering import unit_embeddings, explained_variance, silhouette


def density_partition(values, min_cluster_fraction=.03, min_samples=5):
    """HDBSCAN on unit-vector Euclidean geometry, never a prescribed K."""
    if not 0 < min_cluster_fraction < 1 or min_samples < 1:
        raise ValueError('Invalid density parameters')
    unit = unit_embeddings(values)
    minimum = max(10, int(np.ceil(len(unit) * min_cluster_fraction)))
    model = HDBSCAN(min_cluster_size=minimum, min_samples=min_samples,
        metric='euclidean', cluster_selection_method='eom', allow_single_cluster=False,
        n_jobs=1).fit(unit)
    return model.labels_, model.probabilities_, minimum


def density_scores(values, descriptors, labels):
    """Scores conditional on assigned points; always accompany with coverage."""
    assigned = labels >= 0
    clusters = len(np.unique(labels[assigned]))
    distances = squareform(pdist(unit_embeddings(values), metric='euclidean'))
    score = silhouette(distances[np.ix_(assigned, assigned)], labels[assigned]) if clusters >= 2 else np.nan
    physical = (explained_variance(descriptors[assigned], labels[assigned]) if clusters >= 2
                else np.full(descriptors.shape[1], np.nan))
    return dict(clusters=clusters, coverage=float(assigned.mean()), silhouette=score), physical, distances


def density_stability(values, days, labels, fraction=.03, min_samples=5, repeats=8, seed=20261005):
    """ARI on points assigned in both partitions, plus assignment-set Jaccard.

    All-noise/one-cluster cases never receive perfect stability by convention.
    Whole days are resampled; relative minimum size remains the same.
    """
    unique = np.unique(days)
    if len(unique) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    scores, assignment_agreement = [], []
    for _ in range(repeats):
        selected_days = rng.choice(unique, max(2, int(.8 * len(unique))), replace=False)
        selected = np.flatnonzero(np.isin(days, selected_days))
        new, _, _ = density_partition(values[selected], fraction, min_samples)
        original = labels[selected]
        union = (original >= 0) | (new >= 0)
        common = (original >= 0) & (new >= 0)
        if union.any():
            assignment_agreement.append(float(common.sum() / union.sum()))
        if common.sum() >= 20 and len(np.unique(original[common])) >= 2 and len(np.unique(new[common])) >= 2:
            scores.append(adjusted_rand_score(original[common], new[common]))
    return (float(np.mean(scores)) if scores else np.nan,
            float(np.mean(assignment_agreement)) if assignment_agreement else np.nan)
