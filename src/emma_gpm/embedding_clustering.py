"""Hierarchical geometry and descriptor agreement, without a trained probe."""
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score


def pool_encoder_tokens(tokens, temporal_tokens=8):
    """Spatial mean per tubelet, then mean and OLS change over the 75-min clip.

    Flattened token order is temporal, row, column. Returns raw representations;
    unit-length normalization is applied later for angular geometry only.
    """
    import torch
    if tokens.ndim != 3 or tokens.shape[1] % temporal_tokens:
        raise ValueError('Invalid encoder token grid')
    time_means = tokens.float().reshape(tokens.shape[0], temporal_tokens, -1, tokens.shape[-1]).mean(2)
    time = torch.arange(temporal_tokens, device=tokens.device, dtype=torch.float32) * 10 + 2.5
    time = time - time.mean()
    mean = time_means.mean(1)
    change = 75 * (time_means * time[None, :, None]).sum(1) / time.square().sum()
    return {'mean': mean, 'mean_slope': torch.cat((mean, change), dim=-1)}


def unit_embeddings(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(norms < 1e-12):
        raise ValueError('Embedding is nonfinite or has zero norm')
    return values / norms


def hierarchy(values, method):
    values = unit_embeddings(values)
    if method == 'average':
        distances = np.maximum(pdist(values, metric='cosine'), 0)
    elif method == 'ward':
        distances = pdist(values, metric='euclidean')
    else:
        raise ValueError('Use average/cosine or Ward/Euclidean on unit embeddings')
    return linkage(distances, method=method), squareform(distances)


def explained_variance(descriptors, labels):
    """Per-descriptor eta² adjusted for expected random grouping at this K.

    Negative scores are retained; constant variables are reported as NaN.
    Adjustment is a random-partition reference, not an independence p-value.
    """
    values = np.asarray(descriptors, dtype=np.float64)
    total = np.square(values - values.mean(0)).sum(0)
    within = np.zeros(values.shape[1])
    for label in np.unique(labels):
        group = values[labels == label]
        within += np.square(group - group.mean(0)).sum(0)
    eta = np.divide(total - within, total, out=np.full_like(total, np.nan), where=total > 1e-12)
    expected = (len(np.unique(labels)) - 1) / (len(values) - 1)
    return (eta - expected) / (1 - expected)


def cross_day_neighbor_gain(distances, descriptors, days, k=10):
    """1 - neighbor squared difference / eligible random-pair difference.

    All neighbors are from different UTC start days. Reference uses every
    eligible pair per query, so meteorological near-duplicates cannot win via
    their own day. Scaling, fitted on training descriptors, cancels per column.
    """
    n = len(distances)
    gains = np.full_like(descriptors, np.nan, dtype=np.float64)
    neighbors = np.full((n, k), -1, dtype=int)
    for index in range(n):
        eligible = np.flatnonzero(days != days[index])
        if len(eligible) < k:
            continue
        selected = eligible[np.argsort(distances[index, eligible], kind='stable')[:k]]
        neighbors[index] = selected
        baseline = np.square(descriptors[eligible] - descriptors[index]).mean(0)
        error = np.square(descriptors[selected] - descriptors[index]).mean(0)
        gains[index] = 1 - np.divide(error, baseline, out=np.full_like(error, np.nan), where=baseline > 1e-12)
    return gains, neighbors


def day_mean_interval(values, days, seed=20261005, repeats=400):
    """Equal-day mean and day-block bootstrap percentile interval, descriptive."""
    day_values = np.asarray([np.nanmean(values[days == day], axis=0) for day in np.unique(days)])
    mean = np.nanmean(day_values, axis=0)
    rng = np.random.default_rng(seed)
    sampled = day_values[rng.integers(len(day_values), size=(repeats, len(day_values)))].mean(1)
    low, high = np.nanquantile(sampled, [.025, .975], axis=0)
    return mean, low, high


def stability_scores(values, days, method, counts, reference, repeats=8, seed=20261005):
    """Recluster 80% of whole days, compare to full partition on common clips."""
    unique = np.unique(days)
    if len(unique) < 3:
        return {count: np.nan for count in counts}
    rng = np.random.default_rng(seed)
    scores = {count: [] for count in counts}
    for _ in range(repeats):
        kept_days = rng.choice(unique, max(2, int(.8 * len(unique))), replace=False)
        indices = np.flatnonzero(np.isin(days, kept_days))
        tree, _ = hierarchy(values[indices], method)
        for count in counts:
            if count >= len(indices):
                continue
            labels = cut_tree(tree, n_clusters=count).ravel()
            scores[count].append(adjusted_rand_score(reference[count][indices], labels))
    return {count: float(np.mean(score)) if score else np.nan for count, score in scores.items()}


def silhouette(distances, labels, seed=20261005):
    if not 1 < len(np.unique(labels)) < len(labels):
        return np.nan
    # Deterministic subset; no PCA/2D layout is used to measure separation.
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(labels), min(1024, len(labels)), replace=False))
    if not 1 < len(np.unique(labels[chosen])) < len(chosen):
        return np.nan
    return float(silhouette_score(distances[np.ix_(chosen, chosen)], labels[chosen], metric='precomputed'))
