"""Cluster means and their physical differences, independent of explained variance."""
import numpy as np


def physical_profiles(values, labels, training_mean, training_std, families, min_count=5):
    """RMS distance between means in units of training descriptor std.

    This standardization is for diagnostic descriptors, never encoder inputs.
    Constant training descriptors are excluded. Negative labels are unassigned.
    Pairs involving groups smaller than min_count are not evaluated.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    mean, std = np.asarray(training_mean), np.asarray(training_std)
    if (values.ndim != 2 or labels.shape != (len(values),) or
        mean.shape != (values.shape[1],) or std.shape != mean.shape or
        len(families) != values.shape[1] or min_count < 1 or
        not np.isfinite(values).all() or not np.isfinite(mean).all() or
        not np.isfinite(std).all() or np.any(std < 0)):
        raise ValueError('Invalid descriptor/label/training-stat alignment')
    groups = np.unique(labels[labels >= 0])
    counts = np.array([np.sum(labels == group) for group in groups])
    means = (np.stack([values[labels == group].mean(0) for group in groups])
             if len(groups) else np.empty((0, values.shape[1])))
    distances = {}
    for family in dict.fromkeys(families):
        columns = np.array([name == family for name in families]) & (std > 1e-12)
        if not columns.any():
            distances[family] = np.full((len(groups), len(groups)), np.nan)
            continue
        standardized = (means[:, columns] - mean[columns]) / std[columns]
        difference = standardized[:, None, :] - standardized[None, :, :]
        distance = np.sqrt(np.square(difference).mean(2))
        distance[(counts[:, None] < min_count) | (counts[None, :] < min_count)] = np.nan
        distances[family] = distance
    return groups, counts, means, distances


def physical_dispersion(values, labels, training_mean, training_std, families, min_count=5):
    """Empirical RMS pair distances: diagonal within, off-diagonal across groups.

    Draws are independent with replacement within each empirical cluster.
    E[(X_i-X_j)^2] = (mu_i-mu_j)^2 + var_i + var_j (ddof=0).
    Centroid summary weights valid unordered pairs equally; within summary
    weights valid clusters by clip count. No assigned small group is repaired.
    """
    groups, counts, means, centroids = physical_profiles(
        values, labels, training_mean, training_std, families, min_count)
    values, labels = np.asarray(values, dtype=float), np.asarray(labels)
    std = np.asarray(training_std)
    variances = (np.stack([values[labels == group].var(0) for group in groups])
                 if len(groups) else np.empty_like(means))
    matrices, summaries = {}, {}
    for family, centroid in centroids.items():
        columns = np.array([name == family for name in families]) & (std > 1e-12)
        if columns.any():
            variance = (variances[:, columns] / std[columns]**2).mean(1)
            matrix = np.sqrt(centroid**2 + variance[:, None] + variance[None, :])
        else:
            matrix = centroid.copy()
        matrices[family] = matrix
        pairs = centroid[np.triu_indices(len(groups), 1)]
        pairs = pairs[np.isfinite(pairs)]
        between = np.sqrt(np.mean(pairs**2)) if len(pairs) else np.nan
        diagonal = np.diag(matrix)
        valid = np.isfinite(diagonal)
        within = np.sqrt(np.average(diagonal[valid]**2, weights=counts[valid])) if valid.any() else np.nan
        summaries[family] = dict(between=between, within=within,
            ratio=between/within if np.isfinite(within) and within > 1e-12 else np.nan)
    # Global is an arithmetic mean of the four family summaries, not all
    # descriptors pooled (families have different numbers of descriptors).
    summaries['global'] = {}
    for metric in ['between', 'within', 'ratio']:
        entries = np.array([row[metric] for row in summaries.values() if metric in row])
        summaries['global'][metric] = float(entries.mean()) if np.isfinite(entries).all() else np.nan
    return matrices, summaries


def physical_separation(values, labels, training_mean, training_std, families, min_count=5):
    """Centroid RMS divided by RMS of the two empirical internal spreads.

    Zero pair dispersion is undefined off-diagonal, not infinite separation.
    Valid diagonals are zero by convention, not evidence of compactness.
    Summary is the mean over valid unordered pairs; global weights families
    equally and requires all family summaries to be valid.
    """
    groups, counts, means, centroids = physical_profiles(
        values, labels, training_mean, training_std, families, min_count)
    clip_distances, _ = physical_dispersion(
        values, labels, training_mean, training_std, families, min_count)
    separation, dispersion, summary = {}, {}, {}
    for family, centroid in centroids.items():
        diagonal = np.diag(clip_distances[family])
        variance = diagonal**2 / 2
        denominator = np.sqrt(variance[:, None] + variance[None, :])
        matrix = np.divide(centroid, denominator, out=np.full_like(centroid, np.nan),
                           where=np.isfinite(denominator) & (denominator > 1e-12))
        valid = np.isfinite(diagonal)
        indices = np.flatnonzero(valid)
        matrix[indices, indices] = 0.
        separation[family] = matrix
        dispersion[family] = np.sqrt(variance)
        pairs = matrix[np.triu_indices(len(groups), 1)]
        pairs = pairs[np.isfinite(pairs)]
        summary[family] = float(pairs.mean()) if len(pairs) else np.nan
    scores = np.array(list(summary.values()))
    summary['global'] = float(scores.mean()) if len(scores) and np.isfinite(scores).all() else np.nan
    return groups, counts, means, separation, dispersion, summary
