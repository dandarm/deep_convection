"""SEVIRI-only variability sampling; no cloud classes or teacher labels."""
import numpy as np
import torch

from .advection import estimate_translation, warp_translation


def stratified_indices(scores, count, *, uniform_fraction=.5, seed=20261024):
    """Without replacement: explicit uniform subset, then variability strata.

Stratum draw probabilities are proportional to 1/2/4 for low/medium/high,
independently of stratum size. This is not an estimate of weather prevalence.
The returned order is deterministic and independent of subsequent trainers.
"""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 1 <= count <= len(scores):
        raise ValueError("Finite candidate scores and count within the pool required")
    if not 0 < uniform_fraction <= 1:
        raise ValueError("A nonzero uniform sampling component is required")
    rng = np.random.default_rng(seed)
    edges = np.quantile(scores, [1/3, 2/3])
    strata = np.searchsorted(edges, scores, side="right")
    uniform_count = min(count, max(1, int(np.ceil(count * uniform_fraction))))
    uniform = rng.choice(len(scores), uniform_count, replace=False)
    remaining = np.setdiff1d(np.arange(len(scores)), uniform)
    weights = np.zeros(len(remaining))
    for level, mass in enumerate((1., 2., 4.)):
        members = strata[remaining] == level
        if members.any():
            weights[members] = mass / members.sum()
    extra_count = count - uniform_count
    extra = (rng.choice(remaining, extra_count, replace=False, p=weights / weights.sum())
             if extra_count else np.array([], dtype=int))
    selected = np.concatenate((uniform, extra))
    rng.shuffle(selected)
    return selected, dict(seed=seed, uniform_count=uniform_count, variability_count=extra_count,
                         uniform_fraction=uniform_fraction, quantile_edges_k=edges.tolist(),
                         pool_stratum_counts=np.bincount(strata, minlength=3).tolist(),
                         selected_stratum_counts=np.bincount(strata[selected], minlength=3).tolist())


@torch.no_grad()
def compensated_variability(history_k, max_shift=12):
    """Mean |IR108 residual| after adjacent historical translation compensation.

Both images in each pair are historical. Only pixels with an observed source
footprint contribute. This is an observational sampling score, not convection.
"""
    sums = torch.zeros(len(history_k), device=history_k.device)
    counts = torch.zeros_like(sums)
    for index in range(1, history_k.shape[1]):
        previous, current = history_k[:, index - 1, 4], history_k[:, index, 4]
        shift = estimate_translation(previous, current, max_shift)
        warped, mask = warp_translation(previous[:, None], shift)
        sums += ((current - warped[:, 0]).abs() * mask[:, 0]).sum((-2, -1))
        counts += mask[:, 0].sum((-2, -1))
    if (counts == 0).any():
        raise ValueError("No valid overlap for compensated variability")
    return sums / counts
