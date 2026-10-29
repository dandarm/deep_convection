"""Empirical rules and CI diagnosis classification.

References:
- ATBD v2.1 Section 2.2.1.2.7.3 & Table 6 (Probability assessment - Empirical rules)
- User Manual v1.0 Section 2.1.3 & Table 3 (CI NetCDF output containers)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple
from .config import CiProbabilityClass


def diagnose_ci_probability_pixel(
    n_growth: int,
    n_glaciation: int,
    n_height: int,
) -> int:
    """Evaluates the NWC SAF Table 6 empirical decision rules for a single parameter count tuple.

    Returns
    -------
    int : 0 (NO_PROB), 1 (VLOWPROB), 2 (LOWPROB), 3 (MODPROB), or 4 (HIGHPROB).
    """
    if n_growth >= 2:
        if n_glaciation >= 3:
            if n_height >= 4:
                return CiProbabilityClass.HIGHPROB
            elif n_height >= 3:
                return CiProbabilityClass.MODPROB
            else:
                return CiProbabilityClass.LOWPROB
        elif n_glaciation >= 2:
            if n_height >= 4:
                return CiProbabilityClass.MODPROB
            elif n_height >= 3:
                return CiProbabilityClass.LOWPROB
            else:
                return CiProbabilityClass.VLOWPROB
        else:
            return CiProbabilityClass.NO_PROB

    elif n_growth >= 1:
        if n_glaciation >= 3:
            if n_height >= 4:
                return CiProbabilityClass.MODPROB
            else:
                return CiProbabilityClass.LOWPROB
        elif n_glaciation >= 2:
            if n_height >= 4:
                return CiProbabilityClass.LOWPROB
            elif n_height >= 3:
                return CiProbabilityClass.VLOWPROB
            else:
                return CiProbabilityClass.NO_PROB
        else:
            return CiProbabilityClass.NO_PROB

    elif n_growth == 0:
        if n_glaciation >= 3:
            if n_height >= 3:
                return CiProbabilityClass.LOWPROB
            else:
                return CiProbabilityClass.VLOWPROB
        else:
            return CiProbabilityClass.NO_PROB

    return CiProbabilityClass.NO_PROB


def evaluate_ci_diagnosis(
    n_growth: np.ndarray,
    n_glaciation: np.ndarray,
    n_height: np.ndarray,
) -> np.ndarray:
    """Vectorized evaluation of Table 6 across full 2D arrays.

    Parameters
    ----------
    n_growth : np.ndarray (shape H, W)
    n_glaciation : np.ndarray (shape H, W)
    n_height : np.ndarray (shape H, W)

    Returns
    -------
    ci_prob : np.ndarray (uint8, shape H, W)
        Values in {0, 1, 2, 3, 4}.
    """
    ci_prob = np.zeros(n_growth.shape, dtype=np.uint8)

    # Branch 1: n_growth >= 2
    g2 = (n_growth >= 2)
    # Sub-branch 1.1: glaciation >= 3
    ci_prob[g2 & (n_glaciation >= 3) & (n_height >= 4)] = CiProbabilityClass.HIGHPROB
    ci_prob[g2 & (n_glaciation >= 3) & (n_height == 3)] = CiProbabilityClass.MODPROB
    ci_prob[g2 & (n_glaciation >= 3) & (n_height < 3)] = CiProbabilityClass.LOWPROB
    # Sub-branch 1.2: glaciation == 2
    ci_prob[g2 & (n_glaciation == 2) & (n_height >= 4)] = CiProbabilityClass.MODPROB
    ci_prob[g2 & (n_glaciation == 2) & (n_height == 3)] = CiProbabilityClass.LOWPROB
    ci_prob[g2 & (n_glaciation == 2) & (n_height < 3)] = CiProbabilityClass.VLOWPROB

    # Branch 2: n_growth == 1
    g1 = (n_growth == 1)
    # Sub-branch 2.1: glaciation >= 3
    ci_prob[g1 & (n_glaciation >= 3) & (n_height >= 4)] = CiProbabilityClass.MODPROB
    ci_prob[g1 & (n_glaciation >= 3) & (n_height < 4)] = CiProbabilityClass.LOWPROB
    # Sub-branch 2.2: glaciation == 2
    ci_prob[g1 & (n_glaciation == 2) & (n_height >= 4)] = CiProbabilityClass.LOWPROB
    ci_prob[g1 & (n_glaciation == 2) & (n_height == 3)] = CiProbabilityClass.VLOWPROB

    # Branch 3: n_growth == 0
    g0 = (n_growth == 0)
    ci_prob[g0 & (n_glaciation >= 3) & (n_height >= 3)] = CiProbabilityClass.LOWPROB
    ci_prob[g0 & (n_glaciation >= 3) & (n_height < 3)] = CiProbabilityClass.VLOWPROB

    return ci_prob


def enforce_lead_time_monotonicity(
    prob30: np.ndarray,
    prob60: np.ndarray,
    prob90: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enforces the physical monotonicity constraint: P_30 <= P_60 <= P_90.

    ATBD Note p.23: "a final step will check that the probability value of CI
    for the range [0-60'] is above the probability of CI for the range [0-30']
    and below the probability of CI for the range [0-90']".
    """
    final_prob30 = prob30.copy()
    final_prob60 = np.maximum(prob60, final_prob30)
    final_prob90 = np.maximum(prob90, final_prob60)

    return final_prob30, final_prob60, final_prob90
