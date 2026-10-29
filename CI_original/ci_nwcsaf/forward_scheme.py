"""Lagrangian forward advection scheme for GEO-CI.

References:
- ATBD v2.1 Section 2.2.1.2.8 & Figure 4 (Forward scheme)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.ndimage import maximum_filter
from .config import NwcSafCiConfig


def apply_forward_scheme(
    base_diagnosis: np.ndarray,
    motion_u_px_per_min: np.ndarray,
    motion_v_px_per_min: np.ndarray,
    lead_time_min: float,
    footprint_dilation_px: int = 1,
) -> np.ndarray:
    """Spreads the CI diagnosis along the forward trajectory defined by the 2D movement field.

    Generates a continuous solid forecast corridor (swath) along the advected path,
    ensuring no discrete disconnected dots or string-of-pearls artifacts.

    Parameters
    ----------
    base_diagnosis : np.ndarray
        Diagnosed CI probability classes (H, W) in {0, 1, 2, 3, 4}.
    motion_u_px_per_min, motion_v_px_per_min : np.ndarray
        2D movement field in pixels per minute.
    lead_time_min : float
        Forecast horizon (e.g. 30.0, 60.0, or 90.0 minutes).
    footprint_dilation_px : int
        Cross-track footprint dilation width in pixels.

    Returns
    -------
    spread_prob : np.ndarray
        Forward-advected CI probability map of shape (H, W).
    """
    h, w = base_diagnosis.shape
    spread_prob = base_diagnosis.copy()

    # Find all pixels with positive initiation probability
    ci_rows, ci_cols = np.where(base_diagnosis > 0)
    if len(ci_rows) == 0:
        return spread_prob

    u_act = motion_u_px_per_min[ci_rows, ci_cols]
    v_act = motion_v_px_per_min[ci_rows, ci_cols]
    p_act = base_diagnosis[ci_rows, ci_cols][:, None]

    # Maximum displacement across all initiating pixels
    max_disp = float(np.max(np.hypot(u_act * lead_time_min, v_act * lead_time_min)))

    # Continuous sub-pixel stepping (step <= 0.75 px) guarantees zero spatial gaps along trajectory
    num_steps = max(2, int(np.ceil(max_disp / 0.75)))
    alphas = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)

    r0 = ci_rows[:, None]
    c0 = ci_cols[:, None]

    r_path = np.round(r0 + v_act[:, None] * (lead_time_min * alphas)).astype(np.int32)
    c_path = np.round(c0 + u_act[:, None] * (lead_time_min * alphas)).astype(np.int32)

    rr = r_path.ravel()
    cc = c_path.ravel()
    pp = np.repeat(p_act, num_steps, axis=1).ravel()

    valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    if np.any(valid):
        np.maximum.at(spread_prob, (rr[valid], cc[valid]), pp[valid])

    # Footprint dilation along and across the corridor
    if footprint_dilation_px > 0:
        size = 2 * footprint_dilation_px + 1
        footprint = np.ones((size, size), dtype=bool)
        spread_prob = maximum_filter(spread_prob, footprint=footprint)

    return spread_prob


def generate_ci_forecast_horizons(
    base_diagnosis: np.ndarray,
    motion_u_px_per_min: np.ndarray,
    motion_v_px_per_min: np.ndarray,
    config: Optional[NwcSafCiConfig] = None,
) -> Dict[str, np.ndarray]:
    """Generates the three official CI forecast containers: ci_prob30, ci_prob60, ci_prob90.

    Enforces the cumulative probability monotonicity constraint:
    ci_prob30 <= ci_prob60 <= ci_prob90.
    """
    if config is None:
        config = NwcSafCiConfig()

    prob30 = apply_forward_scheme(base_diagnosis, motion_u_px_per_min, motion_v_px_per_min, 30.0)
    prob60 = apply_forward_scheme(base_diagnosis, motion_u_px_per_min, motion_v_px_per_min, 60.0)
    prob90 = apply_forward_scheme(base_diagnosis, motion_u_px_per_min, motion_v_px_per_min, 90.0)

    # Monotonicity: P30 <= P60 <= P90
    prob60 = np.maximum(prob60, prob30)
    prob90 = np.maximum(prob90, prob60)

    return {
        "ci_prob30": prob30,
        "ci_prob60": prob60,
        "ci_prob90": prob90,
    }
