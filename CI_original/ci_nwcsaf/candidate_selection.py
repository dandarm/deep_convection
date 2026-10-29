"""Area of Interest (AOI) definition and Eligible-CI candidate pixel selection.

References:
- ATBD v2.1 Section 2.2.1.2.1 (Area of interest and eligible-CI)
- ATBD v2.1 Section 2.2.1.2.1.1 (NWP convective mask)
- ATBD v2.1 Section 2.2.1.2.1.2 (Cloud type as filtering mask)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from .config import NwcSafCiConfig, CHANNEL_INDICES


def compute_pure_ir_cloud_mask(
    seviri_cube: np.ndarray,
    clear_sky_temp_est: Optional[np.ndarray] = None,
    split_window_thresh_k: float = 0.3,
    cold_cloud_threshold_k: float = 283.15,
) -> np.ndarray:
    """Computes a pure-IR cloud mask using SEVIRI IR10.8 and IR12.0 split-window channels.

    Parameters
    ----------
    seviri_cube : np.ndarray
        Array of shape (7, H, W) in Kelvin.
    clear_sky_temp_est : np.ndarray, optional
        Estimated clear-sky surface temperature (H, W). If None, estimated from high percentiles.
    split_window_thresh_k : float
        Minimum positive split-window difference (IR10.8 - IR12.0) for cloudy atmosphere.
    cold_cloud_threshold_k : float
        Absolute brightness temperature below which a pixel is definitively cloudy (default 283.15 K / 10°C).

    Returns
    -------
    np.ndarray
        Boolean mask of shape (H, W), True for cloudy pixels.
    """
    ir108 = seviri_cube[CHANNEL_INDICES["IR_108"]]
    ir120 = seviri_cube[CHANNEL_INDICES["IR_120"]]

    split_diff = ir108 - ir120

    # Rule 1: Definitively cold cloud (T < 283.15 K)
    cold_clouds = ir108 < cold_cloud_threshold_k

    # Rule 2: Split-window differential absorption (typical of water/ice clouds over warm ground)
    split_clouds = (split_diff > split_window_thresh_k) & (ir108 < 295.0)

    # Rule 3: Clear-sky background depression if reference temperature is available
    if clear_sky_temp_est is not None:
        depression_clouds = ir108 < (clear_sky_temp_est - 5.0)
    else:
        depression_clouds = np.zeros_like(cold_clouds, dtype=bool)

    cloud_mask = cold_clouds | split_clouds | depression_clouds
    return cloud_mask


def compute_nwp_convective_mask(
    lifted_index: Optional[np.ndarray] = None,
    k_index: Optional[np.ndarray] = None,
    showalter_index: Optional[np.ndarray] = None,
    config: Optional[NwcSafCiConfig] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Computes the 3-state NWP convective mask (0: stable, 1: unclear, 2: unstable).

    From ATBD Section 2.2.1.2.1.1:
    - Full stable case (Mask = 0): LI > 0 AND SHW > 3 AND KI < 20
    - Unstable case (Mask = 2): LI < -3 OR SHW < -3 OR KI > 30
    - Unclear case (Mask = 1): all other cases

    Regions with Mask == 0 are excluded from CI processing.
    """
    if config is None:
        config = NwcSafCiConfig()

    if lifted_index is None and k_index is None and showalter_index is None:
        # No NWP provided: return unclear (1) everywhere so all cloudy pixels are evaluated
        if shape is None:
            raise ValueError("shape must be provided if no NWP fields are given")
        return np.ones(shape, dtype=np.uint8)

    # Determine shape from any available field
    ref_field = next(f for f in (lifted_index, k_index, showalter_index) if f is not None)
    out_shape = ref_field.shape

    # Default neutral values if specific index is missing
    li = lifted_index if lifted_index is not None else np.full(out_shape, -1.0)
    ki = k_index if k_index is not None else np.full(out_shape, 25.0)
    shw = showalter_index if showalter_index is not None else np.full(out_shape, 0.0)

    # Full stable check
    is_stable = (li > config.li_stable_thresh) & (shw > config.shw_stable_thresh) & (ki < config.ki_stable_thresh)

    # Unstable check
    is_unstable = (li < config.li_conv_thresh) | (shw < config.shw_conv_thresh) | (ki > config.ki_conv_thresh)

    mask = np.ones(out_shape, dtype=np.uint8)
    mask[is_stable] = 0
    mask[is_unstable] = 2
    return mask


def select_eligible_ci_pixels(
    seviri_cube: np.ndarray,
    config: Optional[NwcSafCiConfig] = None,
    cloud_mask: Optional[np.ndarray] = None,
    nwp_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Selects eligible-CI pixels from a SEVIRI 7-channel cube.

    Filters:
    1. Must be cloudy (via provided cloud_mask or autonomous pure-IR cloud mask).
    2. Must be in the warm/developing cloud temperature window:
       T_cold (248.15 K = -25°C) <= T_IR10.8 <= T_warm (283.15 K = +10°C).
       Pixels with T_IR10.8 < 248.15 K are excluded (already mature storms in RDT domain).
    3. Must not be in a fully stable NWP environment (NWP mask != 0).

    Returns
    -------
    eligible_mask : np.ndarray (bool)
        Boolean mask of shape (H, W), True for eligible-CI pixels.
    cloud_mask : np.ndarray (bool)
        Cloud mask of shape (H, W).
    """
    if config is None:
        config = NwcSafCiConfig()

    ir108 = seviri_cube[CHANNEL_INDICES["IR_108"]]

    # Step 1: Cloud mask
    if cloud_mask is None:
        cloud_mask = compute_pure_ir_cloud_mask(seviri_cube)

    # Step 2: Warm pre-convective temperature cap
    # ATBD: IR10.8 BT between TCOLD (-25°C) and TWARM (+10°C)
    in_temp_window = (ir108 >= config.t_cold_k) & (ir108 <= config.t_warm_k)

    # Step 3: NWP convective instability filter (if provided)
    if nwp_mask is not None:
        nwp_eligible = (nwp_mask > 0)
    else:
        nwp_eligible = np.ones_like(ir108, dtype=bool)

    eligible_mask = cloud_mask & in_temp_window & nwp_eligible
    return eligible_mask, cloud_mask
