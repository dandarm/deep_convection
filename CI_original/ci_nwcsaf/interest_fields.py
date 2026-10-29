"""Computation of Brightness Temperature Differences (BTD) and Lagrangian trend interest fields.

References:
- ATBD v2.1 Section 2.2.1.2.4 (Pixel image-analysis: BTD and trends)
- ATBD v2.1 Section 2.2.1.2.6 (Definition of Pixel pre-CI)
- ATBD v2.1 Section 2.2.1.2.7.1 & Tables 4, 5 (Interest fields thresholds)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.ndimage import map_coordinates

from .config import NwcSafCiConfig, CHANNEL_INDICES


@dataclass
class CiInterestFields:
    """Contains calculated static BTDs and dynamic Lagrangian trends."""
    # Static BTDs (Kelvin or Celsius)
    ir108: np.ndarray       # IR 10.8 µm brightness temperature (H, W)
    wbtd: np.ndarray        # WV6.2 - WV7.3 (H, W)
    btd: np.ndarray         # WV6.2 - IR10.8 (H, W)
    btd4: np.ndarray        # IR10.8 - IR8.7 (H, W)
    btd5: np.ndarray        # IR12.0 - IR10.8 (H, W)
    btd6: np.ndarray        # IR13.4 - IR10.8 (H, W)

    # Dynamic Lagrangian trends (over 15-min window, in K/15m)
    txbt_15: np.ndarray     # d(IR10.8)/dt (cooling rate)
    txbtd_15: np.ndarray    # d(WV6.2 - IR10.8)/dt
    txbtd4_15: np.ndarray   # d(IR10.8 - IR8.7)/dt
    txbtd5_15: np.ndarray   # d(IR12.0 - IR10.8)/dt
    txbtd6_15: np.ndarray   # d(IR13.4 - IR10.8)/dt

    # Glaciation proxy
    supercooled_flag: np.ndarray  # Boolean mask of pixels in freezing/glaciation zone


def compute_static_btds(seviri_cube: np.ndarray) -> Dict[str, np.ndarray]:
    """Computes the 5 core NWC SAF BTD fields from a 7-channel SEVIRI cube."""
    wv62 = seviri_cube[CHANNEL_INDICES["WV_062"]]
    wv73 = seviri_cube[CHANNEL_INDICES["WV_073"]]
    ir87 = seviri_cube[CHANNEL_INDICES["IR_087"]]
    ir108 = seviri_cube[CHANNEL_INDICES["IR_108"]]
    ir120 = seviri_cube[CHANNEL_INDICES["IR_120"]]
    ir134 = seviri_cube[CHANNEL_INDICES["IR_134"]]

    return {
        "ir108": ir108,
        "wbtd": wv62 - wv73,
        "btd": wv62 - ir108,
        "btd4": ir108 - ir87,
        "btd5": ir120 - ir108,
        "btd6": ir134 - ir108,
    }


def lagrangian_backward_advection(
    field_past: np.ndarray,
    motion_u_px_per_min: np.ndarray,
    motion_v_px_per_min: np.ndarray,
    delta_t_min: float = 15.0,
) -> np.ndarray:
    """Samples field_past at backward-advected parcel positions (x - u*dt, y - v*dt)."""
    h, w = field_past.shape
    grid_y, grid_x = np.indices((h, w), dtype=np.float32)

    past_y = np.clip(grid_y - motion_v_px_per_min * delta_t_min, 0.0, h - 1.0)
    past_x = np.clip(grid_x - motion_u_px_per_min * delta_t_min, 0.0, w - 1.0)

    coordinates = np.array([past_y, past_x])
    advected_field = map_coordinates(field_past, coordinates, order=1, mode="nearest")
    return advected_field


def compute_ci_interest_fields(
    current_cube: np.ndarray,
    past_cube: Optional[np.ndarray],
    motion_u_px_per_min: np.ndarray,
    motion_v_px_per_min: np.ndarray,
    delta_t_min: float = 15.0,
    config: Optional[NwcSafCiConfig] = None,
) -> CiInterestFields:
    """Computes all static and dynamic interest fields required for CI diagnosis.

    Parameters
    ----------
    current_cube : np.ndarray
        Current SEVIRI 7-channel cube of shape (7, H, W) in Kelvin.
    past_cube : Optional[np.ndarray]
        SEVIRI cube from previous slot (e.g. 15 min or 5 min ago). If None, trends are zero.
    motion_u_px_per_min, motion_v_px_per_min : np.ndarray
        2D movement field in pixels per minute.
    delta_t_min : float
        Actual time elapsed between past_cube and current_cube in minutes.
    config : Optional[NwcSafCiConfig]
        Configuration object.

    Returns
    -------
    CiInterestFields dataclass containing all evaluated fields.
    """
    if config is None:
        config = NwcSafCiConfig()

    curr_btds = compute_static_btds(current_cube)
    ir108 = curr_btds["ir108"]

    h, w = ir108.shape

    # Supercooled water / freezing zone: cloud top temperature between 0°C (273.15 K) and -25°C (248.15 K)
    supercooled_flag = (ir108 <= config.t_freeze_k) & (ir108 >= config.t_cold_k)

    if past_cube is None:
        # Cold start / no history available: zero trends
        txbt_15 = np.zeros((h, w), dtype=np.float32)
        txbtd_15 = np.zeros((h, w), dtype=np.float32)
        txbtd4_15 = np.zeros((h, w), dtype=np.float32)
        txbtd5_15 = np.zeros((h, w), dtype=np.float32)
        txbtd6_15 = np.zeros((h, w), dtype=np.float32)
    else:
        past_btds = compute_static_btds(past_cube)
        scale_15 = 15.0 / max(1.0, float(delta_t_min))

        # Backward advect each past field along the motion vector
        adv_past_ir108 = lagrangian_backward_advection(past_btds["ir108"], motion_u_px_per_min, motion_v_px_per_min, delta_t_min)
        adv_past_btd = lagrangian_backward_advection(past_btds["btd"], motion_u_px_per_min, motion_v_px_per_min, delta_t_min)
        adv_past_btd4 = lagrangian_backward_advection(past_btds["btd4"], motion_u_px_per_min, motion_v_px_per_min, delta_t_min)
        adv_past_btd5 = lagrangian_backward_advection(past_btds["btd5"], motion_u_px_per_min, motion_v_px_per_min, delta_t_min)
        adv_past_btd6 = lagrangian_backward_advection(past_btds["btd6"], motion_u_px_per_min, motion_v_px_per_min, delta_t_min)

        # 15-minute equivalent Lagrangian trends
        txbt_15 = (curr_btds["ir108"] - adv_past_ir108) * scale_15
        txbtd_15 = (curr_btds["btd"] - adv_past_btd) * scale_15
        txbtd4_15 = (curr_btds["btd4"] - adv_past_btd4) * scale_15
        txbtd5_15 = (curr_btds["btd5"] - adv_past_btd5) * scale_15
        txbtd6_15 = (curr_btds["btd6"] - adv_past_btd6) * scale_15

    return CiInterestFields(
        ir108=ir108,
        wbtd=curr_btds["wbtd"],
        btd=curr_btds["btd"],
        btd4=curr_btds["btd4"],
        btd5=curr_btds["btd5"],
        btd6=curr_btds["btd6"],
        txbt_15=txbt_15,
        txbtd_15=txbtd_15,
        txbtd4_15=txbtd4_15,
        txbtd5_15=txbtd5_15,
        txbtd6_15=txbtd6_15,
        supercooled_flag=supercooled_flag,
    )


def count_relevant_parameters(
    fields: CiInterestFields,
    eligible_mask: np.ndarray,
    config: Optional[NwcSafCiConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates threshold criteria across the 3 physical families for each pixel.

    Parameters
    ----------
    fields : CiInterestFields
    eligible_mask : np.ndarray (bool)
    config : Optional[NwcSafCiConfig]

    Returns
    -------
    n_growth : np.ndarray (uint8, shape H,W) in 0..3
    n_glaciation : np.ndarray (uint8, shape H,W) in 0..3
    n_height : np.ndarray (uint8, shape H,W) in 0..4
    pre_ci_mask : np.ndarray (bool, shape H,W)
    """
    if config is None:
        config = NwcSafCiConfig()

    is_night_or_pure = (config.mode == "night_or_pure_ir")

    # Select thresholds according to mode (Table 4 day vs Table 5 night/pure IR)
    btd4_min = config.btd4_min_night_k if is_night_or_pure else config.btd4_min_day_k
    btd4_max = config.btd4_max_night_k if is_night_or_pure else config.btd4_max_day_k

    btd_min = config.btd_min_night_k if is_night_or_pure else config.btd_min_day_k
    btd_max = config.btd_max_night_k if is_night_or_pure else config.btd_max_day_k

    btd5_min = config.btd5_min_night_k if is_night_or_pure else config.btd5_min_day_k
    btd5_max = config.btd5_max_night_k if is_night_or_pure else config.btd5_max_day_k

    btd6_min = config.btd6_min_night_k if is_night_or_pure else config.btd6_min_day_k
    btd6_max = config.btd6_max_night_k if is_night_or_pure else config.btd6_max_day_k

    wbtd_min = config.wbtd_min_night_k if is_night_or_pure else config.wbtd_min_day_k
    wbtd_max = config.wbtd_max_night_k if is_night_or_pure else config.wbtd_max_day_k

    # 1. Glaciation Family (3 parameters)
    # Param 1: BT IR10.8 > -25°C and > -20°C (or supercooled transition)
    # ATBD: IR10.8 in [248.15, 253.15] K or cooling in supercooled zone
    glac_p1 = (fields.ir108 >= config.t_cold_k) & (fields.ir108 <= config.t_glaciation_focus_k)
    # Param 2: BTFZG (freezing level transition: in supercooled zone [248.15, 273.15] with active cooling)
    glac_p2 = fields.supercooled_flag & (fields.txbt_15 <= -1.0)
    # Param 3: BTD4 (IR10.8 - IR8.7)
    glac_p3 = (fields.btd4 >= btd4_min) & (fields.btd4 <= btd4_max)

    n_glaciation = (glac_p1.astype(np.uint8) + glac_p2.astype(np.uint8) + glac_p3.astype(np.uint8))

    # 2. Height / Vertical Depth Family (4 parameters)
    # Param 1: BTD (WV6.2 - IR10.8)
    h_p1 = (fields.btd >= btd_min) & (fields.btd <= btd_max)
    # Param 2: BTD5 (IR12.0 - IR10.8)
    h_p2 = (fields.btd5 >= btd5_min) & (fields.btd5 <= btd5_max)
    # Param 3: BTD6 (IR13.4 - IR10.8)
    h_p3 = (fields.btd6 >= btd6_min) & (fields.btd6 <= btd6_max)
    # Param 4: WBTD (WV6.2 - WV7.3)
    h_p4 = (fields.wbtd >= wbtd_min) & (fields.wbtd <= wbtd_max)

    n_height = (h_p1.astype(np.uint8) + h_p2.astype(np.uint8) + h_p3.astype(np.uint8) + h_p4.astype(np.uint8))

    # 3. Growth / Updraft Family (3 parameters)
    # Indicator 1: Strong IR10.8 cooling rate: TxBT_15 in [-50, -4] K/15m
    grow_p1 = (fields.txbt_15 >= config.txbt_15_min_k) & (fields.txbt_15 <= config.txbt_15_max_k)
    # Indicator 2: WV/IR cloud top deepening: TxBTD_15 > 3.0 K/15m OR TxBTD6_15 > 3.0 K/15m
    grow_p2 = (fields.txbtd_15 >= config.txbtd_15_min_k) | (fields.txbtd6_15 >= config.txbtd6_15_min_k)
    # Indicator 3: Optical / microphysical thickening: TxBTD4 in [0, 10] OR TxBTD5 in [0, 10]
    grow_p3 = ((fields.txbtd4_15 >= config.txbtd4_15_min_k) & (fields.txbtd4_15 <= config.txbtd4_15_max_k)) | \
              ((fields.txbtd5_15 >= config.txbtd5_15_min_k) & (fields.txbtd5_15 <= config.txbtd5_15_max_k))

    n_growth = (grow_p1.astype(np.uint8) + grow_p2.astype(np.uint8) + grow_p3.astype(np.uint8))

    # Zero-out non-eligible pixels
    n_growth[~eligible_mask] = 0
    n_glaciation[~eligible_mask] = 0
    n_height[~eligible_mask] = 0

    # Pre-CI definition: eligible pixel having at least one relevant parameter
    pre_ci_mask = eligible_mask & ((n_growth + n_glaciation + n_height) >= 1)

    return n_growth, n_glaciation, n_height, pre_ci_mask
