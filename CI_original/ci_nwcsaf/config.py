"""Configuration and operational parameters for NWC SAF GEO-CI (Convection Initiation).

References:
- ATBD v2.1 (NWC/CDOP2/GEO/MFT/SCI/ATBD/Convection, Issue 2.1)
- User Manual v1.0 (NWC/CDOP3/GEO/MF-PI/SCI/UM/Convection, Issue 1.0)
- NWC SAF GEO v2025 CI Product Description
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Tuple


class CiProbabilityClass(IntEnum):
    """Discrete CI probability classes according to NWC SAF DOF & ATBD Table 7."""
    NO_PROB = 0      # 0% probability (non-convective / stable)
    VLOWPROB = 1     # 0-25% probability of initiation
    LOWPROB = 2      # 25-50% probability of initiation
    MODPROB = 3      # 50-75% probability of initiation
    HIGHPROB = 4     # 75-100% probability of initiation


# SEVIRI 7-channel standard index map (matching emma_gpm.seviri.SEVIRI_CHANNELS)
SEVIRI_CHANNELS: Tuple[str, ...] = (
    "WV_062",
    "WV_073",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
)

CHANNEL_INDICES: Dict[str, int] = {name: idx for idx, name in enumerate(SEVIRI_CHANNELS)}


@dataclass(frozen=True)
class NwcSafCiConfig:
    """Operational parameters for NWC SAF GEO-CI Convection Initiation processor."""

    # 1. Candidate selection & temperature window (in Kelvin)
    # TCOLD = -25.0 °C (248.15 K), TWARM = +10.0 °C (283.15 K)
    t_cold_k: float = 248.15
    t_warm_k: float = 283.15
    t_freeze_k: float = 273.15  # 0.0 °C
    t_glaciation_focus_k: float = 253.15  # -20.0 °C

    # Slicing parameters for warm cloud cell detection
    delta_t_k: float = 1.0  # Temperature slicing step
    delta_t_tour_k: float = 3.0  # Minimum vertical prominence (budding extension)
    s_min_km2: float = 1.0  # Minimum detection area
    s_max_km2: float = 400_000.0  # Maximum detection area
    pixel_size_km: float = 3.0  # Nominal pixel pitch in km (at subsatellite point)

    # Tracking & motion parameters
    min_aor_tracking: float = 0.15  # Area Overlap Ratio threshold
    motion_filter_radius_px: int = 24  # Influence box for blending (ATBD sec 2.2.1.2.2)
    default_u_mps: float = 0.0  # Default zonal advection if no motion available
    default_v_mps: float = 0.0  # Default meridional advection

    # Physical Interest Fields Mode: 'night_or_pure_ir' (default) or 'day_with_cmic'
    mode: str = "night_or_pure_ir"

    # Glaciation Family Thresholds
    # BTD4 = IR10.8 - IR8.7
    btd4_min_night_k: float = -2.1
    btd4_max_night_k: float = 0.0
    btd4_min_day_k: float = -10.0
    btd4_max_day_k: float = 0.0
    btfzg_max_min: float = 30.0  # Time since crossing 0°C (freezing level)

    # Height / Vertical Extension Family Thresholds
    # BTD = WV6.2 - IR10.8
    btd_min_night_k: float = -34.0
    btd_max_night_k: float = -12.0
    btd_min_day_k: float = -35.0
    btd_max_day_k: float = -10.0

    # BTD5 = IR12.0 - IR10.8
    btd5_min_night_k: float = -2.2
    btd5_max_night_k: float = 0.0
    btd5_min_day_k: float = -3.0
    btd5_max_day_k: float = 0.0

    # BTD6 = IR13.4 - IR10.8
    btd6_min_night_k: float = -17.0
    btd6_max_night_k: float = -6.0
    btd6_min_day_k: float = -25.0
    btd6_max_day_k: float = -5.0

    # WBTD = WV6.2 - WV7.3
    wbtd_min_night_k: float = -17.0
    wbtd_max_night_k: float = -7.0
    wbtd_min_day_k: float = -25.0
    wbtd_max_day_k: float = -3.0

    # Growth / Updraft Family Thresholds (Lagrangian trends over 15-min)
    # TxBT 15' = d(IR10.8)/dt (cooling rate: between -50 and -4 K/15m)
    txbt_15_min_k: float = -50.0
    txbt_15_max_k: float = -4.0

    # TxBTD 15' = d(WV6.2 - IR10.8)/dt (> +3.0 K/15m)
    txbtd_15_min_k: float = 3.0

    # TxBTD4 15' = d(IR10.8 - IR8.7)/dt in ]0, 10[ K/15m
    txbtd4_15_min_k: float = 0.0
    txbtd4_15_max_k: float = 10.0

    # TxBTD5 15' = d(IR12.0 - IR10.8)/dt in ]0, 10[ K/15m
    txbtd5_15_min_k: float = 0.0
    txbtd5_15_max_k: float = 10.0

    # TxBTD6 15' = d(IR13.4 - IR10.8)/dt (> +3.0 K/15m)
    txbtd6_15_min_k: float = 3.0

    # Forecast Horizons (Lead times in minutes)
    forecast_horizons_min: Tuple[int, ...] = (30, 60, 90)

    # Optional NWP convective instability thresholds (for LI, KI, SHW)
    li_conv_thresh: float = -3.0
    li_stable_thresh: float = 0.0
    ki_conv_thresh: float = 30.0
    ki_stable_thresh: float = 20.0
    shw_conv_thresh: float = -3.0
    shw_stable_thresh: float = 3.0
