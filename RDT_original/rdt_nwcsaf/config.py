"""Official NWC SAF GEO-RDT-CW configuration parameters and vertical categories.

References:
- NWC/CDOP2/GEO/MFT/SCI/ATBD/Convection v1.1 (2016) & v2.1 (2019)
- NWC/CDOP3/GEO/MF-PI/SCI/UM/Convection v1.0 (2019), Table 5 (pp. 30-31)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math


VERTICAL_CATEGORY_NAMES = {
    0: "Mature",
    1: "Mature_Transition",
    2: "Cold_Transition",
    3: "Warm1_Transition",
    4: "Warm2_Transition",
    5: "Warm",
}

PHASE_NAMES = {
    0: "Triggering",
    1: "Growing",
    2: "Mature",
    3: "Decaying",
    4: "Triggering_from_Split",
}

SEVERITY_NAMES = {
    0: "NIL",
    1: "Low",
    2: "Moderate",
    3: "High",
    4: "Very_High",
    5: "Unknown",
}

HAZARD_NAMES = {
    0: "NIL",
    1: "Turbulence",
    2: "Lightning",
    3: "Icing",
    4: "High_Ice_Water_Content",
    5: "Hail",
    6: "Heavy_Rain_Rate",
    7: "Unknown",
}

SEVIRI_7CH_NAMES = (
    "WV_062",
    "WV_073",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
)

SEMANTIC_CLASS_NAMES = {
    0: "background",
    1: "non_convective_cloud",
    2: "convective_triggering_growing",
    3: "convective_mature_decaying",
    4: "overshooting_top",
}

QUALITY_OBSERVATION_VALID = 1
QUALITY_TEMPORAL_VALID = 2
QUALITY_GROWTH_VALID = 4
QUALITY_PHASE_VALID = 8


@dataclass(frozen=True)
class NwcSafRdtConfig:
    """Configuration matching NWC SAF GEO-RDT-CW (v4.0/v5.0) operational parameters."""

    version: str = "nwcsaf_rdt_cw_v5.0"
    expected_cadence_minutes: float = 5.0
    history_slots: int = 18  # 18 slots * 5 min = 90 min history buffer (CAL VL regime)

    # --- 1. Pre-processing & stretched pixel mask (ATBD Sec 3.2.1.2.1.1, UM Table 5) ---
    nominal_pixel_area_km2: float = 9.0
    stretch_area_factor: float = 5.0  # Mask pixels > 5 * nominal_area (~45 km2 for MSG)
    enforce_stretched_mask: bool = True
    cma_enabled: bool = False
    ct_enabled: bool = False

    # --- 2. Multi-thresholding & cell identification (ATBD Sec 3.2.1.2.1, UM Table 5) ---
    tcold_k: float = 198.15  # TCOLD = -75.0 degC
    twarm_k: float = 278.15  # TWARM = +5.0 degC
    deltat_k: float = 1.0  # DELTATEMPE = 1.0 degC
    tower_prominence_k: float = 6.0  # DeltaT_tower = 6.0 degC (ATBD Fig. 10, p. 27)
    smin_km2: float = 60.0  # SMIN = 60 km2 recommended for MSG SEVIRI
    smax_km2: float = 400_000.0  # SMAX = 400,000 km2
    btlimit_k: float = 213.15  # BTLIMIT = -60.0 degC (upper/coldest limit for Base of Tower)
    limittropo: bool = True  # LIMITTROPO = 1 (use NWP tropopause T or UTLS estimate as BT limit)

    # Single-tower excessive base refinement (ATBD v1.1 Sec 3.2.1.2.1.2, Figs 10-12; v2.1 Fig 11)
    constriction_ratio_max: float = 0.7  # Case A: min(S_{n+1}/S_n) <= 0.7 for budding from deck
    max_base_top_ratio: float = 4.0  # Case B: S_BT < 4 * S_ST (v1.1 exact derivation; 3.0 in v2.1 text)

    # --- 3. Tracking & trajectory management (ATBD Sec 3.2.1.2.2) ---
    ovl_min: float = 0.15  # 15% minimum overlap ratio A(C cap C') / min(A(C), A(C'))
    corr_window_pixels: int = 25  # Local IR10.8 cross-correlation window radius/size
    corr_max_shift_pixels: int = 12  # Maximum displacement search in pixels
    corr_min_rho: float = 0.5  # Minimum correlation coefficient rho_max >= 0.5 (ATBD Fig. 14)
    small_cell_dilation_max_pixels: int = 5  # Step 4 enlargement only for small cells <= 5 pixels
    small_cell_dilation_iter: int = 1  # 1-pixel edge enlargement (3x3 square dilation)
    velocity_smoothing_alpha: float = 0.5  # V(C') = 0.5 * CC'/dt + 0.5 * V(C) (ATBD Fig. 14)

    # --- 4. Convective discrimination (ATBD Sec 3.2.1.2.3) ---
    discri_scheme: str = "CAL"  # "CAL" (v2018 calibrated), "GEN" (generic), or "EMPIRICAL"
    nwp_conv_filter: bool = True  # NWPCONV = 1 (when NWP LI/SHW/KI fields are supplied)
    li_conv_threshold: float = -3.0  # LICONV = -3
    li_noconv_threshold: float = 0.0  # LINOCONV = 0
    ki_conv_threshold: float = 30.0  # KICONV = 30
    ki_noconv_threshold: float = 20.0  # KINOCONV = 20
    shw_conv_threshold: float = -3.0  # SHWCONV = -3
    shw_noconv_threshold: float = 3.0  # SHWNOCONV = 3

    # Vertical stratification thresholds (ATBD Sec 3.2.1.2.3.1 & Annex Line I ch7, p. 58)
    cat_mature_tmin_k: float = 233.15  # -40.0 degC
    cat_mature_min_age_min: float = 30.0  # 30 min in v2.1 CAL (45 min in v1.1 GEN)
    cat1_tbt_k: float = 238.15  # -35.0 degC
    cat2_tmin_k: float = 238.15  # -35.0 degC
    cat2_tbt_k: float = 248.15  # -25.0 degC
    cat3_tmin_k: float = 248.15  # -25.0 degC
    cat3_tbt_k: float = 258.15  # -15.0 degC
    cat4_tmin_k: float = 258.15  # -15.0 degC
    cat4_tbt_k: float = 268.15  # -5.0 degC

    # Empirical declassification rules (ATBD Sec 3.2.1.2.3.5, pp. 43-44)
    declass_warm_hold_min: float = 30.0  # 30 min hold in Cat 5 Warm
    declass_transition_hold_min: float = 60.0  # 60 min hold in Cat 1-4 Transition & Cat 0 Mature
    declass_strong_warming_k_per_h: float = 12.0  # Strong warming rate (degC/h) triggering declass
    declass_mature_btd_wv_ir_k: float = -1.0  # BTD(WV6.2-IR10.8) < -1.0 degC with negative 60-min trend
    low_warming_max_k_per_h: float = 6.0  # Low warming threshold for Phase of Development Table 8

    # Forced discrimination overrides (ATBD Sec 3.2.1.2.3.6 - 3.2.1.2.3.8, p. 44)
    force_conv_on_otd: bool = True  # Mandatory OTD -> Convective override
    lgh_forced_min_strokes: int = 1  # LGH = 1
    tropical_discri_temp_k: float | None = None  # TROPICALDISCRI (e.g., 198.15 K = -75 degC when enabled)

    # --- 5. Overshooting Top Detection (OTD, ATBD Sec 3.2.1.2.5.8, pp. 49-50 & v1.1 p. 47) ---
    ot_cold_threshold_midlat_k: float = 223.15  # -50.0 degC (|lat| > 30 deg)
    ot_cold_threshold_tropics_k: float = 203.15  # -70.0 degC (|lat| <= 30 deg)
    ot_min_cell_btd_wv_ir_k: float = 0.0  # max(WV6.2 - IR10.8) > 0 degC
    ot_size_midlat_km: float = 20.0  # L_OT = 20 km in mid-latitudes
    ot_size_tropics_km: float = 50.0  # L_OT = 50 km in tropics
    ot_sep_midlat_km: float = 50.0  # Minimum inter-OT distance = 50 km in mid-latitudes
    ot_sep_tropics_km: float = 200.0  # Minimum inter-OT distance = 200 km in tropics
    ot_anvil_prominence_k: float = 6.0  # Surrounding anvil pixels warmer by > 6.0 degC
    ot_min_warm_surrounding: int = 2  # At least 2 of 16 explored ray pixels (v1.1 p. 47)
    ot_extension_delta_k: float = 3.0  # Horizontal OT extension: BT(p) - BT(p_OT) < 3.0 degC
    ot_confirm_tropo_strong_k: float = 5.0  # Condition A: BT_OT <= T_tropo - 5.0 degC
    ot_confirm_btd_wv_ir_k: float = 1.5  # Condition B1: WV6.2 - IR10.8 > +1.5 degC
    ot_confirm_vis_refl: float = 0.70  # Condition B2: VIS06/HRV reflectance > 70%
    ot_confirm_mean_diff_k: float = 6.0  # Condition B3: BT_OT <= T_mean(cell) - 6.0 degC

    # --- 6. Forecasting & output options (ATBD Sec 3.2.1.2.4, UM Table 5) ---
    fcst_max_min: int = 60  # FCST = 60 min
    fcst_step_min: int = 15  # FCSTEP = 15 min (+15, +30, +45, +60 min)
    smooth_pts: int = 0  # SMOOTHPTS = 0 (or >1 for contour vertex moving average)
    dilat: bool = False  # DILAT = 0 (radial expansion of forecast contours)
    parallax: bool = False  # PARALLAX = 0/1 (geostationary parallax shift at output)
    sub_satellite_lon_deg: float = 9.5  # MSG RSS sub-satellite longitude (9.5 deg E)

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and value is not None and not math.isfinite(value):
                raise ValueError(f"Non-finite parameter {key}: {value}")
        if not (150.0 <= self.tcold_k < self.twarm_k <= 320.0):
            raise ValueError("Require 150 <= tcold_k < twarm_k <= 320 K")
        if self.deltat_k <= 0 or self.tower_prominence_k < self.deltat_k:
            raise ValueError("Require deltat_k > 0 and tower_prominence_k >= deltat_k")
        if not (0.0 < self.smin_km2 < self.smax_km2):
            raise ValueError("Require 0 < smin_km2 < smax_km2")
        if not (0.0 < self.ovl_min <= 1.0):
            raise ValueError("Require 0 < ovl_min <= 1")
        if self.discri_scheme not in {"CAL", "GEN", "EMPIRICAL"}:
            raise ValueError(f"Unsupported discri_scheme: {self.discri_scheme}")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def classify_vertical_category(
    t_min_k: float,
    t_bt_k: float,
    age_below_minus40_min: float,
    cfg: NwcSafRdtConfig,
) -> int:
    """Return NWC SAF vertical category 0..5 (ATBD Sec 3.2.1.2.3.1 & Annex Line I ch7, p. 58).

    Category 0 (Mature): T_min <= -40 C for at least 30 min (CAL) / 45 min (GEN)
    Category 1 (Mature Transition): T_min <= -40 C (<30 min) OR T_BT <= -35 C
    Category 2 (Cold Transition): -40 C < T_min <= -35 C OR T_BT <= -25 C
    Category 3 (Warm1 Transition): -35 C < T_min <= -25 C OR T_BT <= -15 C
    Category 4 (Warm2 Transition): -25 C < T_min <= -15 C OR T_BT <= -5 C
    Category 5 (Warm / Hot): T_min > -15 C and T_BT > -5 C
    """
    if t_min_k <= cfg.cat_mature_tmin_k and age_below_minus40_min >= cfg.cat_mature_min_age_min - 1e-6:
        return 0
    if t_min_k <= cfg.cat_mature_tmin_k or t_bt_k <= cfg.cat1_tbt_k:
        return 1
    if t_min_k <= cfg.cat2_tmin_k or t_bt_k <= cfg.cat2_tbt_k:
        return 2
    if t_min_k <= cfg.cat3_tmin_k or t_bt_k <= cfg.cat3_tbt_k:
        return 3
    if t_min_k <= cfg.cat4_tmin_k or t_bt_k <= cfg.cat4_tbt_k:
        return 4
    return 5
