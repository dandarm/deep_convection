"""Configuration and enumerations for the NWC SAF Cloud Mask (GEO-CMA) & Cloud Type (GEO-CT) processor.

Derived from the official EUMETSAT NWC SAF documentation:
- Algorithm Theoretical Basis Document for the Cloud Product Processors of the NWC/GEO
  (NWC/CDOP2/MTG/MFL/SCI/ATBD/Cloud, Issue 1.1.1, 30 May 2025; and GEO v1.1 / v2.1)
- User Manual for the Cloud Product Processors of the NWC/GEO: Science Part
  (NWC/CDOP3/MTG/MFL/SCI/UM/Cloud, Issue 1.1.1, 30 May 2025, applicable to
  GEO-CMA-v6.0 [NWC-004] and GEO-CT-v5.0 [NWC-008] in NWC/GEO v2025)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CmaClass(IntEnum):
    """Official NWC SAF GEO-CMA binary cloud mask categories (`cma` container)."""

    CLOUD_FREE = 0
    CLOUDY = 1
    FILL_VALUE = 255


class CmaCloudSnowClass(IntEnum):
    """Official NWC SAF GEO-CMA Cloud and Snow Mask categories (`cma_cloudsnow` container)."""

    CLOUD_FREE = 0
    CLOUD = 1  # Cloud (except thin ice cloud over snow)
    THIN_ICE_CLOUD_OVER_SNOW = 2
    SNOW_OR_ICE = 3
    FILL_VALUE = 255


class CmaDustClass(IntEnum):
    """Official NWC SAF GEO-CMA Dust Detection categories (`cma_dust` container)."""

    NO_DUST = 0
    DUST = 1
    UNDEFINED = 2
    FILL_VALUE = 255


class CmaVolcanicClass(IntEnum):
    """Official NWC SAF GEO-CMA Volcanic Plume Detection categories (`cma_volcanic` container)."""

    NO_VOLCANIC_PLUME = 0
    VOLCANIC_PLUME = 1
    UNDEFINED = 2
    FILL_VALUE = 255


class CmaSmokeClass(IntEnum):
    """Official NWC SAF GEO-CMA Smoke Detection categories (`cma_smoke` container)."""

    NO_SMOKE = 0
    SMOKE = 1
    UNDEFINED = 2
    FILL_VALUE = 255


class CmaTestBit(IntEnum):
    """Official 28-bit test list flags (`cma_testlist`) defined in ATBD Section 2.2.2.4."""

    VIS_REFLECTANCE = 0  # Bit 0: R0.6um (land) or R0.8um (sea)
    NIR_16_SEA = 1  # Bit 1: R1.6um (sea)
    SUNGLINT_38 = 2  # Bit 2: Sunglint test using 3.8um
    CIRRUS_138 = 3  # Bit 3: R1.38um
    T108_OR_SST = 4  # Bit 4: T10.8um or SST
    T108_MINUS_T120 = 5  # Bit 5: T10.8um - T12.0um
    T108_MINUS_T38 = 6  # Bit 6: T10.8um - T3.8um (or Pure-IR WV/CO2 surrogate)
    T120_MINUS_T38 = 7  # Bit 7: T12.0um - T3.8um
    T38_MINUS_T108 = 8  # Bit 8: T3.8um - T10.8um
    T108_MINUS_T87 = 9  # Bit 9: T10.8um - T8.7um (low clouds at high secant)
    T87_MINUS_T108 = 10  # Bit 10: T8.7um - T10.8um (thin ice clouds)
    T87_MINUS_T38 = 11  # Bit 11: T8.7um - T3.8um
    SNOW_T38 = 12  # Bit 12: Snow with only T3.8um / IR snow proxy
    SNOW_R16 = 13  # Bit 13: Snow with R1.6um
    SNOW_R16_R22 = 14  # Bit 14: Snow with R1.6um and R2.2um
    LOCAL_SPATIAL_TEXTURE = 15  # Bit 15: Local Spatial Texture
    RTTOV_T108 = 16  # Bit 16: T10.8um with RTTOV / Pure-IR atmospheric profile
    RTTOV_T38 = 17  # Bit 17: T3.8um with RTTOV
    RTTOV_T87_MINUS_T38 = 18  # Bit 18: T8.7um - T3.8um with RTTOV
    RTTOV_T108_MINUS_T120 = 19  # Bit 19: T10.8um - T12.0um with RTTOV
    RTTOV_T108_MINUS_T87 = 20  # Bit 20: T10.8um - T8.7um with RTTOV
    RTTOV_T108_MINUS_T38 = 21  # Bit 21: T10.8um - T3.8um with RTTOV
    TEMPORAL_DIFFERENCING = 22  # Bit 22: 15-min Temporal-differencing
    STATIONARY_TWILIGHT = 23  # Bit 23: Stationary cloud 1h temporal coherence
    SPATIAL_EXTENSION_TWILIGHT = 24  # Bit 24: Spatial extension of stationary clouds
    HIGH_RES_VIS = 25  # Bit 25: Use of high resolution visible
    SPATIAL_FILTER_TO_CLEAR = 26  # Bit 26: Spatial filtering: cloud reclassified as cloud-free
    SPATIAL_FILTER_TO_CLOUD = 27  # Bit 27: Spatial filtering: cloud-free reclassified as cloud


class CmaStatusBit(IntEnum):
    """Official 11-bit status flags (`cma_status_flag`) defined in ATBD Section 2.2.2.4."""

    LOW_LEVEL_THERMAL_INVERSION = 0
    COLD_SNOWY_GROUND_SUSPECTED = 1
    TEMPORAL_ALGORITHM_PASSED = 2
    HR_DATA_USED = 3
    RTTOV_ONLINE_USED = 4
    SST_ANALYSIS_AVAILABLE = 5
    SNOW_MAP_AVAILABLE = 6
    SEA_ICE_MAP_AVAILABLE = 7
    NO_METHOD_FOR_DUST = 8
    NO_METHOD_FOR_VOLCANIC = 9
    NO_METHOD_FOR_SMOKE = 10


class CmaQuality(IntEnum):
    """Official NWC SAF retrieval quality indicator (`cma_quality` / `ct_quality`)."""

    NA = 0
    GOOD = 1
    QUESTIONABLE = 2
    BAD = 3
    INTERPOLATED = 4


class CtClass(IntEnum):
    """Official 15-class NWC SAF GEO-CT Cloud Type taxonomy (`ct` container, ATBD Section 3.2.2.4)."""

    NON_PROCESSED = 0
    CLOUD_FREE_LAND = 1
    CLOUD_FREE_SEA = 2
    LAND_SNOW = 3
    SEA_ICE = 4
    VERY_LOW_CLOUDS = 5  # Opaque, P_top > 800 hPa, T10.8 > maxT108low
    LOW_CLOUDS = 6  # Opaque, 650 <= P_top <= 800 hPa, maxT108med < T10.8 <= maxT108low
    MID_LEVEL_CLOUDS = 7  # Opaque, 450 <= P_top < 650 hPa, maxT108hi < T10.8 <= maxT108med
    HIGH_OPAQUE_CLOUDS = 8  # Opaque, 300 <= P_top < 450 hPa, maxT108vh < T10.8 <= maxT108hi
    VERY_HIGH_OPAQUE_CLOUDS = 9  # Opaque, P_top < 300 hPa, T10.8 <= maxT108vh
    FRACTIONAL_CLOUDS = 10  # Sub-pixel broken clouds
    HIGH_SEMITRANSPARENT_THIN = 11  # Thin cirrus
    HIGH_SEMITRANSPARENT_MODERATELY_THICK = 12  # Moderately thick cirrus
    HIGH_SEMITRANSPARENT_THICK = 13  # Thick cirrus
    HIGH_SEMITRANSPARENT_ABOVE_LOW_OR_MEDIUM = 14  # Multilayer cloud (cirrus over lower deck)
    HIGH_SEMITRANSPARENT_ABOVE_SNOW_OR_ICE = 15  # Thin cirrus above snow/ice
    FILL_VALUE = 255


class CtCumuliformClass(IntEnum):
    """Official NWC SAF GEO-CT Stratiform/Cumuliform classification (`ct_cumuliform` container)."""

    STRATIFORM = 1
    CUMULIFORM = 2
    MIXED = 3
    CLOUD_FREE = 4
    UNDEFINED = 5
    FILL_VALUE = 255


class CtMultilayerClass(IntEnum):
    """Official NWC SAF GEO-CT Multilayer cloud indicator (`ct_multilayer` container)."""

    NO_MULTILAYER = 0
    MULTILAYER = 1
    CLOUD_FREE = 2
    UNDEFINED = 3
    FILL_VALUE = 255


class CtStatusBit(IntEnum):
    """Official 6-bit status flags (`ct_status_flag`) defined in ATBD Section 3.2.2.4."""

    LOW_LEVEL_THERMAL_INVERSION = 0
    TROPOPAUSE_TEMP_AVAILABLE = 1
    CHANNEL_138_USED = 2
    HR_DATA_USED = 3
    NO_STRATIFORM_CUMULIFORM_METHOD = 4
    NO_MULTILAYER_METHOD = 5


class RdtBridgeCtClass(IntEnum):
    """6-class taxonomy bridging 15-class GEO-CT with RDT/CI convective pre-screening."""

    CLEAR = 0
    LOW_WARM = 1
    MEDIUM_OPAQUE = 2
    HIGH_OPAQUE = 3
    VERY_HIGH_OPAQUE = 4
    SEMITRANSPARENT_CIRRUS = 5


@dataclass(frozen=True)
class NwcSafCmaCtConfig:
    """Configuration for the NWC SAF GEO-CMA v6.0 & GEO-CT v5.0 (v2025) processor.

    All temperatures and brightness temperature differences are in Kelvin (K).
    Supports both Full NWP/ancillary mode and Pure-SEVIRI 7-channel IR autonomous mode.
    """

    # --- Satellite viewing geometry & cadence ---
    sub_satellite_lon_deg: float = 9.5  # Meteosat RSS nominal sub-satellite longitude
    default_secant: float = 1.35  # Typical Mediterranean viewing angle secant (~42 deg)
    max_secant: float = 5.0
    cadence_minutes: float = 5.0

    # --- Causal clear-sky surface temperature tracker (Pure-SEVIRI IR mode) ---
    surface_ema_alpha: float = 0.08
    surface_quantile_fallback: float = 92.0
    min_sea_freezing_sst_k: float = 270.15  # ATBD Section 2.2.1.2.2.1

    # --- GEO-CMA: Thermal contrast & SST tests (ATBD Sections 2.2.1.2.2.1 - 2.2.1.2.2.2) ---
    cma_sst_offset_low_sec_k: float = 1.5  # Offset at secant <= 2.0 (theta_sat <= 60 deg)
    cma_sst_offset_high_sec_k: float = 2.5  # Offset at secant >= 5.0 (theta_sat >= 78.5 deg)
    cma_sst_coast_extra_offset_k: float = 0.5
    cma_t108_base_offset_sea_k: float = 2.8  # Clear-sky absorption + skin margin over sea
    cma_t108_base_offset_land_k: float = 4.5  # Clear-sky absorption + skin margin over land
    cma_t108_wv_secant_slope_k: float = 0.45  # Additional absorption per unit (secant - 1)
    cma_extreme_cooling_fskt_k: float = 263.0  # ATBD p. 16 extreme cooling threshold
    cma_extreme_cooling_t108_pivot_k: float = 255.0
    cma_extreme_cooling_offset_k: float = 5.0
    cma_definite_cloud_t108_k: float = 260.0  # Unambiguously cold cloud top anywhere in mid-latitudes

    # --- GEO-CMA: Split-window & Tri-spectral BTD tests (ATBD Sections 2.2.1.2.2.3 - 2.2.1.2.2.9) ---
    # T10.8 - T12.0 > T108T120_thr (thin cirrus and cloud edges)
    cma_swd_base_sea_k: float = 1.15
    cma_swd_base_land_k: float = 1.45
    cma_swd_secant_slope_k: float = 0.35
    cma_swd_max_land_t108_k: float = 303.15  # ATBD Section 2.2.1.2.2.3: T10.8 < 303.15 K over land

    # T8.7 - T10.8 > T87T108_thr (ice clouds / thin cirrus, ATBD Section 2.2.1.2.2.4)
    cma_ice_btd_base_sea_k: float = -0.25
    cma_ice_btd_base_land_k: float = 0.10
    cma_ice_btd_secant_slope_k: float = 0.20

    # T10.8 - T8.7 > T108T87_thr (low water clouds at high viewing angles, ATBD Section 2.2.1.2.2.9)
    # Formula in ATBD p. 23: T10.8T8.7threshold = 3.7 + 0.3 * sec(theta_sat) for sec > 1.5
    cma_low_cloud_t108_t87_intercept_k: float = 3.7
    cma_low_cloud_t108_t87_secant_slope_k: float = 0.3
    cma_low_cloud_min_secant: float = 1.5

    # Pure-SEVIRI 7-channel IR low/mid cloud spectral tests (replacing 3.8 um when unavailable)
    # In warm low stratus/fog or mid-level clouds, emissivity & CO2/WV differential absorption
    # produce characteristic signatures in (IR_108 - IR_087), (WV_073 - IR_108), and (IR_134 - IR_108)
    cma_pure_ir_low_stratus_t108_t87_k: float = 3.1  # Low water cloud droplet emissivity contrast
    cma_pure_ir_wv073_t108_opaque_k: float = -13.5  # Mid/high cloud where WV_073 approaches IR_108

    # --- GEO-CMA: Local Spatial Texture test (ATBD Section 2.2.1.2.2.17) ---
    cma_texture_sd_t108_sea_k: float = 0.6  # ATBD p. 30: SD(T10.8) > 0.6 K over sea
    cma_texture_sd_btd_sea_k: float = 0.25  # Coherent BTD texture over sea
    cma_texture_noise_t108_k: float = 0.15  # Instrumental noise at 10.8 um
    cma_texture_sd_t108_land_k: float = 1.2  # ATBD p. 30: 1.0 K night / 2.0 K day over land
    cma_texture_sd_btd_land_k: float = 0.45

    # --- GEO-CMA: Quality Security Margins (ATBD Section 2.2.1.2.2.18, Table on p. 31) ---
    cma_margin_sst_k: float = 2.0
    cma_margin_t108_k: float = 3.0
    cma_margin_swd_k: float = 0.5
    cma_margin_btd_k: float = 0.5
    cma_margin_texture_rel: float = 0.20  # 0.2 * threshold

    # --- GEO-CMA: Temporal tests (ATBD Sections 2.2.1.2.4 & 2.2.1.2.5) ---
    cma_temporal_window_minutes: float = 15.0
    cma_temporal_dt108_sea_k: float = -0.6  # ATBD p. 37: Delta_15min(T10.8) < -0.6 K over sea
    cma_temporal_dt108_land_k: float = -1.8  # ATBD p. 37: varies -0.6 K to -3.0 K over land
    cma_stationary_window_minutes: float = 60.0
    cma_stationary_dt108_max_k: float = 1.0  # ATBD p. 38: |Delta_1h(T10.8)| < 1.0 K
    cma_stationary_dbtd87_land_max_k: float = 0.5  # ATBD p. 38: |Delta_1h(T10.8 - T8.7)| < 0.5 K
    cma_stationary_dswd_sea_max_k: float = 0.6  # ATBD p. 38: |Delta_1h(T10.8 - T12.0)| < 0.6 K
    cma_stationary_min_seed_pixels: int = 8  # ATBD p. 40: connected group > 8 pixels
    cma_stationary_max_grow_pixels: int = 10000  # ATBD p. 40: reject if > 10000 pixels

    # --- GEO-CMA: Final Spatial Filtering (ATBD Section 2.2.1.2.7, pp. 49-52) ---
    enable_spatial_filtering: bool = True
    cma_cold_inversion_t108thr_max_k: float = 250.0  # ATBD p. 50: t108thr < 250 K
    cma_cold_inversion_t73_minus_t108_min_k: float = 0.5  # ATBD p. 50: T7.3 - T10.8 > 0.5 K

    # --- GEO-CMA: Dust & Volcanic Ash Detection (ATBD Sections 2.2.1.2.8 & 2.2.1.2.9) ---
    enable_dust_detection: bool = True
    enable_volcanic_detection: bool = True
    # Dust features: negative split window (T12.0 - T10.8 > thr), positive T8.7 - T10.8, low SD(T10.8)
    dust_t120_minus_t108_min_k: float = 0.3
    dust_t87_minus_t108_min_k: float = -1.5
    dust_sd_t108_max_sea_k: float = 0.4  # ATBD p. 53: SD(T10.8) < 0.4 K over sea
    dust_sd_t108_max_land_k: float = 3.0  # ATBD p. 53: SD(T10.8) < 3.0 K over land
    # Volcanic ash features (ATBD p. 54): strongly positive T12.0 - T10.8 > thr_volcan
    volcanic_t120_minus_t108_cold_k: float = 1.2  # ATBD p. 54: 1.2 K at T10.8 < 223.15 K
    volcanic_secant_max_offset_k: float = 1.0  # ATBD p. 54: 0 K to 1 K for secant up to 5

    # --- GEO-CT: Opaque Height Threshold Parameters (ATBD Section 3.2.1.2.2.4, p. 77) ---
    # Exact NWC SAF linear combinations of T500, T700, T850, Ttropo:
    # maxT108vh = 0.4 * T500 + 0.6 * Ttropo - 5 K (or maxT108hi - 25 K if Ttropo unavailable)
    # maxT108hi = 0.5 * T500 - 0.2 * T700 + 178 K
    # maxT108med = 0.8 * T850 + 0.2 * T700 - 8 K
    # maxT108low = 1.2 * T850 - 0.2 * T700 - 5 K
    # T108T73thrlow = 4.0 * sec(theta_sat) + 8.5 K
    ct_vh_fallback_offset_k: float = 25.0
    ct_wv73_low_guard_slope_k: float = 4.0
    ct_wv73_low_guard_intercept_k: float = 8.5
    ct_inversion_t87_t108_5k_base: float = -1.2  # ATBD p. 77: within 5 K of maxT108med
    ct_inversion_t87_t108_8k_base: float = -1.7  # ATBD p. 77: within 8 K of maxT108med
    ct_inversion_t87_t108_10k_base: float = -2.2  # ATBD p. 77: within 10 K of maxT108med

    # --- GEO-CT: Semitransparent & Fractional Thresholds (ATBD Sections 3.2.1.2.2.1 - 3.2.1.2.2.3) ---
    # T108T120opaque: threshold separating opaque from semitransparent/fractional clouds
    ct_t108_t120_opaque_base_k: float = 1.35
    ct_t108_t120_opaque_secant_slope_k: float = 0.28
    # T108T120thick: threshold separating thick/moderately-thick semitransparent from opaque
    ct_t108_t120_thick_base_k: float = 1.15
    ct_t108_t120_thick_secant_slope_k: float = 0.22
    # T87T108opaque: ice cloud signature separating high thin cirrus from warm fractional clouds
    ct_t87_t108_opaque_base_k: float = -0.10
    ct_t87_t108_opaque_secant_slope_k: float = 0.18
    # Delta offset added to maxT108low for warm thin cirrus upper temperature limit (ATBD p. 71)
    ct_thin_cirrus_delta_k: float = 12.0
    # Spatial log-variance threshold VarilogT10.8 = Var_3x3(log(Tsurf - T10.8)) (ATBD p. 71)
    ct_varilog_t108_thr: float = 0.035

    # --- GEO-CT: Multilayer Detection (Class 14: High semitransparent above low/medium clouds) ---
    # In Pure-SEVIRI 7-channel IR mode, multilayer clouds (thin ice cirrus overlying an opaque
    # low/mid deck) exhibit high split-window (T10.8 - T12.0 > T108T120thick) and high ice signature
    # (T8.7 - T10.8), while CO2 slicing (IR_134 - IR_108) and mid-tropospheric WV (WV_073 - IR_108)
    # indicate a colder/higher effective radiative base than clear-sky transmission through thin cirrus.
    enable_multilayer_detection: bool = True
    ct_multilayer_swd_min_k: float = 1.65
    ct_multilayer_ice_btd_min_k: float = 0.15
    ct_multilayer_wv073_t108_min_k: float = -18.5
    ct_multilayer_co2_t134_t108_min_k: float = -12.5
    ct_multilayer_local_sd_max_k: float = 3.5

    # --- GEO-CT: Stratiform vs. Cumuliform Separation (`ct_cumuliform`, ATBD Section 3.2.2.4) ---
    ct_cumuliform_sd_t108_min_k: float = 1.8
    ct_mixed_sd_t108_min_k: float = 1.0

    # --- GEO-CT: Quality Assessment Security Margin (ATBD Section 3.2.1.2.2.5, p. 78) ---
    ct_quality_security_margin_k: float = 0.2  # Official 0.2 K security margin in ATBD p. 78
