"""Configuration, class definitions, and bitmasks for NWC SAF GEO-CMA v2025 (v6.0).

Implements the exact definitions from:
- NWC/CDOP2/GEO/MFL/SCI/ATBD/Cloud, Issue 1, Rev. 1 (15 Oct 2016)
- NWC/CDOP3/GEO/MFL/SCI/ATBD/Cloud, Issue 1, Rev. 0 (v2025 / GEO-CMA v6.0, NWC-004)

In NWC SAF GEO-CMA, alongside the binary Cloud Mask (`cma`: 0 = Cloud-free, 1 = Cloudy),
two 4-category (1 negative class + 3 positive classes) representations are defined:

1. `CmaCoverClass` (`cma_cover` — Classic NWC SAF Cloud Cover Category, ATBD Section 2.2.1.2.2, p. 13):
   - 0 (Negative class): Cloud-free (cielo sereno)
   - 1 (Positive class 1): Cloud contaminated (parzialmente nuvoloso / sub-pixel / bordo nube / cirro sottile)
   - 2 (Positive class 2): Cloud filled (pienamente coperto da nube opaca ad alta confidenza)
   - 3 (Positive class 3): Snow/Ice contaminated (neve continentale o ghiaccio marino)

2. `CmaCloudSnowClass` (`cma_cloudsnow` — v2025 Official Cloud & Snow Container, ATBD Section 2.2.2.4, p. 59):
   - 0 (Negative class): Cloud-free (cielo sereno)
   - 1 (Positive class 1): Cloud (except thin ice cloud over snow)
   - 2 (Positive class 2): Thin ice cloud over snow/ice
   - 3 (Positive class 3): Snow/Ice
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CmaBinaryClass(IntEnum):
    """NWC SAF Binary Cloud Mask (`cma`, Table 4, p. 10)."""

    CLOUD_FREE = 0
    CLOUDY = 1
    FILL_VALUE = 255


class CmaCoverClass(IntEnum):
    """NWC SAF 4-Class Cloud Cover Detector (`cma_cover`, ATBD Section 2.2.1.2.2, p. 13).

    Structured as 1 negative class (`0 = CLOUD_FREE`) + 3 positive classes:
    - `1 = CLOUD_CONTAMINATED`: Partially cloudy / sub-pixel cloud / cloud edge / thin semitransparent cirrus
      or pixel near the radiometric detection threshold (within the security margin).
    - `2 = CLOUD_FILLED`: Fully cloudy / opaque cloud deck clearly exceeding the radiometric security margin.
    - `3 = SNOW_OR_ICE`: Clear-sky snow on land or sea ice over water.
    """

    CLOUD_FREE = 0
    CLOUD_CONTAMINATED = 1
    CLOUD_FILLED = 2
    SNOW_OR_ICE = 3
    FILL_VALUE = 255


class CmaCloudSnowClass(IntEnum):
    """NWC SAF v2025 Cloud and Snow Mask (`cma_cloudsnow`, ATBD Section 2.2.2.4, p. 59).

    Structured as 1 negative class (`0 = CLOUD_FREE`) + 3 positive classes:
    - `1 = CLOUD`: Cloud (except thin ice cloud over snow/ice).
    - `2 = THIN_ICE_CLOUD_OVER_SNOW`: Thin ice cloud overlying snow or sea ice.
    - `3 = SNOW_OR_ICE`: Clear-sky snow on land or sea ice over water.
    """

    CLOUD_FREE = 0
    CLOUD = 1
    THIN_ICE_CLOUD_OVER_SNOW = 2
    SNOW_OR_ICE = 3
    FILL_VALUE = 255


class CmaDustClass(IntEnum):
    """NWC SAF Dust Plume Mask (`cma_dust`, ATBD Section 2.2.2.4, p. 59)."""

    NO_DUST = 0
    DUST = 1
    UNDEFINED = 2
    FILL_VALUE = 255


class CmaVolcanicClass(IntEnum):
    """NWC SAF Volcanic Ash Plume Mask (`cma_volcanic`, ATBD Section 2.2.2.4, p. 59)."""

    NO_VOLCANIC_PLUME = 0
    VOLCANIC_PLUME = 1
    UNDEFINED = 2
    FILL_VALUE = 255


class CmaSmokeClass(IntEnum):
    """NWC SAF Smoke Plume Mask (`cma_smoke`, ATBD Section 2.2.2.4, p. 59)."""

    NO_SMOKE = 0
    SMOKE = 1
    UNDEFINED = 2
    FILL_VALUE = 255


class CmaQuality(IntEnum):
    """NWC SAF Cloud Mask Processing Quality (`cma_quality`, ATBD Section 2.2.1.2.4.1, p. 53)."""

    NODATA = 0
    INTERNAL_CONSISTENCY = 1
    TEMPORAL_CONSISTENCY = 2
    GOOD = 3
    QUESTIONABLE = 4


class CmaStatusBit(IntEnum):
    """NWC SAF `cma_status_flag` bit definitions (ATBD Section 2.2.2.4, p. 60)."""

    MONOTONOUS_SNOW_DETECTION = 0
    INVERSION_OR_COLD_AIR_PIXEL = 1
    TWILIGHT = 2
    NIGHT = 3
    DAY = 4
    SUNG_LINT = 5
    LAND = 6
    COAST = 7
    SEA = 8
    NWP_SNOW_OR_SEA_ICE = 9
    ALL_SATELLITE_CHANNELS_AVAILABLE = 10


class CmaTestBit(IntEnum):
    """NWC SAF `cma_testlist` 28-bit definitions (ATBD Section 2.2.2.4, pp. 59-60)."""

    SNOW_AND_ICE = 0
    THIN_ICE_CLOUD_OVER_SNOW = 1
    T108_BRIGHTNESS_TEMPERATURE = 2
    SST_ESTIMATION = 3
    R06_OR_R08_REFLECTANCE = 4
    T108_T120_SPLIT_WINDOW = 5
    T108_T38_NIGHT = 6
    T38_T108_DAY = 7
    T120_T38_NIGHT = 8
    T87_T108 = 9
    T108_T120_OR_T87_T108_GROSS = 10
    T38_T108_GROSS_DAY = 11
    SNOW_FALSE_DETECTION_CANCELLATION = 12
    SOLAR_CONTAMINATION_ON_38 = 13
    R16_OR_R38_OVER_R06 = 14
    LOCAL_SPATIAL_TEXTURE = 15
    R13_CIRRUS_REFLECTANCE = 16
    T108_T134_TROPOSPHERIC_CO2 = 17
    T73_T108_WATER_VAPOUR = 18
    SUNGLINT_RESTORAL = 19
    DESERT_T108_T87_RESTORAL = 20
    CLOUD_BORDER_REFLECTANCE_RATIO = 21
    TEMPORAL_DIFFERENCING_T108 = 22
    TEMPORAL_COHERENCE_1H = 23
    SPATIAL_CLOUD_EXPANSION = 24
    RESTORAL_CLEAR_OVER_SNOW = 25
    SPATIAL_FILTER_TO_CLEAR = 26
    ISOLATED_PIXEL_FILTER = 27


@dataclass(frozen=True)
class NwcSafCmaConfig:
    """Configuration for NWC SAF GEO-CMA v2025 (v6.0) in Pure-IR and Full NWP modes."""

    # Geostationary sub-satellite longitude (deg East). MSG RSS = 9.5 deg E, 0 deg = 0.0
    sub_lon_deg: float = 9.5

    # Viewing angle limit (deg) above which viewing path is degraded
    max_viewing_zenith_deg: float = 78.0

    # --- 1. Opaque Thermal & Atmospheric Absorption Tests ---
    # T10.8 threshold offset below surface skin temperature (K) at nadir:
    # T108_thr = T_surf - (t108_base_offset_k + t108_sec_slope_k * (sec(theta) - 1))
    t108_land_base_offset_k: float = 8.0
    t108_sea_base_offset_k: float = 4.5
    t108_sec_slope_k: float = 3.5

    # Absolute cold cloud floor (K): any pixel with T10.8 below this is cloudy
    # unless flagged as snow/ice or thermal inversion
    t108_absolute_cloud_floor_k: float = 258.0

    # Opaque high/mid cloud H2O 7.3 um & CO2 13.4 um checks (ATBD Bits 17, 18)
    # T7.3 - T10.8 > thr when T10.8 is moderately cold indicates cloud reaching mid-troposphere
    t73_t108_cloud_thr_k: float = -8.5
    t73_t108_max_t108_k: float = 268.0
    # T10.8 - T13.4 < thr when T10.8 is cold indicates cloud top near/above upper troposphere
    t108_t134_cloud_thr_k: float = 12.0
    t108_t134_max_t108_k: float = 263.0

    # --- 2. Semitransparent Cirrus & Cloud Microphysics BTD Tests ---
    # Saunders & Kriebel (1988) / NWC SAF table lookup approximation for T10.8 - T12.0:
    # btd_108_120_thr = btd_108_120_base_k + btd_108_120_sec_slope_k * (sec(theta) - 1)
    #                 + btd_108_120_warm_slope * max(0, T10.8 - 270.0)
    btd_108_120_sea_base_k: float = 1.35
    btd_108_120_land_base_k: float = 1.75
    btd_108_120_sec_slope_k: float = 0.55
    btd_108_120_warm_slope: float = 0.045

    # T8.7 - T10.8 ice cirrus test (Bit 9): positive BTD over non-desert surface
    btd_087_108_ice_cloud_thr_k: float = 0.35
    btd_087_108_max_t108_k: float = 278.0

    # Combined gross cirrus test (Bit 10): T10.8 - T12.0 + (T8.7 - T10.8)
    btd_gross_cirrus_thr_k: float = 1.90

    # --- 3. Low Stratus / Fog IR Emissivity Deficit Test (Pure-IR Night/Day proxy) ---
    # Over sea or moist land, water droplets have lower emissivity at 8.7 um than 10.8 um:
    # T8.7 - T10.8 < -stratus_btd_108_087_thr_k with low spatial variance and small split-window
    stratus_btd_087_108_sea_thr_k: float = -2.10
    stratus_btd_087_108_land_thr_k: float = -3.20
    stratus_max_btd_108_120_k: float = 1.60
    stratus_max_t108_k: float = 286.0

    # --- 4. Thin Ice Cloud Over Snow/Ice Test (ATBD Section 2.2.1.2.2.13, p. 28) ---
    # Over snow/ice pixels: T10.8 - T12.0 > thr and T7.3 - T10.8 > thr
    thin_ice_over_snow_btd_108_120_thr_k: float = 1.35
    thin_ice_over_snow_btd_073_108_thr_k: float = -12.0

    # --- 5. Local Spatial Texture Test (ATBD Section 2.2.1.2.2.17, p. 30, Bit 15) ---
    # 3x3 local standard deviation of T10.8 and T10.8 - T12.0
    spatial_sd_t108_sea_thr_k: float = 0.85
    spatial_sd_t108_land_thr_k: float = 2.50
    spatial_sd_t108_coast_thr_k: float = 3.50
    spatial_sd_btd_108_120_sea_thr_k: float = 0.35
    spatial_sd_t108_max_t108_offset_k: float = 3.0  # must also be cooler than T_surf - offset

    # --- 6. Desert Restoral & Thermal Inversion Restoral ---
    # Desert false-cloud restoral (ATBD Section 2.2.1.2.2.19, p. 32, Bit 20):
    # Hot, dry sandy surfaces have lower emissivity at 8.7 um (T10.8 - T8.7 >> 0)
    desert_restoral_min_t108_k: float = 288.0
    desert_restoral_btd_108_087_thr_k: float = 2.80

    # Cold air / nocturnal thermal inversion filter (ATBD Section 2.2.1.2.3.1, p. 50, Bit 26):
    # Over land, if pixel was ONLY detected by T10.8 test, and T7.3 - T10.8 > inv_btd_073_108_thr_k
    # or local 3x3 SD(T10.8) < inv_max_sd_t108_k with very small T10.8 - T12.0
    inversion_btd_073_108_thr_k: float = -4.0
    inversion_max_sd_t108_k: float = 0.65
    inversion_max_btd_108_120_k: float = 0.80
    inversion_min_t108_k: float = 248.0

    # --- 7. Temporal Tests (ATBD Sections 2.2.1.2.2.21-22, pp. 34-41, Bits 22-24) ---
    # 15-min rapid cooling test (Bit 22): T10.8(t) - T10.8(t - 15 min) < -thr
    temporal_15min_cooling_thr_k: float = 3.5
    temporal_15min_max_t108_offset_k: float = 2.0  # T10.8(t) < T_surf - offset

    # 1-hour stationary cloud test (Bit 23) & spatial expansion (Bit 24)
    enable_temporal_tests: bool = True
    spatial_expansion_temp_margin_k: float = 2.5

    # --- 8. Spatial Post-Filtering (ATBD Section 2.2.1.2.3.2, p. 51, Bit 27) ---
    # Isolated cloudy pixel surrounded by >= 7 clear neighbors (or vice versa) is flipped
    enable_isolated_pixel_filter: bool = True
    isolated_neighbor_count_thr: int = 7

    # --- 9. Cloud Cover Separation (`Cloud contaminated = 1` vs `Cloud filled = 2`) ---
    # ATBD Section 2.2.1.2.2 (pp. 12-13, 30-31):
    # A cloudy pixel is `CLOUD_FILLED (2)` when a primary radiometric test exceeds its
    # security margin (e.g. T10.8_thr - T10.8 >= cloud_filled_thermal_margin_k), the cloud is
    # not strongly semitransparent (T10.8 - T12.0 <= cloud_filled_max_semitrans_btd_k unless
    # deeply cold T10.8 < cloud_filled_deep_cold_t108_k), and is not an isolated sub-pixel edge.
    # Otherwise, a cloudy pixel is `CLOUD_CONTAMINATED (1)`.
    cloud_filled_thermal_margin_k: float = 3.0
    cloud_filled_max_semitrans_btd_k: float = 2.40
    cloud_filled_deep_cold_t108_k: float = 242.0
    cloud_filled_min_3x3_cloudy_neighbors: int = 5

    # --- 10. Dust & Volcanic Ash Plume Detection (ATBD Section 2.2.1.2.2.23, pp. 42-49) ---
    # Dust: negative split window (T12.0 - T10.8 > thr), positive (T10.8 - T8.7 > thr),
    # moderate/warm T10.8, low spatial texture
    dust_btd_120_108_sea_thr_k: float = 0.0
    dust_btd_120_108_land_thr_k: float = 0.35
    dust_btd_108_087_thr_k: float = 1.50
    dust_min_t108_k: float = 265.0
    dust_max_sd_t108_k: float = 2.20

    # Volcanic ash: strong reverse absorption (T12.0 - T10.8 > thr) at cold/mid temperatures
    # where silicate ash absorbs strongly at 10.8 um relative to 12.0 um
    volcanic_btd_120_108_thr_k: float = 1.20
    volcanic_btd_087_108_max_k: float = 0.50
    volcanic_max_t108_k: float = 265.0

    # --- 11. Autonomous Pure-IR Surface & Profile Estimation ---
    # Causal exponential moving average factor for clear-sky surface temperature tracking
    surface_ema_alpha: float = 0.25
    # Fallback warm-envelope percentile over clear/warm pixels when no NWP T_surf is given
    surface_warm_percentile: float = 92.0
    # Security margin (K) near threshold for QUESTIONABLE quality flag (ATBD p. 53)
    quality_security_margin_k: float = 1.2

    # Standard lapse rate (K/km) and tropopause proxy constraints
    standard_lapse_rate_k_per_km: float = 6.5
