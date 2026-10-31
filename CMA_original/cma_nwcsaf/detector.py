"""NWC SAF GEO-CMA v2025 (v6.0, NWC-004) Multi-Spectral & Multi-Temporal Cloud Mask Detector.

Implements the complete NWC SAF GEO-CMA decision chain on the 7 SEVIRI thermal/absorption
channels (`WV_062`, `WV_073`, `IR_087`, `IR_097`, `IR_108`, `IR_120`, `IR_134`):
  1. Snow / Sea-Ice and Thin Ice Cloud over Snow/Ice tests (Bits 0, 1, 12, 25)
  2. Opaque Thermal & Atmospheric Absorption tests:
     - T10.8 Brightness Temperature test (Bit 2)
     - SST split-window estimation test over Sea (Bit 3)
     - Tropospheric CO2 T10.8 - T13.4 test (Bit 17)
     - Water Vapour T7.3 - T10.8 test (Bit 18)
  3. Semitransparent Cirrus & Low-Stratus BTD tests:
     - Split-Window T10.8 - T12.0 test (Bit 5)
     - Ice/Stratus T8.7 - T10.8 test (Bit 9)
     - Combined Gross Cirrus BTD test (Bit 10)
  4. Local Spatial Texture (3x3 standard deviation of T10.8 & T10.8 - T12.0, Bit 15)
  5. Desert T10.8 - T8.7 False-Cloud Restoral test (Bit 20)
  6. Temporal Differencing (15-min rapid cooling dT10.8 < -3.5 K, Bit 22)
  7. 1-Hour Stationary Cloud Elimination & Spatial Cloud Expansion (Bits 23, 24)
  8. Spatial Cold-Air / Thermal Inversion Filter to Clear (Bit 26)
  9. Isolated Pixel Post-Filter (Bit 27)
 10. Multi-Category Cloud Cover Separation (`cma_cover`: 0=Cloud-free, 1=Cloud contaminated,
     2=Cloud filled, 3=Snow/Ice) and v2025 Cloud/Snow Mask (`cma_cloudsnow`: 0=Cloud-free,
     1=Cloud, 2=Thin ice cloud over snow/ice, 3=Snow/Ice)
 11. Dust (`cma_dust`), Volcanic Ash (`cma_volcanic`), and Smoke (`cma_smoke`) Plume Detection.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter

from .ancillary import AncillaryData
from .config import (
    CmaBinaryClass,
    CmaCloudSnowClass,
    CmaCoverClass,
    CmaDustClass,
    CmaQuality,
    CmaSmokeClass,
    CmaStatusBit,
    CmaTestBit,
    CmaVolcanicClass,
    NwcSafCmaConfig,
)


@dataclass
class CmaOutput:
    """Complete output of NWC SAF GEO-CMA v6.0 for a single 2D frame `(H, W)`."""

    cma: np.ndarray               # uint8: 0 = Cloud-free, 1 = Cloudy, 255 = Fill (`CmaBinaryClass`)
    cma_cover: np.ndarray         # uint8: 0 = Cloud-free, 1 = Cloud contaminated, 2 = Cloud filled, 3 = Snow/Ice (`CmaCoverClass`)
    cma_cloudsnow: np.ndarray     # uint8: 0 = Cloud-free, 1 = Cloud, 2 = Thin ice over snow, 3 = Snow/Ice (`CmaCloudSnowClass`)
    cma_dust: np.ndarray          # uint8: 0 = No dust, 1 = Dust, 2 = Undefined, 255 = Fill
    cma_volcanic: np.ndarray      # uint8: 0 = No volcanic plume, 1 = Volcanic plume, 2 = Undefined
    cma_smoke: np.ndarray         # uint8: 0 = No smoke, 1 = Smoke, 2 = Undefined
    cma_testlist: np.ndarray      # uint32: 28-bit bitmask of triggered GEO-CMA tests (`CmaTestBit`)
    cma_status_flag: np.ndarray   # uint16: 11-bit processing status flag (`CmaStatusBit`)
    cma_quality: np.ndarray       # uint8: 0 = Nodata, 3 = Good, 4 = Questionable (`CmaQuality`)
    t108_threshold_k: np.ndarray  # float32: Dynamic T10.8 threshold (K)
    btd_108_120_threshold_k: np.ndarray  # float32: Dynamic split-window threshold (K)
    sd_t108_3x3: np.ndarray       # float32: Local 3x3 standard deviation of IR_108 (K)


def compute_local_3x3_stats(
    arr: np.ndarray, same_surface_mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local 3x3 mean and standard deviation, restricted to `same_surface_mask` if provided.

    Following ATBD Section 2.2.1.2.2.17 (p. 30), spatial texture is only computed across
    neighboring pixels sharing the same surface class (land vs. sea, excluding snow/ice boundaries).
    """
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    if same_surface_mask is not None:
        valid = finite & np.asarray(same_surface_mask, dtype=bool)
    else:
        valid = finite
    a_safe = np.where(valid, a, 0.0)
    w = valid.astype(np.float64)

    mean_w = uniform_filter(a_safe, size=3, mode="nearest")
    cnt = uniform_filter(w, size=3, mode="nearest")
    mean = np.where(cnt > 0.0, mean_w / np.maximum(cnt, 1e-6), np.nan)

    sq_w = uniform_filter(a_safe * a_safe, size=3, mode="nearest")
    mean_sq = np.where(cnt > 0.0, sq_w / np.maximum(cnt, 1e-6), np.nan)
    var = np.maximum(0.0, mean_sq - mean * mean)
    # Require at least 6 of 9 neighbors in the 3x3 window to belong to the same surface class
    sd = np.where(cnt >= (6.0 / 9.0 - 1e-4), np.sqrt(var), 0.0)
    return mean.astype(np.float32), sd.astype(np.float32)


def run_cma_frame(
    bt_7ch: dict[str, np.ndarray],
    anc: AncillaryData,
    cfg: NwcSafCmaConfig,
    *,
    bt_108_t_minus_15m: np.ndarray | None = None,
    bt_108_t_minus_60m: np.ndarray | None = None,
    cma_t_minus_60m: np.ndarray | None = None,
) -> CmaOutput:
    """Execute the NWC SAF GEO-CMA v2025 (v6.0) algorithm on a single 7-channel SEVIRI frame."""
    wv_062 = np.asarray(bt_7ch["WV_062"], dtype=np.float32)
    wv_073 = np.asarray(bt_7ch["WV_073"], dtype=np.float32)
    ir_087 = np.asarray(bt_7ch["IR_087"], dtype=np.float32)
    ir_097 = np.asarray(bt_7ch["IR_097"], dtype=np.float32)
    ir_108 = np.asarray(bt_7ch["IR_108"], dtype=np.float32)
    ir_120 = np.asarray(bt_7ch["IR_120"], dtype=np.float32)
    ir_134 = np.asarray(bt_7ch["IR_134"], dtype=np.float32)

    shape = ir_108.shape
    valid_px = (
        np.isfinite(wv_062)
        & np.isfinite(wv_073)
        & np.isfinite(ir_087)
        & np.isfinite(ir_097)
        & np.isfinite(ir_108)
        & np.isfinite(ir_120)
        & np.isfinite(ir_134)
    )

    testlist = np.zeros(shape, dtype=np.uint32)
    status_flag = np.zeros(shape, dtype=np.uint16)

    # Populate status flags
    status_flag[anc.is_land] |= 1 << CmaStatusBit.LAND
    status_flag[anc.is_coast] |= 1 << CmaStatusBit.COAST
    status_flag[anc.is_sea] |= 1 << CmaStatusBit.SEA
    status_flag[anc.snow_ice_mask > 0] |= 1 << CmaStatusBit.NWP_SNOW_OR_SEA_ICE
    status_flag[anc.thermal_inversion] |= 1 << CmaStatusBit.INVERSION_OR_COLD_AIR_PIXEL
    status_flag[valid_px] |= 1 << CmaStatusBit.ALL_SATELLITE_CHANNELS_AVAILABLE

    is_night = anc.solar_zenith_deg >= 90.0
    is_twilight = (anc.solar_zenith_deg >= 80.0) & (anc.solar_zenith_deg < 90.0)
    is_day = anc.solar_zenith_deg < 80.0
    status_flag[is_night] |= 1 << CmaStatusBit.NIGHT
    status_flag[is_twilight] |= 1 << CmaStatusBit.TWILIGHT
    status_flag[is_day] |= 1 << CmaStatusBit.DAY
    status_flag[anc.sunglint_mask] |= 1 << CmaStatusBit.SUNG_LINT

    # Spectral Brightness Temperature Differences (BTDs)
    btd_108_120 = ir_108 - ir_120
    btd_087_108 = ir_087 - ir_108
    btd_108_087 = ir_108 - ir_087
    btd_073_108 = wv_073 - ir_108
    btd_108_134 = ir_108 - ir_134
    btd_120_108 = ir_120 - ir_108

    # Local 3x3 spatial statistics (computed separately on homogeneous land and sea masks
    # so land/sea or snow/clear boundaries do not trigger artificial spatial texture spikes)
    land_clear_snow = anc.is_land & (anc.snow_ice_mask == 0)
    sea_clear_snow = anc.is_sea & (anc.snow_ice_mask == 0)
    mean_t108_land, sd_t108_land = compute_local_3x3_stats(ir_108, land_clear_snow)
    mean_t108_sea, sd_t108_sea = compute_local_3x3_stats(ir_108, sea_clear_snow)
    mean_t108_3x3 = np.where(anc.is_sea, mean_t108_sea, mean_t108_land)
    sd_t108_3x3 = np.where(anc.is_sea, sd_t108_sea, sd_t108_land)
    _, sd_btd_108_120_3x3 = compute_local_3x3_stats(btd_108_120, sea_clear_snow)

    # -------------------------------------------------------------------------
    # STEP 1: Snow / Sea-Ice and Thin Ice Cloud over Snow/Ice (Bits 0, 1, 12, 25)
    # -------------------------------------------------------------------------
    has_snow_ice = valid_px & (anc.snow_ice_mask > 0)
    testlist[has_snow_ice] |= 1 << CmaTestBit.SNOW_AND_ICE

    # Over snow/ice, detect overlying thin ice cloud via split-window & WV 7.3 (ATBD p. 28)
    thin_ice_over_snow = (
        has_snow_ice
        & (btd_108_120 > cfg.thin_ice_over_snow_btd_108_120_thr_k)
        & (btd_073_108 > cfg.thin_ice_over_snow_btd_073_108_thr_k)
    )
    # Also detect thick cold opaque cloud over snow/ice when T10.8 is deeply depressed
    thick_cloud_over_snow = has_snow_ice & (
        (ir_108 < np.minimum(anc.t_700hpa_k, 242.0))
        | ((btd_073_108 > -5.5) & (ir_108 < 252.0))
    )
    testlist[thin_ice_over_snow] |= 1 << CmaTestBit.THIN_ICE_CLOUD_OVER_SNOW

    # Snow pixels that are NOT covered by thin ice or thick cold cloud are restored to snow-clear
    clear_over_snow = has_snow_ice & (~thin_ice_over_snow) & (~thick_cloud_over_snow)
    testlist[clear_over_snow] |= 1 << CmaTestBit.RESTORAL_CLEAR_OVER_SNOW

    # Non-snow pixels eligible for standard cloud detection
    non_snow_px = valid_px & (~clear_over_snow)

    # -------------------------------------------------------------------------
    # STEP 2: Opaque Thermal & Atmospheric Absorption Tests (Bits 2, 3, 17, 18)
    # -------------------------------------------------------------------------
    sec_m1 = np.maximum(0.0, anc.sec_zenith - 1.0)
    base_offset = np.where(
        anc.is_sea,
        cfg.t108_sea_base_offset_k,
        cfg.t108_land_base_offset_k,
    )
    # Increase offset slightly over high terrain or coastal pixels
    coast_extra = np.where(anc.is_coast, 2.0, 0.0)
    t108_thr = (anc.t_surf_k - (base_offset + cfg.t108_sec_slope_k * sec_m1 + coast_extra)).astype(
        np.float32
    )
    t108_thr = np.maximum(t108_thr, cfg.t108_absolute_cloud_floor_k)

    test_t108 = non_snow_px & (ir_108 < t108_thr)
    testlist[test_t108] |= 1 << CmaTestBit.T108_BRIGHTNESS_TEMPERATURE

    # Sea Surface Temperature (SST) split-window check over open sea (Bit 3, ATBD p. 17)
    # Over sea, split-window SST estimate: SST_est = T10.8 + 1.15*(T10.8 - T12.0)
    sst_est = ir_108 + 1.15 * np.maximum(0.0, btd_108_120)
    sst_thr = anc.t_surf_k - (3.5 + 1.5 * sec_m1)
    test_sst = non_snow_px & anc.is_sea & (sst_est < sst_thr)
    testlist[test_sst] |= 1 << CmaTestBit.SST_ESTIMATION

    # Tropospheric CO2 test T10.8 - T13.4 (Bit 17) & Water Vapour test T7.3 - T10.8 (Bit 18)
    test_co2 = (
        non_snow_px
        & (ir_108 < cfg.t108_t134_max_t108_k)
        & (btd_108_134 < cfg.t108_t134_cloud_thr_k)
    )
    testlist[test_co2] |= 1 << CmaTestBit.T108_T134_TROPOSPHERIC_CO2

    test_wv73 = (
        non_snow_px
        & (~anc.thermal_inversion)
        & (ir_108 < cfg.t73_t108_max_t108_k)
        & (btd_073_108 > cfg.t73_t108_cloud_thr_k)
    )
    testlist[test_wv73] |= 1 << CmaTestBit.T73_T108_WATER_VAPOUR

    # -------------------------------------------------------------------------
    # STEP 3: Semitransparent Cirrus & Low-Stratus BTD Tests (Bits 5, 9, 10)
    # -------------------------------------------------------------------------
    btd_108_120_base = np.where(
        anc.is_sea, cfg.btd_108_120_sea_base_k, cfg.btd_108_120_land_base_k
    )
    warm_term = cfg.btd_108_120_warm_slope * np.maximum(0.0, ir_108 - 270.0)
    btd_108_120_thr = (
        btd_108_120_base + cfg.btd_108_120_sec_slope_k * sec_m1 + warm_term
    ).astype(np.float32)

    test_split_win = non_snow_px & (btd_108_120 > btd_108_120_thr) & (ir_108 < anc.t_surf_k - 1.0)
    testlist[test_split_win] |= 1 << CmaTestBit.T108_T120_SPLIT_WINDOW

    # Ice cirrus positive BTD (T8.7 - T10.8 > thr) or low stratus negative BTD (T8.7 - T10.8 < thr)
    test_ice_87_108 = (
        non_snow_px
        & (btd_087_108 > cfg.btd_087_108_ice_cloud_thr_k)
        & (ir_108 < cfg.btd_087_108_max_t108_k)
    )
    stratus_thr = np.where(
        anc.is_sea, cfg.stratus_btd_087_108_sea_thr_k, cfg.stratus_btd_087_108_land_thr_k
    )
    test_stratus_87_108 = (
        non_snow_px
        & (btd_087_108 < stratus_thr)
        & (btd_108_120 > 0.05)
        & (btd_108_120 < cfg.stratus_max_btd_108_120_k)
        & (ir_108 < np.minimum(cfg.stratus_max_t108_k, anc.t_surf_k - 1.5))
    )
    test_87_108 = test_ice_87_108 | test_stratus_87_108
    testlist[test_87_108] |= 1 << CmaTestBit.T87_T108

    # Combined gross cirrus check: (T10.8 - T12.0) + (T8.7 - T10.8) = T8.7 - T12.0 (Bit 10)
    test_gross_cirrus = (
        non_snow_px
        & ((btd_108_120 + np.maximum(0.0, btd_087_108)) > cfg.btd_gross_cirrus_thr_k)
        & (ir_108 < anc.t_surf_k - 1.5)
    )
    testlist[test_gross_cirrus] |= 1 << CmaTestBit.T108_T120_OR_T87_T108_GROSS

    # -------------------------------------------------------------------------
    # STEP 4: Local Spatial Texture Test (Bit 15, ATBD Section 2.2.1.2.2.17, p. 30)
    # -------------------------------------------------------------------------
    sd_thr = np.where(
        anc.is_coast,
        cfg.spatial_sd_t108_coast_thr_k,
        np.where(anc.is_sea, cfg.spatial_sd_t108_sea_thr_k, cfg.spatial_sd_t108_land_thr_k),
    )
    # Only flag pixels that are on the colder side of the local 3x3 neighborhood
    # so warm clear pixels adjacent to a cold cloud edge are not falsely flagged as cloudy
    colder_than_local_mean = ir_108 < (mean_t108_3x3 - 0.15)
    test_spatial_texture = (
        non_snow_px
        & (~anc.is_coast)
        & colder_than_local_mean
        & (ir_108 < (anc.t_surf_k - cfg.spatial_sd_t108_max_t108_offset_k))
        & (
            (sd_t108_3x3 > sd_thr)
            | (anc.is_sea & (sd_btd_108_120_3x3 > cfg.spatial_sd_btd_108_120_sea_thr_k))
        )
    )
    testlist[test_spatial_texture] |= 1 << CmaTestBit.LOCAL_SPATIAL_TEXTURE

    # Combine initial single-frame cloud detections
    cloudy = (
        thin_ice_over_snow
        | thick_cloud_over_snow
        | test_t108
        | test_sst
        | test_co2
        | test_wv73
        | test_split_win
        | test_87_108
        | test_gross_cirrus
        | test_spatial_texture
    )

    # -------------------------------------------------------------------------
    # STEP 5: Desert False-Cloud Restoral (Bit 20, ATBD Section 2.2.1.2.2.19, p. 32)
    # -------------------------------------------------------------------------
    # Over hot/warm desert land, low 8.7 um quartz emissivity yields large T10.8 - T8.7 (> 2.8 K)
    # that can mimic split-window or spatial texture tests if no upper-tropospheric cloud is present.
    desert_restoral = (
        cloudy
        & anc.is_land
        & (ir_108 > cfg.desert_restoral_min_t108_k)
        & (btd_108_087 > cfg.desert_restoral_btd_108_087_thr_k)
        & (~test_co2)
        & (~test_wv73)
        & (btd_108_120 < 1.6)
    )
    cloudy = cloudy & (~desert_restoral)
    testlist[desert_restoral] |= 1 << CmaTestBit.DESERT_T108_T87_RESTORAL

    # -------------------------------------------------------------------------
    # STEP 6: Temporal Differencing Test (Bit 22, ATBD Section 2.2.1.2.2.21, p. 34)
    # -------------------------------------------------------------------------
    # Rapid cooling over 15 minutes (3 RSS steps): T10.8(t) - T10.8(t - 15 min) < -3.5 K
    test_temporal_15m = np.zeros(shape, dtype=bool)
    if cfg.enable_temporal_tests and bt_108_t_minus_15m is not None:
        prev15 = np.asarray(bt_108_t_minus_15m, dtype=np.float32)
        dt_15m = ir_108 - prev15
        test_temporal_15m = (
            non_snow_px
            & np.isfinite(prev15)
            & (dt_15m < -cfg.temporal_15min_cooling_thr_k)
            & (ir_108 < (anc.t_surf_k - cfg.temporal_15min_max_t108_offset_k))
        )
        cloudy = cloudy | test_temporal_15m
        testlist[test_temporal_15m] |= 1 << CmaTestBit.TEMPORAL_DIFFERENCING_T108

    # -------------------------------------------------------------------------
    # STEP 7: 1-Hour Stationary Cloud Elimination & Spatial Cloud Expansion (Bits 23, 24)
    # -------------------------------------------------------------------------
    # ATBD Section 2.2.1.2.2.22 (pp. 35-41):
    # - Bit 23: If a land pixel was only marginally detected by T10.8 and its T10.8 has remained
    #   completely stationary over 1 hour (|dT10.8(1h)| < 0.35 K) with very low spatial variance
    #   and small BTD_108_120, it is a stationary surface cold anomaly -> restore to clear.
    # - Bit 24: Region growing around high-confidence cloudy cores whose T10.8 cooled relative
    #   to t - 60 min and is within `spatial_expansion_temp_margin_k` of `t108_thr`.
    test_spatial_expansion = np.zeros(shape, dtype=bool)
    if cfg.enable_temporal_tests and bt_108_t_minus_60m is not None:
        prev60 = np.asarray(bt_108_t_minus_60m, dtype=np.float32)
        dt_60m = ir_108 - prev60
        only_marginal_t108 = (
            cloudy
            & anc.is_land
            & test_t108
            & (~test_co2)
            & (~test_wv73)
            & (~test_split_win)
            & (~test_gross_cirrus)
            & (ir_108 > (t108_thr - 3.5))
        )
        stationary_false_cloud = (
            only_marginal_t108
            & np.isfinite(prev60)
            & (np.abs(dt_60m) < 0.35)
            & (sd_t108_3x3 < 0.50)
            & (btd_108_120 < 0.80)
        )
        if cma_t_minus_60m is not None:
            stationary_false_cloud &= np.asarray(cma_t_minus_60m) == CmaBinaryClass.CLOUDY
        cloudy = cloudy & (~stationary_false_cloud)
        testlist[stationary_false_cloud] |= 1 << CmaTestBit.TEMPORAL_COHERENCE_1H

        # Spatial expansion (region growing) around confident cloudy cores
        confident_core = cloudy & (ir_108 < (t108_thr - 3.0))
        dilated_core = binary_dilation(confident_core, iterations=1)
        test_spatial_expansion = (
            non_snow_px
            & (~cloudy)
            & dilated_core
            & np.isfinite(prev60)
            & (dt_60m < -1.5)
            & (ir_108 < (t108_thr + cfg.spatial_expansion_temp_margin_k))
        )
        cloudy = cloudy | test_spatial_expansion
        testlist[test_spatial_expansion] |= 1 << CmaTestBit.SPATIAL_CLOUD_EXPANSION

    # -------------------------------------------------------------------------
    # STEP 8: Cold Air / Nocturnal Thermal Inversion Spatial Filter (Bit 26, ATBD p. 50)
    # -------------------------------------------------------------------------
    # Over land, if a pixel was detected ONLY by T10.8 (no cirrus BTD, no upper-tropospheric CO2/WV
    # signature, no texture spike) and satisfies the low-level thermal inversion condition,
    # restore it to clear sky.
    only_t108_land = (
        cloudy
        & anc.is_land
        & test_t108
        & (~test_co2)
        & (~test_wv73)
        & (~test_split_win)
        & (~test_87_108)
        & (~test_gross_cirrus)
        & (~test_spatial_texture)
    )
    inversion_restoral = only_t108_land & anc.thermal_inversion
    cloudy = cloudy & (~inversion_restoral)
    testlist[inversion_restoral] |= 1 << CmaTestBit.SPATIAL_FILTER_TO_CLEAR

    # -------------------------------------------------------------------------
    # STEP 9: Isolated Pixel Filter (Bit 27, ATBD Section 2.2.1.2.3.2, p. 51)
    # -------------------------------------------------------------------------
    if cfg.enable_isolated_pixel_filter:
        c_float = cloudy.astype(np.float32)
        # 3x3 sum of neighbors excluding center pixel
        sum_3x3 = np.rint(uniform_filter(c_float, size=3, mode="nearest") * 9.0).astype(np.int32)
        neighbor_cloudy_cnt = sum_3x3 - cloudy.astype(np.int32)
        neighbor_clear_cnt = 8 - neighbor_cloudy_cnt

        # Isolated cloudy pixel surrounded by >= 7 clear neighbors (unless very cold convective core)
        isolated_cloudy = (
            cloudy
            & (neighbor_clear_cnt >= cfg.isolated_neighbor_count_thr)
            & (ir_108 > (t108_thr - 8.0))
        )
        # Isolated clear hole surrounded by >= 8 cloudy neighbors
        isolated_clear = (
            non_snow_px
            & (~cloudy)
            & (neighbor_cloudy_cnt >= 8)
            & (ir_108 < (t108_thr + 2.0))
        )
        flipped = isolated_cloudy | isolated_clear
        cloudy = (cloudy & (~isolated_cloudy)) | isolated_clear
        testlist[flipped] |= 1 << CmaTestBit.ISOLATED_PIXEL_FILTER

    # -------------------------------------------------------------------------
    # STEP 10: Assemble Binary `cma`, 4-Class `cma_cover`, and 4-Class `cma_cloudsnow`
    # -------------------------------------------------------------------------
    cma_bin = np.full(shape, CmaBinaryClass.FILL_VALUE, dtype=np.uint8)
    cma_bin[valid_px & (~cloudy)] = CmaBinaryClass.CLOUD_FREE
    cma_bin[valid_px & cloudy] = CmaBinaryClass.CLOUDY

    # v2025 Official Cloud & Snow Mask (`cma_cloudsnow`, ATBD Section 2.2.2.4, p. 59):
    #   0 = Cloud-free (negative class)
    #   1 = Cloud (except thin ice cloud over snow)
    #   2 = Thin ice cloud over snow/ice
    #   3 = Snow/Ice
    cma_cs = np.full(shape, CmaCloudSnowClass.FILL_VALUE, dtype=np.uint8)
    cma_cs[valid_px & (~cloudy) & (~clear_over_snow)] = CmaCloudSnowClass.CLOUD_FREE
    cma_cs[valid_px & clear_over_snow] = CmaCloudSnowClass.SNOW_OR_ICE
    cma_cs[valid_px & cloudy & (~thin_ice_over_snow)] = CmaCloudSnowClass.CLOUD
    cma_cs[valid_px & cloudy & thin_ice_over_snow] = CmaCloudSnowClass.THIN_ICE_CLOUD_OVER_SNOW

    # Classic NWC SAF 4-Class Cloud Cover Category (`cma_cover`, ATBD Section 2.2.1.2.2, p. 13):
    #   0 = Cloud-free (negative class)
    #   1 = Cloud contaminated (sub-pixel / cloud edge / thin semitransparent cirrus / near-threshold)
    #   2 = Cloud filled (opaque / thick cloud deck clearly exceeding radiometric security margins)
    #   3 = Snow/Ice contaminated (clear-sky snow or sea ice)
    #
    # Following ATBD Section 2.2.1.2.2 (pp. 12-13, 30-31):
    # A cloudy pixel is classified as `CLOUD_FILLED (2)` when:
    #   1) At least one primary opaque thermal/absorption test (`T10.8`, `SST`, `CO2`, `WV7.3`,
    #      or opaque low-stratus `T8.7 - T10.8`) is *really successful* (exceeds its security margin:
    #      e.g. `t108_thr - ir_108 >= cfg.cloud_filled_thermal_margin_k`, or `test_co2`, or `test_wv73`),
    #   2) It is not a thin semitransparent cirrus over warm surface
    #      (`btd_108_120 <= cfg.cloud_filled_max_semitrans_btd_k` or `ir_108 < cfg.cloud_filled_deep_cold_t108_k`),
    #   3) It is not a thin ice cloud over snow (`~thin_ice_over_snow`), and
    #   4) Its 3x3 neighborhood has at least `cfg.cloud_filled_min_3x3_cloudy_neighbors` cloudy pixels
    #      (so sub-pixel cloud edges are assigned `CLOUD_CONTAMINATED = 1`).
    cloudy_3x3_count = np.rint(
        uniform_filter(cloudy.astype(np.float32), size=3, mode="nearest") * 9.0
    ).astype(np.int32)

    opaque_thermal_strong = (
        ((t108_thr - ir_108) >= cfg.cloud_filled_thermal_margin_k)
        | (anc.is_sea & ((sst_thr - sst_est) >= cfg.cloud_filled_thermal_margin_k))
        | test_co2
        | test_wv73
        | (test_stratus_87_108 & ((stratus_thr - btd_087_108) >= 0.40))
        | thick_cloud_over_snow
    )
    not_pure_thin_cirrus = (btd_108_120 <= cfg.cloud_filled_max_semitrans_btd_k) | (
        ir_108 < cfg.cloud_filled_deep_cold_t108_k
    )
    is_cloud_filled = (
        valid_px
        & cloudy
        & (~thin_ice_over_snow)
        & opaque_thermal_strong
        & not_pure_thin_cirrus
        & (cloudy_3x3_count >= cfg.cloud_filled_min_3x3_cloudy_neighbors)
    )
    is_cloud_contaminated = valid_px & cloudy & (~is_cloud_filled)

    cma_cov = np.full(shape, CmaCoverClass.FILL_VALUE, dtype=np.uint8)
    cma_cov[valid_px & (~cloudy) & (~clear_over_snow)] = CmaCoverClass.CLOUD_FREE
    cma_cov[valid_px & clear_over_snow] = CmaCoverClass.SNOW_OR_ICE
    cma_cov[is_cloud_contaminated] = CmaCoverClass.CLOUD_CONTAMINATED
    cma_cov[is_cloud_filled] = CmaCoverClass.CLOUD_FILLED

    # -------------------------------------------------------------------------
    # STEP 11: Dust, Volcanic Ash, and Smoke Plume Detection (ATBD pp. 42-49)
    # -------------------------------------------------------------------------
    cma_dust = np.full(shape, CmaDustClass.FILL_VALUE, dtype=np.uint8)
    cma_dust[valid_px] = CmaDustClass.NO_DUST

    dust_btd_120_108_thr = np.where(
        anc.is_sea, cfg.dust_btd_120_108_sea_thr_k, cfg.dust_btd_120_108_land_thr_k
    )
    is_dust = (
        valid_px
        & (~clear_over_snow)
        & (btd_120_108 > dust_btd_120_108_thr)
        & (btd_108_087 > cfg.dust_btd_108_087_thr_k)
        & (ir_108 > cfg.dust_min_t108_k)
        & (sd_t108_3x3 < cfg.dust_max_sd_t108_k)
    )
    cma_dust[is_dust] = CmaDustClass.DUST

    cma_volc = np.full(shape, CmaVolcanicClass.FILL_VALUE, dtype=np.uint8)
    cma_volc[valid_px] = CmaVolcanicClass.NO_VOLCANIC_PLUME
    is_volcanic = (
        valid_px
        & (~clear_over_snow)
        & (btd_120_108 > cfg.volcanic_btd_120_108_thr_k)
        & (btd_087_108 < cfg.volcanic_btd_087_108_max_k)
        & (ir_108 <= cfg.volcanic_max_t108_k)
    )
    cma_volc[is_volcanic] = CmaVolcanicClass.VOLCANIC_PLUME

    # Smoke requires NIR/VIS channels (R0.6, R1.6); in Pure-IR mode it is marked UNDEFINED
    cma_smoke = np.full(shape, CmaSmokeClass.FILL_VALUE, dtype=np.uint8)
    cma_smoke[valid_px] = CmaSmokeClass.UNDEFINED

    # -------------------------------------------------------------------------
    # STEP 12: Quality Assessment (ATBD Section 2.2.1.2.4.1, p. 53)
    # -------------------------------------------------------------------------
    # Mark pixel as QUESTIONABLE if measurement is within `quality_security_margin_k` of threshold,
    # or if spatial/inversion/desert restoral intervened, or at extreme viewing angles (> 75 deg)
    near_t108_thr = np.abs(ir_108 - t108_thr) < cfg.quality_security_margin_k
    near_btd_thr = np.abs(btd_108_120 - btd_108_120_thr) < (0.25 * cfg.quality_security_margin_k)
    questionable = (
        valid_px
        & (
            near_t108_thr
            | (cloudy & near_btd_thr & (~test_t108))
            | desert_restoral
            | inversion_restoral
            | (anc.sat_zenith_deg > 75.0)
        )
    )
    quality = np.full(shape, CmaQuality.NODATA, dtype=np.uint8)
    quality[valid_px] = CmaQuality.GOOD
    quality[questionable] = CmaQuality.QUESTIONABLE

    return CmaOutput(
        cma=cma_bin,
        cma_cover=cma_cov,
        cma_cloudsnow=cma_cs,
        cma_dust=cma_dust,
        cma_volcanic=cma_volc,
        cma_smoke=cma_smoke,
        cma_testlist=testlist,
        cma_status_flag=status_flag,
        cma_quality=quality,
        t108_threshold_k=t108_thr,
        btd_108_120_threshold_k=btd_108_120_thr,
        sd_t108_3x3=sd_t108_3x3,
    )
