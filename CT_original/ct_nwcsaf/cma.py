"""Authoritative NWC SAF GEO-CMA (Cloud Mask v6.0 / v2025) implementation.

Implements Section 2 of the EUMETSAT NWC SAF Cloud ATBD (Issue 1.1.1, 30 May 2025):
- Multispectral threshold sequence (SST split-window, T10.8 thermal contrast with extreme
  cooling correction, T10.8-T12.0 split-window, T8.7-T10.8 ice phase, T10.8-T8.7 high-secant
  low cloud, and 7-channel SEVIRI water-vapour/CO2 low/mid cloud spectral tests)
- Local Spatial Texture test (3x3 standard deviation coherency across T10.8 and BTD)
- Temporal-differencing test (15-min rapid cooling Delta_15min(T10.8) for fast-moving clouds)
- Stationary low/mid cloud 1-hour temporal coherence & 8-connected region growing
- Final spatial filtering (cold nocturnal inversion restoral T7.3-T10.8 > 0.5 K, coastal
  3x3/7x7 clear restoral, snow-edge restoral, and isolated pixel morphological filtering)
- Dust cloud (`cma_dust`) and Volcanic ash plume (`cma_volcanic`) detection
- 28-bit `cma_testlist`, 11-bit `cma_status_flag`, and security-margin `cma_quality`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage

from .ancillary import AtmosphericProfileFields, extract_seviri_7ch
from .config import (
    CmaClass,
    CmaCloudSnowClass,
    CmaDustClass,
    CmaQuality,
    CmaSmokeClass,
    CmaStatusBit,
    CmaTestBit,
    CmaVolcanicClass,
    CtClass,
    NwcSafCmaCtConfig,
)


@dataclass
class CmaResult:
    """Output containers and diagnostics of the NWC SAF GEO-CMA processor."""

    cma: np.ndarray  # uint8: 0 = Cloud-free, 1 = Cloudy, 255 = FillValue
    cma_cloudsnow: np.ndarray  # uint8: 0 = Clear, 1 = Cloud, 2 = Thin ice over snow, 3 = Snow/Ice
    cma_dust: np.ndarray  # uint8: 0 = No dust, 1 = Dust, 2 = Undefined
    cma_volcanic: np.ndarray  # uint8: 0 = No volcanic plume, 1 = Volcanic plume, 2 = Undefined
    cma_smoke: np.ndarray  # uint8: 2 = Undefined (per ATBD Section 2.2.1.2.10)
    cma_testlist: np.ndarray  # uint32: 28-bit bitmask of triggered detection tests
    cma_status_flag: np.ndarray  # uint16: 11-bit processing status flags
    cma_quality: np.ndarray  # uint8: CmaQuality (0=NA, 1=Good, 2=Questionable, 3=Bad, 4=Interpolated)
    # Intermediate fields exposed for GEO-CT and diagnostics
    t108_threshold_k: np.ndarray
    swd_threshold_k: np.ndarray
    ice_btd_threshold_k: np.ndarray
    local_sd_t108_k: np.ndarray
    local_sd_btd_k: np.ndarray


def compute_local_3x3_stats(
    field: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 3x3 local standard deviation, mean, minimum, and maximum over valid pixels."""
    arr = np.where(valid_mask, np.asarray(field, dtype=np.float64), 0.0)
    w = valid_mask.astype(np.float64)
    count = ndimage.uniform_filter(w, size=3, mode="nearest") * 9.0
    sum_x = ndimage.uniform_filter(arr, size=3, mode="nearest") * 9.0
    sum_x2 = ndimage.uniform_filter(arr * arr, size=3, mode="nearest") * 9.0

    safe_count = np.maximum(count, 1.0)
    mean = sum_x / safe_count
    var = np.maximum(0.0, (sum_x2 / safe_count) - mean * mean)
    sd = np.where(count >= 2.0, np.sqrt(var), 0.0).astype(np.float32)
    mean_f32 = np.where(count >= 1.0, mean, field).astype(np.float32)

    arr_for_min = np.where(valid_mask, field, np.inf)
    arr_for_max = np.where(valid_mask, field, -np.inf)
    local_min = ndimage.minimum_filter(arr_for_min, size=3, mode="nearest").astype(np.float32)
    local_max = ndimage.maximum_filter(arr_for_max, size=3, mode="nearest").astype(np.float32)
    return sd, mean_f32, local_min, local_max


def _box_clear_fraction(clear_mask: np.ndarray, size: int) -> np.ndarray:
    """Compute fraction of clear pixels inside a `size x size` neighborhood."""
    return ndimage.uniform_filter(clear_mask.astype(np.float32), size=size, mode="nearest")


def compute_nwcsaf_cma(
    cube_7ch: np.ndarray,
    profile: AtmosphericProfileFields,
    cfg: NwcSafCmaCtConfig,
    *,
    prev_15m_ir108: np.ndarray | None = None,
    prev_1h_cube_7ch: np.ndarray | None = None,
    prev_1h_cma: np.ndarray | None = None,
    prev_1h_ct: np.ndarray | None = None,
    prev_1h_quality: np.ndarray | None = None,
) -> CmaResult:
    """Run the full NWC SAF GEO-CMA v6.0 / v2025 Cloud Mask algorithm on a 7-channel SEVIRI scene."""
    ch = extract_seviri_7ch(cube_7ch)
    wv062 = ch["WV_062"]
    wv073 = ch["WV_073"]
    ir087 = ch["IR_087"]
    ir097 = ch["IR_097"]
    ir108 = ch["IR_108"]
    ir120 = ch["IR_120"]
    ir134 = ch["IR_134"]

    shape = ir108.shape
    valid = (
        np.isfinite(wv062)
        & np.isfinite(wv073)
        & np.isfinite(ir087)
        & np.isfinite(ir097)
        & np.isfinite(ir108)
        & np.isfinite(ir120)
        & np.isfinite(ir134)
    )

    secant = profile.secant
    land = profile.land_mask
    sea = ~land
    coast = profile.coast_mask
    t_surf = profile.t_surf_k
    tcwv = profile.tcwv_kg_m2

    testlist = np.zeros(shape, dtype=np.uint32)
    status_flag = np.zeros(shape, dtype=np.uint16)

    # Set initial status flags (ATBD Section 2.2.2.4, p. 61)
    status_flag = np.where(
        profile.low_level_inversion,
        status_flag | (np.uint16(1) << CmaStatusBit.LOW_LEVEL_THERMAL_INVERSION),
        status_flag,
    )
    status_flag = np.where(
        profile.snow_ice_mask | (t_surf < cfg.cma_extreme_cooling_fskt_k),
        status_flag | (np.uint16(1) << CmaStatusBit.COLD_SNOWY_GROUND_SUSPECTED),
        status_flag,
    )
    status_flag |= np.uint16(1) << CmaStatusBit.NO_METHOD_FOR_SMOKE
    if not cfg.enable_dust_detection:
        status_flag |= np.uint16(1) << CmaStatusBit.NO_METHOD_FOR_DUST
    if not cfg.enable_volcanic_detection:
        status_flag |= np.uint16(1) << CmaStatusBit.NO_METHOD_FOR_VOLCANIC

    # Track whether at least one test is "really successful" (exceeds security margin, ATBD p. 31)
    really_successful_cloud = np.zeros(shape, dtype=bool)
    close_to_threshold_clear = np.zeros(shape, dtype=bool)

    # -------------------------------------------------------------------------
    # STEP 0: Snow/Ice & Thin Ice Cloud over Snow/Ice (ATBD 2.2.1.2.2.15 - 16)
    # -------------------------------------------------------------------------
    snow_detected = valid & profile.snow_ice_mask
    # Thin cirrus above snow/ice: high split-window T10.8 - T12.0 or high ice BTD T8.7 - T10.8
    swd = ir108 - ir120
    ice_btd = ir087 - ir108
    thin_cirrus_over_snow = (
        snow_detected
        & ((swd > 1.35) | (ice_btd > 0.35) | ((wv062 - ir108) > -12.0))
    )
    pure_snow_or_ice = snow_detected & (~thin_cirrus_over_snow)
    testlist = np.where(
        snow_detected,
        testlist | (np.uint32(1) << CmaTestBit.SNOW_T38),
        testlist,
    )

    # Pixels not covered by pure snow/ice proceed to cloud detection tests
    processable = valid & (~pure_snow_or_ice)

    # -------------------------------------------------------------------------
    # STEP 1: Test on SST over Sea (ATBD Section 2.2.1.2.2.1, p. 14)
    # -------------------------------------------------------------------------
    # SST(T10.8, T12.0) = 0.977*(T10.8 - 273.15)
    #   + (0.075*(sst_clim - 273.15) + 1.127*(sec - 1))*(T10.8 - T12.0)
    #   + 1.156 + 273.15
    sst_retrieval = (
        0.977 * (ir108 - 273.15)
        + (0.075 * (t_surf - 273.15) + 1.127 * (secant - 1.0)) * swd
        + 1.156
        + 273.15
    ).astype(np.float32)

    # Secant-dependent offset: 1.5 K for sec <= 2.0 (theta <= 60 deg) up to 2.5 K for sec >= 5.0
    sec_norm = np.clip((secant - 2.0) / 3.0, 0.0, 1.0)
    sst_offset = (
        cfg.cma_sst_offset_low_sec_k
        + sec_norm * (cfg.cma_sst_offset_high_sec_k - cfg.cma_sst_offset_low_sec_k)
        + np.where(coast, cfg.cma_sst_coast_extra_offset_k, 0.0)
    ).astype(np.float32)
    sst_threshold = t_surf - sst_offset

    # Clear-sky sounding guard: when all 4 sounding absorption channels (WV_062, WV_073,
    # IR_097, IR_134) show a deep transparent troposphere and T10.8 is within 3.5 K of T_surf,
    # moderate BTDs are surface/boundary-layer continuum effects rather than clouds.
    clear_sounding_votes = (
        ((wv062 - ir108) <= -18.0).astype(np.int32)
        + ((wv073 - ir108) <= -8.0).astype(np.int32)
        + ((ir097 - ir108) <= -10.0).astype(np.int32)
        + ((ir134 - ir108) <= -8.0).astype(np.int32)
    )
    deep_clear_sounding = (
        (not profile.nwp_available)
        & (clear_sounding_votes >= 3)
        & (ir108 >= t_surf - 3.5)
        & (swd < 2.1)
    )

    sst_applicable = (
        processable
        & sea
        & (t_surf > cfg.min_sea_freezing_sst_k)
        & (profile.nwp_available | (~deep_clear_sounding))
    )
    sst_diff = sst_threshold - sst_retrieval  # > 0 means cloud
    test_sst = sst_applicable & (sst_diff > 0.0)
    really_successful_cloud |= test_sst & (sst_diff > cfg.cma_margin_sst_k)
    close_to_threshold_clear |= sst_applicable & (~test_sst) & (np.abs(sst_diff) < cfg.cma_margin_sst_k)

    # -------------------------------------------------------------------------
    # STEP 2: Test on T10.8um (ATBD Section 2.2.1.2.2.2, pp. 14-16)
    # -------------------------------------------------------------------------
    wv_factor = np.clip((tcwv - 15.0) / 30.0, 0.0, 1.5)
    base_offset = np.where(land, cfg.cma_t108_base_offset_land_k, cfg.cma_t108_base_offset_sea_k)
    t108_thr = (
        t_surf
        - base_offset
        - cfg.cma_t108_wv_secant_slope_k * (secant - 1.0) * (1.0 + 0.5 * wv_factor)
    ).astype(np.float32)

    # Extreme cooling modification over land (ATBD p. 16)
    extreme_cooling = land & (t_surf < cfg.cma_extreme_cooling_fskt_k)
    t108_thr = np.where(
        extreme_cooling & (t108_thr >= cfg.cma_extreme_cooling_t108_pivot_k),
        t108_thr - cfg.cma_extreme_cooling_offset_k,
        np.where(
            extreme_cooling & (t108_thr < cfg.cma_extreme_cooling_t108_pivot_k),
            t108_thr
            - cfg.cma_extreme_cooling_offset_k
            - 0.4 * (cfg.cma_extreme_cooling_t108_pivot_k - t108_thr),
            t108_thr,
        ),
    ).astype(np.float32)

    # Guarantee that unambiguously cold cloud tops (< 260 K) are always caught even over cold sea/land
    t108_thr = np.maximum(t108_thr, cfg.cma_definite_cloud_t108_k).astype(np.float32)

    t108_diff = t108_thr - ir108  # > 0 means cloud
    test_t108 = processable & (t108_diff > 0.0)
    really_successful_cloud |= test_t108 & (t108_diff > cfg.cma_margin_t108_k)
    close_to_threshold_clear |= processable & (~test_t108) & (np.abs(t108_diff) < cfg.cma_margin_t108_k)

    testlist = np.where(
        test_sst | test_t108,
        testlist | (np.uint32(1) << CmaTestBit.T108_OR_SST),
        testlist,
    )

    # -------------------------------------------------------------------------
    # STEP 3: Test on T10.8um - T12.0um Split-Window (ATBD Section 2.2.1.2.2.3)
    # -------------------------------------------------------------------------
    # Per ATBD Figure 2 (p. 17), clear-sky T10.8 - T12.0 increases strongly with
    # total column water vapour and warm surface temperature due to continuum absorption.
    swd_base = np.where(land, cfg.cma_swd_base_land_k, cfg.cma_swd_base_sea_k)
    warm_moist_allowance = np.where(
        profile.nwp_available,
        0.0,
        0.045 * np.clip(t_surf - 280.0, 0.0, 25.0),
    )
    swd_thr = (
        swd_base
        + cfg.cma_swd_secant_slope_k * (secant - 1.0) * (1.0 + 0.35 * wv_factor)
        + warm_moist_allowance
    ).astype(np.float32)

    swd_applicable = (
        processable
        & (sea | (ir108 < cfg.cma_swd_max_land_t108_k))
        & (~deep_clear_sounding)
    )
    swd_diff = swd - swd_thr  # > 0 means cloud
    test_swd = swd_applicable & (swd_diff > 0.0)
    really_successful_cloud |= test_swd & (swd_diff > cfg.cma_margin_swd_k)
    close_to_threshold_clear |= swd_applicable & (~test_swd) & (np.abs(swd_diff) < cfg.cma_margin_swd_k)
    testlist = np.where(
        test_swd,
        testlist | (np.uint32(1) << CmaTestBit.T108_MINUS_T120),
        testlist,
    )

    # -------------------------------------------------------------------------
    # STEP 4: Test on T8.7um - T10.8um Thin Cirrus / Ice (ATBD Section 2.2.1.2.2.4)
    # -------------------------------------------------------------------------
    ice_base = np.where(land, cfg.cma_ice_btd_base_land_k, cfg.cma_ice_btd_base_sea_k)
    ice_btd_thr = (
        ice_base
        + cfg.cma_ice_btd_secant_slope_k * (secant - 1.0)
        + np.where(extreme_cooling & (t108_thr < 250.0), 0.4, 0.0)
    ).astype(np.float32)

    # Guard against warm desert quartz reststrahlen surface effect where T10.8 is near T_surf
    # and WV_073 is very dry/warm relative to upper clouds
    ice_applicable = processable & (ir108 < t_surf - 1.5)
    ice_diff = ice_btd - ice_btd_thr
    test_ice_btd = ice_applicable & (ice_diff > 0.0)
    really_successful_cloud |= test_ice_btd & (ice_diff > cfg.cma_margin_btd_k)
    close_to_threshold_clear |= ice_applicable & (~test_ice_btd) & (np.abs(ice_diff) < cfg.cma_margin_btd_k)
    testlist = np.where(
        test_ice_btd,
        testlist | (np.uint32(1) << CmaTestBit.T87_MINUS_T108),
        testlist,
    )

    # -------------------------------------------------------------------------
    # STEP 5: Test on T10.8um - T8.7um & 7-Ch IR Low/Mid Cloud Detection
    # (ATBD Section 2.2.1.2.2.9, p. 23 & Section 2.2.1.2.3.5, p. 36)
    # -------------------------------------------------------------------------
    t108_minus_t87 = ir108 - ir087
    # Official ATBD formula for high satellite viewing angles (secant > 1.5):
    # T10.8T8.7threshold = 3.7 + 0.3 * sec(theta_sat)
    high_sec_thr = (
        cfg.cma_low_cloud_t108_t87_intercept_k + cfg.cma_low_cloud_t108_t87_secant_slope_k * secant
    ).astype(np.float32)
    test_t108_t87_high_sec = (
        processable
        & (secant > cfg.cma_low_cloud_min_secant)
        & (t108_minus_t87 > high_sec_thr)
    )
    testlist = np.where(
        test_t108_t87_high_sec,
        testlist | (np.uint32(1) << CmaTestBit.T108_MINUS_T87),
        testlist,
    )
    really_successful_cloud |= test_t108_t87_high_sec & (
        (t108_minus_t87 - high_sec_thr) > cfg.cma_margin_btd_k
    )

    # Pure-SEVIRI 7-channel IR low/mid cloud spectral test (replacing 3.8 um when unavailable):
    # 1) Warm low water cloud (stratus/fog): water droplets have lower emissivity at 8.7 um than
    #    at 10.8 um -> T10.8 - T8.7 > 2.4 K, while T10.8 is at least 1.0 K cooler than T_surf.
    # 2) Mid-level cloud under warm atmosphere: WV_073 - IR_108 > -16.0 K and IR_134 - IR_108 > -11.0 K
    #    with T10.8 < T_surf - 2.0 K.
    test_pure_ir_low_stratus = (
        processable
        & (t108_minus_t87 > cfg.cma_pure_ir_low_stratus_t108_t87_k)
        & (ir108 < t_surf - 1.5)
        & (~deep_clear_sounding)
    )
    test_pure_ir_mid_wv_co2 = (
        processable
        & ((wv073 - ir108) > cfg.cma_pure_ir_wv073_t108_opaque_k)
        & ((wv073 - ir108) <= cfg.cma_cold_inversion_t73_minus_t108_min_k)
        & ((ir134 - ir108) > -11.5)
        & (ir108 < t_surf - 2.0)
    )
    test_pure_ir_spectral = test_pure_ir_low_stratus | test_pure_ir_mid_wv_co2
    testlist = np.where(
        test_pure_ir_low_stratus,
        testlist | (np.uint32(1) << CmaTestBit.T108_MINUS_T38),
        testlist,
    )
    testlist = np.where(
        test_pure_ir_mid_wv_co2,
        testlist | (np.uint32(1) << CmaTestBit.RTTOV_T108),
        testlist,
    )
    really_successful_cloud |= test_pure_ir_low_stratus & (
        (t108_minus_t87 - cfg.cma_pure_ir_low_stratus_t108_t87_k) > cfg.cma_margin_btd_k
    )
    really_successful_cloud |= test_pure_ir_mid_wv_co2 & (
        ((wv073 - ir108) - cfg.cma_pure_ir_wv073_t108_opaque_k) > 1.0
    )

    # -------------------------------------------------------------------------
    # STEP 6: Local Spatial Texture Test (ATBD Section 2.2.1.2.2.17, pp. 29-30)
    # -------------------------------------------------------------------------
    # Per ATBD p. 30: SD, MIN, MAX are computed using surrounding pixels provided
    # they correspond to the same surface type (sea or land) and are not snow/ice.
    sea_valid = processable & sea & (~snow_detected)
    land_valid = processable & land & (~snow_detected)

    sd_t108_sea, mean_t108_sea, _, max_t108_sea = compute_local_3x3_stats(ir108, sea_valid)
    sd_t108_land, mean_t108_land, _, max_t108_land = compute_local_3x3_stats(ir108, land_valid)
    sd_t108 = np.where(land, sd_t108_land, sd_t108_sea)
    mean_t108 = np.where(land, mean_t108_land, mean_t108_sea)
    max_t108 = np.where(land, max_t108_land, max_t108_sea)

    sd_swd_sea, _, _, _ = compute_local_3x3_stats(swd, sea_valid)
    sd_swd_land, _, _, _ = compute_local_3x3_stats(swd, land_valid)
    sd_btd87_sea, _, _, _ = compute_local_3x3_stats(t108_minus_t87, sea_valid)
    sd_btd87_land, _, _, _ = compute_local_3x3_stats(t108_minus_t87, land_valid)
    sd_btd = np.where(
        land,
        np.maximum(sd_swd_land, sd_btd87_land),
        np.maximum(sd_swd_sea, sd_btd87_sea),
    )

    # Over Sea: SD(T10.8) > 0.6 K and SD(BTD) > 0.25 K and T10.8 < MAX(T10.8) - 2*noise(T10.8)
    #           and pixel is on the colder side of the local 3x3 neighborhood
    # Over Land: SD(T10.8) > 1.2 K and SD(BTD) > 0.45 K and T10.8 < MEAN(T10.8) - 2*noise(T10.8)
    test_texture_sea = (
        sea_valid
        & (sd_t108 > cfg.cma_texture_sd_t108_sea_k)
        & (sd_btd > cfg.cma_texture_sd_btd_sea_k)
        & (ir108 < (max_t108 - 2.0 * cfg.cma_texture_noise_t108_k))
        & (ir108 < (mean_t108 - cfg.cma_texture_noise_t108_k))
    )
    test_texture_land = (
        land_valid
        & (sd_t108 > cfg.cma_texture_sd_t108_land_k)
        & (sd_btd > cfg.cma_texture_sd_btd_land_k)
        & (ir108 < (mean_t108 - 2.0 * cfg.cma_texture_noise_t108_k))
        & (ir108 < (t_surf - 1.5))
    )
    test_texture = test_texture_sea | test_texture_land
    testlist = np.where(
        test_texture,
        testlist | (np.uint32(1) << CmaTestBit.LOCAL_SPATIAL_TEXTURE),
        testlist,
    )
    tex_thr_t108 = np.where(land, cfg.cma_texture_sd_t108_land_k, cfg.cma_texture_sd_t108_sea_k)
    really_successful_cloud |= test_texture & (
        sd_t108 > (1.0 + cfg.cma_margin_texture_rel) * tex_thr_t108
    )

    # Combine spectral & textural cloud detections
    cloudy = (
        thin_cirrus_over_snow
        | test_sst
        | test_t108
        | test_swd
        | test_ice_btd
        | test_t108_t87_high_sec
        | test_pure_ir_spectral
        | test_texture
    )

    # -------------------------------------------------------------------------
    # STEP 7: Temporal Test for Rapidly Moving/Developing Clouds (ATBD 2.2.1.2.4)
    # -------------------------------------------------------------------------
    detected_by_temporal = np.zeros(shape, dtype=bool)
    if prev_15m_ir108 is not None:
        prev15 = np.asarray(prev_15m_ir108, dtype=np.float32)
        if prev15.shape == shape:
            status_flag |= np.uint16(1) << CmaStatusBit.TEMPORAL_ALGORITHM_PASSED
            dt108_15m = ir108 - prev15
            dt_thr = np.where(land, cfg.cma_temporal_dt108_land_k, cfg.cma_temporal_dt108_sea_k)
            # Applied to pixels still classified as clear (or snowy)
            cand_temp = valid & (~cloudy) & np.isfinite(prev15)
            detected_by_temporal = cand_temp & (dt108_15m < dt_thr)
            cloudy |= detected_by_temporal
            testlist = np.where(
                detected_by_temporal,
                testlist | (np.uint32(1) << CmaTestBit.TEMPORAL_DIFFERENCING),
                testlist,
            )

    # -------------------------------------------------------------------------
    # STEP 8: Stationary Low/Mid Cloud 1h Temporal Coherence & Region Growing
    # (ATBD Section 2.2.1.2.5, pp. 38-41)
    # -------------------------------------------------------------------------
    detected_by_stationary = np.zeros(shape, dtype=bool)
    detected_by_expansion = np.zeros(shape, dtype=bool)
    if (
        prev_1h_cube_7ch is not None
        and prev_1h_cma is not None
        and prev_1h_ct is not None
    ):
        ch_1h = extract_seviri_7ch(prev_1h_cube_7ch)
        ir108_1h = ch_1h["IR_108"]
        ir087_1h = ch_1h["IR_087"]
        ir120_1h = ch_1h["IR_120"]
        cma_1h = np.asarray(prev_1h_cma)
        ct_1h = np.asarray(prev_1h_ct)

        status_flag |= np.uint16(1) << CmaStatusBit.TEMPORAL_ALGORITHM_PASSED
        was_low_or_mid_1h = (cma_1h == CmaClass.CLOUDY) & np.isin(
            ct_1h,
            [
                int(CtClass.VERY_LOW_CLOUDS),
                int(CtClass.LOW_CLOUDS),
                int(CtClass.MID_LEVEL_CLOUDS),
            ],
        )
        if prev_1h_quality is not None:
            was_low_or_mid_1h &= np.asarray(prev_1h_quality) == int(CmaQuality.GOOD)

        d1h_t108 = np.abs(ir108 - ir108_1h)
        d1h_btd87 = np.abs((ir108 - ir087) - (ir108_1h - ir087_1h))
        d1h_swd = np.abs((ir108 - ir120) - (ir108_1h - ir120_1h))

        coherence_land = (
            land
            & (d1h_t108 < cfg.cma_stationary_dt108_max_k)
            & (d1h_btd87 < cfg.cma_stationary_dbtd87_land_max_k)
        )
        coherence_sea = (
            sea
            & (d1h_t108 < cfg.cma_stationary_dt108_max_k)
            & (d1h_swd < cfg.cma_stationary_dswd_sea_max_k)
        )
        detected_by_stationary = (
            valid & (~cloudy) & (~pure_snow_or_ice) & was_low_or_mid_1h & (coherence_land | coherence_sea)
        )
        cloudy |= detected_by_stationary
        testlist = np.where(
            detected_by_stationary,
            testlist | (np.uint32(1) << CmaTestBit.STATIONARY_TWILIGHT),
            testlist,
        )

        # Step 2: Spatial expansion of connected stationary seed groups > 8 pixels (ATBD 2.2.1.2.5.2)
        if np.any(detected_by_stationary):
            struct8 = np.ones((3, 3), dtype=bool)
            labels, num_labels = ndimage.label(detected_by_stationary, structure=struct8)
            if num_labels > 0:
                counts = np.bincount(labels.ravel())
                for lbl in range(1, num_labels + 1):
                    if counts[lbl] <= cfg.cma_stationary_min_seed_pixels:
                        continue
                    seed_mask = labels == lbl
                    avg_108 = float(np.mean(ir108[seed_mask]))
                    # Similar thermal characteristics: AVG10.8 - 5.0 K < T10.8 < AVG10.8 + 0.5 K
                    compatible = (
                        valid
                        & (~cloudy)
                        & (~pure_snow_or_ice)
                        & (ir108 > (avg_108 - 5.0))
                        & (ir108 < (avg_108 + 0.5))
                    )
                    grown = ndimage.binary_propagation(
                        seed_mask, structure=struct8, mask=compatible | seed_mask
                    )
                    new_pixels = grown & (~seed_mask)
                    n_new = int(np.sum(new_pixels))
                    if 0 < n_new <= cfg.cma_stationary_max_grow_pixels:
                        detected_by_expansion |= new_pixels

        if np.any(detected_by_expansion):
            cloudy |= detected_by_expansion
            testlist = np.where(
                detected_by_expansion,
                testlist | (np.uint32(1) << CmaTestBit.SPATIAL_EXTENSION_TWILIGHT),
                testlist,
            )

    # -------------------------------------------------------------------------
    # STEP 9: Final Spatial Filtering (ATBD Section 2.2.1.2.7, pp. 49-52)
    # -------------------------------------------------------------------------
    reclassified_to_clear_questionable = np.zeros(shape, dtype=bool)
    reclassified_interpolated = np.zeros(shape, dtype=bool)

    if cfg.enable_spatial_filtering:
        # 9a. Cold nocturnal clear-sky inversion restoral (ATBD p. 50):
        # Restores pixels detected ONLY by T10.8thr test when t108thr < 250 K and T7.3 - T10.8 > 0.5 K
        allowed_inv_bits = (
            (np.uint32(1) << CmaTestBit.T108_OR_SST)
            | (np.uint32(1) << CmaTestBit.RTTOV_T108)
        )
        only_t108_bit = (testlist != 0) & ((testlist & (~allowed_inv_bits)) == 0)
        cold_inv_restore = (
            cloudy
            & only_t108_bit
            & (t108_thr < cfg.cma_cold_inversion_t108thr_max_k)
            & ((wv073 - ir108) > cfg.cma_cold_inversion_t73_minus_t108_min_k)
        )
        if np.any(cold_inv_restore):
            cloudy &= ~cold_inv_restore
            reclassified_to_clear_questionable |= cold_inv_restore
            testlist = np.where(
                cold_inv_restore,
                testlist | (np.uint32(1) << CmaTestBit.SPATIAL_FILTER_TO_CLEAR),
                testlist,
            )

        # 9b. Coastal false-alarm filtering (ATBD p. 50):
        # Over sea (texture-only) or over land (coastal), restored clear when 3x3 box >= 10% clear
        # and 7x7 box >= 50% clear.
        if np.any(coast):
            clear_now = valid & (~cloudy)
            clear_frac_3 = _box_clear_fraction(clear_now, 3)
            clear_frac_7 = _box_clear_fraction(clear_now, 7)
            only_tex_bit = testlist == (np.uint32(1) << CmaTestBit.LOCAL_SPATIAL_TEXTURE)
            coastal_restore = (
                cloudy
                & coast
                & ((sea & only_tex_bit) | land)
                & (clear_frac_3 >= 0.10)
                & (clear_frac_7 >= 0.50)
            )
            if np.any(coastal_restore):
                cloudy &= ~coastal_restore
                reclassified_interpolated |= coastal_restore
                testlist = np.where(
                    coastal_restore,
                    testlist | (np.uint32(1) << CmaTestBit.SPATIAL_FILTER_TO_CLEAR),
                    testlist,
                )

        # 9c. Snow-edge local spatial texture filtering (ATBD p. 51)
        if np.any(pure_snow_or_ice):
            struct3 = np.ones((3, 3), dtype=bool)
            snow_dil = ndimage.binary_dilation(pure_snow_or_ice, structure=struct3, iterations=2)
            snow_ero = ndimage.binary_erosion(pure_snow_or_ice, structure=struct3, iterations=2)
            snow_edge_band = np.logical_xor(snow_dil, snow_ero)
            only_tex_bit = testlist == (np.uint32(1) << CmaTestBit.LOCAL_SPATIAL_TEXTURE)
            clear_now = valid & (~cloudy) & (~pure_snow_or_ice)
            has_clear_3x3 = ndimage.maximum_filter(clear_now, size=3, mode="nearest")
            non_tex_cloudy = cloudy & (~only_tex_bit)
            has_non_tex_cloudy_3x3 = ndimage.maximum_filter(non_tex_cloudy, size=3, mode="nearest")
            snow_edge_restore = (
                cloudy
                & snow_edge_band
                & only_tex_bit
                & has_clear_3x3
                & (~has_non_tex_cloudy_3x3)
            )
            if np.any(snow_edge_restore):
                cloudy &= ~snow_edge_restore
                reclassified_interpolated |= snow_edge_restore
                testlist = np.where(
                    snow_edge_restore,
                    testlist | (np.uint32(1) << CmaTestBit.SPATIAL_FILTER_TO_CLEAR),
                    testlist,
                )

        # 9d. Isolated pixel morphological filtering (ATBD p. 52):
        # - Isolated cloudy pixels (0 cloudy 8-neighbors) detected by low-cloud/3.8um/texture test -> clear
        # - Isolated cloud-free pixels (all 8 neighbors cloudy) -> cloudy
        kernel_neighbors = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
        cloudy_neighbor_count = ndimage.convolve(
            cloudy.astype(np.int32), kernel_neighbors, mode="nearest"
        )
        valid_neighbor_count = ndimage.convolve(
            valid.astype(np.int32), kernel_neighbors, mode="nearest"
        )
        low_conf_bits = (
            (np.uint32(1) << CmaTestBit.T108_MINUS_T38)
            | (np.uint32(1) << CmaTestBit.LOCAL_SPATIAL_TEXTURE)
        )
        only_low_conf = (testlist & (~low_conf_bits)) == 0
        isolated_cloudy_to_clear = (
            valid
            & cloudy
            & only_low_conf
            & (valid_neighbor_count == 8)
            & (cloudy_neighbor_count == 0)
        )
        isolated_clear_to_cloudy = (
            valid
            & (~cloudy)
            & (~pure_snow_or_ice)
            & (valid_neighbor_count == 8)
            & (cloudy_neighbor_count == 8)
        )
        if np.any(isolated_cloudy_to_clear):
            cloudy &= ~isolated_cloudy_to_clear
            reclassified_interpolated |= isolated_cloudy_to_clear
            testlist = np.where(
                isolated_cloudy_to_clear,
                testlist | (np.uint32(1) << CmaTestBit.SPATIAL_FILTER_TO_CLEAR),
                testlist,
            )
        if np.any(isolated_clear_to_cloudy):
            cloudy |= isolated_clear_to_cloudy
            reclassified_interpolated |= isolated_clear_to_cloudy
            testlist = np.where(
                isolated_clear_to_cloudy,
                testlist | (np.uint32(1) << CmaTestBit.SPATIAL_FILTER_TO_CLOUD),
                testlist,
            )

    # -------------------------------------------------------------------------
    # STEP 10: Dust (`cma_dust`) and Volcanic Ash (`cma_volcanic`) Identification
    # (ATBD Sections 2.2.1.2.8 & 2.2.1.2.9, pp. 52-55)
    # -------------------------------------------------------------------------
    t120_minus_t108 = ir120 - ir108  # Reverse split-window difference (positive for silicate dust/ash)

    cma_dust = np.full(shape, int(CmaDustClass.FILL_VALUE), dtype=np.uint8)
    if cfg.enable_dust_detection:
        # ATBD Section 2.2.1.2.8: silicate mineral dust has reverse absorption T12.0 - T10.8 > thr,
        # elevated T8.7 - T10.8 > -(2.5 - 0.17*(1 - sec)), moderate cooling vs surface, and
        # strong spatial homogeneity SD(T10.8) < 0.4 K (sea) / 3.0 K (land)
        dust_btd87_min = -(2.5 - 0.17 * (1.0 - secant))
        dust_sea = (
            valid
            & sea
            & (t120_minus_t108 > cfg.dust_t120_minus_t108_min_k)
            & (ice_btd > dust_btd87_min)
            & ((ir108 - t_surf) > (-10.0 - 5.0 * (secant - 1.0)))
            & (sd_t108 < cfg.dust_sd_t108_max_sea_k)
            & (ir108 > 265.0)
        )
        dust_land = (
            valid
            & land
            & (ir108 > 273.15)
            & (ir108 < 315.15)
            & (t120_minus_t108 > cfg.dust_t120_minus_t108_min_k)
            & (ice_btd > cfg.dust_t87_minus_t108_min_k)
            & (sd_t108 < cfg.dust_sd_t108_max_land_k)
        )
        dust_detected = dust_sea | dust_land
        cma_dust = np.where(
            ~valid,
            int(CmaDustClass.FILL_VALUE),
            np.where(dust_detected, int(CmaDustClass.DUST), int(CmaDustClass.NO_DUST)),
        ).astype(np.uint8)
    else:
        cma_dust = np.where(valid, int(CmaDustClass.UNDEFINED), int(CmaDustClass.FILL_VALUE)).astype(
            np.uint8
        )

    cma_volcanic = np.full(shape, int(CmaVolcanicClass.FILL_VALUE), dtype=np.uint8)
    if cfg.enable_volcanic_detection:
        # ATBD Section 2.2.1.2.9 (p. 54): T12.0 - T10.8 > T12.0T10.8threshold_volcan
        # Decreases linearly from 1.2 K (at T10.8 <= 223.15 K) down to 0.65 K (at T10.8 >= t108_thr + 20 K),
        # plus secant offset (0 K at sec=1 up to 1.0 K at sec=5).
        upper_t = np.maximum(t108_thr + 20.0, 245.0)
        temp_frac = np.clip((ir108 - 223.15) / np.maximum(upper_t - 223.15, 1.0), 0.0, 1.0)
        sec_volc_offset = np.clip((secant - 1.0) / 4.0, 0.0, 1.0) * cfg.volcanic_secant_max_offset_k
        volc_thr = (
            cfg.volcanic_t120_minus_t108_cold_k * (1.0 - temp_frac)
            + 0.65 * temp_frac
            + sec_volc_offset
        ).astype(np.float32)

        # Volcanic ash plume: strong reverse split-window T12.0 - T10.8 > volc_thr,
        # positive tri-spectral signature (T8.7 - T10.8 > -0.5 K), and plume cooler than surface + 20 K
        volc_detected = (
            valid
            & (t120_minus_t108 > volc_thr)
            & (ir108 < (t108_thr + 20.0))
            & (ice_btd > -0.5)
            & (~(cma_dust == int(CmaDustClass.DUST)))
        )
        cma_volcanic = np.where(
            ~valid,
            int(CmaVolcanicClass.FILL_VALUE),
            np.where(
                volc_detected,
                int(CmaVolcanicClass.VOLCANIC_PLUME),
                int(CmaVolcanicClass.NO_VOLCANIC_PLUME),
            ),
        ).astype(np.uint8)
    else:
        cma_volcanic = np.where(
            valid, int(CmaVolcanicClass.UNDEFINED), int(CmaVolcanicClass.FILL_VALUE)
        ).astype(np.uint8)

    cma_smoke = np.where(
        valid, int(CmaSmokeClass.UNDEFINED), int(CmaSmokeClass.FILL_VALUE)
    ).astype(np.uint8)

    # -------------------------------------------------------------------------
    # STEP 11: Assemble `cma`, `cma_cloudsnow`, and `cma_quality` (ATBD 2.2.2.4)
    # -------------------------------------------------------------------------
    cma_out = np.where(
        ~valid,
        int(CmaClass.FILL_VALUE),
        np.where(cloudy, int(CmaClass.CLOUDY), int(CmaClass.CLOUD_FREE)),
    ).astype(np.uint8)

    cma_cloudsnow = np.full(shape, int(CmaCloudSnowClass.FILL_VALUE), dtype=np.uint8)
    cma_cloudsnow = np.where(
        ~valid,
        int(CmaCloudSnowClass.FILL_VALUE),
        np.where(
            thin_cirrus_over_snow,
            int(CmaCloudSnowClass.THIN_ICE_CLOUD_OVER_SNOW),
            np.where(
                pure_snow_or_ice,
                int(CmaCloudSnowClass.SNOW_OR_ICE),
                np.where(
                    cloudy,
                    int(CmaCloudSnowClass.CLOUD),
                    int(CmaCloudSnowClass.CLOUD_FREE),
                ),
            ),
        ),
    ).astype(np.uint8)

    # Quality assessment (ATBD Section 2.2.1.2.2.18, p. 31):
    # - Interpolated (4): pixels reclassified by coastal/snow-edge/isolated spatial filtering
    # - Questionable (2): pixels detected by temporal/stationary tests or cold inversion restoral
    # - Bad (3): cloudy pixels where no test exceeded its security margin, or clear pixels where
    #            at least one test was closer than its security margin
    # - Good (1): all other valid pixels
    quality = np.full(shape, int(CmaQuality.NA), dtype=np.uint8)
    is_good_cloudy = cloudy & really_successful_cloud
    is_bad_cloudy = cloudy & (~really_successful_cloud)
    is_bad_clear = (~cloudy) & (~pure_snow_or_ice) & close_to_threshold_clear
    is_questionable = (
        detected_by_temporal | detected_by_stationary | detected_by_expansion | reclassified_to_clear_questionable
    )

    quality = np.where(valid, int(CmaQuality.GOOD), quality)
    quality = np.where(valid & (is_bad_cloudy | is_bad_clear), int(CmaQuality.BAD), quality)
    quality = np.where(valid & is_good_cloudy, int(CmaQuality.GOOD), quality)
    quality = np.where(valid & is_questionable, int(CmaQuality.QUESTIONABLE), quality)
    quality = np.where(valid & reclassified_interpolated, int(CmaQuality.INTERPOLATED), quality)

    return CmaResult(
        cma=cma_out,
        cma_cloudsnow=cma_cloudsnow,
        cma_dust=cma_dust,
        cma_volcanic=cma_volcanic,
        cma_smoke=cma_smoke,
        cma_testlist=testlist,
        cma_status_flag=status_flag,
        cma_quality=quality.astype(np.uint8),
        t108_threshold_k=t108_thr,
        swd_threshold_k=swd_thr,
        ice_btd_threshold_k=ice_btd_thr,
        local_sd_t108_k=sd_t108,
        local_sd_btd_k=sd_btd,
    )
