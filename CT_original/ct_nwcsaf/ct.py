"""Authoritative NWC SAF GEO-CT (Cloud Type v5.0 / v2025) 15-class processor.

Implements Section 3 of the EUMETSAT NWC SAF Cloud ATBD (Issue 1.1.1, 30 May 2025):
- 15 official NWC SAF cloud & surface categories (`CtClass` 1..15, 0 = Non-processed)
- Dynamic NWP / Pure-SEVIRI atmospheric temperature profile height thresholds:
  `maxT108vh`, `maxT108hi`, `maxT108med`, `maxT108low`, `T108interthr`, `T108T73thrlow`
- Separation of High Semitransparent clouds (Classes 11, 12, 13, 14, 15) and Fractional
  clouds (Class 10) from Opaque clouds (Classes 5, 6, 7, 8, 9) using `T10.8 - T12.0`,
  `T8.7 - T10.8`, spatial log-variance `VarilogT10.8`, and 7-channel SEVIRI CO2/WV slicing
- Water-vapour `T10.8 - T7.3 > T108T73thrlow` guard and secant-dependent `T8.7 - T10.8`
  low-level thermal inversion reclassification (ATBD Section 3.2.1.2.2.4, p. 77)
- `ct_cumuliform` (Stratiform/Cumuliform/Mixed), `ct_multilayer`, 6-bit `ct_status_flag`,
  `ct_quality` (0.2 K security margin, ATBD p. 78), and 6-class RDT/CI bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage

from .ancillary import AtmosphericProfileFields, extract_seviri_7ch
from .cma import CmaResult
from .config import (
    CmaClass,
    CmaCloudSnowClass,
    CmaQuality,
    CtClass,
    CtCumuliformClass,
    CtMultilayerClass,
    CtStatusBit,
    NwcSafCmaCtConfig,
    RdtBridgeCtClass,
)


@dataclass
class CtThresholds:
    """Spatially varying NWC SAF GEO-CT brightness temperature thresholds (in Kelvin)."""

    max_t108_vh_k: np.ndarray
    max_t108_hi_k: np.ndarray
    max_t108_med_k: np.ndarray
    max_t108_low_k: np.ndarray
    t108_inter_thr_k: np.ndarray
    t108_t73_thr_low_k: np.ndarray
    t108_t120_opaque_k: np.ndarray
    t108_t120_thick_k: np.ndarray
    t87_t108_opaque_k: np.ndarray


@dataclass
class CtResult:
    """Output containers and diagnostics of the NWC SAF GEO-CT processor."""

    ct: np.ndarray  # uint8: Official 15-class NWC SAF Cloud Type (1..15, 0=Non-processed, 255=FillValue)
    ct_cumuliform: np.ndarray  # uint8: CtCumuliformClass (1=Stratiform, 2=Cumuliform, 3=Mixed, 4=Clear)
    ct_multilayer: np.ndarray  # uint8: CtMultilayerClass (0=No, 1=Multilayer, 2=Clear)
    ct_status_flag: np.ndarray  # uint8: 6-bit status flag (ATBD Section 3.2.2.4, p. 83)
    ct_quality: np.ndarray  # uint8: CmaQuality (1=Good, 2=Questionable, 3=Bad)
    rdt_bridge_class: np.ndarray  # uint8: 6-class taxonomy (0..5) compatible with RDT & CI
    opaque_cloud_mask: np.ndarray  # bool: True for opaque clouds (Classes 5..9 or thick convective core)
    semitransparent_mask: np.ndarray  # bool: True for semitransparent clouds (Classes 11..15)
    thresholds: CtThresholds


def compute_ct_thresholds(
    profile: AtmosphericProfileFields,
    cfg: NwcSafCmaCtConfig,
) -> CtThresholds:
    """Compute the exact NWC SAF GEO-CT thresholds from Section 3.2.1.2.2 (ATBD pp. 71-77)."""
    secant = profile.secant
    t500 = profile.t_500_k
    t700 = profile.t_700_k
    t850 = profile.t_850_k
    ttropo = profile.t_tropo_k
    tcwv = profile.tcwv_kg_m2

    # Official ATBD Section 3.2.1.2.2.4 formulas (p. 77):
    # maxT108hi = 0.5 * T500hPa - 0.2 * T700hPa + 178 K
    # maxT108med = 0.8 * T850hPa + 0.2 * T700hPa - 8 K
    # maxT108low = 1.2 * T850hPa - 0.2 * T700hPa - 5 K
    # maxT108vh = 0.4 * T500hPa + 0.6 * Ttropo - 5 K (or maxT108hi - 25 K if Ttropo unavailable)
    max_t108_hi = (0.5 * t500 - 0.2 * t700 + 178.0).astype(np.float32)
    max_t108_med = (0.8 * t850 + 0.2 * t700 - 8.0).astype(np.float32)
    max_t108_low = (1.2 * t850 - 0.2 * t700 - 5.0).astype(np.float32)

    if profile.tropo_available:
        max_t108_vh = (0.4 * t500 + 0.6 * ttropo - 5.0).astype(np.float32)
    else:
        max_t108_vh = (max_t108_hi - float(cfg.ct_vh_fallback_offset_k)).astype(np.float32)

    # Enforce strict monotonic ordering across pressure levels even under extreme inversions:
    # maxT108vh < maxT108hi < maxT108med < maxT108low
    max_t108_vh = np.minimum(max_t108_vh, max_t108_hi - 6.0).astype(np.float32)
    max_t108_med = np.maximum(max_t108_med, max_t108_hi + 6.0).astype(np.float32)
    max_t108_low = np.maximum(max_t108_low, max_t108_med + 5.0).astype(np.float32)

    t108_inter_thr = (0.5 * (max_t108_med + max_t108_hi)).astype(np.float32)

    # Official ATBD formula for T108T73thrlow (p. 77):
    # T108T73thrlow = 4.0 * sec(theta_sat) + 8.5 K
    t108_t73_thr_low = (
        cfg.ct_wv73_low_guard_slope_k * secant + cfg.ct_wv73_low_guard_intercept_k
    ).astype(np.float32)

    # Semitransparent & fractional split-window / tri-spectral thresholds (ATBD pp. 71-72)
    wv_factor = np.clip((tcwv - 15.0) / 30.0, 0.0, 1.5)
    t108_t120_opaque = (
        cfg.ct_t108_t120_opaque_base_k
        + cfg.ct_t108_t120_opaque_secant_slope_k * (secant - 1.0) * (1.0 + 0.3 * wv_factor)
    ).astype(np.float32)
    t108_t120_thick = (
        cfg.ct_t108_t120_thick_base_k
        + cfg.ct_t108_t120_thick_secant_slope_k * (secant - 1.0)
    ).astype(np.float32)
    t87_t108_opaque = (
        cfg.ct_t87_t108_opaque_base_k + cfg.ct_t87_t108_opaque_secant_slope_k * (secant - 1.0)
    ).astype(np.float32)

    return CtThresholds(
        max_t108_vh_k=max_t108_vh,
        max_t108_hi_k=max_t108_hi,
        max_t108_med_k=max_t108_med,
        max_t108_low_k=max_t108_low,
        t108_inter_thr_k=t108_inter_thr,
        t108_t73_thr_low_k=t108_t73_thr_low,
        t108_t120_opaque_k=t108_t120_opaque,
        t108_t120_thick_k=t108_t120_thick,
        t87_t108_opaque_k=t87_t108_opaque,
    )


def compute_varilog_t108(
    ir108: np.ndarray,
    t_surf: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Compute `VarilogT10.8 = Var_3x3(log(max(T_surf - T10.8, 1.0)))` per ATBD Section 3.2.1.2.2.1."""
    depression = np.maximum(t_surf - ir108, 1.0)
    log_dep = np.where(valid_mask, np.log(depression), 0.0).astype(np.float64)
    w = valid_mask.astype(np.float64)
    count = ndimage.uniform_filter(w, size=3, mode="nearest") * 9.0
    sum_x = ndimage.uniform_filter(log_dep, size=3, mode="nearest") * 9.0
    sum_x2 = ndimage.uniform_filter(log_dep * log_dep, size=3, mode="nearest") * 9.0
    safe_count = np.maximum(count, 1.0)
    mean = sum_x / safe_count
    var = np.maximum(0.0, (sum_x2 / safe_count) - mean * mean)
    return np.where(count >= 2.0, var, 0.0).astype(np.float32)


def map_ct15_to_rdt_bridge(ct: np.ndarray) -> np.ndarray:
    """Map the 15-class NWC SAF `CtClass` onto the 6-class `RdtBridgeCtClass` for RDT/CI."""
    out = np.full(ct.shape, int(RdtBridgeCtClass.CLEAR), dtype=np.uint8)
    out = np.where(
        np.isin(
            ct,
            [
                int(CtClass.VERY_LOW_CLOUDS),
                int(CtClass.LOW_CLOUDS),
                int(CtClass.FRACTIONAL_CLOUDS),
            ],
        ),
        int(RdtBridgeCtClass.LOW_WARM),
        out,
    )
    out = np.where(
        ct == int(CtClass.MID_LEVEL_CLOUDS),
        int(RdtBridgeCtClass.MEDIUM_OPAQUE),
        out,
    )
    out = np.where(
        ct == int(CtClass.HIGH_OPAQUE_CLOUDS),
        int(RdtBridgeCtClass.HIGH_OPAQUE),
        out,
    )
    out = np.where(
        ct == int(CtClass.VERY_HIGH_OPAQUE_CLOUDS),
        int(RdtBridgeCtClass.VERY_HIGH_OPAQUE),
        out,
    )
    out = np.where(
        np.isin(
            ct,
            [
                int(CtClass.HIGH_SEMITRANSPARENT_THIN),
                int(CtClass.HIGH_SEMITRANSPARENT_MODERATELY_THICK),
                int(CtClass.HIGH_SEMITRANSPARENT_THICK),
                int(CtClass.HIGH_SEMITRANSPARENT_ABOVE_LOW_OR_MEDIUM),
                int(CtClass.HIGH_SEMITRANSPARENT_ABOVE_SNOW_OR_ICE),
            ],
        ),
        int(RdtBridgeCtClass.SEMITRANSPARENT_CIRRUS),
        out,
    )
    return out.astype(np.uint8)


def compute_nwcsaf_ct(
    cube_7ch: np.ndarray,
    profile: AtmosphericProfileFields,
    cma_result: CmaResult,
    cfg: NwcSafCmaCtConfig,
) -> CtResult:
    """Run the full NWC SAF GEO-CT v5.0 / v2025 15-class Cloud Type classification."""
    ch = extract_seviri_7ch(cube_7ch)
    wv062 = ch["WV_062"]
    wv073 = ch["WV_073"]
    ir087 = ch["IR_087"]
    ir097 = ch["IR_097"]
    ir108 = ch["IR_108"]
    ir120 = ch["IR_120"]
    ir134 = ch["IR_134"]

    shape = ir108.shape
    valid = cma_result.cma != int(CmaClass.FILL_VALUE)
    land = profile.land_mask
    sea = ~land
    secant = profile.secant
    thr = compute_ct_thresholds(profile, cfg)

    swd = ir108 - ir120
    ice_btd = ir087 - ir108
    t108_minus_t73 = ir108 - wv073
    varilog_t108 = compute_varilog_t108(ir108, profile.t_surf_k, valid)
    sd_t108 = cma_result.local_sd_t108_k

    ct = np.full(shape, int(CtClass.NON_PROCESSED), dtype=np.uint8)
    status_flag = np.zeros(shape, dtype=np.uint8)

    # Set CT status flags (ATBD Section 3.2.2.4, p. 83)
    status_flag = np.where(
        profile.low_level_inversion,
        status_flag | (np.uint8(1) << CtStatusBit.LOW_LEVEL_THERMAL_INVERSION),
        status_flag,
    )
    if profile.tropo_available:
        status_flag |= np.uint8(1) << CtStatusBit.TROPOPAUSE_TEMP_AVAILABLE
    if not cfg.enable_multilayer_detection:
        status_flag |= np.uint8(1) << CtStatusBit.NO_MULTILAYER_METHOD

    # -------------------------------------------------------------------------
    # STEP 1: Cloud-free & Snow/Ice Surface Categories (Classes 1, 2, 3, 4, 15)
    # -------------------------------------------------------------------------
    cs_mask = cma_result.cma_cloudsnow
    is_clear = valid & (cs_mask == int(CmaCloudSnowClass.CLOUD_FREE))
    is_snow_or_ice = valid & (cs_mask == int(CmaCloudSnowClass.SNOW_OR_ICE))
    is_thin_ice_over_snow = valid & (cs_mask == int(CmaCloudSnowClass.THIN_ICE_CLOUD_OVER_SNOW))
    is_cloud = valid & (cs_mask == int(CmaCloudSnowClass.CLOUD))

    ct = np.where(is_clear & land, int(CtClass.CLOUD_FREE_LAND), ct)
    ct = np.where(is_clear & sea, int(CtClass.CLOUD_FREE_SEA), ct)
    ct = np.where(is_snow_or_ice & land, int(CtClass.LAND_SNOW), ct)
    ct = np.where(is_snow_or_ice & sea, int(CtClass.SEA_ICE), ct)
    ct = np.where(is_thin_ice_over_snow, int(CtClass.HIGH_SEMITRANSPARENT_ABOVE_SNOW_OR_ICE), ct)

    # -------------------------------------------------------------------------
    # STEP 2: Separation of Semitransparent & Fractional Clouds from Opaque Clouds
    # (ATBD Sections 3.2.1.2.2.1 - 3.2.1.2.2.3, pp. 70-76)
    # -------------------------------------------------------------------------
    # 2a. Class 14: High Semitransparent above Low or Medium Clouds (Multilayer)
    # Thin ice anvil/cirrus (high T10.8 - T12.0 and high T8.7 - T10.8) overlying an opaque
    # lower/middle cloud deck: the lower cloud deck blocks warm surface radiation, while CO2 (IR_134)
    # and mid-tropospheric WV (WV_073) show strong upper/mid-level emission relative to T10.8.
    is_multilayer = np.zeros(shape, dtype=bool)
    if cfg.enable_multilayer_detection:
        is_multilayer = (
            is_cloud
            & (ir108 > thr.max_t108_vh_k)
            & (ir108 <= thr.max_t108_low_k)
            & (swd > np.maximum(thr.t108_t120_thick_k, cfg.ct_multilayer_swd_min_k))
            & (ice_btd > cfg.ct_multilayer_ice_btd_min_k)
            & ((wv073 - ir108) > cfg.ct_multilayer_wv073_t108_min_k)
            & ((ir134 - ir108) > cfg.ct_multilayer_co2_t134_t108_min_k)
        )

    remaining_cloud = is_cloud & (~is_multilayer)

    # 2b. Warm semitransparent / fractional regime (maxT108med < T10.8 < maxT108low + Delta)
    # ATBD Section 3.2.1.2.2.1 & 3.2.1.2.2.2 (pp. 71-72):
    # - High semitransparent thin clouds (Class 11):
    #   (maxT108med < T10.8 < maxT108low + Delta) and (T10.8 - T12.0 > T108T120opaque)
    #   and (T8.7 - T10.8 > T87T108opaque)
    #   OR (T108interthr < T10.8 <= maxT108med) and (T10.8 - T12.0 > T108T120thick)
    #   and (T8.7 - T10.8 > T87T108opaque) and (VarilogT10.8 > VarilogT10.8thr)
    warm_thin_band = (ir108 > thr.max_t108_med_k) & (
        ir108 < (thr.max_t108_low_k + cfg.ct_thin_cirrus_delta_k)
    )
    inter_thin_band = (ir108 > thr.t108_inter_thr_k) & (ir108 <= thr.max_t108_med_k)

    is_semitrans_thin = remaining_cloud & (
        (
            warm_thin_band
            & (swd > thr.t108_t120_opaque_k)
            & (ice_btd > thr.t87_t108_opaque_k)
        )
        | (
            inter_thin_band
            & (swd > thr.t108_t120_thick_k)
            & (ice_btd > thr.t87_t108_opaque_k)
            & (varilog_t108 > cfg.ct_varilog_t108_thr)
        )
    )
    remaining_cloud &= ~is_semitrans_thin

    # 2c. Class 10: Fractional Clouds (ATBD Section 3.2.1.2.2.1 & 3.2.1.2.2.2, pp. 71-72):
    # - (maxT108med < T10.8 < maxT108low + Delta) and (T10.8 - T12.0 > T108T120opaque)
    #   and (T8.7 - T10.8 <= T87T108opaque)
    # - OR (T10.8 >= maxT108low + Delta) and (T10.8 - T12.0 > T108T120opaque)
    is_fractional = (
        remaining_cloud
        & (ir108 > thr.max_t108_med_k)
        & (swd > thr.t108_t120_opaque_k)
        & (ice_btd <= thr.t87_t108_opaque_k)
    )
    remaining_cloud &= ~is_fractional

    # 2d. Class 12: High Semitransparent Moderately Thick Clouds (ATBD p. 71):
    # (maxT108hi < T10.8 <= maxT108med) and (T10.8 - T12.0 > T108T120thick)
    is_semitrans_mod_thick = (
        remaining_cloud
        & (ir108 > thr.max_t108_hi_k)
        & (ir108 <= thr.max_t108_med_k)
        & (swd > thr.t108_t120_thick_k)
    )
    remaining_cloud &= ~is_semitrans_mod_thick

    # 2e. Class 13: High Semitransparent Thick Clouds (ATBD p. 71):
    # (T10.8 <= maxT108hi) and (T10.8 - T12.0 > T108T120thick)
    # Note: Deep convective cores near the tropopause (T10.8 <= maxT108vh with small WV_062 - IR_108
    # deficit) are opaque anvils/cores; semitransparent thick clouds have larger split-window and
    # lack saturated opaque core characteristics.
    is_deep_opaque_core = (ir108 <= thr.max_t108_vh_k) & ((wv062 - ir108) > -3.5) & (swd < 2.2)
    is_semitrans_thick = (
        remaining_cloud
        & (ir108 <= thr.max_t108_hi_k)
        & (swd > thr.t108_t120_thick_k)
        & (~is_deep_opaque_core)
    )
    remaining_cloud &= ~is_semitrans_thick

    # Assign semitransparent and fractional classes
    ct = np.where(is_multilayer, int(CtClass.HIGH_SEMITRANSPARENT_ABOVE_LOW_OR_MEDIUM), ct)
    ct = np.where(is_semitrans_thin, int(CtClass.HIGH_SEMITRANSPARENT_THIN), ct)
    ct = np.where(is_semitrans_mod_thick, int(CtClass.HIGH_SEMITRANSPARENT_MODERATELY_THICK), ct)
    ct = np.where(is_semitrans_thick, int(CtClass.HIGH_SEMITRANSPARENT_THICK), ct)
    ct = np.where(is_fractional, int(CtClass.FRACTIONAL_CLOUDS), ct)

    # -------------------------------------------------------------------------
    # STEP 3: Opaque Cloud Height Classification (Classes 5, 6, 7, 8, 9)
    # (ATBD Section 3.2.1.2.2.4, pp. 76-77)
    # -------------------------------------------------------------------------
    # Remaining cloudy pixels are opaque clouds:
    # - Class 9 (Very high opaque): T10.8 <= maxT108vh
    # - Class 8 (High opaque): maxT108vh < T10.8 <= maxT108hi
    # - Class 7 (Mid-level): maxT108hi < T10.8 <= maxT108med AND (T10.8 - T7.3 <= T108T73thrlow)
    #                        AND not reclassified by low-level thermal inversion test
    # - Class 6 (Low): (maxT108med < T10.8 <= maxT108low)
    #                  OR ((maxT108hi < T10.8 <= maxT108med) and (T10.8 - T7.3 > T108T73thrlow))
    #                  OR reclassified by low-level thermal inversion test
    # - Class 5 (Very low): T10.8 > maxT108low
    is_vh_opaque = remaining_cloud & (ir108 <= thr.max_t108_vh_k)
    is_hi_opaque = remaining_cloud & (ir108 > thr.max_t108_vh_k) & (ir108 <= thr.max_t108_hi_k)

    in_med_temp_band = remaining_cloud & (ir108 > thr.max_t108_hi_k) & (ir108 <= thr.max_t108_med_k)

    # Water vapour guard (ATBD p. 76): T10.8 - T7.3 > T108T73thrlow -> Low clouds (Class 6)
    wv73_reclassify_to_low = in_med_temp_band & (t108_minus_t73 > thr.t108_t73_thr_low_k)

    # Low-level thermal inversion reclassification test (ATBD p. 77):
    # - If T10.8 > maxT108med - 5 K and T8.7 - T10.8 < -1.2 - (sec - 1) -> Low cloud
    # - If T10.8 > maxT108med - 8 K and T8.7 - T10.8 < -1.7 - (sec - 1) -> Low cloud
    # - If T10.8 > maxT108med - 10 K and T8.7 - T10.8 < -2.2 - (sec - 1) -> Low cloud
    sec_minus_1 = secant - 1.0
    inv_cond_5k = (ir108 > (thr.max_t108_med_k - 5.0)) & (
        ice_btd < (cfg.ct_inversion_t87_t108_5k_base - sec_minus_1)
    )
    inv_cond_8k = (ir108 > (thr.max_t108_med_k - 8.0)) & (
        ice_btd < (cfg.ct_inversion_t87_t108_8k_base - sec_minus_1)
    )
    inv_cond_10k = (ir108 > (thr.max_t108_med_k - 10.0)) & (
        ice_btd < (cfg.ct_inversion_t87_t108_10k_base - sec_minus_1)
    )
    inversion_reclassify_to_low = in_med_temp_band & (inv_cond_5k | inv_cond_8k | inv_cond_10k)

    is_mid_opaque = in_med_temp_band & (~wv73_reclassify_to_low) & (~inversion_reclassify_to_low)
    is_low_opaque = (
        (remaining_cloud & (ir108 > thr.max_t108_med_k) & (ir108 <= thr.max_t108_low_k))
        | wv73_reclassify_to_low
        | inversion_reclassify_to_low
    )
    is_very_low_opaque = remaining_cloud & (ir108 > thr.max_t108_low_k)

    ct = np.where(is_vh_opaque, int(CtClass.VERY_HIGH_OPAQUE_CLOUDS), ct)
    ct = np.where(is_hi_opaque, int(CtClass.HIGH_OPAQUE_CLOUDS), ct)
    ct = np.where(is_mid_opaque, int(CtClass.MID_LEVEL_CLOUDS), ct)
    ct = np.where(is_low_opaque, int(CtClass.LOW_CLOUDS), ct)
    ct = np.where(is_very_low_opaque, int(CtClass.VERY_LOW_CLOUDS), ct)

    # -------------------------------------------------------------------------
    # STEP 4: Auxiliary Containers (`ct_cumuliform`, `ct_multilayer`, `ct_quality`)
    # -------------------------------------------------------------------------
    # Stratiform vs. Cumuliform (`ct_cumuliform`, ATBD Section 3.2.2.4, p. 82)
    # Cumuliform clouds exhibit strong local thermal gradients (SD(T10.8) >= 1.8 K) or
    # fractional/convective core structure; Stratiform decks are spatially homogeneous (SD < 1.0 K).
    is_any_cloud = is_cloud | is_thin_ice_over_snow
    is_clear_or_snow = is_clear | is_snow_or_ice

    cumuliform_mask = is_any_cloud & (
        (sd_t108 >= cfg.ct_cumuliform_sd_t108_min_k)
        | is_fractional
        | (is_vh_opaque & ((wv062 - ir108) > -1.0))
    )
    mixed_mask = is_any_cloud & (~cumuliform_mask) & (sd_t108 >= cfg.ct_mixed_sd_t108_min_k)
    stratiform_mask = is_any_cloud & (~cumuliform_mask) & (~mixed_mask)

    ct_cumuliform = np.full(shape, int(CtCumuliformClass.FILL_VALUE), dtype=np.uint8)
    ct_cumuliform = np.where(is_clear_or_snow, int(CtCumuliformClass.CLOUD_FREE), ct_cumuliform)
    ct_cumuliform = np.where(stratiform_mask, int(CtCumuliformClass.STRATIFORM), ct_cumuliform)
    ct_cumuliform = np.where(Mixed := mixed_mask, int(CtCumuliformClass.MIXED), ct_cumuliform)
    _ = Mixed
    ct_cumuliform = np.where(cumuliform_mask, int(CtCumuliformClass.CUMULIFORM), ct_cumuliform)

    # Multilayer container (`ct_multilayer`, ATBD Section 3.2.2.4, p. 82)
    ct_multilayer = np.full(shape, int(CtMultilayerClass.FILL_VALUE), dtype=np.uint8)
    if cfg.enable_multilayer_detection:
        ct_multilayer = np.where(is_clear_or_snow, int(CtMultilayerClass.CLOUD_FREE), ct_multilayer)
        ct_multilayer = np.where(
            is_any_cloud & (~is_multilayer),
            int(CtMultilayerClass.NO_MULTILAYER),
            ct_multilayer,
        )
        ct_multilayer = np.where(is_multilayer, int(CtMultilayerClass.MULTILAYER), ct_multilayer)
    else:
        ct_multilayer = np.where(valid, int(CtMultilayerClass.UNDEFINED), ct_multilayer)

    # Quality assessment (`ct_quality`, ATBD Section 3.2.1.2.2.5, p. 78):
    # A pixel is flagged as QUESTIONABLE (low confidence) if the absolute difference between any
    # classification feature and its threshold is smaller than the security margin of 0.2 K,
    # or if CMA quality was not GOOD.
    margin = float(cfg.ct_quality_security_margin_k)
    near_t108_boundary = (
        (np.abs(ir108 - thr.max_t108_vh_k) < margin)
        | (np.abs(ir108 - thr.max_t108_hi_k) < margin)
        | (np.abs(ir108 - thr.max_t108_med_k) < margin)
        | (np.abs(ir108 - thr.max_t108_low_k) < margin)
    )
    near_btd_boundary = (
        ((ir108 > thr.max_t108_med_k) & (np.abs(swd - thr.t108_t120_opaque_k) < margin))
        | ((ir108 <= thr.max_t108_med_k) & (np.abs(swd - thr.t108_t120_thick_k) < margin))
        | (
            (ir108 > thr.t108_inter_thr_k)
            & (swd > thr.t108_t120_thick_k)
            & (np.abs(ice_btd - thr.t87_t108_opaque_k) < margin)
        )
        | (in_med_temp_band & (np.abs(t108_minus_t73 - thr.t108_t73_thr_low_k) < margin))
    )
    near_any_ct_boundary = is_any_cloud & (near_t108_boundary | near_btd_boundary)

    ct_quality = np.full(shape, int(CmaQuality.NA), dtype=np.uint8)
    ct_quality = np.where(valid, cma_result.cma_quality, ct_quality)
    ct_quality = np.where(
        valid & (ct_quality == int(CmaQuality.GOOD)) & near_any_ct_boundary,
        int(CmaQuality.QUESTIONABLE),
        ct_quality,
    )

    # RDT/CI 6-class bridge and masks
    rdt_bridge = map_ct15_to_rdt_bridge(ct)
    opaque_mask = np.isin(
        ct,
        [
            int(CtClass.VERY_LOW_CLOUDS),
            int(CtClass.LOW_CLOUDS),
            int(CtClass.MID_LEVEL_CLOUDS),
            int(CtClass.HIGH_OPAQUE_CLOUDS),
            int(CtClass.VERY_HIGH_OPAQUE_CLOUDS),
        ],
    )
    semitrans_mask = np.isin(
        ct,
        [
            int(CtClass.HIGH_SEMITRANSPARENT_THIN),
            int(CtClass.HIGH_SEMITRANSPARENT_MODERATELY_THICK),
            int(CtClass.HIGH_SEMITRANSPARENT_THICK),
            int(CtClass.HIGH_SEMITRANSPARENT_ABOVE_LOW_OR_MEDIUM),
            int(CtClass.HIGH_SEMITRANSPARENT_ABOVE_SNOW_OR_ICE),
        ],
    )

    _ = ir097  # Retained in 7-channel signature

    return CtResult(
        ct=ct.astype(np.uint8),
        ct_cumuliform=ct_cumuliform.astype(np.uint8),
        ct_multilayer=ct_multilayer.astype(np.uint8),
        ct_status_flag=status_flag.astype(np.uint8),
        ct_quality=ct_quality.astype(np.uint8),
        rdt_bridge_class=rdt_bridge,
        opaque_cloud_mask=opaque_mask,
        semitransparent_mask=semitrans_mask,
        thresholds=thr,
    )
