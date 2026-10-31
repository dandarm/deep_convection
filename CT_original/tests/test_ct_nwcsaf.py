"""Comprehensive unit and synthetic verification tests for NWC SAF GEO-CMA v6.0 & GEO-CT v5.0."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

_CT_ROOT = Path(__file__).resolve().parents[1]
if str(_CT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CT_ROOT))

from ct_nwcsaf import (
    CmaClass,
    CmaCloudSnowClass,
    CmaDustClass,
    CmaQuality,
    CmaStatusBit,
    CmaTestBit,
    CmaVolcanicClass,
    CtClass,
    CtCumuliformClass,
    CtMultilayerClass,
    CtStatusBit,
    NwcSafCmaCtConfig,
    NwcSafCmaCtTracker,
    RdtBridgeCtClass,
    build_atmospheric_profile_fields,
    compute_ct_thresholds,
    compute_nwcsaf_cma,
    compute_nwcsaf_ct,
    compute_satellite_secant,
    run_nwcsaf_cma_ct_sequence,
)


def _make_uniform_7ch(
    shape: tuple[int, int] = (24, 24),
    *,
    wv062: float = 236.0,
    wv073: float = 254.0,
    ir087: float = 292.0,
    ir097: float = 266.0,
    ir108: float = 293.0,
    ir120: float = 292.4,
    ir134: float = 269.0,
) -> np.ndarray:
    """Create a synthetic `(7, H, W)` SEVIRI brightness temperature cube in Kelvin."""
    vals = [wv062, wv073, ir087, ir097, ir108, ir120, ir134]
    cube = np.empty((7, shape[0], shape[1]), dtype=np.float32)
    for idx, v in enumerate(vals):
        cube[idx].fill(float(v))
    return cube


def _standard_nwp() -> dict[str, float]:
    """Standard mid-latitude NWP vertical temperature profile in Kelvin."""
    return {
        "T_surf": 294.0,
        "T_950": 290.0,
        "T_850": 284.0,
        "T_700": 274.0,
        "T_500": 256.0,
        "T_tropo": 215.0,
        "TCWV": 22.0,
    }


def test_satellite_secant_geometry() -> None:
    cfg = NwcSafCmaCtConfig(sub_satellite_lon_deg=9.5)
    lat = np.array([[0.0, 38.0, 55.0]], dtype=np.float32)
    lon = np.array([[9.5, 15.0, 25.0]], dtype=np.float32)
    sec = compute_satellite_secant(lat, lon, (1, 3), cfg)
    assert sec.shape == (1, 3)
    assert np.isclose(sec[0, 0], 1.0, atol=1e-4)
    assert 1.2 < sec[0, 1] < 1.6
    assert sec[0, 2] > sec[0, 1]


def test_exact_atbd_ct_height_threshold_formulas() -> None:
    """Verify exact NWC SAF ATBD Section 3.2.1.2.2.4 (p. 77) linear equations."""
    cfg = NwcSafCmaCtConfig(default_secant=1.5)
    cube = _make_uniform_7ch((8, 8))
    nwp = _standard_nwp()
    profile, _ = build_atmospheric_profile_fields(cube, cfg, nwp_fields=nwp)
    thr = compute_ct_thresholds(profile, cfg)

    expected_hi = 0.5 * 256.0 - 0.2 * 274.0 + 178.0  # 128 - 54.8 + 178 = 251.2 K
    expected_med = 0.8 * 284.0 + 0.2 * 274.0 - 8.0  # 227.2 + 54.8 - 8 = 274.0 K
    expected_low = 1.2 * 284.0 - 0.2 * 274.0 - 5.0  # 340.8 - 54.8 - 5 = 281.0 K
    expected_vh = 0.4 * 256.0 + 0.6 * 215.0 - 5.0  # 102.4 + 129.0 - 5 = 226.4 K
    expected_wv73_guard = 4.0 * 1.5 + 8.5  # 14.5 K

    assert np.allclose(thr.max_t108_hi_k, expected_hi, atol=1e-4)
    assert np.allclose(thr.max_t108_med_k, expected_med, atol=1e-4)
    assert np.allclose(thr.max_t108_low_k, expected_low, atol=1e-4)
    assert np.allclose(thr.max_t108_vh_k, expected_vh, atol=1e-4)
    assert np.allclose(thr.t108_t73_thr_low_k, expected_wv73_guard, atol=1e-4)


def test_clear_land_sea_snow_and_sea_ice_classes() -> None:
    """Verify Classes 1 (Clear Land), 2 (Clear Sea), 3 (Snow Land), 4 (Sea Ice), and 15."""
    cfg = NwcSafCmaCtConfig(enable_spatial_filtering=False)
    cube = _make_uniform_7ch((12, 12))
    land_mask = np.zeros((12, 12), dtype=bool)
    land_mask[:6, :] = True  # Top half land, bottom half sea

    snow_ice_mask = np.zeros((12, 12), dtype=bool)
    snow_ice_mask[0:2, 0:4] = True  # Snow on land
    snow_ice_mask[6:8, 0:4] = True  # Ice on sea
    snow_ice_mask[2:4, 0:4] = True  # Snow with thin ice cloud aloft

    # Put thin ice cloud signature (high split-window) over [2:4, 0:4]
    cube[4, 2:4, 0:4] = 258.0  # IR_108
    cube[5, 2:4, 0:4] = 255.5  # IR_120 -> SWD = 2.5 K > 1.35 K

    profile, _ = build_atmospheric_profile_fields(
        cube,
        cfg,
        land_mask=land_mask,
        snow_ice_mask=snow_ice_mask,
        nwp_fields=_standard_nwp(),
    )
    cma_res = compute_nwcsaf_cma(cube, profile, cfg)
    ct_res = compute_nwcsaf_ct(cube, profile, cma_res, cfg)

    assert np.all(ct_res.ct[4:6, :] == int(CtClass.CLOUD_FREE_LAND))
    assert np.all(ct_res.ct[8:, :] == int(CtClass.CLOUD_FREE_SEA))
    assert np.all(ct_res.ct[0:2, 0:4] == int(CtClass.LAND_SNOW))
    assert np.all(ct_res.ct[6:8, 0:4] == int(CtClass.SEA_ICE))
    assert np.all(ct_res.ct[2:4, 0:4] == int(CtClass.HIGH_SEMITRANSPARENT_ABOVE_SNOW_OR_ICE))
    assert np.all(
        cma_res.cma_cloudsnow[2:4, 0:4] == int(CmaCloudSnowClass.THIN_ICE_CLOUD_OVER_SNOW)
    )


def test_opaque_cloud_vertical_hierarchy_classes_5_to_9() -> None:
    """Verify Opaque Classes 5 (Very low), 6 (Low), 7 (Mid), 8 (High), and 9 (Very high)."""
    cfg = NwcSafCmaCtConfig(enable_spatial_filtering=False)
    cube = _make_uniform_7ch((20, 20))
    nwp = _standard_nwp()
    # Thresholds with standard NWP:
    # maxT108vh = 226.4 K, maxT108hi = 251.2 K, maxT108med = 274.0 K, maxT108low = 281.0 K

    # Row 0..4: Class 5 Very Low Opaque (T10.8 = 285 K > 281 K, detected by low stratus BTD T10.8-T8.7 = 3.2 K)
    cube[4, 0:4, :] = 285.0  # IR_108
    cube[5, 0:4, :] = 284.5  # IR_120 (opaque SWD = 0.5 K)
    cube[2, 0:4, :] = 281.8  # IR_087 (T10.8 - T8.7 = 3.2 K > 2.4 K)
    cube[1, 0:4, :] = 275.0  # WV_073

    # Row 4..8: Class 6 Low Opaque (T10.8 = 277.0 K in (274.0, 281.0])
    cube[4, 4:8, :] = 277.0
    cube[5, 4:8, :] = 276.5
    cube[2, 4:8, :] = 275.8
    cube[1, 4:8, :] = 266.0

    # Row 8..12: Class 7 Mid-Level Opaque (T10.8 = 262.0 K in (251.2, 274.0], T10.8 - T7.3 = 8.0 K <= 13.9 K)
    cube[4, 8:12, :] = 262.0
    cube[5, 8:12, :] = 261.5
    cube[2, 8:12, :] = 261.5
    cube[1, 8:12, :] = 254.0  # T10.8 - T7.3 = 8.0 K

    # Row 12..16: Class 8 High Opaque (T10.8 = 238.0 K in (226.4, 251.2])
    cube[4, 12:16, :] = 238.0
    cube[5, 12:16, :] = 237.4
    cube[2, 12:16, :] = 237.8
    cube[1, 12:16, :] = 236.0
    cube[0, 12:16, :] = 233.0

    # Row 16..20: Class 9 Very High Opaque (T10.8 = 216.0 K <= 226.4 K)
    cube[4, 16:20, :] = 216.0
    cube[5, 16:20, :] = 215.5
    cube[2, 16:20, :] = 216.2
    cube[1, 16:20, :] = 216.5
    cube[0, 16:20, :] = 217.0

    profile, _ = build_atmospheric_profile_fields(cube, cfg, nwp_fields=nwp)
    cma_res = compute_nwcsaf_cma(cube, profile, cfg)
    ct_res = compute_nwcsaf_ct(cube, profile, cma_res, cfg)

    assert np.all(cma_res.cma == int(CmaClass.CLOUDY))
    assert np.all(ct_res.ct[0:4, :] == int(CtClass.VERY_LOW_CLOUDS))
    assert np.all(ct_res.ct[4:8, :] == int(CtClass.LOW_CLOUDS))
    assert np.all(ct_res.ct[8:12, :] == int(CtClass.MID_LEVEL_CLOUDS))
    assert np.all(ct_res.ct[12:16, :] == int(CtClass.HIGH_OPAQUE_CLOUDS))
    assert np.all(ct_res.ct[16:20, :] == int(CtClass.VERY_HIGH_OPAQUE_CLOUDS))

    # Verify RDT 6-class bridge mapping
    assert np.all(ct_res.rdt_bridge_class[0:8, :] == int(RdtBridgeCtClass.LOW_WARM))
    assert np.all(ct_res.rdt_bridge_class[8:12, :] == int(RdtBridgeCtClass.MEDIUM_OPAQUE))
    assert np.all(ct_res.rdt_bridge_class[12:16, :] == int(RdtBridgeCtClass.HIGH_OPAQUE))
    assert np.all(ct_res.rdt_bridge_class[16:20, :] == int(RdtBridgeCtClass.VERY_HIGH_OPAQUE))
    assert np.all(ct_res.opaque_cloud_mask)


def test_wv73_and_thermal_inversion_guards_reclassify_medium_to_low() -> None:
    """Verify ATBD Section 3.2.1.2.2.4 (pp. 76-77) WV7.3 guard and T8.7-T10.8 inversion test."""
    cfg = NwcSafCmaCtConfig(default_secant=1.25, enable_spatial_filtering=False)
    # T108T73thrlow = 4.0 * 1.25 + 8.5 = 13.5 K
    # maxT108med = 274.0 K
    cube = _make_uniform_7ch((12, 12))
    nwp = _standard_nwp()

    # Top half [0:6]: T10.8 = 266 K (in mid band), but WV_073 = 248 K -> T10.8 - T7.3 = 18 K > 13.5 K
    cube[4, 0:6, :] = 266.0
    cube[5, 0:6, :] = 265.5
    cube[2, 0:6, :] = 265.5
    cube[1, 0:6, :] = 248.0

    # Bottom half [6:12]: T10.8 = 270.5 K (within 5 K of maxT108med = 274 K),
    # T10.8 - T7.3 = 8.5 K <= 13.5 K, but strong negative T8.7 - T10.8 = -2.0 K < -1.2 - 0.25 = -1.45 K
    cube[4, 6:12, :] = 270.5
    cube[5, 6:12, :] = 270.0
    cube[1, 6:12, :] = 262.0
    cube[2, 6:12, :] = 268.5  # T8.7 - T10.8 = -2.0 K

    profile, _ = build_atmospheric_profile_fields(cube, cfg, nwp_fields=nwp)
    cma_res = compute_nwcsaf_cma(cube, profile, cfg)
    ct_res = compute_nwcsaf_ct(cube, profile, cma_res, cfg)

    assert np.all(ct_res.ct[0:6, :] == int(CtClass.LOW_CLOUDS))
    assert np.all(ct_res.ct[6:12, :] == int(CtClass.LOW_CLOUDS))


def test_semitransparent_thin_mod_thick_multilayer_and_fractional_classes() -> None:
    """Verify Classes 10 (Fractional), 11 (Thin), 12 (Mod. thick), 13 (Thick), and 14 (Multilayer)."""
    cfg = NwcSafCmaCtConfig(default_secant=1.25, enable_spatial_filtering=False)
    cube = _make_uniform_7ch((25, 20))
    nwp = _standard_nwp()

    # Rows 0..5: Class 10 Fractional Clouds (T10.8 = 279 K > maxT108med=274 K,
    # high SWD = 2.1 K > T108T120opaque, but low ice signature T8.7 - T10.8 = -0.8 K <= T87T108opaque)
    cube[4, 0:5, :] = 279.0
    cube[5, 0:5, :] = 276.9  # SWD = 2.1 K
    cube[2, 0:5, :] = 278.2  # T8.7 - T10.8 = -0.8 K

    # Rows 5..10: Class 11 High Semitransparent Thin Cirrus (T10.8 = 279 K > maxT108med,
    # high SWD = 2.4 K > T108T120opaque, and positive ice BTD T8.7 - T10.8 = +0.9 K > T87T108opaque,
    # with dry/warm lower atmosphere WV_073 - IR_108 = -24 K so not multilayer)
    cube[4, 5:10, :] = 279.0
    cube[5, 5:10, :] = 276.6  # SWD = 2.4 K
    cube[2, 5:10, :] = 279.9  # T8.7 - T10.8 = +0.9 K
    cube[1, 5:10, :] = 255.0  # WV_073 - IR_108 = -24.0 K
    cube[6, 5:10, :] = 262.0  # IR_134 - IR_108 = -17.0 K

    # Rows 10..15: Class 12 High Semitransparent Moderately Thick (T10.8 = 262 K in (251.2, 274.0],
    # SWD = 2.2 K > T108T120thick, WV_073 - IR_108 = -20 K so single-layer cirrus)
    cube[4, 10:15, :] = 262.0
    cube[5, 10:15, :] = 259.8  # SWD = 2.2 K
    cube[2, 10:15, :] = 262.4
    cube[1, 10:15, :] = 242.0  # WV_073 - IR_108 = -20.0 K
    cube[6, 10:15, :] = 246.0  # IR_134 - IR_108 = -16.0 K

    # Rows 15..20: Class 13 High Semitransparent Thick (T10.8 = 242 K <= maxT108hi=251.2 K,
    # SWD = 2.3 K > T108T120thick, WV_073 - IR_108 = -20 K)
    cube[4, 15:20, :] = 242.0
    cube[5, 15:20, :] = 239.7  # SWD = 2.3 K
    cube[2, 15:20, :] = 242.5
    cube[1, 15:20, :] = 222.0
    cube[6, 15:20, :] = 226.0

    # Rows 20..25: Class 14 High Semitransparent above Low/Medium Clouds (Multilayer):
    # T10.8 = 260 K, SWD = 2.3 K, ice_btd = +0.8 K, AND elevated WV_073 (249 K -> -11 K > -18.5 K)
    # and CO2 IR_134 (252 K -> -8 K > -12.5 K) from lower cloud deck!
    cube[4, 20:25, :] = 260.0
    cube[5, 20:25, :] = 257.7  # SWD = 2.3 K
    cube[2, 20:25, :] = 260.8  # ice_btd = +0.8 K
    cube[1, 20:25, :] = 249.0  # WV_073 - IR_108 = -11.0 K
    cube[6, 20:25, :] = 252.0  # IR_134 - IR_108 = -8.0 K

    profile, _ = build_atmospheric_profile_fields(cube, cfg, nwp_fields=nwp)
    cma_res = compute_nwcsaf_cma(cube, profile, cfg)
    ct_res = compute_nwcsaf_ct(cube, profile, cma_res, cfg)

    assert np.all(ct_res.ct[0:5, :] == int(CtClass.FRACTIONAL_CLOUDS))
    assert np.all(ct_res.ct[5:10, :] == int(CtClass.HIGH_SEMITRANSPARENT_THIN))
    assert np.all(ct_res.ct[10:15, :] == int(CtClass.HIGH_SEMITRANSPARENT_MODERATELY_THICK))
    assert np.all(ct_res.ct[15:20, :] == int(CtClass.HIGH_SEMITRANSPARENT_THICK))
    assert np.all(ct_res.ct[20:25, :] == int(CtClass.HIGH_SEMITRANSPARENT_ABOVE_LOW_OR_MEDIUM))
    assert np.all(ct_res.ct_multilayer[20:25, :] == int(CtMultilayerClass.MULTILAYER))
    assert np.all(ct_res.semitransparent_mask[5:25, :])
    assert np.all(
        ct_res.rdt_bridge_class[5:25, :] == int(RdtBridgeCtClass.SEMITRANSPARENT_CIRRUS)
    )


def test_cma_temporal_15min_and_stationary_1h_region_growing() -> None:
    """Verify ATBD 2.2.1.2.4 (15-min cooling) and 2.2.1.2.5 (1h stationary + region growing)."""
    cfg = NwcSafCmaCtConfig(enable_spatial_filtering=False)
    nwp = _standard_nwp()

    # 1. Test 15-min temporal differencing over sea:
    # A subtle fast-moving cloud cools pixel [4:8, 4:8] by -1.1 K (from 293.5 K to 292.4 K),
    # which is still warmer than the static SST/T10.8 threshold (294 - 2.8 = 291.2 K),
    # so static tests leave it clear, but Delta_15min(T10.8) = -1.1 K < -0.6 K catches it!
    cube_prev15 = _make_uniform_7ch((12, 12), ir108=293.5, ir120=293.1, ir087=292.5)
    cube_now = _make_uniform_7ch((12, 12), ir108=293.5, ir120=293.1, ir087=292.5)
    cube_now[4, 4:8, 4:8] = 292.4
    cube_now[5, 4:8, 4:8] = 292.0
    cube_now[2, 4:8, 4:8] = 291.4

    profile, _ = build_atmospheric_profile_fields(cube_now, cfg, nwp_fields=nwp)
    cma_no_temp = compute_nwcsaf_cma(cube_now, profile, cfg)
    assert np.all(cma_no_temp.cma[4:8, 4:8] == int(CmaClass.CLOUD_FREE))

    cma_with_temp = compute_nwcsaf_cma(
        cube_now, profile, cfg, prev_15m_ir108=cube_prev15[4]
    )
    assert np.all(cma_with_temp.cma[4:8, 4:8] == int(CmaClass.CLOUDY))
    assert np.all(
        (cma_with_temp.cma_testlist[4:8, 4:8] & (np.uint32(1) << CmaTestBit.TEMPORAL_DIFFERENCING))
        != 0
    )
    assert np.all(cma_with_temp.cma_quality[4:8, 4:8] == int(CmaQuality.QUESTIONABLE))

    # 2. Test 1-hour stationary cloud coherence & region growing (ATBD Section 2.2.1.2.5):
    # At t - 1h, a 4x4 block [2:6, 2:6] (16 pixels > 8 minimum seed size) was classified as
    # LOW_CLOUDS with GOOD quality. At t = 0, its temperature changed by only 0.2 K, and adjacent
    # pixels [2:6, 6:8] have the exact same T10.8 (within [-5.0, +0.5] K of seed mean).
    cube_1h = _make_uniform_7ch((12, 12), ir108=293.5, ir120=293.1, ir087=292.5)
    cube_1h[4, 2:6, 2:6] = 292.3
    cube_1h[5, 2:6, 2:6] = 291.9
    cube_1h[2, 2:6, 2:6] = 291.3
    cma_1h = np.zeros((12, 12), dtype=np.uint8)
    cma_1h[2:6, 2:6] = int(CmaClass.CLOUDY)
    ct_1h = np.full((12, 12), int(CtClass.CLOUD_FREE_SEA), dtype=np.uint8)
    ct_1h[2:6, 2:6] = int(CtClass.LOW_CLOUDS)
    qual_1h = np.full((12, 12), int(CmaQuality.GOOD), dtype=np.uint8)

    cube_t0 = _make_uniform_7ch((12, 12), ir108=293.5, ir120=293.1, ir087=292.5)
    cube_t0[4, 2:6, 2:8] = 292.2  # Both seed [2:6, 2:6] and neighbor [2:6, 6:8]
    cube_t0[5, 2:6, 2:8] = 291.8
    cube_t0[2, 2:6, 2:8] = 291.2

    cma_stat = compute_nwcsaf_cma(
        cube_t0,
        profile,
        cfg,
        prev_1h_cube_7ch=cube_1h,
        prev_1h_cma=cma_1h,
        prev_1h_ct=ct_1h,
        prev_1h_quality=qual_1h,
    )
    assert np.all(cma_stat.cma[2:6, 2:6] == int(CmaClass.CLOUDY))
    assert np.all(cma_stat.cma[2:6, 6:8] == int(CmaClass.CLOUDY))
    assert np.all(
        (cma_stat.cma_testlist[2:6, 2:6] & (np.uint32(1) << CmaTestBit.STATIONARY_TWILIGHT)) != 0
    )
    assert np.all(
        (
            cma_stat.cma_testlist[2:6, 6:8]
            & (np.uint32(1) << CmaTestBit.SPATIAL_EXTENSION_TWILIGHT)
        )
        != 0
    )


def test_cma_spatial_filtering_cold_inversion_and_isolated_pixels() -> None:
    """Verify ATBD Section 2.2.1.2.7 cold nocturnal inversion restoral and isolated pixel filter."""
    cfg = NwcSafCmaCtConfig(
        enable_spatial_filtering=True,
        cma_definite_cloud_t108_k=235.0,
    )
    # Cold surface with nocturnal inversion: T_surf = 252 K -> t108_thr = 239.28 K < 250 K
    # Pixel has T10.8 = 237.0 K (< t108_thr) but WV_073 = 239.0 K -> T7.3 - T10.8 = +2.0 K > 0.5 K!
    cube = _make_uniform_7ch(
        (10, 10),
        wv062=228.0,
        wv073=239.0,
        ir087=236.5,
        ir097=235.0,
        ir108=237.0,
        ir120=236.8,
        ir134=236.0,
    )
    nwp_cold = {
        "T_surf": 252.0,
        "T_950": 255.0,
        "T_850": 250.0,
        "T_700": 242.0,
        "T_500": 228.0,
        "T_tropo": 208.0,
        "TCWV": 6.0,
    }
    land_mask = np.ones((10, 10), dtype=bool)
    profile, _ = build_atmospheric_profile_fields(
        cube, cfg, land_mask=land_mask, nwp_fields=nwp_cold
    )
    cma_res = compute_nwcsaf_cma(cube, profile, cfg)
    assert np.all(cma_res.cma == int(CmaClass.CLOUD_FREE))
    assert np.all(
        (cma_res.cma_testlist & (np.uint32(1) << CmaTestBit.SPATIAL_FILTER_TO_CLEAR)) != 0
    )
    assert np.all(cma_res.cma_quality == int(CmaQuality.QUESTIONABLE))

    # Isolated pixel removal: single warm low-stratus pixel surrounded by 8 clear neighbors
    cfg_iso = NwcSafCmaCtConfig(enable_spatial_filtering=True)
    cube_iso = _make_uniform_7ch((9, 9))
    cube_iso[4, 4, 4] = 291.5  # Slightly cooler than T_surf=294, warmer than t108_thr
    cube_iso[5, 4, 4] = 291.2
    cube_iso[2, 4, 4] = 288.2  # Triggers T10.8 - T8.7 = 3.3 K > 3.1 K (Bit 6 only)
    cube_iso[1, 4, 4] = 276.0  # Moist low-cloud WV_073 signature
    cube_iso[6, 4, 4] = 278.0
    prof_iso, _ = build_atmospheric_profile_fields(cube_iso, cfg_iso, nwp_fields=_standard_nwp())
    cma_iso = compute_nwcsaf_cma(cube_iso, prof_iso, cfg_iso)
    assert cma_iso.cma[4, 4] == int(CmaClass.CLOUD_FREE)
    assert (cma_iso.cma_testlist[4, 4] & (np.uint32(1) << CmaTestBit.SPATIAL_FILTER_TO_CLEAR)) != 0
    assert cma_iso.cma_quality[4, 4] == int(CmaQuality.INTERPOLATED)


def test_cma_dust_and_volcanic_ash_detection() -> None:
    """Verify ATBD Sections 2.2.1.2.8 (Dust) and 2.2.1.2.9 (Volcanic Ash) reverse split-window."""
    cfg = NwcSafCmaCtConfig(enable_spatial_filtering=False)
    cube = _make_uniform_7ch((12, 12))
    nwp = _standard_nwp()

    # Rows 0..4: Saharan Dust plume over sea (warm T10.8 = 288 K, reverse split-window
    # T12.0 - T10.8 = +0.8 K -> IR_120 = 288.8 K, T8.7 - T10.8 = -0.3 K, spatially uniform SD=0)
    cube[4, 0:4, :] = 288.0
    cube[5, 0:4, :] = 288.8  # T12.0 - T10.8 = +0.8 K > 0.3 K
    cube[2, 0:4, :] = 287.7

    # Rows 6..10: Cold Volcanic Ash plume (T10.8 = 222 K, strong reverse split-window
    # T12.0 - T10.8 = +2.2 K -> IR_120 = 224.2 K, T8.7 - T10.8 = +0.5 K)
    cube[4, 6:10, :] = 222.0
    cube[5, 6:10, :] = 224.2  # T12.0 - T10.8 = +2.2 K > 1.2 K
    cube[2, 6:10, :] = 222.5

    profile, _ = build_atmospheric_profile_fields(cube, cfg, nwp_fields=nwp)
    cma_res = compute_nwcsaf_cma(cube, profile, cfg)

    assert np.all(cma_res.cma_dust[0:3, :] == int(CmaDustClass.DUST))
    assert np.all(cma_res.cma_dust[6:10, :] == int(CmaDustClass.NO_DUST))
    assert np.all(cma_res.cma_volcanic[6:10, :] == int(CmaVolcanicClass.VOLCANIC_PLUME))
    assert np.all(cma_res.cma_volcanic[0:3, :] == int(CmaVolcanicClass.NO_VOLCANIC_PLUME))


def test_ct_quality_security_margin_and_stateful_sequence() -> None:
    """Verify ATBD p. 78 0.2 K security margin for `ct_quality` and sequence runner."""
    cfg = NwcSafCmaCtConfig(enable_spatial_filtering=False)
    nwp = _standard_nwp()
    # maxT108hi = 251.2 K
    cube = _make_uniform_7ch((8, 8))
    # Top half: T10.8 = 251.12 K (within 0.08 K < 0.2 K of maxT108hi = 251.2 K) -> QUESTIONABLE
    cube[4, 0:4, :] = 251.12
    cube[5, 0:4, :] = 250.62
    cube[2, 0:4, :] = 250.90
    # Bottom half: T10.8 = 244.0 K (well away from all thresholds) -> GOOD
    cube[4, 4:8, :] = 244.0
    cube[5, 4:8, :] = 243.5
    cube[2, 4:8, :] = 243.8

    seq = np.stack([cube, cube, cube, cube], axis=0)
    results = run_nwcsaf_cma_ct_sequence(seq, cfg, nwp_fields=nwp)
    assert len(results) == 4
    last = results[-1]
    assert np.all(last.ct_quality[0:4, :] == int(CmaQuality.QUESTIONABLE))
    assert np.all(last.ct_quality[4:8, :] == int(CmaQuality.GOOD))
    assert (
        int(last.ct_result.ct_status_flag[0, 0])
        & (1 << int(CtStatusBit.TROPOPAUSE_TEMP_AVAILABLE))
    ) != 0
