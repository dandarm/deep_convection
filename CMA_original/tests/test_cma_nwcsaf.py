"""Unit and synthetic physical tests for NWC SAF GEO-CMA v2025 (v6.0) in `CMA_original`."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_CMA_ROOT = Path(__file__).resolve().parents[1]
if str(_CMA_ROOT) not in sys.path:
    sys.path.insert(0, str(_CMA_ROOT))

from cma_nwcsaf import (
    CmaBinaryClass,
    CmaCloudSnowClass,
    CmaCoverClass,
    CmaDustClass,
    CmaQuality,
    CmaTestBit,
    CmaVolcanicClass,
    NwcSafCmaConfig,
    NwcSafCmaTracker,
    estimate_autonomous_ir_ancillary,
    run_cma_frame,
    run_nwcsaf_cma_sequence,
)


def _make_clear_sky_7ch(h: int = 32, w: int = 32, t_surf: float = 294.0) -> dict[str, np.ndarray]:
    """Create a synthetic clear-sky 7-channel SEVIRI scene in Kelvin."""
    return {
        "WV_062": np.full((h, w), 236.0, dtype=np.float32),
        "WV_073": np.full((h, w), 254.0, dtype=np.float32),
        "IR_087": np.full((h, w), t_surf - 1.2, dtype=np.float32),
        "IR_097": np.full((h, w), 258.0, dtype=np.float32),
        "IR_108": np.full((h, w), t_surf - 0.6, dtype=np.float32),
        "IR_120": np.full((h, w), t_surf - 1.3, dtype=np.float32),
        "IR_134": np.full((h, w), 266.0, dtype=np.float32),
    }


def test_cma_cover_four_classes_and_cma_cloudsnow_four_classes() -> None:
    """Verify the 1 negative class (`0=Cloud-free`) + 3 positive classes in `cma_cover` and `cma_cloudsnow`."""
    h, w = 32, 32
    bt = _make_clear_sky_7ch(h, w, t_surf=294.0)

    # Region 1 (rows 0..7, cols 0..15): Clear sky -> CLOUD_FREE (0) in both cma_cover and cma_cloudsnow
    # Region 2 (rows 8..15, cols 8..15): Thin semitransparent cirrus ->
    #   cma_cover == 1 (CLOUD_CONTAMINATED), cma_cloudsnow == 1 (CLOUD)
    r_cirrus = (slice(8, 16), slice(8, 16))
    bt["IR_108"][r_cirrus] = 289.0  # slightly cooler than clear sky, above opaque T10.8 threshold
    bt["IR_120"][r_cirrus] = 285.8  # BTD 10.8 - 12.0 = +3.2 K (strong semitransparent cirrus)
    bt["IR_087"][r_cirrus] = 290.1  # BTD 8.7 - 10.8 = +1.1 K (ice crystals)

    # Region 3 (rows 18..26, cols 4..14): Opaque cold deep cloud deck ->
    #   cma_cover == 2 (CLOUD_FILLED), cma_cloudsnow == 1 (CLOUD)
    r_opaque = (slice(18, 26), slice(4, 14))
    bt["IR_108"][r_opaque] = 230.0
    bt["IR_120"][r_opaque] = 229.5
    bt["IR_087"][r_opaque] = 229.2
    bt["WV_062"][r_opaque] = 228.5
    bt["WV_073"][r_opaque] = 229.5
    bt["IR_134"][r_opaque] = 228.0

    # Region 4 (rows 18..26, cols 18..24): Clear-sky snow on land ->
    #   cma_cover == 3 (SNOW_OR_ICE), cma_cloudsnow == 3 (SNOW_OR_ICE)
    snow_mask = np.zeros((h, w), dtype=np.uint8)
    r_snow_clear = (slice(18, 26), slice(18, 24))
    snow_mask[r_snow_clear] = 2
    bt["IR_108"][r_snow_clear] = 264.0
    bt["IR_120"][r_snow_clear] = 263.6  # small BTD 10.8 - 12.0 = 0.4 K
    bt["WV_073"][r_snow_clear] = 246.0  # WV7.3 - IR10.8 = -18.0 K (dry clear atmosphere above snow)

    # Region 5 (rows 18..26, cols 25..30): Thin ice cloud over snow ->
    #   cma_cover == 1 (CLOUD_CONTAMINATED), cma_cloudsnow == 2 (THIN_ICE_CLOUD_OVER_SNOW)
    r_thin_ice_snow = (slice(18, 26), slice(25, 30))
    snow_mask[r_thin_ice_snow] = 2
    bt["IR_108"][r_thin_ice_snow] = 259.0
    bt["IR_120"][r_thin_ice_snow] = 256.8  # BTD 10.8 - 12.0 = +2.2 K > 1.35 K
    bt["WV_073"][r_thin_ice_snow] = 252.0  # BTD 7.3 - 10.8 = -7.0 K > -12.0 K

    cfg = NwcSafCmaConfig()
    anc = estimate_autonomous_ir_ancillary(
        bt,
        cfg,
        is_land=np.ones((h, w), dtype=bool),
        snow_ice_mask=snow_mask,
        t_surf_k=np.full((h, w), 294.0, dtype=np.float32),
    )
    out = run_cma_frame(bt, anc, cfg)

    # 1. Clear sky (negative class = 0)
    assert np.all(out.cma[2:6, 2:12] == CmaBinaryClass.CLOUD_FREE)
    assert np.all(out.cma_cover[2:6, 2:12] == CmaCoverClass.CLOUD_FREE)
    assert np.all(out.cma_cloudsnow[2:6, 2:12] == CmaCloudSnowClass.CLOUD_FREE)

    # 2. Thin semitransparent cirrus -> CLOUD_CONTAMINATED (1) in cma_cover, CLOUD (1) in cma_cloudsnow
    assert np.all(out.cma[10:14, 10:14] == CmaBinaryClass.CLOUDY)
    assert np.all(out.cma_cover[10:14, 10:14] == CmaCoverClass.CLOUD_CONTAMINATED)
    assert np.all(out.cma_cloudsnow[10:14, 10:14] == CmaCloudSnowClass.CLOUD)

    # 3. Opaque cold cloud -> CLOUD_FILLED (2) in cma_cover, CLOUD (1) in cma_cloudsnow
    assert np.all(out.cma[20:24, 6:12] == CmaBinaryClass.CLOUDY)
    assert np.all(out.cma_cover[20:24, 6:12] == CmaCoverClass.CLOUD_FILLED)
    assert np.all(out.cma_cloudsnow[20:24, 6:12] == CmaCloudSnowClass.CLOUD)

    # 4. Clear-sky snow -> SNOW_OR_ICE (3) in both cma_cover and cma_cloudsnow, binary cma == 0
    assert np.all(out.cma[20:24, 19:23] == CmaBinaryClass.CLOUD_FREE)
    assert np.all(out.cma_cover[20:24, 19:23] == CmaCoverClass.SNOW_OR_ICE)
    assert np.all(out.cma_cloudsnow[20:24, 19:23] == CmaCloudSnowClass.SNOW_OR_ICE)

    # 5. Thin ice cloud over snow -> CLOUD_CONTAMINATED (1) in cma_cover, THIN_ICE_CLOUD_OVER_SNOW (2) in cma_cloudsnow
    assert np.all(out.cma[20:24, 26:29] == CmaBinaryClass.CLOUDY)
    assert np.all(out.cma_cover[20:24, 26:29] == CmaCoverClass.CLOUD_CONTAMINATED)
    assert np.all(
        out.cma_cloudsnow[20:24, 26:29] == CmaCloudSnowClass.THIN_ICE_CLOUD_OVER_SNOW
    )


def test_desert_restoral_and_thermal_inversion_restoral() -> None:
    """Verify Step 5 (Desert Restoral Bit 20) and Step 8 (Thermal Inversion Filter Bit 26)."""
    h, w = 24, 24
    bt = _make_clear_sky_7ch(h, w, t_surf=305.0)
    # Desert patch with slightly depressed T10.8 (293 K vs 305 K T_surf) but strong quartz emissivity
    # signature: T10.8 - T8.7 = +4.2 K (> 2.8 K) and small split-window BTD
    r_desert = (slice(4, 10), slice(4, 10))
    bt["IR_108"][r_desert] = 293.0
    bt["IR_120"][r_desert] = 292.4
    bt["IR_087"][r_desert] = 288.8  # T10.8 - T8.7 = +4.2 K

    # Nocturnal cold-air inversion patch over land (rows 14..20, cols 14..20):
    # Cold surface T10.8 = 266 K (< T_surf - 8 K), spatially homogeneous, small split-window,
    # and WV_073 - IR_108 = -2.0 K (> -4.0 K threshold)
    r_inv = (slice(14, 20), slice(14, 20))
    bt["IR_108"][r_inv] = 266.0
    bt["IR_120"][r_inv] = 265.6
    bt["IR_087"][r_inv] = 264.8
    bt["WV_073"][r_inv] = 264.0  # BTD 7.3 - 10.8 = -2.0 K

    cfg = NwcSafCmaConfig()
    anc = estimate_autonomous_ir_ancillary(
        bt,
        cfg,
        is_land=np.ones((h, w), dtype=bool),
        t_surf_k=np.full((h, w), 305.0, dtype=np.float32),
    )
    out = run_cma_frame(bt, anc, cfg)

    # Desert patch is restored to clear and flags Bit 20
    assert np.all(out.cma[5:9, 5:9] == CmaBinaryClass.CLOUD_FREE)
    assert np.all((out.cma_testlist[5:9, 5:9] & (1 << CmaTestBit.DESERT_T108_T87_RESTORAL)) != 0)
    assert np.all(out.cma_quality[5:9, 5:9] == CmaQuality.QUESTIONABLE)

    # Inversion patch is restored to clear and flags Bit 26
    assert np.all(out.cma[15:19, 15:19] == CmaBinaryClass.CLOUD_FREE)
    assert np.all((out.cma_testlist[15:19, 15:19] & (1 << CmaTestBit.SPATIAL_FILTER_TO_CLEAR)) != 0)


def test_temporal_15min_cooling_and_1h_coherence_and_expansion() -> None:
    """Verify Bit 22 (15-min cooling), Bit 23 (1h stationary restoral), and Bit 24 (spatial expansion)."""
    h, w = 24, 24
    cfg = NwcSafCmaConfig()
    bt_now = _make_clear_sky_7ch(h, w, t_surf=294.0)
    bt_15m = bt_now["IR_108"].copy()
    bt_60m = bt_now["IR_108"].copy()
    cma_60m = np.zeros((h, w), dtype=np.uint8)

    # 1. Rapid 15-min cooling patch (rows 3..7, cols 3..7): cools from 293.4 K to 288.0 K (-5.4 K)
    # Still above T10.8 static threshold (284.8 K), so only caught by 15-min temporal test!
    r_cool = (slice(3, 7), slice(3, 7))
    bt_now["IR_108"][r_cool] = 288.0
    bt_now["IR_120"][r_cool] = 287.3
    bt_now["IR_087"][r_cool] = 287.2

    # 2. Stationary marginal cold land anomaly (rows 10..15, cols 3..8):
    # Slightly below static T10.8 threshold (283.5 K < 284.8 K) for a full hour (dT_60m = 0.0 K)
    r_stat = (slice(10, 15), slice(3, 8))
    bt_now["IR_108"][r_stat] = 283.5
    bt_now["IR_120"][r_stat] = 283.1
    bt_now["IR_087"][r_stat] = 282.8
    bt_15m[r_stat] = 283.5
    bt_60m[r_stat] = 283.5
    cma_60m[r_stat] = CmaBinaryClass.CLOUDY

    # 3. Confident cold core (rows 14..18, cols 14..18) + warm cloud collar at (row 13, cols 14..18)
    # that cooled by -3.3 K over 60 minutes -> caught by Bit 24 (Spatial Cloud Expansion)
    r_core = (slice(14, 18), slice(14, 18))
    bt_now["IR_108"][r_core] = 245.0
    bt_now["IR_120"][r_core] = 244.5
    bt_now["IR_087"][r_core] = 244.2
    bt_now["WV_073"][r_core] = 244.0
    bt_now["IR_134"][r_core] = 243.0
    bt_now["IR_108"][13, 14:18] = 286.2  # 1.4 K warmer than static threshold 284.8 K
    bt_now["IR_120"][13, 14:18] = 285.6
    bt_now["IR_087"][13, 14:18] = 285.4
    bt_15m[13, 14:18] = 287.5  # gradual cooling (-1.3 K in last 15m, -4.3 K over 60m)
    bt_60m[13, 14:18] = 290.5  # cooled by -4.3 K over 1 hour

    anc = estimate_autonomous_ir_ancillary(
        bt_now,
        cfg,
        is_land=np.ones((h, w), dtype=bool),
        t_surf_k=np.full((h, w), 294.0, dtype=np.float32),
    )
    out = run_cma_frame(
        bt_now,
        anc,
        cfg,
        bt_108_t_minus_15m=bt_15m,
        bt_108_t_minus_60m=bt_60m,
        cma_t_minus_60m=cma_60m,
    )

    # 15-min cooling patch is detected as CLOUD_CONTAMINATED (1) and sets Bit 22
    assert np.all(out.cma[4:6, 4:6] == CmaBinaryClass.CLOUDY)
    assert np.all(out.cma_cover[4:6, 4:6] == CmaCoverClass.CLOUD_CONTAMINATED)
    assert np.all(
        (out.cma_testlist[4:6, 4:6] & (1 << CmaTestBit.TEMPORAL_DIFFERENCING_T108)) != 0
    )

    # 1h stationary anomaly is restored to CLOUD_FREE (0) and sets Bit 23
    assert np.all(out.cma[11:14, 4:7] == CmaBinaryClass.CLOUD_FREE)
    assert np.all((out.cma_testlist[11:14, 4:7] & (1 << CmaTestBit.TEMPORAL_COHERENCE_1H)) != 0)

    # Collar row 13 is captured by Bit 24 (Spatial Cloud Expansion)
    assert np.all(out.cma[13, 15:17] == CmaBinaryClass.CLOUDY)
    assert np.all(
        (out.cma_testlist[13, 15:17] & (1 << CmaTestBit.SPATIAL_CLOUD_EXPANSION)) != 0
    )


def test_dust_and_volcanic_ash_detection_and_isolated_pixel_filter() -> None:
    """Verify Dust (`cma_dust`), Volcanic Ash (`cma_volcanic`), and Isolated Pixel Filter (`Bit 27`)."""
    h, w = 24, 24
    bt = _make_clear_sky_7ch(h, w, t_surf=298.0)

    # 1. Saharan dust plume (rows 3..8, cols 3..8):
    # Reverse split-window (T12.0 - T10.8 = +1.2 K > 0.35 K), T10.8 - T8.7 = +2.5 K (> 1.5 K),
    # warm T10.8 = 285 K (> 265 K), spatially smooth
    r_dust = (slice(3, 8), slice(3, 8))
    bt["IR_108"][r_dust] = 285.0
    bt["IR_120"][r_dust] = 286.2
    bt["IR_087"][r_dust] = 282.5

    # 2. Volcanic ash plume (rows 3..8, cols 14..19):
    # Cold plume (T10.8 = 248 K <= 265 K) with strong silicate reverse absorption:
    # T12.0 - T10.8 = +2.4 K (> 1.2 K) and T8.7 - T10.8 = -1.5 K (< 0.5 K)
    r_volc = (slice(3, 8), slice(14, 19))
    bt["IR_108"][r_volc] = 248.0
    bt["IR_120"][r_volc] = 250.4
    bt["IR_087"][r_volc] = 246.5

    # 3. Single isolated noisy cloudy pixel at (18, 18) surrounded by 8 clear neighbors
    bt["IR_108"][18, 18] = 283.0  # moderately below threshold 290 K, but isolated

    cfg = NwcSafCmaConfig()
    anc = estimate_autonomous_ir_ancillary(
        bt,
        cfg,
        is_land=np.ones((h, w), dtype=bool),
        t_surf_k=np.full((h, w), 298.0, dtype=np.float32),
    )
    out = run_cma_frame(bt, anc, cfg)

    assert np.all(out.cma_dust[4:7, 4:7] == CmaDustClass.DUST)
    assert np.all(out.cma_volcanic[4:7, 15:18] == CmaVolcanicClass.VOLCANIC_PLUME)

    # Isolated pixel at (18, 18) is filtered back to clear and sets Bit 27
    assert out.cma[18, 18] == CmaBinaryClass.CLOUD_FREE
    assert (out.cma_testlist[18, 18] & (1 << CmaTestBit.ISOLATED_PIXEL_FILTER)) != 0


def test_autonomous_pure_ir_sequence_runner() -> None:
    """Verify `run_nwcsaf_cma_sequence` and `NwcSafCmaTracker` on a 6-frame `(T, 7, H, W)` sequence."""
    t_steps, h, w = 6, 20, 20
    clip = np.empty((t_steps, 7, h, w), dtype=np.float32)
    for t in range(t_steps):
        frame = _make_clear_sky_7ch(h, w, t_surf=292.0)
        # Moving cold convective core that expands over time
        r0 = 4 + t
        frame["WV_062"][r0 : r0 + 5, 6:12] = 222.0
        frame["WV_073"][r0 : r0 + 5, 6:12] = 224.0
        frame["IR_087"][r0 : r0 + 5, 6:12] = 225.0
        frame["IR_097"][r0 : r0 + 5, 6:12] = 226.0
        frame["IR_108"][r0 : r0 + 5, 6:12] = 225.0
        frame["IR_120"][r0 : r0 + 5, 6:12] = 224.5
        frame["IR_134"][r0 : r0 + 5, 6:12] = 223.0
        for idx, ch in enumerate(
            ("WV_062", "WV_073", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134")
        ):
            clip[t, idx] = frame[ch]

    res = run_nwcsaf_cma_sequence(clip)
    assert res.cma.shape == (t_steps, h, w)
    assert res.cma_cover.shape == (t_steps, h, w)
    assert res.cma_cloudsnow.shape == (t_steps, h, w)

    # Clear background remains CLOUD_FREE (0)
    assert np.all(res.cma[:, 0:3, 0:4] == CmaBinaryClass.CLOUD_FREE)
    assert np.all(res.cma_cover[:, 0:3, 0:4] == CmaCoverClass.CLOUD_FREE)

    # Core center is CLOUD_FILLED (2) in cma_cover and CLOUD (1) in cma_cloudsnow
    for t in range(t_steps):
        r0 = 4 + t
        assert np.all(res.cma[t, r0 + 1 : r0 + 4, 7:11] == CmaBinaryClass.CLOUDY)
        assert np.all(res.cma_cover[t, r0 + 1 : r0 + 4, 7:11] == CmaCoverClass.CLOUD_FILLED)
        assert np.all(res.cma_cloudsnow[t, r0 + 1 : r0 + 4, 7:11] == CmaCloudSnowClass.CLOUD)
