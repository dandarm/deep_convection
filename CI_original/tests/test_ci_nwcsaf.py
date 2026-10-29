"""Comprehensive unit and integration test suite for NWC SAF GEO-CI (Convection Initiation)."""

from __future__ import annotations

import numpy as np
import pytest

from CI_original.ci_nwcsaf.config import (
    NwcSafCiConfig,
    CiProbabilityClass,
    SEVIRI_CHANNELS,
    CHANNEL_INDICES,
)
from CI_original.ci_nwcsaf.candidate_selection import (
    select_eligible_ci_pixels,
    compute_pure_ir_cloud_mask,
    compute_nwp_convective_mask,
)
from CI_original.ci_nwcsaf.detection_tracking import (
    CiCloudCell,
    detect_warm_cloud_cells,
    WarmCellTracker,
)
from CI_original.ci_nwcsaf.interest_fields import (
    compute_static_btds,
    compute_ci_interest_fields,
    count_relevant_parameters,
    lagrangian_backward_advection,
)
from CI_original.ci_nwcsaf.discrimination import (
    diagnose_ci_probability_pixel,
    evaluate_ci_diagnosis,
    enforce_lead_time_monotonicity,
)
from CI_original.ci_nwcsaf.forward_scheme import (
    apply_forward_scheme,
    generate_ci_forecast_horizons,
)
from CI_original.ci_nwcsaf.ancillary import compute_pure_ir_amv_motion
from CI_original.ci_nwcsaf.pipeline import NwcSafCiTracker, run_nwcsaf_ci_sequence


def create_synthetic_seviri_cube(
    h: int = 64,
    w: int = 64,
    bg_surface_k: float = 295.0,
) -> np.ndarray:
    """Helper creating a baseline 7-channel SEVIRI cube in Kelvin."""
    cube = np.zeros((7, h, w), dtype=np.float32)
    # WV_062: high dry troposphere (~235 K)
    cube[CHANNEL_INDICES["WV_062"]] = 235.0
    # WV_073: mid troposphere (~250 K)
    cube[CHANNEL_INDICES["WV_073"]] = 250.0
    # IR_087: surface/cloud (~bg - 1)
    cube[CHANNEL_INDICES["IR_087"]] = bg_surface_k - 1.0
    # IR_097: ozone (~255 K)
    cube[CHANNEL_INDICES["IR_097"]] = 255.0
    # IR_108: window channel
    cube[CHANNEL_INDICES["IR_108"]] = bg_surface_k
    # IR_120: split-window (~bg - 0.5)
    cube[CHANNEL_INDICES["IR_120"]] = bg_surface_k - 0.5
    # IR_134: CO2 absorption (~bg - 12)
    cube[CHANNEL_INDICES["IR_134"]] = bg_surface_k - 12.0
    return cube


def test_config_and_channel_mappings():
    """Verifies default configuration parameters and channel index alignments."""
    config = NwcSafCiConfig()
    assert config.t_cold_k == 248.15  # -25°C
    assert config.t_warm_k == 283.15  # +10°C
    assert config.delta_t_tour_k == 3.0  # 3 K prominence
    assert config.forecast_horizons_min == (30, 60, 90)

    # 7 standard SEVIRI channels
    assert len(SEVIRI_CHANNELS) == 7
    assert SEVIRI_CHANNELS[0] == "WV_062"
    assert SEVIRI_CHANNELS[4] == "IR_108"
    assert CHANNEL_INDICES["IR_108"] == 4


def test_candidate_selection_filtering():
    """Verifies that clear sky and very cold mature storm anvils are excluded."""
    cube = create_synthetic_seviri_cube(h=40, w=40, bg_surface_k=295.0)

    # Add a warm developing cumulus cell at center: T = 265 K (eligible)
    cube[CHANNEL_INDICES["IR_108"], 15:25, 15:25] = 265.0
    cube[CHANNEL_INDICES["IR_120"], 15:25, 15:25] = 264.0

    # Add a mature cold convective anvil: T = 215 K (ineligible, RDT domain)
    cube[CHANNEL_INDICES["IR_108"], 0:10, 0:10] = 215.0
    cube[CHANNEL_INDICES["IR_120"], 0:10, 0:10] = 214.0

    eligible_mask, cloud_mask = select_eligible_ci_pixels(cube)

    # Warm cumulus must be eligible
    assert eligible_mask[20, 20] == True

    # Cold mature storm must be excluded from CI
    assert eligible_mask[5, 5] == False

    # Clear warm surface (295 K) must be excluded
    assert eligible_mask[35, 35] == False


def test_nwp_convective_mask_evaluation():
    """Verifies Lifted Index, Showalter, and K-Index combination into NWP convective mask."""
    shape = (20, 20)
    li = np.zeros(shape)
    ki = np.zeros(shape)
    shw = np.zeros(shape)

    # Region 1: Fully stable (LI = +2, SHW = +5, KI = 15) -> Mask 0
    li[:10, :10] = 2.0
    shw[:10, :10] = 5.0
    ki[:10, :10] = 15.0

    # Region 2: Unstable (LI = -5, SHW = -4, KI = 35) -> Mask 2
    li[10:, 10:] = -5.0
    shw[10:, 10:] = -4.0
    ki[10:, 10:] = 35.0

    # Region 3: Neutral / Unclear (LI = -1, SHW = 1, KI = 24) -> Mask 1
    li[:10, 10:] = -1.0
    shw[:10, 10:] = 1.0
    ki[:10, 10:] = 24.0

    mask = compute_nwp_convective_mask(lifted_index=li, k_index=ki, showalter_index=shw)

    assert mask[5, 5] == 0
    assert mask[15, 15] == 2
    assert mask[5, 15] == 1


def test_warm_cloud_cell_detection():
    """Verifies detection of warm cloud cells with minimum vertical prominence >= 3 K."""
    config = NwcSafCiConfig(delta_t_tour_k=3.0)
    ir108 = np.full((50, 50), 285.0, dtype=np.float32)

    # Insert a developing cloud cell: base at 275 K, core minimum at 260 K (prominence = 15 K >= 3 K)
    for r in range(50):
        for c in range(50):
            dist = np.hypot(r - 25, c - 25)
            if dist <= 8:
                ir108[r, c] = 260.0 + dist * 1.5  # core 260 K, edge 272 K

    cells = detect_warm_cloud_cells(ir108, config=config)
    assert len(cells) >= 1
    cell = cells[0]
    assert np.isclose(cell.t_min_k, 260.0, atol=1.0)
    assert cell.t_base_k - cell.t_min_k >= 3.0
    assert np.isclose(cell.center_yx[0], 25.0, atol=1.5)
    assert np.isclose(cell.center_yx[1], 25.0, atol=1.5)


def test_motion_tracking_and_amv():
    """Verifies that cell tracking links consecutive slots and estimates non-zero displacement."""
    config = NwcSafCiConfig()
    tracker = WarmCellTracker(config)

    # Frame 1: cell at (20, 20)
    cell_t0 = CiCloudCell(
        cell_id=1,
        t_min_k=262.0,
        t_base_k=272.0,
        center_yx=(20.0, 20.0),
        area_px=25,
        area_km2=225.0,
        pixel_coords=np.where(np.ones((5, 5))),
    )
    # Mock pixel coords in full grid
    rr, cc = np.indices((5, 5))
    cell_t0.pixel_coords = (rr.ravel() + 18, cc.ravel() + 18)

    tracker.previous_cells = [cell_t0]

    # Frame 2: cell moved by +4 rows and +3 cols over 15 minutes
    cell_t1 = CiCloudCell(
        cell_id=2,
        t_min_k=258.0,
        t_base_k=270.0,
        center_yx=(24.0, 23.0),
        area_px=25,
        area_km2=225.0,
        pixel_coords=(rr.ravel() + 22, cc.ravel() + 21),
    )

    u_field, v_field = tracker.track_and_update_motion(
        [cell_t1],
        shape=(50, 50),
        delta_t_min=15.0,
    )

    # Cell must have parent linked
    assert cell_t1.parent_id == 1
    # Measured velocity: dy = 4 / 15 px/min, dx = 3 / 15 px/min
    assert np.isclose(cell_t1.v_mps, 4.0 / 15.0, atol=0.05)
    assert np.isclose(cell_t1.u_mps, 3.0 / 15.0, atol=0.05)
    # The motion field at cell location must reflect positive displacement
    assert v_field[24, 23] > 0.1
    assert u_field[24, 23] > 0.1


def test_static_and_dynamic_interest_fields():
    """Verifies BTD definitions and Lagrangian cooling rate computation."""
    cube_curr = create_synthetic_seviri_cube(h=30, w=30)
    cube_past = create_synthetic_seviri_cube(h=30, w=30)

    # In current cube, cloud at (15, 15) cooled significantly to 255 K
    cube_curr[CHANNEL_INDICES["IR_108"], 15, 15] = 255.0
    cube_curr[CHANNEL_INDICES["WV_062"], 15, 15] = 238.0
    cube_curr[CHANNEL_INDICES["WV_073"], 15, 15] = 248.0
    cube_curr[CHANNEL_INDICES["IR_087"], 15, 15] = 256.0
    cube_curr[CHANNEL_INDICES["IR_120"], 15, 15] = 254.0
    cube_curr[CHANNEL_INDICES["IR_134"], 15, 15] = 247.0

    # In past cube 15 min ago, cloud was at (15, 12) with warmer temperature 270 K (shifted by u=0.2 px/min)
    cube_past[CHANNEL_INDICES["IR_108"], 15, 12] = 270.0
    cube_past[CHANNEL_INDICES["WV_062"], 15, 12] = 236.0

    motion_u = np.full((30, 30), 3.0 / 15.0, dtype=np.float32)  # dx = +3 px in 15 min
    motion_v = np.zeros((30, 30), dtype=np.float32)

    fields = compute_ci_interest_fields(
        current_cube=cube_curr,
        past_cube=cube_past,
        motion_u_px_per_min=motion_u,
        motion_v_px_per_min=motion_v,
        delta_t_min=15.0,
    )

    # Static BTD checks
    assert np.isclose(fields.wbtd[15, 15], 238.0 - 248.0)  # -10.0
    assert np.isclose(fields.btd[15, 15], 238.0 - 255.0)   # -17.0
    assert np.isclose(fields.btd4[15, 15], 255.0 - 256.0)  # -1.0
    assert np.isclose(fields.btd5[15, 15], 254.0 - 255.0)  # -1.0
    assert np.isclose(fields.btd6[15, 15], 247.0 - 255.0)  # -8.0

    # Lagrangian 15-min cooling: T_curr (255) - T_past (270) = -15 K/15m
    assert np.isclose(fields.txbt_15[15, 15], -15.0, atol=0.5)


def test_empirical_decision_rules_table6():
    """Verifies all branches of NWC SAF Table 6 empirical decision rules."""
    # Branch 1: High growth (>= 2)
    assert diagnose_ci_probability_pixel(n_growth=3, n_glaciation=3, n_height=4) == CiProbabilityClass.HIGHPROB  # 4
    assert diagnose_ci_probability_pixel(n_growth=2, n_glaciation=3, n_height=4) == CiProbabilityClass.HIGHPROB  # 4
    assert diagnose_ci_probability_pixel(n_growth=2, n_glaciation=3, n_height=3) == CiProbabilityClass.MODPROB   # 3
    assert diagnose_ci_probability_pixel(n_growth=2, n_glaciation=3, n_height=2) == CiProbabilityClass.LOWPROB   # 2
    assert diagnose_ci_probability_pixel(n_growth=2, n_glaciation=2, n_height=4) == CiProbabilityClass.MODPROB   # 3
    assert diagnose_ci_probability_pixel(n_growth=2, n_glaciation=2, n_height=3) == CiProbabilityClass.LOWPROB   # 2
    assert diagnose_ci_probability_pixel(n_growth=2, n_glaciation=2, n_height=1) == CiProbabilityClass.VLOWPROB  # 1

    # Branch 2: Moderate growth (== 1)
    assert diagnose_ci_probability_pixel(n_growth=1, n_glaciation=3, n_height=4) == CiProbabilityClass.MODPROB   # 3
    assert diagnose_ci_probability_pixel(n_growth=1, n_glaciation=3, n_height=3) == CiProbabilityClass.LOWPROB   # 2
    assert diagnose_ci_probability_pixel(n_growth=1, n_glaciation=2, n_height=4) == CiProbabilityClass.LOWPROB   # 2
    assert diagnose_ci_probability_pixel(n_growth=1, n_glaciation=2, n_height=3) == CiProbabilityClass.VLOWPROB  # 1

    # Branch 3: Static / Zero growth (== 0)
    assert diagnose_ci_probability_pixel(n_growth=0, n_glaciation=3, n_height=4) == CiProbabilityClass.LOWPROB   # 2
    assert diagnose_ci_probability_pixel(n_growth=0, n_glaciation=3, n_height=2) == CiProbabilityClass.VLOWPROB  # 1
    assert diagnose_ci_probability_pixel(n_growth=0, n_glaciation=1, n_height=4) == CiProbabilityClass.NO_PROB   # 0

    # Test vectorized function matches scalar
    g_arr = np.array([[3, 2, 0]])
    gl_arr = np.array([[3, 2, 3]])
    h_arr = np.array([[4, 3, 2]])
    vec_res = evaluate_ci_diagnosis(g_arr, gl_arr, h_arr)
    assert vec_res[0, 0] == CiProbabilityClass.HIGHPROB
    assert vec_res[0, 1] == CiProbabilityClass.LOWPROB
    assert vec_res[0, 2] == CiProbabilityClass.VLOWPROB


def test_forward_scheme_and_lead_time_monotonicity():
    """Verifies trajectory advection and P30 <= P60 <= P90 constraint."""
    diag = np.zeros((30, 30), dtype=np.uint8)
    diag[10, 10] = CiProbabilityClass.MODPROB  # 3

    u = np.full((30, 30), 0.2, dtype=np.float32)  # +6 px in 30 min, +12 px in 60 min, +18 px in 90 min
    v = np.zeros((30, 30), dtype=np.float32)

    horizons = generate_ci_forecast_horizons(diag, u, v)
    p30 = horizons["ci_prob30"]
    p60 = horizons["ci_prob60"]
    p90 = horizons["ci_prob90"]

    # In p30, pixel at (10, 10 + 6) must be active
    assert p30[10, 16] == CiProbabilityClass.MODPROB
    # In p60, pixel at (10, 10 + 12) must be active
    assert p60[10, 22] == CiProbabilityClass.MODPROB
    # In p90, pixel at (10, 10 + 18) must be active
    assert p90[10, 28] == CiProbabilityClass.MODPROB

    # Strict monotonicity across all pixels: p30 <= p60 <= p90
    assert np.all(p30 <= p60)
    assert np.all(p60 <= p90)


def test_end_to_end_tracker_on_synthetic_initiation_event():
    """Runs the full stateful pipeline on a 3-step sequence simulating convective initiation."""
    tracker = NwcSafCiTracker()

    cube_t0 = create_synthetic_seviri_cube(h=40, w=40)
    cube_t1 = create_synthetic_seviri_cube(h=40, w=40)

    # Step 0: Warm shallow cumulus forming at (20, 20) with T = 274 K
    cube_t0[CHANNEL_INDICES["IR_108"], 18:22, 18:22] = 274.0
    cube_t0[CHANNEL_INDICES["IR_120"], 18:22, 18:22] = 273.0

    res0 = tracker.step(cube_t0, timestamp="2020-09-01 12:00:00")
    assert res0.eligible_mask[20, 20] == True
    # Initial cold start: zero trends, low initiation probability
    assert res0.ci_status_flag & (1 << 0)  # Bit 0 set

    # Step 1: 15 minutes later, cell deepened rapidly to 252 K (-22 K cooling), glaciating
    # Advected slightly to (20, 22)
    cube_t1[CHANNEL_INDICES["IR_108"], 18:22, 20:24] = 252.0
    cube_t1[CHANNEL_INDICES["IR_087"], 18:22, 20:24] = 253.0  # BTD4 = -1.0 K (in [-2.1, 0])
    cube_t1[CHANNEL_INDICES["WV_062"], 18:22, 20:24] = 236.0  # BTD = -16.0 K (in [-34, -12])
    cube_t1[CHANNEL_INDICES["WV_073"], 18:22, 20:24] = 246.0  # WBTD = -10.0 K (in [-17, -7])
    cube_t1[CHANNEL_INDICES["IR_120"], 18:22, 20:24] = 251.0  # BTD5 = -1.0 K (in [-2.2, 0])
    cube_t1[CHANNEL_INDICES["IR_134"], 18:22, 20:24] = 244.0  # BTD6 = -8.0 K (in [-17, -6])

    res1 = tracker.step(cube_t1, timestamp="2020-09-01 12:15:00")

    # Cell must be detected
    assert len(res1.detected_cells) >= 1
    # Strong cooling must be detected
    assert res1.n_growth[20, 22] >= 1
    # Glaciation and height criteria satisfied
    assert res1.n_glaciation[20, 22] >= 2
    assert res1.n_height[20, 22] >= 3

    # Convection Initiation diagnosed!
    assert res1.ci_prob30[20, 22] >= CiProbabilityClass.LOWPROB
    # Monotonicity obeyed
    assert res1.ci_prob90[20, 22] >= res1.ci_prob60[20, 22] >= res1.ci_prob30[20, 22]


def test_night_vs_day_interest_field_thresholds():
    """Verifies that daytime and night/pure-IR modes apply Table 4 and Table 5 respectively."""
    config_pure = NwcSafCiConfig(mode="night_or_pure_ir")
    config_day = NwcSafCiConfig(mode="day_with_cmic")

    cube = create_synthetic_seviri_cube(h=20, w=20)
    # At (10, 10): BTD4 = IR10.8 - IR8.7 = -5.0 K
    # Under Table 5 (Night/pure-IR), valid range is [-2.1, 0.0] -> FAIL
    # Under Table 4 (Day with CMIC), valid range is [-10.0, 0.0] -> PASS
    cube[CHANNEL_INDICES["IR_108"], 10, 10] = 260.0
    cube[CHANNEL_INDICES["IR_087"], 10, 10] = 265.0  # diff = -5.0

    eligible = np.zeros((20, 20), dtype=bool)
    eligible[10, 10] = True

    fields = compute_ci_interest_fields(cube, None, np.zeros((20, 20)), np.zeros((20, 20)), config=config_pure)

    _, n_glac_pure, _, _ = count_relevant_parameters(fields, eligible, config=config_pure)
    _, n_glac_day, _, _ = count_relevant_parameters(fields, eligible, config=config_day)

    # In pure-IR mode, BTD4 = -5.0 is rejected (0 or 1 depending on IR108)
    # In day mode, BTD4 = -5.0 is accepted (+1 count)
    assert n_glac_day[10, 10] > n_glac_pure[10, 10]


def test_amv_cross_correlation_synthetic_shift():
    """Verifies that pure-IR AMVs recover an induced spatial displacement."""
    np.random.seed(42)
    # Create textured cloud field
    base = np.random.uniform(250.0, 280.0, (64, 64)).astype(np.float32)
    # Smooth slightly to create realistic cloud texture
    from scipy.ndimage import gaussian_filter
    img_past = gaussian_filter(base, sigma=2.0)

    # Shift current image by +3 rows and +2 cols
    shift_r = 3
    shift_c = 2
    img_curr = np.roll(np.roll(img_past, shift_r, axis=0), shift_c, axis=1)

    u_amv, v_amv = compute_pure_ir_amv_motion(img_curr, img_past, delta_t_min=15.0, patch_size=16, search_radius=6)

    # Center region should have positive u and v velocities
    center_u = u_amv[24:40, 24:40]
    center_v = v_amv[24:40, 24:40]

    assert np.mean(center_v) > 0.05
    assert np.mean(center_u) > 0.05


def test_clear_sky_graceful_handling():
    """Verifies that an entirely cloud-free scene produces clean zero outputs without crashing."""
    tracker = NwcSafCiTracker()
    cube = create_synthetic_seviri_cube(h=32, w=32, bg_surface_k=305.0)

    res = tracker.step(cube)
    assert np.all(res.ci_prob30 == 0)
    assert np.all(res.ci_prob60 == 0)
    assert np.all(res.ci_prob90 == 0)
    assert len(res.detected_cells) == 0


def test_run_nwcsaf_ci_sequence_utility():
    """Verifies that run_nwcsaf_ci_sequence processes an array sequence of cubes."""
    cubes = [create_synthetic_seviri_cube(h=24, w=24) for _ in range(3)]
    results = run_nwcsaf_ci_sequence(cubes)
    assert len(results) == 3
    for r in results:
        assert r.ci_prob30.shape == (24, 24)
        assert r.ci_prob60.shape == (24, 24)
        assert r.ci_prob90.shape == (24, 24)
