"""Unit and algorithmic verification tests for the NWC SAF GEO-RDT-CW implementation."""

from __future__ import annotations

import numpy as np
import pytest

from rdt_nwcsaf import (
    NwcSafRdtConfig,
    NwcSafRdtTracker,
    classify_vertical_category,
    detect_cell_overshooting_tops,
    detect_rdt_cells,
    run_nwcsaf_rdt_sequence,
    track_rdt_cells,
)


def _make_7ch_frame(
    ir108: np.ndarray,
    *,
    btd_wv62_ir108: np.ndarray | float = -4.0,
    wbtd_wv62_wv73: np.ndarray | float = -12.0,
    btd_ir87_ir108: np.ndarray | float = -1.0,
    btd_ir108_ir120: np.ndarray | float = 0.5,
) -> np.ndarray:
    """Construct a synthetic 7-channel SEVIRI frame (7, H, W) in Kelvin."""
    ir108_f = np.asarray(ir108, dtype=np.float32)
    h, w = ir108_f.shape
    wv062 = ir108_f + np.broadcast_to(np.asarray(btd_wv62_ir108, dtype=np.float32), (h, w))
    wv073 = wv062 - np.broadcast_to(np.asarray(wbtd_wv62_wv73, dtype=np.float32), (h, w))
    ir087 = ir108_f + np.broadcast_to(np.asarray(btd_ir87_ir108, dtype=np.float32), (h, w))
    ir097 = ir108_f + 8.0
    ir120 = ir108_f - np.broadcast_to(np.asarray(btd_ir108_ir120, dtype=np.float32), (h, w))
    ir134 = ir108_f + 12.0
    return np.stack([wv062, wv073, ir087, ir097, ir108_f, ir120, ir134], axis=0)


def test_config_and_vertical_category_slicing() -> None:
    cfg = NwcSafRdtConfig()
    # Category 0: T_min <= -40 C (233.15 K) for >= 30 min
    assert classify_vertical_category(225.0, 245.0, 35.0, cfg) == 0
    # Category 1: T_min <= -40 C but age < 30 min (Mature Transition)
    assert classify_vertical_category(225.0, 245.0, 15.0, cfg) == 1
    # Category 1 via cold base T_BT <= -35 C (238.15 K)
    assert classify_vertical_category(235.0, 237.0, 0.0, cfg) == 1
    # Category 2 (Cold Transition): -40 C < T_min <= -35 C (233.15..238.15 K)
    assert classify_vertical_category(236.0, 250.0, 0.0, cfg) == 2
    # Category 3 (Warm1 Transition): -35 C < T_min <= -25 C (238.15..248.15 K)
    assert classify_vertical_category(244.0, 260.0, 0.0, cfg) == 3
    # Category 4 (Warm2 Transition): -25 C < T_min <= -15 C (248.15..258.15 K)
    assert classify_vertical_category(253.0, 268.0, 0.0, cfg) == 4
    # Category 5 (Warm): T_min > -15 C (258.15 K)
    assert classify_vertical_category(262.0, 272.0, 0.0, cfg) == 5


def test_detection_sibling_tower_separation_and_btlimit() -> None:
    cfg = NwcSafRdtConfig()
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w]
    area = np.full((h, w), 16.0, dtype=np.float32)  # 16 km2/pixel -> 4 pixels = 64 km2 >= SMIN (60 km2)

    # Create two sibling convective towers embedded in a shared anvil, plus a sub-tower
    # inside Tower 1 that would only split below BTLIMIT (-60 C = 213.15 K):
    # - Tower 1 main core at (20, 18) [199 K] and inner sub-core at (20, 23) [202 K]
    #   which merge at 210 K (< BTLIMIT=213.15 K, so BTLIMIT must keep them together!)
    # - Tower 2 at (20, 44) with T_min = 218 K, separating from Tower 1 around 235 K
    r1a = np.hypot(yy - 20.0, xx - 18.0)
    r1b = np.hypot(yy - 20.0, xx - 23.0)
    r2 = np.hypot(yy - 20.0, xx - 44.0)
    ir1 = np.minimum(199.0 + 3.2 * r1a, 202.0 + 3.2 * r1b)
    ir2 = 218.0 + 3.0 * r2
    ir = np.minimum(np.minimum(ir1, ir2), 290.0).astype(np.float32)

    frame = _make_7ch_frame(ir, btd_wv62_ir108=1.0)
    labels, cells, _, _ = detect_rdt_cells(frame, area, cfg, tropopause_temp_k=212.0)

    # Tower 1 (with its sub-core below BTLIMIT kept unified) and Tower 2 are detected as 2 cells
    assert len(cells) == 2
    c_cold = min(cells.values(), key=lambda c: c["t_min_k"])
    c_warm = max(cells.values(), key=lambda c: c["t_min_k"])

    # Sub-cores (20,18) and (20,23) must belong to the same mature cell because their split is < BTLIMIT (213.15 K)
    assert labels[20, 18] == labels[20, 23]
    assert c_cold["t_min_k"] == pytest.approx(199.0, abs=1.0)
    assert c_cold["t_bt_k"] >= cfg.btlimit_k
    assert c_warm["t_min_k"] == pytest.approx(218.0, abs=1.0)
    assert labels[20, 18] != labels[20, 44]


def test_detection_single_tower_budding_and_pyramidal_refinement() -> None:
    cfg = NwcSafRdtConfig()
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w]
    area = np.full((h, w), 16.0, dtype=np.float32)

    # Single tower budding out of a wide, flat warm stratiform deck at 262 K:
    # Core around (32, 32) drops sharply from 250 K down to 232 K, surrounded by a huge 262 K plateau
    r = np.hypot(yy - 32.0, xx - 32.0)
    ir = np.full((h, w), 290.0, dtype=np.float32)
    ir[r <= 22.0] = 262.0  # huge flat deck (area ~ 1520 * 16 = 24,320 km2)
    ir[r <= 6.0] = 232.0 + 3.0 * r[r <= 6.0]  # compact convective tower (232..250 K)

    frame = _make_7ch_frame(ir)
    _, cells, _, _ = detect_rdt_cells(frame, area, cfg)
    assert len(cells) == 1
    cell = next(iter(cells.values()))
    assert cell["is_single_tower"] is True
    # Base of Tower must be cut above the 262 K flat deck (Case A constriction <= 0.7)
    assert cell["t_bt_k"] < 261.5
    assert cell["area_km2"] < 4000.0


def test_detection_stretched_pixel_and_cloud_mask_filtering() -> None:
    cfg = NwcSafRdtConfig(
        nominal_pixel_area_km2=9.0, stretch_area_factor=5.0, cma_enabled=True
    )
    h, w = 32, 32
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - 16.0, xx - 16.0)
    ir = (220.0 + 2.5 * r).astype(np.float32)
    frame = _make_7ch_frame(ir)

    # All pixels stretched beyond 5 * 9 = 45 km2 -> no cell should be detected
    stretched_area = np.full((h, w), 50.0, dtype=np.float32)
    _, cells_stretched, _, valid_mask = detect_rdt_cells(frame, stretched_area, cfg)
    assert not valid_mask.any()
    assert len(cells_stretched) == 0

    # Normal pixel area (16 km2), but ct_cloudy_mask = False -> filtered out
    normal_area = np.full((h, w), 16.0, dtype=np.float32)
    clear_mask = np.zeros((h, w), dtype=bool)
    _, cells_clear, _, _ = detect_rdt_cells(
        frame, normal_area, cfg, ct_cloudy_mask=clear_mask
    )
    assert len(cells_clear) == 0


def test_tracking_four_steps_and_common_threshold_father() -> None:
    cfg = NwcSafRdtConfig()
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w]
    area = np.full((h, w), 16.0, dtype=np.float32)

    # Slot 0: Two parent cells close enough to merge at Slot 1:
    # Parent 1 at (26, 20) [larger/colder, T_min=218 K] and Parent 2 at (33, 20) [smaller, T_min=224 K]
    r_p1 = np.hypot(yy - 26.0, xx - 20.0)
    r_p2 = np.hypot(yy - 33.0, xx - 20.0)
    ir0 = np.minimum(np.minimum(218.0 + 3.2 * r_p1, 224.0 + 3.5 * r_p2), 290.0).astype(np.float32)
    f0 = _make_7ch_frame(ir0)
    l0, c0, h0, _ = detect_rdt_cells(f0, area, cfg)
    assert len(c0) == 2

    # Slot 1: Both cells translate east by +2 pixels and merge into a single elongated system
    # whose cold core spans y=26..33 at x=22 (with single minimum at (26, 22) and plateau < 6 K to (33, 22))
    r_c1a = np.hypot(yy - 26.0, xx - 22.0)
    r_c1b = np.hypot(yy - 33.0, xx - 22.0)
    ir1 = np.minimum(np.minimum(216.0 + 2.5 * r_c1a, 219.5 + 2.5 * r_c1b), 290.0).astype(np.float32)
    f1 = _make_7ch_frame(ir1)
    l1, c1, h1, _ = detect_rdt_cells(f1, area, cfg)
    assert len(c1) == 1

    parents, _, _, primary_parent, events, _ = track_rdt_cells(
        c0, l0, h0, ir0, c1, l1, h1, ir1, area, cfg
    )
    only_cid = next(iter(c1.keys()))
    assert len(parents[only_cid]) == 2
    assert events[only_cid] == "merge_main"
    # Larger/colder parent at common threshold must be selected as primary father
    expected_father = min(c0.keys(), key=lambda pid: c0[pid]["t_min_k"])
    assert primary_parent[only_cid] == expected_father


def test_overshooting_top_detection_two_step() -> None:
    cfg = NwcSafRdtConfig()
    h, w = 48, 48
    yy, xx = np.mgrid[0:h, 0:w]
    area = np.full((h, w), 9.0, dtype=np.float32)  # 3 km pixel resolution

    # Mature cumulonimbus anvil with radial gradient (212..224 K) and a sharp
    # overshooting dome at (24, 24) reaching 204 K (<= T_tropo - 5 K and > 6 K colder than anvil at 2*L_OT)
    r = np.hypot(yy - 24.0, xx - 24.0)
    ir = np.full((h, w), 290.0, dtype=np.float32)
    anvil_mask = r <= 15.0
    ir[anvil_mask] = 211.0 + 1.0 * r[anvil_mask]  # 211 K near centre, ~224 K at r=13 (2*L_OT=40 km ~ 13 pix)
    ir[r <= 1.5] = 204.0  # OT core at 204 K (17..20 K colder than surrounding anvil rays)

    btd = np.full((h, w), -1.0, dtype=np.float32)
    btd[anvil_mask] = 0.5
    btd[r <= 2.0] = 2.5  # WV6.2 - IR10.8 > +1.5 K over the OT dome

    frame = _make_7ch_frame(ir, btd_wv62_ir108=btd)
    _, cells, _, _ = detect_rdt_cells(frame, area, cfg, tropopause_temp_k=215.0)
    assert len(cells) == 1
    cell = next(iter(cells.values()))

    ot_res = detect_cell_overshooting_tops(cell, frame, area, cfg, cell_lat_deg=45.0)
    assert ot_res["confirmed_count"] >= 1
    assert ot_res["confirmed_pixels"] >= 1
    assert bool(ot_res["confirmed_mask"].any()) is True


def test_full_pipeline_lifecycle_and_forecasts() -> None:
    cfg = NwcSafRdtConfig()
    h, w = 48, 48
    yy, xx = np.mgrid[0:h, 0:w]
    area = np.full((h, w), 16.0, dtype=np.float32)

    # Create a 5-slot (20 min) sequence of an initiating, rapidly cooling & expanding convective cell
    frames = []
    times = [
        np.datetime64("2020-06-07T12:00:00") + np.timedelta64(5 * t, "m")
        for t in range(5)
    ]
    t_mins = [252.0, 242.0, 231.0, 218.0, 208.0]
    radii = [4.0, 5.5, 7.0, 8.5, 10.0]
    btds = [-6.0, -3.0, -0.5, 1.2, 2.4]

    for t_idx in range(5):
        cy = 24.0
        cx = 16.0 + 1.5 * t_idx  # steady eastward motion
        r = np.hypot(yy - cy, xx - cx)
        ir = np.full((h, w), 290.0, dtype=np.float32)
        rad = radii[t_idx]
        mask = r <= rad
        ir[mask] = t_mins[t_idx] + (260.0 - t_mins[t_idx]) * (r[mask] / rad) ** 1.5
        # Add sharp OT core on last frame
        if t_idx == 4:
            ir[r <= 1.2] = 204.0
        btd_2d = np.full((h, w), btds[t_idx], dtype=np.float32)
        frames.append(_make_7ch_frame(ir, btd_wv62_ir108=btd_2d, wbtd_wv62_wv73=-8.0, btd_ir87_ir108=0.5))

    clip_tchw = np.stack(frames, axis=0)
    out = run_nwcsaf_rdt_sequence(
        clip_tchw, times, area, cfg, lat_deg=45.0, tropopause_temp_k=214.0
    )

    assert out["semantic_label"].shape == (5, h, w)
    # Track ID must be maintained across all 5 slots
    track_ids_per_slot = [
        set(np.unique(out["track_id_map"][t])) - {0} for t in range(5)
    ]
    assert all(len(s) == 1 for s in track_ids_per_slot)
    assert len(set.union(*track_ids_per_slot)) == 1

    # By slot 2-4 the rapidly cooling cell must be discriminated as convective (semantic class >= 2)
    assert int(out["semantic_label"][2].max()) >= 2
    assert int(out["semantic_label"][3].max()) >= 2
    # On slot 4 with confirmed OT, class 6 (Overshooting Top) must appear
    assert int(out["semantic_label"][4].max()) == 6
    # Lagrangian forecasts (+15, +30, +45, +60 min) must be generated for convective slots
    assert len(out["forecast_table"]) >= 4


def test_wmo_tropopause_computation() -> None:
    from rdt_nwcsaf import compute_wmo_tropopause

    levels = np.array([500, 400, 300, 250, 200, 150, 100, 70], dtype=np.float32)
    z_km = np.array([5.5, 7.2, 9.2, 10.4, 11.8, 13.6, 16.2, 18.4], dtype=np.float32)
    # Standard mid-latitude profile cooling at 6.5 K/km up to 11.8 km (200 hPa, 215 K), then isothermal/inversion
    t_1d = np.array([256.0, 245.0, 232.0, 224.0, 215.0, 215.5, 216.0, 217.0], dtype=np.float32)

    temp_3d = np.broadcast_to(t_1d[:, None, None], (len(levels), 4, 4)).copy()
    geopot_3d = np.broadcast_to((z_km * 9.80665 * 1000.0)[:, None, None], (len(levels), 4, 4)).copy()

    t_tropo, p_tropo, z_tropo = compute_wmo_tropopause(levels, temp_3d, geopot_3d)
    assert t_tropo.shape == (4, 4)
    assert np.all((t_tropo >= 213.0) & (t_tropo <= 218.0))
    assert np.all((p_tropo >= 170.0) & (p_tropo <= 230.0))
    assert np.all((z_tropo >= 11.0) & (z_tropo <= 13.0))


def test_nwcsaf_cloud_type_and_hrw_amv() -> None:
    from rdt_nwcsaf import compute_hrw_amv_guess_field, compute_nwcsaf_cloud_type

    h, w = 48, 48
    ir108 = np.full((h, w), 288.0, dtype=np.float32)
    ir120 = np.full((h, w), 287.0, dtype=np.float32)
    ir087 = np.full((h, w), 286.5, dtype=np.float32)
    wv062 = np.full((h, w), 236.0, dtype=np.float32)
    wv073 = np.full((h, w), 252.0, dtype=np.float32)

    # Region 1: Detached thin semitransparent cirrus (IR108=250K, large split-window BTD=4.0K, WV62-IR108=-14K)
    ir108[5:12, 5:12] = 250.0
    ir120[5:12, 5:12] = 246.0
    ir087[5:12, 5:12] = 251.0

    # Region 2: Deep opaque convective core (IR108=212K, small split-window BTD=0.3K, WV62-IR108=+1.0K)
    ir108[28:38, 28:38] = 212.0
    ir120[28:38, 28:38] = 211.7
    ir087[28:38, 28:38] = 212.5
    wv062[28:38, 28:38] = 213.0

    ct = compute_nwcsaf_cloud_type(
        ir108, ir120=ir120, ir087=ir087, wv062=wv062, wv073=wv073, t_tropo_k=215.0, t_surf_k=292.0
    )
    assert bool(ct.thin_cirrus_mask[8, 8]) is True
    assert bool(ct.opaque_cloud_mask[8, 8]) is False
    assert bool(ct.opaque_cloud_mask[32, 32]) is True
    assert int(ct.ct_class[32, 32]) == 4

    # Test HRW + NWP 700 hPa guess motion field
    u700 = np.full((h, w), 10.0, dtype=np.float32)  # 10 m/s eastward -> +1 px per 300s at 3 km
    v700 = np.full((h, w), 0.0, dtype=np.float32)
    dy_px, dx_px = compute_hrw_amv_guess_field(
        ir108, ir108, prev_wv062=wv062, curr_wv062=wv062, u700_ms=u700, v700_ms=v700
    )
    assert dy_px.shape == (h, w)
    assert dx_px.shape == (h, w)
    assert float(np.mean(dx_px)) > 0.2


def test_calibrated_rdt_model_and_seviri_surrogates() -> None:
    from rdt_nwcsaf import fit_calibrated_rdt_model, predict_seviri_only_nwp_surrogates

    rng = np.random.default_rng(42)
    records = []
    for vcat in range(6):
        for has_hist in (0, 1):
            for i in range(12):
                y = 1 if i >= 6 else 0
                base = 0.8 if y == 1 else -0.8
                feats = [1.0] + (base + 0.3 * rng.standard_normal(11)).tolist()
                records.append({"vertical_cat": vcat, "has_history": has_hist, "features": feats, "label": y})

    cal_model = fit_calibrated_rdt_model(records, [], [])
    assert len(cal_model.weights) == 12
    assert len(cal_model.thresholds) == 6

    ch = {
        "WV_062": np.full((16, 16), 232.0, dtype=np.float32),
        "WV_073": np.full((16, 16), 248.0, dtype=np.float32),
        "IR_087": np.full((16, 16), 282.0, dtype=np.float32),
        "IR_097": np.full((16, 16), 245.0, dtype=np.float32),
        "IR_108": np.full((16, 16), 285.0, dtype=np.float32),
        "IR_120": np.full((16, 16), 283.0, dtype=np.float32),
        "IR_134": np.full((16, 16), 260.0, dtype=np.float32),
    }
    t_tropo, nwpconv = predict_seviri_only_nwp_surrogates(cal_model, ch, lat_deg=45.0)
    assert t_tropo.shape == (16, 16)
    assert nwpconv.shape == (16, 16)
    assert np.all((t_tropo >= 198.0) & (t_tropo <= 228.0))

