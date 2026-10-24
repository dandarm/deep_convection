"""NWC SAF GEO-RDT-CW Overshooting Top Detection (OTD) Algorithm.

Implements ATBD Section 3.2.1.2.5.8 (pp. 49-50, Figure 27), ATBD v1.1 Section 3.2.1.2.5.7 (p. 47),
and Annex Line O (p. 61):
1. Step 1 (Pre-selection of OT candidates):
   - Cell eligibility: T_min < COLD_Threshold (-50 C mid-lat, -70 C tropics) and max(WV6.2 - IR10.8) > 0 C.
   - Candidate extrema: primary 'IRMin' and secondary 'BTDMax' / 'WBTDMax' separated by >= D_sep
     (50 km mid-lat, 200 km tropics).
   - 16-pixel exploration along 8 compass directions at 2*L_OT and 4*L_OT inside cell C:
     require at least 2 surrounding anvil pixels warmer than p_OT by > 6.0 C (ATBD v1.1 p. 47).
   - Horizontal OT extension inside the 2*L_OT x 2*L_OT window: pixels in C with BT(p) - BT(p_OT) < 3.0 C.
2. Step 2 (Confirmation of OT candidates):
   - Confirmed if Condition A: BT_10.8(p_OT) <= T_tropo - 5.0 C (when NWP T_tropo available)
   - OR Condition B: (BT_10.8(p_OT) < T_tropo or T_tropo unavailable) AND >= 2 of:
     (1) WV6.2 - IR10.8 > +1.5 C
     (2) VIS06/HRV reflectance > 70% (daytime when VIS available)
     (3) BT_10.8(p_OT) <= T_mean(cell) - 6.0 C
"""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy import ndimage

from .config import NwcSafRdtConfig
from .detection import CONNECTIVITY_8


# 8 compass unit vectors: N, NE, E, SE, S, SW, W, NW in (dy, dx) north-up/west-left image coordinates
COMPASS_8_DIRS = (
    (-1.0, 0.0),
    (-0.70710678, 0.70710678),
    (0.0, 1.0),
    (0.70710678, 0.70710678),
    (1.0, 0.0),
    (0.70710678, -0.70710678),
    (0.0, -1.0),
    (-0.70710678, -0.70710678),
)


def _estimate_satellite_anvil_tropo_k(
    local_ir: np.ndarray,
    local_mask: np.ndarray,
) -> float:
    """Estimate anvil equilibrium reference temperature when external NWP T_tropo is absent."""
    vals = local_ir[local_mask]
    if len(vals) < 10:
        return np.nan
    q15, q75 = np.quantile(vals, [0.15, 0.75])
    anvil_band = vals[(vals >= q15) & (vals <= q75)]
    if len(anvil_band) == 0:
        return float(np.median(vals))
    return float(np.median(anvil_band))


def detect_cell_overshooting_tops(
    cell: dict[str, Any],
    frame_7ch: np.ndarray,
    area_km2: np.ndarray,
    cfg: NwcSafRdtConfig,
    *,
    cell_lat_deg: float = 45.0,
    vis_refl_2d: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run NWC SAF two-step OTD on a single detected RDT-CW cell.

    Returns dict with:
        candidate_mask: bool array over cell['bbox']
        confirmed_mask: bool array over cell['bbox']
        candidates: list of dicts (Line O records) for each pre-selected candidate
        confirmed_count: number of confirmed OT cores in the cell
        confirmed_pixels: total pixels in confirmed OT extensions
    """
    bbox = cell["bbox"]
    local_mask = cell["mask"]
    local_ir = frame_7ch[4][bbox]
    local_wv62 = frame_7ch[0][bbox]
    local_wv73 = frame_7ch[1][bbox]
    local_area = area_km2[bbox]
    local_btd = local_wv62 - local_ir
    local_wbtd = local_wv62 - local_wv73

    empty_mask = np.zeros(local_mask.shape, dtype=bool)
    is_tropics = abs(float(cell_lat_deg)) <= 30.0
    cold_thresh_k = (
        cfg.ot_cold_threshold_tropics_k if is_tropics else cfg.ot_cold_threshold_midlat_k
    )
    l_ot_km = cfg.ot_size_tropics_km if is_tropics else cfg.ot_size_midlat_km
    d_sep_km = cfg.ot_sep_tropics_km if is_tropics else cfg.ot_sep_midlat_km

    # Step 1.1: Cell-level eligibility pre-conditions (ATBD p. 49)
    if (
        cell["t_min_k"] >= cold_thresh_k
        or cell["btd_wv62_ir108_max_k"] <= cfg.ot_min_cell_btd_wv_ir_k
    ):
        return {
            "candidate_mask": empty_mask,
            "confirmed_mask": empty_mask.copy(),
            "candidates": [],
            "confirmed_count": 0,
            "confirmed_pixels": 0,
        }

    mean_pix_km = float(np.sqrt(max(float(local_area[local_mask].mean()), 1.0)))

    # Build ordered list of candidate pixels in local bbox coordinates:
    # 1. Primary candidate: coldest IR10.8 pixel ("IRMin")
    # 2. Secondary candidates: local maxima of BTD ("BTDMax") and WBTD ("WBTDMax")
    raw_seeds: list[tuple[int, int, str]] = []
    ir_masked = np.where(local_mask, local_ir, np.inf)
    iy_min, ix_min = np.unravel_index(int(np.argmin(ir_masked)), local_ir.shape)
    raw_seeds.append((int(iy_min), int(ix_min), "IRMin"))

    # Local maxima of BTD(WV6.2 - IR10.8) > 0 and IR10.8 < cold_thresh_k
    btd_masked = np.where(local_mask & (local_ir < cold_thresh_k), local_btd, -np.inf)
    if np.isfinite(btd_masked).any():
        by_max, bx_max = np.unravel_index(int(np.argmax(btd_masked)), local_btd.shape)
        if btd_masked[by_max, bx_max] > cfg.ot_min_cell_btd_wv_ir_k:
            raw_seeds.append((int(by_max), int(bx_max), "BTDMax"))

    wbtd_masked = np.where(local_mask & (local_ir < cold_thresh_k), local_wbtd, -np.inf)
    if np.isfinite(wbtd_masked).any():
        wy_max, wx_max = np.unravel_index(int(np.argmax(wbtd_masked)), local_wbtd.shape)
        if local_btd[wy_max, wx_max] > cfg.ot_min_cell_btd_wv_ir_k:
            raw_seeds.append((int(wy_max), int(wx_max), "WBTDMax"))

    accepted_centers_yx: list[tuple[int, int]] = []
    candidate_records: list[dict[str, Any]] = []
    candidate_mask = np.zeros(local_mask.shape, dtype=bool)
    confirmed_mask = np.zeros(local_mask.shape, dtype=bool)

    nwp_tropo_k = float(cell.get("tropopause_temp_k", np.nan))
    has_nwp_tropo = bool(np.isfinite(nwp_tropo_k))
    anvil_ref_k = (
        nwp_tropo_k
        if has_nwp_tropo
        else _estimate_satellite_anvil_tropo_k(local_ir, local_mask)
    )

    h_loc, w_loc = local_mask.shape
    for sy, sx, source_type in raw_seeds:
        if not local_mask[sy, sx]:
            continue
        t_ot = float(local_ir[sy, sx])
        if t_ot >= cold_thresh_k:
            continue

        # Enforce minimum separation distance D_sep (50 km mid-lat, 200 km tropics)
        too_close = False
        for py, px in accepted_centers_yx:
            dist_km = np.hypot(sy - py, sx - px) * mean_pix_km
            if dist_km < d_sep_km:
                too_close = True
                break
        if too_close:
            continue

        # Step 1.3: Explore 16 surrounding pixels along 8 compass directions at 2*L_OT and 4*L_OT
        # restricted to pixels belonging to the same cloud cell C (ATBD Fig. 27 Right, p. 50)
        dist_1_pix = max(1.0, (2.0 * l_ot_km) / mean_pix_km)
        dist_2_pix = max(2.0, (4.0 * l_ot_km) / mean_pix_km)

        n_explored_in_cell = 0
        n_warm_surr = 0
        warm_diffs: list[float] = []

        for u_y, u_x in COMPASS_8_DIRS:
            ray_sampled = False
            for d_pix in (dist_1_pix, dist_2_pix):
                ry = int(round(sy + u_y * d_pix))
                rx = int(round(sx + u_x * d_pix))
                if 0 <= ry < h_loc and 0 <= rx < w_loc and local_mask[ry, rx]:
                    n_explored_in_cell += 1
                    ray_sampled = True
                    diff_k = float(local_ir[ry, rx] - t_ot)
                    if diff_k > cfg.ot_anvil_prominence_k:
                        n_warm_surr += 1
                        warm_diffs.append(diff_k)
            # If the cell is compact (< 2*L_OT radius along this ray), sample the furthest
            # in-cell pixel along this compass ray (at >= 0.5*L_OT) so compact convective
            # cells still test their surrounding anvil rim within C.
            if not ray_sampled:
                min_r = max(1, int(round(0.35 * l_ot_km / mean_pix_km)))
                max_r = int(round(dist_1_pix))
                furthest_in_cell: tuple[int, int] | None = None
                for step_r in range(min_r, max_r + 1):
                    ry = int(round(sy + u_y * step_r))
                    rx = int(round(sx + u_x * step_r))
                    if 0 <= ry < h_loc and 0 <= rx < w_loc and local_mask[ry, rx]:
                        furthest_in_cell = (ry, rx)
                if furthest_in_cell is not None:
                    n_explored_in_cell += 1
                    diff_k = float(local_ir[furthest_in_cell] - t_ot)
                    if diff_k > cfg.ot_anvil_prominence_k:
                        n_warm_surr += 1
                        warm_diffs.append(diff_k)

        if n_warm_surr < cfg.ot_min_warm_surrounding:
            continue

        # Step 1.4: Horizontal extension of the OT inside a 2*L_OT x 2*L_OT square window
        # centred on p_OT where BT_10.8(p) - BT_10.8(p_OT) < 3.0 C (ATBD p. 49)
        half_win_pix = max(1, int(round(l_ot_km / mean_pix_km)))
        wy0, wy1 = max(0, sy - half_win_pix), min(h_loc, sy + half_win_pix + 1)
        wx0, wx1 = max(0, sx - half_win_pix), min(w_loc, sx + half_win_pix + 1)

        win_cand = np.zeros_like(local_mask, dtype=bool)
        win_cand[wy0:wy1, wx0:wx1] = (
            local_mask[wy0:wy1, wx0:wx1]
            & ((local_ir[wy0:wy1, wx0:wx1] - t_ot) < cfg.ot_extension_delta_k)
        )
        comp_ot, _ = ndimage.label(win_cand, structure=CONNECTIVITY_8)
        ot_part = int(comp_ot[sy, sx])
        ot_ext_mask = (comp_ot == ot_part) if ot_part > 0 else win_cand
        ot_ext_mask[sy, sx] = True

        accepted_centers_yx.append((sy, sx))
        candidate_mask |= ot_ext_mask

        # Step 2: Confirmation of OT candidate (ATBD p. 50)
        btd_ot = float(max(local_btd[sy, sx], local_btd[ot_ext_mask].max()))
        wbtd_ot = float(max(local_wbtd[sy, sx], local_wbtd[ot_ext_mask].max()))
        vis_ot = (
            float(vis_refl_2d[bbox][ot_ext_mask].max())
            if vis_refl_2d is not None
            else np.nan
        )

        # Condition A: NWP tropopause available and BT_10.8(p_OT) <= T_tropo - 5.0 C
        cond_a = has_nwp_tropo and (t_ot <= nwp_tropo_k - cfg.ot_confirm_tropo_strong_k)

        # Condition B: (BT_10.8(p_OT) < T_tropo or T_tropo unavailable) AND >= 2 of 3 criteria
        tropo_gate_b = (not has_nwp_tropo) or (t_ot < nwp_tropo_k)
        # When NWP T_tropo is unavailable, also check that t_ot is colder than the satellite anvil median
        if not has_nwp_tropo and np.isfinite(anvil_ref_k):
            tropo_gate_b = t_ot < anvil_ref_k

        crit_b1 = btd_ot > cfg.ot_confirm_btd_wv_ir_k  # WV6.2 - IR10.8 > +1.5 C
        crit_b2 = bool(np.isfinite(vis_ot) and vis_ot > cfg.ot_confirm_vis_refl)  # VIS > 70%
        crit_b3 = t_ot <= (cell["t_mean_k"] - cfg.ot_confirm_mean_diff_k)  # BT_OT <= T_mean - 6 C
        cond_b = tropo_gate_b and ((int(crit_b1) + int(crit_b2) + int(crit_b3)) >= 2)

        is_confirmed = bool(cond_a or cond_b)
        if is_confirmed:
            confirmed_mask |= ot_ext_mask

        candidate_records.append(
            {
                "source_type": source_type,
                "local_y": sy,
                "local_x": sx,
                "global_y": sy + bbox[0].start,
                "global_x": sx + bbox[1].start,
                "t_ot_k": t_ot,
                "btd_wv62_ir108_k": btd_ot,
                "wbtd_wv62_wv73_k": wbtd_ot,
                "n_explored_in_cell": n_explored_in_cell,
                "n_warm_surrounding": n_warm_surr,
                "mean_warm_diff_k": float(np.mean(warm_diffs)) if warm_diffs else 0.0,
                "ot_pixels": int(ot_ext_mask.sum()),
                "ot_area_km2": float(local_area[ot_ext_mask].sum()),
                "cond_a_strong_tropo": cond_a,
                "cond_b_multispectral": cond_b,
                "confirmed": is_confirmed,
            }
        )

    confirmed_records = [r for r in candidate_records if r["confirmed"]]
    return {
        "candidate_mask": candidate_mask,
        "confirmed_mask": confirmed_mask,
        "candidates": candidate_records,
        "confirmed_count": len(confirmed_records),
        "confirmed_pixels": int(confirmed_mask.sum()),
    }
