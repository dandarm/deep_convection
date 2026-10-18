"""NWC SAF GEO-RDT-CW Cloud System Tracking & Trajectory Management.

Implements ATBD Section 3.2.1.2.2 (pp. 30-34, Figures 13-19) and ATBD v1.1 Section 3.1.5 (pp. 21-22):
1. Step 1: Forward advection of previous cells C by +V(C)*dt and overlap check ovl >= 0.15.
2. Step 2: Local IR10.8 cross-correlation motion estimation for unlinked cells C':
   - Accept cross-correlation vector if Pearson rho_max >= 0.5;
   - Otherwise fall back to nearest tracked neighbour velocity or 2D guess field.
3. Step 3: Backward advection of unlinked cell C' by -V_est(C')*dt and overlap check.
4. Step 4: 1-pixel morphological edge enlargement (3x3 dilation) of backward-advected C'
   for small cells (<= 5 pixels) and final overlap check.
5. Common temperature threshold surface S(T_common) rule for main father/child selection
   in merges and splits, and full trajectory reset on many-to-many complex links (v1.1 p. 22).
6. Temperature-weighted gravity centre family displacement and 0.5 temporal velocity smoothing.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy import ndimage, signal

from .config import NwcSafRdtConfig
from .detection import CONNECTIVITY_8, ThresholdSliceHierarchy


def common_threshold_area(
    cell: dict[str, Any],
    common_temp_k: float,
    hierarchy: ThresholdSliceHierarchy | None,
    cfg: NwcSafRdtConfig,
) -> float:
    """Return horizontal surface S(T_common) of cell at common_temp_k (Line H)."""
    key = int(round((common_temp_k - cfg.tcold_k) / cfg.deltat_k))
    profile = cell.get("vertical_profile", {})
    if key in profile:
        return float(profile[key])
    if hierarchy is not None:
        return hierarchy.area_containing_seed(common_temp_k, cell["seed_yx"])
    return float(cell["area_km2"])


def _precompute_ir_anomaly_sat(
    prev_ir: np.ndarray, twarm_k: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute cold cloud anomaly and 2D summed-area tables once per frame."""
    anom = np.maximum(
        twarm_k - np.nan_to_num(prev_ir, nan=twarm_k), 0.0
    ).astype(np.float64, copy=False)
    sat1 = np.pad(anom, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    sat2 = np.pad(anom * anom, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return anom, sat1, sat2


def estimate_local_correlation_motion(
    prev_ir: np.ndarray,
    curr_ir: np.ndarray,
    center_yx: np.ndarray,
    cfg: NwcSafRdtConfig,
    *,
    _precomp: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, float]:
    """Estimate displacement (dy, dx) in pixels from t to t+dt via local IR10.8 cross-correlation.

    Returns (displacement_yx_pixels, max_pearson_rho).
    ATBD Fig. 14 (p. 31): accept cross-correlation velocity when rho_max >= 0.5.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    h, w = curr_ir.shape
    cy, cx = int(round(float(center_yx[0]))), int(round(float(center_yx[1])))
    r_win = int(cfg.corr_window_pixels)
    r_shift = int(cfg.corr_max_shift_pixels)

    y0 = max(0, cy - r_win)
    y1 = min(h, cy + r_win + 1)
    x0 = max(0, cx - r_win)
    x1 = min(w, cx + r_win + 1)
    ph, pw = y1 - y0, x1 - x0
    if ph < 5 or pw < 5:
        return np.zeros(2, dtype=float), 0.0

    # Extract cold cloud anomaly below T_warm
    patch_curr = np.maximum(cfg.twarm_k - np.nan_to_num(curr_ir[y0:y1, x0:x1], nan=cfg.twarm_k), 0.0)
    patch_curr_zm = patch_curr - patch_curr.mean()
    norm_curr = float(np.linalg.norm(patch_curr_zm))
    if norm_curr < 1e-6:
        return np.zeros(2, dtype=float), 0.0

    # Valid shift ranges so that py0 = y0 - dy >= 0 and py1 = y1 - dy <= h
    # => y1 - h <= dy <= y0, intersected with [-r_shift, +r_shift]
    dy_min = max(-r_shift, y1 - h)
    dy_max = min(r_shift, y0)
    dx_min = max(-r_shift, x1 - w)
    dx_max = min(r_shift, x0)
    if dy_min > dy_max or dx_min > dx_max:
        return np.zeros(2, dtype=float), 0.0

    # Extract bounding region in prev_ir covering all valid shifts:
    # py0 ranges from (y0 - dy_max) to (y0 - dy_min)
    sy0 = y0 - dy_max
    sy1 = y1 - dy_min
    sx0 = x0 - dx_max
    sx1 = x1 - dx_min
    patch_zm64 = patch_curr_zm.astype(np.float64, copy=False)
    n_pix = float(ph * pw)

    if _precomp is not None:
        anom_full, sat1_full, sat2_full = _precomp
        search_prev = anom_full[sy0:sy1, sx0:sx1]
        s1 = sat1_full[sy0 : sy1 + 1, sx0 : sx1 + 1]
        s2 = sat2_full[sy0 : sy1 + 1, sx0 : sx1 + 1]
    else:
        search_prev = np.maximum(
            cfg.twarm_k - np.nan_to_num(prev_ir[sy0:sy1, sx0:sx1], nan=cfg.twarm_k),
            0.0,
        ).astype(np.float64, copy=False)
        s1 = np.pad(search_prev, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
        s2 = np.pad(search_prev * search_prev, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    win_sum = s1[ph:, pw:] - s1[:-ph, pw:] - s1[ph:, :-pw] + s1[:-ph, :-pw]
    win_sum2 = s2[ph:, pw:] - s2[:-ph, pw:] - s2[ph:, :-pw] + s2[:-ph, :-pw]
    var_prev = np.maximum(win_sum2 - (win_sum * win_sum) / n_pix, 0.0)
    norms_prev = np.sqrt(var_prev)
    valid_mask = norms_prev >= 1e-6
    if not np.any(valid_mask):
        return np.zeros(2, dtype=float), 0.0

    # Since patch_zm64 has zero mean, sum((w - mean(w)) * patch_zm64) == sum(w * patch_zm64)
    windows = sliding_window_view(search_prev, (ph, pw))
    dots = np.einsum("ijkl,kl->ij", windows, patch_zm64)
    rhos = np.full(norms_prev.shape, -2.0, dtype=float)
    rhos[valid_mask] = dots[valid_mask] / (norm_curr * norms_prev[valid_mask])

    best_rho = float(rhos.max())
    if best_rho < 0.0:
        return np.zeros(2, dtype=float), 0.0

    # Note: window index (iy, ix) corresponds to py0 = sy0 + iy = y0 - dy_max + iy => dy = dy_max - iy
    iy_cand, ix_cand = np.nonzero(np.abs(rhos - best_rho) <= 1e-9)
    dys = dy_max - iy_cand
    dxs = dx_max - ix_cand
    dist2 = dys * dys + dxs * dxs
    best_idx = int(np.argmin(dist2))
    best_dy = float(dys[best_idx])
    best_dx = float(dxs[best_idx])
    return np.array([best_dy, best_dx], dtype=float), best_rho


def _fallback_velocity_for_orphan(
    cell_curr: dict[str, Any],
    linked_curr_velocities: list[tuple[np.ndarray, np.ndarray]],
    previous_cells: dict[int, dict[str, Any]],
    guess_motion_yx: np.ndarray | None,
) -> np.ndarray:
    """Return fallback velocity (dy, dx) in pixels/slot from nearest tracked cell or 2D guess field."""
    pos = cell_curr["weight_centroid_yx"]
    # 1. Nearest already-linked cell in current slot
    if linked_curr_velocities:
        nearest_vel = min(
            linked_curr_velocities,
            key=lambda item: float(np.sum((item[0] - pos) ** 2)),
        )[1]
        return np.asarray(nearest_vel, dtype=float).copy()
    # 2. Nearest tracked cell from previous slot
    if previous_cells:
        nearest_prev = min(
            previous_cells.values(),
            key=lambda p: float(np.sum((p["weight_centroid_yx"] - pos) ** 2)),
        )
        if "motion_yx" in nearest_prev:
            return np.asarray(nearest_prev["motion_yx"], dtype=float).copy()
    # 3. Pre-calculated 2D guess motion field (HRW + NWP 700 hPa)
    if guess_motion_yx is not None:
        if guess_motion_yx.ndim == 3 and guess_motion_yx.shape[0] == 2:
            cy = int(np.clip(round(float(pos[0])), 0, guess_motion_yx.shape[1] - 1))
            cx = int(np.clip(round(float(pos[1])), 0, guess_motion_yx.shape[2] - 1))
            return np.asarray(guess_motion_yx[:, cy, cx], dtype=float)
        return np.asarray(guess_motion_yx, dtype=float).reshape(2)
    return np.zeros(2, dtype=float)


def _overlap_forward(
    prev_cell: dict[str, Any],
    shift_yx: np.ndarray,
    curr_labels: np.ndarray,
    curr_cells: dict[int, dict[str, Any]],
    area_km2: np.ndarray,
    cfg: NwcSafRdtConfig,
) -> list[tuple[int, float, float]]:
    """Translate prev_cell forward by shift_yx and return [(curr_id, ovl, inter_km2)]."""
    h, w = curr_labels.shape
    dy, dx = int(round(float(shift_yx[0]))), int(round(float(shift_yx[1])))
    ys, xs = prev_cell["bbox"]
    y0, x0 = ys.start + dy, xs.start + dx
    y1, x1 = ys.stop + dy, xs.stop + dx
    cy0, cx0 = max(0, y0), max(0, x0)
    cy1, cx1 = min(h, y1), min(w, x1)
    if cy0 >= cy1 or cx0 >= cx1:
        return []

    sub_mask = prev_cell["mask"][cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
    sub_labels = curr_labels[cy0:cy1, cx0:cx1]
    sub_area = area_km2[cy0:cy1, cx0:cx1]
    valid = sub_mask & np.isfinite(sub_area)
    if not valid.any():
        return []

    valid_labels = sub_labels[valid]
    max_label = int(valid_labels.max())
    if max_label == 0:
        return []
    adv_area = float(sub_area[valid].sum())
    inter_sums = np.bincount(
        valid_labels, weights=sub_area[valid], minlength=max_label + 1
    )
    matches: list[tuple[int, float, float]] = []
    for cid in np.flatnonzero(inter_sums[1:]) + 1:
        cid_int = int(cid)
        inter_km2 = float(inter_sums[cid_int])
        denom = min(adv_area, float(curr_cells[cid_int]["area_km2"]))
        ovl = inter_km2 / denom if denom > 0.0 else 0.0
        if ovl >= cfg.ovl_min:
            matches.append((cid_int, ovl, inter_km2))
    return matches


def _overlap_backward(
    curr_cell: dict[str, Any],
    shift_yx: np.ndarray,
    prev_labels: np.ndarray,
    prev_cells: dict[int, dict[str, Any]],
    area_km2: np.ndarray,
    cfg: NwcSafRdtConfig,
    *,
    dilate_pixels: int = 0,
) -> list[tuple[int, float, float]]:
    """Translate curr_cell backward by -shift_yx (optionally dilated) and check overlap with prev_labels."""
    h, w = prev_labels.shape
    dy, dx = -int(round(float(shift_yx[0]))), -int(round(float(shift_yx[1])))
    ys, xs = curr_cell["bbox"]
    mask = curr_cell["mask"]
    pad = int(dilate_pixels)
    if pad > 0:
        padded = np.pad(mask, pad_width=pad, mode="constant", constant_values=False)
        mask = ndimage.binary_dilation(
            padded, structure=CONNECTIVITY_8, iterations=pad
        )
        y0, x0 = ys.start - pad + dy, xs.start - pad + dx
        y1, x1 = ys.stop + pad + dy, xs.stop + pad + dx
    else:
        y0, x0 = ys.start + dy, xs.start + dx
        y1, x1 = ys.stop + dy, xs.stop + dx

    cy0, cx0 = max(0, y0), max(0, x0)
    cy1, cx1 = min(h, y1), min(w, x1)
    if cy0 >= cy1 or cx0 >= cx1:
        return []

    sub_mask = mask[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
    sub_labels = prev_labels[cy0:cy1, cx0:cx1]
    sub_area = area_km2[cy0:cy1, cx0:cx1]
    valid = sub_mask & np.isfinite(sub_area)
    if not valid.any():
        return []

    valid_labels = sub_labels[valid]
    max_label = int(valid_labels.max())
    if max_label == 0:
        return []
    back_area = float(sub_area[valid].sum())
    inter_sums = np.bincount(
        valid_labels, weights=sub_area[valid], minlength=max_label + 1
    )
    matches: list[tuple[int, float, float]] = []
    for pid in np.flatnonzero(inter_sums[1:]) + 1:
        pid_int = int(pid)
        inter_km2 = float(inter_sums[pid_int])
        denom = min(back_area, float(prev_cells[pid_int]["area_km2"]))
        ovl = inter_km2 / denom if denom > 0.0 else 0.0
        if ovl >= cfg.ovl_min:
            matches.append((pid_int, ovl, inter_km2))
    return matches


def track_rdt_cells(
    prev_cells: dict[int, dict[str, Any]],
    prev_labels: np.ndarray | None,
    prev_hierarchy: ThresholdSliceHierarchy | None,
     prev_ir108: np.ndarray | None,
    curr_cells: dict[int, dict[str, Any]],
    curr_labels: np.ndarray,
    curr_hierarchy: ThresholdSliceHierarchy,
    curr_ir108: np.ndarray,
    area_km2: np.ndarray,
    cfg: NwcSafRdtConfig,
    *,
    guess_motion_yx: np.ndarray | None = None,
) -> tuple[
    dict[int, list[int]],
    dict[int, list[int]],
    dict[tuple[int, int], dict[str, Any]],
    dict[int, int],
    dict[int, str],
    dict[int, int | None],
]:
    """Perform NWC SAF RDT-CW 4-step tracking and primary/convective father assignment.

    Returns:
        parents: {curr_id: [prev_id, ...]}
        children: {prev_id: [curr_id, ...]}
        edges: {(prev_id, curr_id): {'aor': float, 'intersection_km2': float, 'method': str}}
        primary_parent: {curr_id: prev_id} (main 'father' trajectory link)
        events: {curr_id: 'new' | 'continuation' | 'merge_main' | 'split_main' | 'split_new' | 'complex'}
        convective_parent: {curr_id: prev_id | None} (any convective parent for inheritance)
    """
    parents: dict[int, list[int]] = {cid: [] for cid in curr_cells}
    children: dict[int, list[int]] = {pid: [] for pid in prev_cells}
    edges: dict[tuple[int, int], dict[str, Any]] = {}

    _precomp_sat = (
        _precompute_ir_anomaly_sat(prev_ir108, cfg.twarm_k)
        if prev_ir108 is not None
        else None
    )

    if not prev_cells or prev_labels is None or not curr_cells:
        events = {cid: "new" for cid in curr_cells}
        for cid, cell in curr_cells.items():
            if prev_ir108 is not None:
                corr_v, rho = estimate_local_correlation_motion(
                    prev_ir108,
                    curr_ir108,
                    cell["weight_centroid_yx"],
                    cfg,
                    _precomp=_precomp_sat,
                )
                cell["provisional_motion_yx"] = (
                    corr_v
                    if rho >= cfg.corr_min_rho
                    else _fallback_velocity_for_orphan(cell, [], prev_cells, guess_motion_yx)
                )
            else:
                cell["provisional_motion_yx"] = _fallback_velocity_for_orphan(
                    cell, [], prev_cells, guess_motion_yx
                )
        return parents, children, edges, {}, events, {cid: None for cid in curr_cells}

    # --- Step 1: Forward advection of previous cells C by +V(C)*dt (ATBD Fig. 14) ---
    for pid, pcell in prev_cells.items():
        vel_yx = np.asarray(pcell.get("motion_yx", np.zeros(2)), dtype=float)
        matches = _overlap_forward(pcell, vel_yx, curr_labels, curr_cells, area_km2, cfg)
        for cid, ovl, inter_km2 in matches:
            parents[cid].append(pid)
            children[pid].append(cid)
            edges[(pid, cid)] = {
                "aor": ovl,
                "intersection_km2": inter_km2,
                "method": "forward_advection",
            }

    # Provisional velocities of cells already linked in Step 1 (for neighbour fallback)
    step1_velocities: list[tuple[np.ndarray, np.ndarray]] = []
    for cid, ccell in curr_cells.items():
        if parents[cid]:
            p0 = prev_cells[parents[cid][0]]
            disp = ccell["weight_centroid_yx"] - p0["weight_centroid_yx"]
            step1_velocities.append((ccell["weight_centroid_yx"], disp))

    # --- Steps 2, 3 & 4 for unlinked current cells C' (ATBD Figs. 14, 15, 16) ---
    for cid, ccell in curr_cells.items():
        if parents[cid]:
            continue

        # Step 2: Estimate velocity via cross-correlation (if rho >= 0.5) else neighbour/guess
        if prev_ir108 is not None:
            corr_vel, rho_max = estimate_local_correlation_motion(
                prev_ir108,
                curr_ir108,
                ccell["weight_centroid_yx"],
                cfg,
                _precomp=_precomp_sat,
            )
        else:
            corr_vel, rho_max = np.zeros(2, dtype=float), 0.0

        if rho_max >= cfg.corr_min_rho:
            v_est = corr_vel
            est_source = "cross_correlation"
        else:
            v_est = _fallback_velocity_for_orphan(
                ccell, step1_velocities, prev_cells, guess_motion_yx
            )
            est_source = "neighbour_or_guess"

        ccell["provisional_motion_yx"] = v_est
        ccell["provisional_motion_source"] = est_source
        ccell["corr_rho_max"] = float(rho_max)

        # Step 3: Backward advection of C' by -V_est * dt
        back_matches = _overlap_backward(
            ccell, v_est, prev_labels, prev_cells, area_km2, cfg, dilate_pixels=0
        )
        if back_matches:
            for pid, ovl, inter_km2 in back_matches:
                if pid not in parents[cid]:
                    parents[cid].append(pid)
                    children[pid].append(cid)
                    edges[(pid, cid)] = {
                        "aor": ovl,
                        "intersection_km2": inter_km2,
                        "method": f"backward_{est_source}",
                    }
            continue

        # Step 4: 1-pixel edge enlargement for small cells (<= 5 pixels, ATBD Fig. 16, p. 32)
        if (
            ccell["pixel_count"] <= cfg.small_cell_dilation_max_pixels
            and cfg.small_cell_dilation_iter > 0
        ):
            dilated_matches = _overlap_backward(
                ccell,
                v_est,
                prev_labels,
                prev_cells,
                area_km2,
                cfg,
                dilate_pixels=cfg.small_cell_dilation_iter,
            )
            for pid, ovl, inter_km2 in dilated_matches:
                if pid not in parents[cid]:
                    parents[cid].append(pid)
                    children[pid].append(cid)
                    edges[(pid, cid)] = {
                        "aor": ovl,
                        "intersection_km2": inter_km2,
                        "method": "backward_enlarged_small_cell",
                    }

    # --- Bipartite Family Decomposition & Common-Threshold Main Father Selection ---
    primary_parent: dict[int, int] = {}
    events: dict[int, str] = {}
    convective_parent: dict[int, int | None] = {cid: None for cid in curr_cells}
    visited_curr: set[int] = set()

    for cid in curr_cells:
        if cid in visited_curr:
            continue
        fam_prev: set[int] = set()
        fam_curr: set[int] = {cid}
        queue = [cid]
        while queue:
            curr_node = queue.pop()
            for pid in parents[curr_node]:
                if pid not in fam_prev:
                    fam_prev.add(pid)
                    for child_node in children[pid]:
                        if child_node not in fam_curr:
                            fam_curr.add(child_node)
                            queue.append(child_node)
        visited_curr.update(fam_curr)

        if not fam_prev:
            events[cid] = "new"
            continue

        # Common (colder) threshold temperature across the family (ATBD Sec 3.2.1.2.3.4, p. 43)
        # Note: in ATBD, each cell has pixels T <= T_BT, so the threshold present inside both
        # cells' vertical profiles below their respective bases is min(T_BT). If min(T_BT) is
        # colder than one cell's T_min, max(T_BT) is the warmest shared contour level.
        min_t_bt = min(
            [curr_cells[c]["t_bt_k"] for c in fam_curr]
            + [prev_cells[p]["t_bt_k"] for p in fam_prev]
        )
        max_t_bt = max(
            [curr_cells[c]["t_bt_k"] for c in fam_curr]
            + [prev_cells[p]["t_bt_k"] for p in fam_prev]
        )

        if len(fam_prev) == 1 and len(fam_curr) == 1:
            p_only = next(iter(fam_prev))
            c_only = next(iter(fam_curr))
            primary_parent[c_only] = p_only
            events[c_only] = "continuation"
            if prev_cells[p_only].get("is_convective", False):
                convective_parent[c_only] = p_only

        elif len(fam_curr) == 1:
            # Merge: K >= 2 parents -> 1 child C'. Keep trajectory of largest parent at common threshold
            c_only = next(iter(fam_curr))

            def _parent_rank(pid: int) -> tuple[float, float, float, int]:
                s_min_bt = common_threshold_area(prev_cells[pid], min_t_bt, prev_hierarchy, cfg)
                s_max_bt = common_threshold_area(prev_cells[pid], max_t_bt, prev_hierarchy, cfg)
                s_comm = s_min_bt if s_min_bt > 0.0 else s_max_bt
                return (
                    s_comm,
                    float(edges[(pid, c_only)]["intersection_km2"]),
                    float(prev_cells[pid]["area_km2"]),
                    -pid,
                )

            main_p = max(fam_prev, key=_parent_rank)
            primary_parent[c_only] = main_p
            events[c_only] = "merge_main"
            # Convective inheritance: prefer primary parent if convective, else any secondary convective parent
            if prev_cells[main_p].get("is_convective", False):
                convective_parent[c_only] = main_p
            else:
                conv_ps = [
                    pid for pid in sorted(fam_prev) if prev_cells[pid].get("is_convective", False)
                ]
                if conv_ps:
                    convective_parent[c_only] = conv_ps[0]

        elif len(fam_prev) == 1:
            # Split: 1 parent P -> M >= 2 children. Largest child at common threshold carries trajectory
            p_only = next(iter(fam_prev))

            def _child_rank(ccid: int) -> tuple[float, float, float, int]:
                s_min_bt = common_threshold_area(curr_cells[ccid], min_t_bt, curr_hierarchy, cfg)
                s_max_bt = common_threshold_area(curr_cells[ccid], max_t_bt, curr_hierarchy, cfg)
                s_comm = s_min_bt if s_min_bt > 0.0 else s_max_bt
                return (
                    s_comm,
                    float(edges[(p_only, ccid)]["intersection_km2"]),
                    float(curr_cells[ccid]["area_km2"]),
                    -ccid,
                )

            main_c = max(fam_curr, key=_child_rank)
            primary_parent[main_c] = p_only
            for ccid in fam_curr:
                events[ccid] = "split_main" if ccid == main_c else "split_new"
                if prev_cells[p_only].get("is_convective", False):
                    convective_parent[ccid] = p_only

        else:
            # Complex many-to-many (K >= 2 and M >= 2, ATBD v1.1 Sec 3.1.5, p. 22):
            # All incoming trajectories are closed ('c') and current cells start new trajectories.
            for ccid in fam_curr:
                events[ccid] = "complex"
                # Check if any direct parent of ccid was convective
                conv_ps = [
                    pid
                    for pid in sorted(parents[ccid])
                    if prev_cells[pid].get("is_convective", False)
                ]
                if conv_ps:
                    convective_parent[ccid] = conv_ps[0]

        # Compute family-weighted velocity & apply 0.5 exponential smoothing (ATBD Fig. 14, p. 31)
        prev_list = sorted(fam_prev)
        curr_list = sorted(fam_curr)
        w_prev = np.array([prev_cells[p]["area_km2"] for p in prev_list], dtype=float)
        w_curr = np.array([curr_cells[c]["area_km2"] for c in curr_list], dtype=float)
        g_prev = np.average(
            [prev_cells[p]["weight_centroid_yx"] for p in prev_list], axis=0, weights=w_prev
        )
        g_curr = np.average(
            [curr_cells[c]["weight_centroid_yx"] for c in curr_list], axis=0, weights=w_curr
        )
        inst_disp_yx = g_curr - g_prev

        prev_vels = [
            np.asarray(prev_cells[p].get("motion_yx", np.zeros(2)), dtype=float)
            for p in prev_list
            if np.any(prev_cells[p].get("motion_yx", np.zeros(2)) != 0.0)
        ]
        if prev_vels:
            prev_vel_weights = [
                prev_cells[p]["area_km2"]
                for p in prev_list
                if np.any(prev_cells[p].get("motion_yx", np.zeros(2)) != 0.0)
            ]
            v_prev_mean = np.average(prev_vels, axis=0, weights=prev_vel_weights)
            alpha = cfg.velocity_smoothing_alpha
            smoothed_vel_yx = alpha * inst_disp_yx + (1.0 - alpha) * v_prev_mean
        else:
            smoothed_vel_yx = inst_disp_yx

        if np.linalg.norm(smoothed_vel_yx) > cfg.corr_max_shift_pixels * 1.5:
            smoothed_vel_yx = np.zeros(2, dtype=float)

        for ccid in fam_curr:
            curr_cells[ccid]["family_inst_disp_yx"] = inst_disp_yx
            curr_cells[ccid]["motion_yx"] = smoothed_vel_yx

    # Set initial motion for newly born cells
    for cid, ccell in curr_cells.items():
        if "motion_yx" not in ccell:
            ccell["motion_yx"] = np.asarray(
                ccell.get("provisional_motion_yx", np.zeros(2)), dtype=float
            )
            ccell["family_inst_disp_yx"] = ccell["motion_yx"].copy()

    return parents, children, edges, primary_parent, events, convective_parent
