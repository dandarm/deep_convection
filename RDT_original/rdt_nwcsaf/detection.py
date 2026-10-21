"""NWC SAF GEO-RDT-CW Cloud System Detection & Multi-Thresholding Cell Identification.

Implements ATBD Section 3.2.1.2.1 (pp. 26-30) and Annex Lines S, S2, H, X:
1. Stretched pixel masking (> 5 * nominal_area) and optional Cloud Type (CT/CMa) filtering.
2. Adaptive multi-thresholding from T_warm (+5 C) to T_cold (-75 C) in 1 K steps.
3. Thermal tower identification (DeltaT_tower >= 6 K) and separation of sibling towers
   at the warmest isolating threshold (T_merge - DeltaT).
4. Mature system Base of Tower limitation (LIMITTROPO & BTLIMIT = -60 C).
5. Top of Tower (ST) contour and single-tower excessive base refinement:
   - Case A (Budding from flat cloud deck): min(S_{n+1}/S_n) <= 0.7
   - Case B (Flat pyramidal single tower): S_BT <= max_base_top_ratio * S_ST
6. Extraction of vertical area profile (Line H), weighted/geometric gravity centres,
   thermal volume, moment ellipse, inner/peripheral gradients, and multi-spectral predictors.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy import ndimage

from .config import NwcSafRdtConfig


CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


class ThresholdSliceHierarchy:
    """Caches 8-connected components across discrete IR10.8 temperature thresholds."""

    def __init__(
        self,
        ir108: np.ndarray,
        area_km2: np.ndarray,
        valid_mask: np.ndarray,
        cfg: NwcSafRdtConfig,
    ) -> None:
        self.ir = ir108
        self.area = area_km2
        self.valid = valid_mask
        self.cfg = cfg
        self._cache: dict[
            int,
            tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[slice, slice] | None] | None],
        ] = {}
        self._qual_cache: dict[
            int,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}

        # Precompute 3x3 regional minima of IR108 <= Twarm sorted from warmest to coldest.
        # Every connected component at any threshold T <= Twarm attains its exact minimum at
        # one of these regional minima, allowing O(N_min) vectorized minimum & seed lookup.
        ir_v = np.where(valid_mask, ir108, np.inf)
        cloudy_mask = valid_mask & (ir108 <= cfg.twarm_k)
        self._cloudy_idx = np.flatnonzero(cloudy_mask)
        self._cloudy_area = area_km2.ravel()[self._cloudy_idx].astype(float)
        lmin_mask = (
            cloudy_mask
            & (ir_v == ndimage.minimum_filter(ir_v, size=3, mode="constant", cval=np.inf))
        )
        lmin_idx = np.flatnonzero(lmin_mask)
        lmin_vals = ir108.ravel()[lmin_idx].astype(float)
        order_desc = np.argsort(-lmin_vals, kind="mergesort")
        self._lmin_idx_desc = lmin_idx[order_desc]
        self._lmin_vals_desc = lmin_vals[order_desc]

    def _key(self, threshold_k: float) -> int:
        return int(round((threshold_k - self.cfg.tcold_k) / self.cfg.deltat_k))

    def threshold_from_key(self, key: int) -> float:
        return float(self.cfg.tcold_k + key * self.cfg.deltat_k)

    def _ensure_level(
        self, key: int, need_bboxes: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[slice, slice] | None] | None]:
        if key not in self._cache:
            thresh = self.threshold_from_key(key)
            binary = self.valid & (self.ir <= thresh)
            labels, num = ndimage.label(binary, structure=CONNECTIVITY_8)
            if num == 0:
                self._cache[key] = (
                    labels,
                    np.zeros(1, dtype=float),
                    np.array([np.inf], dtype=float),
                    [None],
                )
                self._qual_cache[key] = (
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.int64),
                    labels,
                    np.zeros(1, dtype=float),
                    np.array([np.inf], dtype=float),
                )
            else:
                idx_bin = np.flatnonzero(binary)
                areas = np.bincount(
                    labels.ravel()[idx_bin], weights=self.area.ravel()[idx_bin], minlength=num + 1
                ).astype(float)
                areas[0] = 0.0
                minima = np.full(num + 1, np.inf, dtype=float)
                seeds = np.zeros(num + 1, dtype=np.int64)

                mask_sub = self._lmin_vals_desc <= thresh
                l_sub = self._lmin_idx_desc[mask_sub]
                v_sub = self._lmin_vals_desc[mask_sub]
                l_ids = labels.ravel()[l_sub]
                minima[l_ids] = v_sub
                seeds[l_ids] = l_sub
                bboxes = ([None] + list(ndimage.find_objects(labels))) if need_bboxes else None
                self._cache[key] = (labels, areas, minima, bboxes)

                qual_mask = (
                    (areas >= self.cfg.smin_km2)
                    & ((thresh - minima) >= self.cfg.tower_prominence_k - 1e-6)
                )
                qual_mask[0] = False
                qual_cids = np.flatnonzero(qual_mask).astype(np.int32)
                qual_seeds = seeds[qual_cids]
                self._qual_cache[key] = (qual_cids, qual_seeds, labels, areas, minima)
        elif need_bboxes and self._cache[key][3] is None:
            labels, areas, minima, _ = self._cache[key]
            bboxes = [None] + list(ndimage.find_objects(labels))
            self._cache[key] = (labels, areas, minima, bboxes)
        return self._cache[key]

    def qualifying_towers_at(
        self, threshold_k: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (qual_cids, qual_seed_flat_idx, labels, areas, minima) at threshold_k."""
        key = self._key(threshold_k)
        if key not in self._qual_cache:
            self._ensure_level(key, need_bboxes=False)
        return self._qual_cache[key]

    def slice_at(self, threshold_k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (labels, areas_km2, minima_k) for valid pixels with IR10.8 <= threshold_k."""
        key = self._key(threshold_k)
        labels, areas, minima, _ = self._ensure_level(key, need_bboxes=False)
        return labels, areas, minima

    def slice_at_with_bboxes(
        self, threshold_k: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[slice, slice] | None]]:
        """Return (labels, areas_km2, minima_k, bboxes_1indexed) for IR10.8 <= threshold_k."""
        key = self._key(threshold_k)
        labels, areas, minima, bboxes = self._ensure_level(key, need_bboxes=True)
        assert bboxes is not None
        return labels, areas, minima, bboxes

    def area_containing_seed(self, threshold_k: float, seed_yx: tuple[int, int]) -> float:
        """Return area (km2) of the connected component at threshold_k containing seed_yx."""
        if threshold_k < self.cfg.tcold_k - 1e-6:
            return 0.0
        threshold_clamped = min(float(threshold_k), self.cfg.twarm_k)
        labels, areas, _ = self.slice_at(threshold_clamped)
        comp = int(labels[seed_yx])
        return float(areas[comp]) if comp > 0 else 0.0

    def mask_containing_seed(self, threshold_k: float, seed_yx: tuple[int, int]) -> np.ndarray:
        """Return boolean mask of the component at threshold_k containing seed_yx."""
        threshold_clamped = min(max(float(threshold_k), self.cfg.tcold_k), self.cfg.twarm_k)
        labels, _, _ = self.slice_at(threshold_clamped)
        comp = int(labels[seed_yx])
        if comp <= 0:
            return np.zeros_like(self.valid, dtype=bool)
        return labels == comp

    def component_containing_seed(
        self, threshold_k: float, seed_yx: tuple[int, int]
    ) -> tuple[int, tuple[slice, slice] | None, np.ndarray | None]:
        """Return (comp_id, bbox, local_mask) for the component at threshold_k containing seed_yx."""
        threshold_clamped = min(max(float(threshold_k), self.cfg.tcold_k), self.cfg.twarm_k)
        labels, _, _, bboxes = self.slice_at_with_bboxes(threshold_clamped)
        comp = int(labels[seed_yx])
        if comp <= 0 or bboxes[comp] is None:
            return 0, None, None
        bbox = bboxes[comp]
        assert bbox is not None
        return comp, bbox, (labels[bbox] == comp)


class _LazyVerticalProfile(dict):  # type: ignore[type-arg]
    """Lazy dictionary of horizontal surface S(T) at 1 K steps (ATBD Annex Line H)."""

    def __init__(self, hierarchy: ThresholdSliceHierarchy, seed_yx: tuple[int, int], t_min_k: float) -> None:
        super().__init__()
        self._hierarchy = hierarchy
        self._seed_yx = seed_yx
        self._min_key = hierarchy._key(max(hierarchy.cfg.tcold_k, float(np.floor(t_min_k))))
        self._max_key = hierarchy._key(hierarchy.cfg.twarm_k)

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        return isinstance(key, int) and (self._min_key <= key <= self._max_key)

    def __getitem__(self, key: int) -> float:
        if not super().__contains__(key):
            if self._min_key <= key <= self._max_key:
                lv = self._hierarchy.threshold_from_key(key)
                val = self._hierarchy.area_containing_seed(lv, self._seed_yx)
                super().__setitem__(key, val)
                return val
            raise KeyError(key)
        return float(super().__getitem__(key))


def _spatial_coherence_sig(diff_values_2d: np.ndarray, cell_mask: np.ndarray, threshold: float) -> float:
    """Compute NWC SAF spatial coherence ratio SIG (ATBD Annex Line X, p. 60).

    Ratio of relevant pixels (D > threshold) that have at least one relevant
    8-connected neighbour to the total number of relevant pixels inside the cell.
    """
    relevant = cell_mask & (diff_values_2d > threshold)
    total_rel = int(relevant.sum())
    if total_rel <= 1:
        return 0.0
    neighbour_count = ndimage.convolve(
        relevant.astype(np.uint8), CONNECTIVITY_8.astype(np.uint8), mode="constant", cval=0
    )
    surrounded = relevant & (neighbour_count >= 2)
    return float(surrounded.sum() / total_rel)


def _compute_ellipse_parameters(
    yy: np.ndarray, xx: np.ndarray, areas: np.ndarray, total_area: float
) -> tuple[float, float, float, float]:
    """Return (minor_axis_km, major_axis_km, angle_deg, eccentricity) of equivalent ellipse."""
    if len(yy) < 2 or total_area <= 0.0:
        radius = np.sqrt(max(total_area, 1e-6) / np.pi)
        return radius, radius, 0.0, 0.0
    pixel_scale_km = float(np.sqrt(total_area / len(areas)))
    y_mean = float(np.dot(yy, areas) / total_area)
    x_mean = float(np.dot(xx, areas) / total_area)
    y_km = (yy - y_mean) * pixel_scale_km
    x_km = (xx - x_mean) * pixel_scale_km
    cov_xx = float(np.dot(x_km * x_km, areas) / total_area) + (pixel_scale_km ** 2) / 12.0
    cov_yy = float(np.dot(y_km * y_km, areas) / total_area) + (pixel_scale_km ** 2) / 12.0
    cov_xy = float(np.dot(x_km * y_km, areas) / total_area)
    trace = cov_xx + cov_yy
    det = max(cov_xx * cov_yy - cov_xy * cov_xy, 0.0)
    disc = np.sqrt(max(trace * trace / 4.0 - det, 0.0))
    lam1 = max(trace / 2.0 + disc, 1e-6)
    lam2 = max(trace / 2.0 - disc, 1e-6)
    major_axis_km = float(4.0 * np.sqrt(lam1))
    minor_axis_km = float(4.0 * np.sqrt(lam2))
    angle_rad = 0.5 * np.arctan2(2.0 * cov_xy, cov_xx - cov_yy)
    angle_deg = float(np.degrees(angle_rad))
    eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor_axis_km / max(major_axis_km, 1e-6)) ** 2)))
    return minor_axis_km, major_axis_km, angle_deg, eccentricity


def detect_rdt_cells(
    frame_7ch: np.ndarray,
    area_km2: np.ndarray,
    cfg: NwcSafRdtConfig,
    *,
    ct_cloudy_mask: np.ndarray | None = None,
    tropopause_temp_k: np.ndarray | float | None = None,
) -> tuple[np.ndarray, dict[int, dict[str, Any]], ThresholdSliceHierarchy, np.ndarray]:
    """Run NWC SAF RDT-CW adaptive cell detection on a 7-channel SEVIRI frame."""
    frame = np.asarray(frame_7ch, dtype=np.float32)
    area = np.asarray(area_km2, dtype=float)
    if frame.ndim != 3 or frame.shape[0] != 7 or area.shape != frame.shape[1:]:
        raise ValueError(f"Expected frame (7,H,W) and area (H,W); got {frame.shape} and {area.shape}")

    ir108 = frame[4]
    wv062 = frame[0]
    wv073 = frame[1]
    ir087 = frame[2]
    ir097 = frame[3]
    ir120 = frame[5]
    ir134 = frame[6]

    finite_obs = (
        np.isfinite(frame).all(axis=0)
        & np.isfinite(area)
        & (area > 0.0)
        & ((frame >= 100.0) & (frame <= 400.0)).all(axis=0)
    )
    if cfg.enforce_stretched_mask:
        max_pixel_area = cfg.stretch_area_factor * cfg.nominal_pixel_area_km2
        not_stretched = area <= max_pixel_area
    else:
        not_stretched = np.ones_like(finite_obs, dtype=bool)

    valid_mask = finite_obs & not_stretched
    if ct_cloudy_mask is not None:
        valid_mask = valid_mask & np.asarray(ct_cloudy_mask, dtype=bool)

    hierarchy = ThresholdSliceHierarchy(ir108, area, valid_mask, cfg)
    if not valid_mask.any():
        return np.zeros(area.shape, dtype=np.int32), {}, hierarchy, valid_mask

    # Spatial gradient of IR108 in K/km for inner and peripheral gradient attributes
    gy_pix, gx_pix = np.gradient(np.where(valid_mask, ir108, np.nan))
    pixel_step_km = np.sqrt(np.maximum(area, 1e-6))
    grad_ir_km = np.hypot(gy_pix, gx_pix) / pixel_step_km
    grad_ir_km = np.where(np.isfinite(grad_ir_km), grad_ir_km, 0.0)

    invalid_neighbour_mask = ndimage.binary_dilation(~valid_mask, structure=CONNECTIVITY_8)

    # Determine local BT floor from BTLIMIT (-60 C = 213.15 K) and optional NWP tropopause
    if tropopause_temp_k is not None and cfg.limittropo:
        tropo_field = np.broadcast_to(np.asarray(tropopause_temp_k, dtype=float), area.shape)
    else:
        tropo_field = np.full(area.shape, -np.inf, dtype=float)

    root_labels, root_areas, root_minima, root_bboxes = hierarchy.slice_at_with_bboxes(cfg.twarm_k)
    selected_towers: list[tuple[tuple[slice, slice], np.ndarray, float, float, tuple[int, int], bool]] = []

    def _local_bt_floor(bbox: tuple[slice, slice], loc_mask: np.ndarray, t_min: float) -> float:
        floor_k = cfg.btlimit_k
        if cfg.limittropo:
            mid_y = (bbox[0].start + bbox[0].stop) // 2
            mid_x = (bbox[1].start + bbox[1].stop) // 2
            tr_val = float(tropo_field[mid_y, mid_x])
            if np.isfinite(tr_val) and tr_val > 150.0:
                floor_k = max(floor_k, tr_val)
            elif t_min <= cfg.cat_mature_tmin_k:
                loc_ir = ir108[bbox]
                loc_wv = wv062[bbox]
                anvil_loading = loc_mask & (loc_ir <= cfg.cat_mature_tmin_k) & (np.abs(loc_wv - loc_ir) <= 4.5)
                if int(anvil_loading.sum()) >= 25:
                    est_tropo = float(np.clip(np.median(loc_ir[anvil_loading]), cfg.btlimit_k, 226.15))
                    floor_k = max(floor_k, est_tropo)
        return max(cfg.tcold_k, floor_k, t_min + cfg.tower_prominence_k)

    def _descend(
        bbox: tuple[slice, slice],
        loc_mask: np.ndarray,
        threshold_k: float,
        comp_id: int,
        minimum_k: float,
        is_isolated_single_tower: bool,
    ) -> None:
        bt_floor_k = _local_bt_floor(bbox, loc_mask, minimum_k)
        lower_levels = np.arange(
            threshold_k - cfg.deltat_k, bt_floor_k - 1e-6, -cfg.deltat_k
        )
        parent_labels_flat = hierarchy.slice_at(threshold_k)[0].ravel()
        cur_area = float(area[bbox][loc_mask].sum())
        for lower_k in lower_levels:
            qual_cids, qual_seeds, _, child_areas, child_minima = hierarchy.qualifying_towers_at(float(lower_k))
            if len(qual_cids) == 0:
                break
            valid_children = qual_cids[parent_labels_flat[qual_seeds] == comp_id]
            if len(valid_children) >= 2:
                child_labels, _, _, child_bboxes = hierarchy.slice_at_with_bboxes(float(lower_k))
                for cid_val in valid_children:
                    cid = int(cid_val)
                    c_bbox = child_bboxes[cid]
                    assert c_bbox is not None
                    c_loc_mask = child_labels[c_bbox] == cid
                    _descend(
                        c_bbox,
                        c_loc_mask,
                        float(lower_k),
                        cid,
                        float(child_minima[cid]),
                        is_isolated_single_tower=False,
                    )
                return
            if len(valid_children) == 1 and cur_area > cfg.smax_km2:
                only_cid = int(valid_children[0])
                child_labels, _, _, child_bboxes = hierarchy.slice_at_with_bboxes(float(lower_k))
                c_bbox = child_bboxes[only_cid]
                assert c_bbox is not None
                bbox = c_bbox
                loc_mask = child_labels[c_bbox] == only_cid
                threshold_k = float(lower_k)
                comp_id = only_cid
                parent_labels_flat = child_labels.ravel()
                minimum_k = float(child_minima[only_cid])
                cur_area = float(child_areas[only_cid])

        # Single tower (or mature cell reaching BT floor without further valid split)
        loc_ir_arr = ir108[bbox]
        masked_ir = np.where(loc_mask, loc_ir_arr, np.inf)
        loc_seed = np.unravel_index(int(np.argmin(masked_ir)), loc_ir_arr.shape)
        seed_yx = (int(bbox[0].start + loc_seed[0]), int(bbox[1].start + loc_seed[1]))
        t_min = float(ir108[seed_yx])

        # Define Top of Tower (ST) level: ~ T_min + 6 K (aligned to discrete grid)
        st_key = int(np.ceil((t_min + cfg.tower_prominence_k - cfg.tcold_k) / cfg.deltat_k))
        t_st = min(float(threshold_k), hierarchy.threshold_from_key(max(0, st_key)))
        s_st = hierarchy.area_containing_seed(t_st, seed_yx)
        s_bt = cur_area

        refined_threshold_k = float(threshold_k)
        refined_bbox = bbox
        refined_loc_mask = loc_mask

        single_tower_floor_k = max(
            t_st,
            bt_floor_k,
            cfg.cat_mature_tmin_k if t_min <= cfg.cat_mature_tmin_k else t_st,
        )
        if s_st >= cfg.smin_km2 and (is_isolated_single_tower or s_bt > cfg.smax_km2):
            candidate_levels = np.arange(
                refined_threshold_k, single_tower_floor_k - 1e-6, -cfg.deltat_k
            )
            if len(candidate_levels) >= 2:
                level_areas = [
                    hierarchy.area_containing_seed(float(lv), seed_yx)
                    for lv in candidate_levels
                ]
                deck_indices = [
                    idx
                    for idx in range(len(candidate_levels) - 1)
                    if candidate_levels[idx] >= cfg.cat1_tbt_k
                ]
                if deck_indices:
                    ratios = [
                        level_areas[idx + 1] / max(level_areas[idx], 1e-6)
                        for idx in deck_indices
                    ]
                    best_pos = int(np.argmin(ratios))
                    min_idx = deck_indices[best_pos]
                    if (
                        ratios[best_pos] <= cfg.constriction_ratio_max
                        and level_areas[min_idx + 1] >= cfg.smin_km2
                    ):
                        refined_threshold_k = float(candidate_levels[min_idx + 1])
                        _, r_bbox, r_loc_mask = hierarchy.component_containing_seed(refined_threshold_k, seed_yx)
                        if r_bbox is not None and r_loc_mask is not None:
                            refined_bbox = r_bbox
                            refined_loc_mask = r_loc_mask
                            s_bt = float(level_areas[min_idx + 1])

                if s_bt > cfg.max_base_top_ratio * s_st or s_bt > cfg.smax_km2:
                    refined_by_case_b = False
                    max_allowed_area = min(cfg.smax_km2, cfg.max_base_top_ratio * s_st)
                    for lv, lv_area in zip(candidate_levels, level_areas):
                        if cfg.smin_km2 <= lv_area <= max_allowed_area:
                            refined_threshold_k = float(lv)
                            _, r_bbox, r_loc_mask = hierarchy.component_containing_seed(refined_threshold_k, seed_yx)
                            if r_bbox is not None and r_loc_mask is not None:
                                refined_bbox = r_bbox
                                refined_loc_mask = r_loc_mask
                                s_bt = float(lv_area)
                                refined_by_case_b = True
                            break
                    if not refined_by_case_b and cfg.smin_km2 <= level_areas[-1] <= cfg.smax_km2:
                        refined_threshold_k = float(candidate_levels[-1])
                        _, r_bbox, r_loc_mask = hierarchy.component_containing_seed(refined_threshold_k, seed_yx)
                        if r_bbox is not None and r_loc_mask is not None:
                            refined_bbox = r_bbox
                            refined_loc_mask = r_loc_mask
                            s_bt = float(level_areas[-1])

        if cfg.smin_km2 <= s_bt <= cfg.smax_km2 and (refined_threshold_k - t_min) >= cfg.tower_prominence_k - 1e-6:
            selected_towers.append(
                (refined_bbox, refined_loc_mask, refined_threshold_k, t_st, seed_yx, is_isolated_single_tower)
            )

    for comp_id in range(1, len(root_areas)):
        comp_area = float(root_areas[comp_id])
        comp_min = float(root_minima[comp_id])
        if (
            comp_area >= cfg.smin_km2
            and (cfg.twarm_k - comp_min) >= cfg.tower_prominence_k - 1e-6
        ):
            r_bbox = root_bboxes[comp_id]
            assert r_bbox is not None
            _descend(
                r_bbox,
                root_labels[r_bbox] == comp_id,
                cfg.twarm_k,
                comp_id,
                comp_min,
                is_isolated_single_tower=True,
            )

    # Sort deterministically by coldest minimum temperature, then seed position
    selected_towers.sort(key=lambda item: (float(ir108[item[4]]), item[4][0], item[4][1]))

    labels = np.zeros(area.shape, dtype=np.int32)
    cells: dict[int, dict[str, Any]] = {}

    # Precompute full-domain multi-spectral BTD fields once to avoid slicing & subtracting per cell
    btd_wv62_ir108_full = wv062 - ir108
    wbtd_wv62_wv73_full = wv062 - wv073
    btd_ir87_ir108_full = ir087 - ir108
    btd_ir120_ir108_full = ir120 - ir108
    btd_ir97_ir108_full = ir097 - ir108
    btd_ir134_ir108_full = ir134 - ir108

    for cand_bbox, cand_loc_mask, t_bt, t_st, seed_yx, is_single in selected_towers:
        loc_seed_yx = (seed_yx[0] - cand_bbox[0].start, seed_yx[1] - cand_bbox[1].start)
        clean_loc = cand_loc_mask & (labels[cand_bbox] == 0)
        if not clean_loc[loc_seed_yx]:
            continue
        if not np.array_equal(clean_loc, cand_loc_mask):
            comp_labels, _ = ndimage.label(clean_loc, structure=CONNECTIVITY_8)
            part_id = int(comp_labels[loc_seed_yx])
            if part_id <= 0:
                continue
            clean_loc = comp_labels == part_id

        s_bt = float(area[cand_bbox][clean_loc].sum())
        if s_bt < cfg.smin_km2:
            continue

        cell_id = len(cells) + 1
        labels[cand_bbox][clean_loc] = cell_id

        yy_loc, xx_loc = np.where(clean_loc)
        yy = yy_loc + cand_bbox[0].start
        xx = xx_loc + cand_bbox[1].start
        bbox = (
            slice(int(yy.min()), int(yy.max()) + 1),
            slice(int(xx.min()), int(xx.max()) + 1),
        )
        local_mask = labels[bbox] == cell_id
        local_area = area[bbox]
        local_ir = ir108[bbox]
        local_seed_yx = (seed_yx[0] - bbox[0].start, seed_yx[1] - bbox[1].start)

        cell_areas = local_area[local_mask]
        cell_ir = local_ir[local_mask]
        t_min = float(cell_ir.min())
        t_mean = float(np.dot(cell_ir, cell_areas) / s_bt)
        t_std = float(np.sqrt(max(0.0, float(np.dot((cell_ir - t_mean) ** 2, cell_areas) / s_bt))))

        geom_centroid = np.array(
            [float(np.dot(yy, cell_areas) / s_bt), float(np.dot(xx, cell_areas) / s_bt)],
            dtype=float,
        )
        thermal_weights = cell_areas * np.maximum(t_bt - cell_ir, 1e-3)
        tw_sum = float(thermal_weights.sum())
        weight_centroid = np.array(
            [float(np.dot(yy, thermal_weights) / tw_sum), float(np.dot(xx, thermal_weights) / tw_sum)],
            dtype=float,
        )

        thermal_volume = float(np.sum(cell_areas * np.maximum(t_bt - cell_ir, 0.0)))

        # Top of Tower (ST) contour & area via cached hierarchy slice
        st_labels_full, _, _ = hierarchy.slice_at(t_st)
        st_comp_id = int(st_labels_full[seed_yx])
        if st_comp_id > 0:
            st_local_mask = local_mask & (st_labels_full[bbox] == st_comp_id)
        else:
            st_local_mask = local_mask & (local_ir <= t_min + cfg.deltat_k)
        s_st = float(local_area[st_local_mask].sum())

        # Inner and peripheral temperature gradients (K/km)
        local_grad = grad_ir_km[bbox]
        inner_grad = float(np.dot(local_grad[local_mask], cell_areas) / s_bt)
        eroded = ndimage.binary_erosion(local_mask, structure=CONNECTIVITY_8, border_value=0)
        periph_mask = local_mask & ~eroded
        if not periph_mask.any():
            periph_mask = local_mask
        periph_grads = local_grad[periph_mask]
        periph_grad_mean = float(np.mean(periph_grads))
        pg_sorted = np.sort(periph_grads)
        n_pg = len(pg_sorted)
        if n_pg == 1:
            periph_grad_p95 = float(pg_sorted[0])
        else:
            pos95 = 0.95 * (n_pg - 1)
            i95 = int(pos95)
            periph_grad_p95 = float(
                pg_sorted[i95] + (pos95 - i95) * (pg_sorted[min(i95 + 1, n_pg - 1)] - pg_sorted[i95])
            )
        n_periph = int(periph_mask.sum())

        minor_km, major_km, angle_deg, eccentricity = _compute_ellipse_parameters(
            yy, xx, cell_areas, s_bt
        )

        touches_domain_edge = bool(
            (yy == 0).any()
            or (yy == area.shape[0] - 1).any()
            or (xx == 0).any()
            or (xx == area.shape[1] - 1).any()
        )
        touches_invalid = bool(invalid_neighbour_mask[bbox][local_mask].any())
        spatial_cutoff = (1 if touches_domain_edge else 0) + (2 if touches_invalid else 0)

        vertical_profile: dict[int, float] = _LazyVerticalProfile(hierarchy, seed_yx, t_min)

        loc_wv062 = wv062[bbox]
        loc_wv073 = wv073[bbox]
        loc_ir087 = ir087[bbox]
        loc_ir097 = ir097[bbox]
        loc_ir120 = ir120[bbox]
        loc_ir134 = ir134[bbox]

        btd_wv62_ir108 = btd_wv62_ir108_full[bbox]
        wbtd_wv62_wv73 = wbtd_wv62_wv73_full[bbox]
        btd_ir87_ir108 = btd_ir87_ir108_full[bbox]
        btd_ir120_ir108 = btd_ir120_ir108_full[bbox]
        btd_ir97_ir108 = btd_ir97_ir108_full[bbox]
        btd_ir134_ir108 = btd_ir134_ir108_full[bbox]

        def _btd_stats(field_2d: np.ndarray, sig_thresh: float) -> dict[str, float]:
            vals = field_2d[local_mask]
            s = np.sort(vals)
            n = len(s)
            if n == 1:
                v0 = float(s[0])
                q90 = q75 = q50 = v0
            else:
                nm1 = n - 1
                p90, p75, p50 = 0.90 * nm1, 0.75 * nm1, 0.50 * nm1
                i90, i75, i50 = int(p90), int(p75), int(p50)
                q90 = float(s[i90] + (p90 - i90) * (s[min(i90 + 1, nm1)] - s[i90]))
                q75 = float(s[i75] + (p75 - i75) * (s[min(i75 + 1, nm1)] - s[i75]))
                q50 = float(s[i50] + (p50 - i50) * (s[min(i50 + 1, nm1)] - s[i50]))
            return {
                "max": float(s[-1]),
                "p90": q90,
                "p75": q75,
                "median": q50,
                "mean": float(s.sum() / n),
                "sig": _spatial_coherence_sig(field_2d, local_mask, sig_thresh),
            }

        s_btd = _btd_stats(btd_wv62_ir108, sig_thresh=-2.0)
        s_wbtd = _btd_stats(wbtd_wv62_wv73, sig_thresh=-10.0)
        s_btd87 = _btd_stats(btd_ir87_ir108, sig_thresh=-1.0)
        s_btd120 = _btd_stats(btd_ir120_ir108, sig_thresh=-1.0)

        local_tropo = (
            float(np.nanmedian(tropo_field[bbox][local_mask]))
            if np.isfinite(tropo_field[bbox][local_mask]).any()
            else np.nan
        )

        cells[cell_id] = {
            "cell_id": cell_id,
            "bbox": bbox,
            "mask": local_mask,
            "st_mask": st_local_mask,
            "seed_yx": seed_yx,
            "pixel_count": int(local_mask.sum()),
            "is_single_tower": bool(is_single),
            "t_bt_k": float(t_bt),
            "t_st_k": float(t_st),
            "t_min_k": t_min,
            "t_mean_k": t_mean,
            "t_std_k": t_std,
            "contrast_k": float(t_mean - t_min),
            "tower_depth_k": float(t_bt - t_min),
            "area_km2": s_bt,
            "st_area_km2": s_st,
            "base_top_area_ratio": float(s_bt / max(s_st, 1e-6)),
            "thermal_volume_km2_k": thermal_volume,
            "geom_centroid_yx": geom_centroid,
            "weight_centroid_yx": weight_centroid,
            "centroid": weight_centroid,
            "ellipse_minor_km": minor_km,
            "ellipse_major_km": major_km,
            "ellipse_angle_deg": angle_deg,
            "ellipse_eccentricity": eccentricity,
            "ellipse_aspect_ratio": float(major_km / max(minor_km, 1e-6)),
            "inner_grad_k_km": inner_grad,
            "periph_grad_mean_k_km": periph_grad_mean,
            "periph_grad_p95_k_km": periph_grad_p95,
            "n_periph_pixels": n_periph,
            "border": bool(touches_domain_edge or touches_invalid),
            "spatial_cutoff_flag": int(spatial_cutoff),
            "vertical_profile": vertical_profile,
            "tmin_wv062_k": float(loc_wv062[local_mask].min()),
            "tmin_wv073_k": float(loc_wv073[local_mask].min()),
            "tmin_ir087_k": float(loc_ir087[local_mask].min()),
            "tmin_ir097_k": float(loc_ir097[local_mask].min()),
            "tmin_ir120_k": float(loc_ir120[local_mask].min()),
            "tmin_ir134_k": float(loc_ir134[local_mask].min()),
            "btd_wv62_ir108_max_k": s_btd["max"],
            "btd_wv62_ir108_p90_k": s_btd["p90"],
            "btd_wv62_ir108_p75_k": s_btd["p75"],
            "btd_wv62_ir108_median_k": s_btd["median"],
            "btd_wv62_ir108_sig": s_btd["sig"],
            "wbtd_wv62_wv73_max_k": s_wbtd["max"],
            "wbtd_wv62_wv73_p90_k": s_wbtd["p90"],
            "wbtd_wv62_wv73_p75_k": s_wbtd["p75"],
            "wbtd_wv62_wv73_sig": s_wbtd["sig"],
            "btd_ir87_ir108_max_k": s_btd87["max"],
            "btd_ir87_ir108_p90_k": s_btd87["p90"],
            "btd_ir87_ir108_p75_k": s_btd87["p75"],
            "btd_ir87_ir108_sig": s_btd87["sig"],
            "btd_ir120_ir108_max_k": s_btd120["max"],
            "btd_ir120_ir108_p90_k": s_btd120["p90"],
            "btd_ir120_ir108_p75_k": s_btd120["p75"],
            "btd_ir120_ir108_sig": s_btd120["sig"],
            "btd_ir108_ir120_median_k": -s_btd120["median"],
            "btd_ir97_ir108_median_k": float(np.sort(btd_ir97_ir108[local_mask])[len(btd_ir97_ir108[local_mask]) // 2]),
            "btd_ir134_ir108_median_k": float(np.sort(btd_ir134_ir108[local_mask])[len(btd_ir134_ir108[local_mask]) // 2]),
            "tropopause_temp_k": local_tropo,
        }

    return labels, cells, hierarchy, valid_mask
