"""Warm cloud cell detection and 2D motion tracking for GEO-CI.

References:
- ATBD v2.1 Section 2.2.1.2.2 (2D movement field)
- ATBD v2.1 Section 2.2.1.2.3 (Object image-analysis: Cloud cell detection and tracking)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.ndimage import label, center_of_mass, gaussian_filter, find_objects
from scipy.spatial import cKDTree

from .config import NwcSafCiConfig, CHANNEL_INDICES


@dataclass
class CiCloudCell:
    """Represents a detected warm convective cloud cell."""
    cell_id: int
    t_min_k: float
    t_base_k: float
    center_yx: Tuple[float, float]
    area_px: int
    area_km2: float
    pixel_coords: Tuple[np.ndarray, np.ndarray]  # (rows, cols)
    u_mps: float = 0.0  # Horizontal velocity (columns/sec or m/s)
    v_mps: float = 0.0  # Vertical velocity (rows/sec or m/s)
    parent_id: Optional[int] = None
    age_frames: int = 1


def detect_warm_cloud_cells(
    ir108_k: np.ndarray,
    config: Optional[NwcSafCiConfig] = None,
    eligible_mask: Optional[np.ndarray] = None,
) -> List[CiCloudCell]:
    """Detects warm developing cloud cells via multi-thresholding on IR10.8 channel.

    Slicing range:
    From TWARM (+10°C / 283.15 K) down to TCOLD (-25°C / 248.15 K) in steps of DELTATEMPE (1.0 K).
    A cell requires vertical prominence DELTATTOUR >= 3.0 K (T_base - T_min >= 3.0 K).
    Area range: SMIN to SMAX.
    """
    if config is None:
        config = NwcSafCiConfig()

    px_area_km2 = config.pixel_size_km ** 2
    min_px = max(1, int(np.ceil(config.s_min_km2 / px_area_km2)))
    max_px = int(np.floor(config.s_max_km2 / px_area_km2))

    # Construct threshold hierarchy from warmest to coldest
    thresholds = np.arange(config.t_warm_k, config.t_cold_k - 0.1, -config.delta_t_k)

    detected_cells: List[CiCloudCell] = []
    occupied_mask = np.zeros_like(ir108_k, dtype=bool)

    structure_8conn = np.ones((3, 3), dtype=int)

    # Start from coldest levels to find core summits, then grow to warmest distinguishing base
    # (or slice from warm to cold and isolate distinct minima)
    # Find local minima with prominence >= delta_t_tour_k
    # Standard NWC SAF approach: Slice top-down or bottom-up
    for thresh in thresholds:
        binary = (ir108_k <= thresh)
        if eligible_mask is not None:
            binary &= eligible_mask

        # Exclude regions already identified as distinct mature towers
        binary_unclaimed = binary & (~occupied_mask)
        if not np.any(binary_unclaimed):
            continue

        labeled, num_features = label(binary_unclaimed, structure=structure_8conn)
        if num_features == 0:
            continue

        slices = find_objects(labeled)
        for feat_id, sl in enumerate(slices, start=1):
            if sl is None:
                continue

            sub_lbl = labeled[sl]
            sub_mask = (sub_lbl == feat_id)
            n_px = int(np.count_nonzero(sub_mask))
            if n_px < min_px or n_px > max_px:
                continue

            sub_ir = ir108_k[sl]
            t_min = float(np.min(sub_ir[sub_mask]))
            prominence = thresh - t_min

            # Must satisfy minimum vertical budding extension (prominence >= 3 K)
            if prominence >= config.delta_t_tour_k:
                sub_y, sub_x = np.where(sub_mask)
                rows = sub_y + sl[0].start
                cols = sub_x + sl[1].start
                cy = float(np.mean(rows))
                cx = float(np.mean(cols))
                cell = CiCloudCell(
                    cell_id=len(detected_cells) + 1,
                    t_min_k=t_min,
                    t_base_k=float(thresh),
                    center_yx=(cy, cx),
                    area_px=n_px,
                    area_km2=float(n_px * px_area_km2),
                    pixel_coords=(rows, cols),
                )
                detected_cells.append(cell)
                occupied_mask[sl][sub_mask] = True

    return detected_cells


class WarmCellTracker:
    """Maintains cloud cell trajectories and builds the 2D movement field."""

    def __init__(self, config: Optional[NwcSafCiConfig] = None):
        self.config = config or NwcSafCiConfig()
        self.previous_cells: List[CiCloudCell] = []
        self.next_cell_id: int = 1

    def track_and_update_motion(
        self,
        current_cells: List[CiCloudCell],
        shape: Tuple[int, int],
        delta_t_min: float = 15.0,
        guess_u: Optional[np.ndarray] = None,
        guess_v: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Matches cells across consecutive frames and updates the 2D movement field.

        Parameters
        ----------
        current_cells : List[CiCloudCell]
            Cells detected in current slot.
        shape : Tuple[int, int]
            Grid shape (H, W).
        delta_t_min : float
            Time step between slots in minutes (typically 15 min or 5 min).
        guess_u, guess_v : Optional[np.ndarray]
            Initial 2D motion field (pixels/minute or m/s). If None, defaults to zero.

        Returns
        -------
        motion_u, motion_v : Tuple[np.ndarray, np.ndarray]
            Updated 2D movement field in pixels per minute (u: x/col, v: y/row).
        """
        h, w = shape
        if guess_u is None:
            motion_u = np.full((h, w), self.config.default_u_mps, dtype=np.float32)
        else:
            motion_u = guess_u.astype(np.float32).copy()

        if guess_v is None:
            motion_v = np.full((h, w), self.config.default_v_mps, dtype=np.float32)
        else:
            motion_v = guess_v.astype(np.float32).copy()

        delta_t_min = max(1.0, float(delta_t_min))

        # Track against previous cells
        if len(self.previous_cells) > 0:
            prev_centers = np.array([c.center_yx for c in self.previous_cells], dtype=np.float32)
            prev_tree = cKDTree(prev_centers)
        else:
            prev_tree = None

        for cell in current_cells:
            cell.cell_id = self.next_cell_id
            self.next_cell_id += 1

            best_parent: Optional[CiCloudCell] = None
            best_aor: float = 0.0

            # Pixel set of current cell (flatten coordinates to ensure 1D elements)
            rows_curr = np.asarray(cell.pixel_coords[0]).ravel()
            cols_curr = np.asarray(cell.pixel_coords[1]).ravel()
            curr_set = set(zip(rows_curr, cols_curr))

            # Pre-filter candidate parents using spatial proximity (radius 25 px ~= 75 km at 3 km/px)
            if prev_tree is not None:
                cand_indices = prev_tree.query_ball_point(cell.center_yx, r=25.0)
            else:
                cand_indices = []

            for idx in cand_indices:
                prev_cell = self.previous_cells[idx]
                rows_prev = np.asarray(prev_cell.pixel_coords[0]).ravel()
                cols_prev = np.asarray(prev_cell.pixel_coords[1]).ravel()

                # Advect prev_cell pixels forward using prev_cell velocity or guess motion
                p_cy = int(np.clip(round(prev_cell.center_yx[0]), 0, h - 1))
                p_cx = int(np.clip(round(prev_cell.center_yx[1]), 0, w - 1))
                vy = prev_cell.v_mps if (prev_cell.v_mps != 0.0 or prev_cell.u_mps != 0.0) else float(motion_v[p_cy, p_cx])
                vx = prev_cell.u_mps if (prev_cell.v_mps != 0.0 or prev_cell.u_mps != 0.0) else float(motion_u[p_cy, p_cx])

                shift_y = int(round(vy * delta_t_min))
                shift_x = int(round(vx * delta_t_min))

                advected_rows = np.clip(rows_prev + shift_y, 0, h - 1)
                advected_cols = np.clip(cols_prev + shift_x, 0, w - 1)
                prev_advected_set = set(zip(advected_rows, advected_cols))

                intersection = len(curr_set & prev_advected_set)
                min_area = min(len(curr_set), len(prev_advected_set))
                aor = intersection / max(1, min_area)
                if aor >= self.config.min_aor_tracking and aor > best_aor:
                    best_aor = aor
                    best_parent = prev_cell

            # Fallback for small cells: centroid proximity matching (within 12 pixels displacement)
            if best_parent is None and prev_tree is not None:
                best_dist = float("inf")
                max_displacement_px = 12.0  # ~36 km in 15 min at 3km/px (typical upper convective speed limit)
                close_indices = prev_tree.query_ball_point(cell.center_yx, r=max_displacement_px)
                for idx in close_indices:
                    prev_cell = self.previous_cells[idx]
                    p_cy, p_cx = prev_cell.center_yx
                    c_cy, c_cx = cell.center_yx
                    dist = float(np.hypot(c_cy - p_cy, c_cx - p_cx))
                    if dist < best_dist:
                        best_dist = dist
                        best_parent = prev_cell

            if best_parent is not None:
                cell.parent_id = best_parent.cell_id
                cell.age_frames = best_parent.age_frames + 1

                # Measured displacement in pixels
                dy = cell.center_yx[0] - best_parent.center_yx[0]
                dx = cell.center_yx[1] - best_parent.center_yx[1]

                # Velocity in pixels per minute
                vel_y = dy / delta_t_min
                vel_x = dx / delta_t_min

                # If parent had no prior motion (cold start), initialize directly; otherwise smooth
                if best_parent.v_mps == 0.0 and best_parent.u_mps == 0.0:
                    cell.v_mps = vel_y
                    cell.u_mps = vel_x
                else:
                    cell.v_mps = 0.5 * vel_y + 0.5 * best_parent.v_mps
                    cell.u_mps = 0.5 * vel_x + 0.5 * best_parent.u_mps
            else:
                # Cold start / orphan: sample from guess motion field at cell centroid
                cy, cx = int(round(cell.center_yx[0])), int(round(cell.center_yx[1]))
                cy = np.clip(cy, 0, h - 1)
                cx = np.clip(cx, 0, w - 1)
                cell.u_mps = float(motion_u[cy, cx])
                cell.v_mps = float(motion_v[cy, cx])

            # Apply cell velocity to all pixels within the cell
            motion_u[cell.pixel_coords] = cell.u_mps
            motion_v[cell.pixel_coords] = cell.v_mps

        # Smooth motion field with Gaussian filter to propagate motion to surrounding influence box
        sigma = self.config.motion_filter_radius_px / 3.0
        motion_u = gaussian_filter(motion_u, sigma=sigma, mode="nearest")
        motion_v = gaussian_filter(motion_v, sigma=sigma, mode="nearest")

        # ATBD Sec 2.2.1.2.3: All pixels belonging to a tracked cloud system are affected the corresponding movement speed
        for cell in current_cells:
            motion_u[cell.pixel_coords] = cell.u_mps
            motion_v[cell.pixel_coords] = cell.v_mps

        self.previous_cells = current_cells
        return motion_u, motion_v
