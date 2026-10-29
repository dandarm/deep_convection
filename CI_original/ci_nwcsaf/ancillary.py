"""Ancillary data processors and pure-IR motion/cloud surrogates.

Provides autonomous pure-IR motion vector estimation (GEO-HRW surrogate)
and cloud masking, with optional bridges for ERA5 NWP convective indices.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.signal import fftconvolve

from .config import NwcSafCiConfig, CHANNEL_INDICES


def compute_pure_ir_amv_motion(
    current_ir108: np.ndarray,
    past_ir108: np.ndarray,
    delta_t_min: float = 15.0,
    patch_size: int = 32,
    search_radius: int = 8,
    step: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes Atmospheric Motion Vectors (AMVs) via local block-matching cross-correlation.

    Parameters
    ----------
    current_ir108 : np.ndarray
        Current IR10.8 brightness temperature (H, W).
    past_ir108 : np.ndarray
        Past IR10.8 brightness temperature (H, W).
    delta_t_min : float
        Time step between images in minutes.
    patch_size : int
        Block size for correlation matching.
    search_radius : int
        Maximum pixel displacement search window.
    step : Optional[int]
        Stride between correlation patches (defaults to patch_size).

    Returns
    -------
    u_px_per_min, v_px_per_min : Tuple[np.ndarray, np.ndarray]
        2D movement field in pixels per minute (u: x/col, v: y/row).
    """
    h, w = current_ir108.shape
    u_grid = np.zeros((h, w), dtype=np.float32)
    v_grid = np.zeros((h, w), dtype=np.float32)

    delta_t_min = max(1.0, float(delta_t_min))
    stride = step if step is not None else patch_size

    # Grid of patch centers
    for r in range(search_radius + patch_size // 2, h - search_radius - patch_size // 2, stride):
        for c in range(search_radius + patch_size // 2, w - search_radius - patch_size // 2, stride):
            r_start = r - patch_size // 2
            r_end = r_start + patch_size
            c_start = c - patch_size // 2
            c_end = c_start + patch_size

            target = current_ir108[r_start:r_end, c_start:c_end]
            # Fast clear-sky rejection: no clouds in patch if all pixels > 0 °C (273.15 K)
            if float(np.min(target)) > 273.15:
                continue

            target_std = float(np.std(target))
            if target_std < 0.5:  # Flat area, no trackable texture
                continue

            target_norm = (target - float(np.mean(target))) / target_std
            box_past = past_ir108[r_start - search_radius : r_end + search_radius, c_start - search_radius : c_end + search_radius]

            box_std = float(np.std(box_past))
            if box_std < 0.5:
                continue
            box_past_norm = (box_past - float(np.mean(box_past))) / box_std

            conv = fftconvolve(box_past_norm, target_norm[::-1, ::-1], mode="valid") / float(patch_size * patch_size)
            max_val = float(np.max(conv))
            if max_val > 0.4:
                best_idx = np.unravel_index(np.argmax(conv), conv.shape)
                dr = search_radius - best_idx[0]
                dc = search_radius - best_idx[1]

                # Velocity in pixels per minute
                vel_y = dr / delta_t_min
                vel_x = dc / delta_t_min

                # Populate block
                r_box_start = max(0, r - stride // 2)
                r_box_end = min(h, r + stride // 2)
                c_box_start = max(0, c - stride // 2)
                c_box_end = min(w, c + stride // 2)

                v_grid[r_box_start:r_box_end, c_box_start:c_box_end] = vel_y
                u_grid[r_box_start:r_box_end, c_box_start:c_box_end] = vel_x

    # Smooth the AMV field
    u_smooth = gaussian_filter(u_grid, sigma=patch_size / 2.0, mode="nearest")
    v_smooth = gaussian_filter(v_grid, sigma=patch_size / 2.0, mode="nearest")

    return u_smooth, v_smooth
