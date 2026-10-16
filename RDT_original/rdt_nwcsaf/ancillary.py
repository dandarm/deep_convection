"""NWC SAF GEO-CMa / GEO-CT and GEO-HRW Ancillary Equivalent Modules.

Implements the physical multi-spectral radiative cloud-type pre-filtering (GEO-CT)
and the Atmospheric Motion Vector (GEO-HRW) + NWP 700 hPa first-guess motion field
required by the complete NWC SAF RDT-CW pipeline (ATBD v2.1 Sections 3.2.1.1 and 3.2.1.2.2).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, zoom


@dataclass(frozen=True)
class NwcSafCloudTypeResult:
    """Output of the NWC SAF GEO-CMa / GEO-CT radiative classification."""

    ct_class: np.ndarray
    """Cloud type category:
    0 = Cloud-free
    1 = Low warm cloud
    2 = Medium opaque cloud
    3 = High opaque cloud
    4 = Very high opaque cloud / deep tower
    5 = Semitransparent thin cirrus
    """
    opaque_cloud_mask: np.ndarray
    thin_cirrus_mask: np.ndarray
    filtered_ir108: np.ndarray


def compute_nwcsaf_cloud_type(
    ir108: np.ndarray,
    *,
    ir120: np.ndarray | None = None,
    ir087: np.ndarray | None = None,
    wv062: np.ndarray | None = None,
    wv073: np.ndarray | None = None,
    t_tropo_k: float | np.ndarray = 215.0,
    t_surf_k: float | np.ndarray = 292.0,
) -> NwcSafCloudTypeResult:
    """Classify pixels into NWC SAF Cloud Types and filter thin semitransparent cirrus.

    Follows the NWC SAF GEO-CT multi-spectral split-window and water-vapour logic:
    - Split-window difference ``BTD(10.8 - 12.0)`` identifies optically thin ice clouds
      (semitransparent cirrus) because differential absorption by small ice crystals
      between 10.8 um and 12.0 um peaks for optical thicknesses tau in [0.5, 3.0].
    - ``BTD(8.7 - 10.8)`` and ``BTD(6.2 - 10.8)`` separate high ice clouds from mid-level
      water clouds and identify deep opaque cumulonimbus anvils.
    - Detached thin semitransparent cirrus not connected to an opaque convective/anvil
      core are masked out from ``filtered_ir108`` before RDT thermal slicing.
    """
    ir108_f = np.asarray(ir108, dtype=np.float64)
    H, W = ir108_f.shape
    finite = np.isfinite(ir108_f)

    t_tropo_arr = np.broadcast_to(np.asarray(t_tropo_k, dtype=np.float64), (H, W))
    t_surf_arr = np.broadcast_to(np.asarray(t_surf_k, dtype=np.float64), (H, W))

    ct_class = np.zeros((H, W), dtype=np.uint8)

    # Clear-sky vs cloudy threshold
    clear_thresh = np.minimum(t_surf_arr - 6.0, 280.15)
    cloudy = finite & (ir108_f <= clear_thresh)

    # Default split-window and BTD arrays if available
    if ir120 is not None:
        btd_108_120 = ir108_f - np.asarray(ir120, dtype=np.float64)
    else:
        btd_108_120 = np.zeros_like(ir108_f)

    if ir087 is not None:
        btd_087_108 = np.asarray(ir087, dtype=np.float64) - ir108_f
    else:
        btd_087_108 = np.zeros_like(ir108_f)

    if wv062 is not None:
        btd_062_108 = np.asarray(wv062, dtype=np.float64) - ir108_f
    else:
        btd_062_108 = np.full_like(ir108_f, -10.0)

    # NWC SAF GEO-CT semitransparent cirrus curve:
    # Thin cirrus exhibits large positive BTD(10.8 - 12.0) (> 2.2 K to 3.5 K depending on BT10.8)
    # while still being significantly warmer than the upper-tropospheric WV6.2 emission
    # (BTD(6.2 - 10.8) < -11 K) and warmer than 228 K.
    # Opaque cumulonimbus anvils and towers have small BTD(10.8 - 12.0) (< 1.8 K)
    # and high WV6.2 - IR10.8 (> -10 K).
    sw_cirrus_thresh = 2.2 + 0.025 * np.clip(ir108_f - 230.0, 0.0, 45.0)
    thin_cirrus = (
        cloudy
        & (ir108_f > 228.15)
        & (ir108_f < 268.15)
        & (btd_108_120 > sw_cirrus_thresh)
        & (btd_062_108 < -11.0)
        & (btd_087_108 > -1.5)
    )

    # Opaque core protection: any cirrus pixel within 3 pixels (~9 km) of a cold opaque
    # cloud (IR10.8 < 235 K and BTD(10.8-12.0) <= sw_cirrus_thresh) is preserved as part
    # of the cumulonimbus anvil skirt so we do not clip real anvil edges.
    cold_opaque_core = cloudy & (ir108_f < 235.15) & (~thin_cirrus)
    if np.any(cold_opaque_core):
        anvil_halo = binary_dilation(cold_opaque_core, structure=np.ones((7, 7), dtype=bool))
        detached_thin_cirrus = thin_cirrus & (~anvil_halo)
    else:
        detached_thin_cirrus = thin_cirrus

    # Assign classes
    ct_class[cloudy & (ir108_f >= 260.15)] = 1  # Low warm cloud
    ct_class[cloudy & (ir108_f >= 240.15) & (ir108_f < 260.15)] = 2  # Medium opaque
    ct_class[cloudy & (ir108_f >= 220.15) & (ir108_f < 240.15)] = 3  # High opaque
    ct_class[
        cloudy & ((ir108_f < 220.15) | (ir108_f <= t_tropo_arr + 8.0))
    ] = 4  # Very high opaque
    ct_class[detached_thin_cirrus] = 5  # Detached semitransparent cirrus

    opaque_cloud_mask = (ct_class >= 1) & (ct_class <= 4)

    # Filtered IR10.8 for RDT detection: replace detached thin cirrus with warm background
    filtered_ir108 = ir108_f.copy()
    filtered_ir108[detached_thin_cirrus] = np.maximum(ir108_f[detached_thin_cirrus], 280.0)

    return NwcSafCloudTypeResult(
        ct_class=ct_class,
        opaque_cloud_mask=opaque_cloud_mask,
        thin_cirrus_mask=detached_thin_cirrus,
        filtered_ir108=filtered_ir108,
    )


def compute_hrw_amv_guess_field(
    prev_ir108: np.ndarray,
    curr_ir108: np.ndarray,
    *,
    prev_wv062: np.ndarray | None = None,
    curr_wv062: np.ndarray | None = None,
    u700_ms: np.ndarray | None = None,
    v700_ms: np.ndarray | None = None,
    dt_seconds: float = 300.0,
    pixel_dx_km: float = 3.0,
    pixel_dy_km: float = 3.0,
    block_size: int = 24,
    stride: int = 12,
    max_shift_px: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a 2D first-guess motion field (dy_px, dx_px) combining GEO-HRW AMVs and NWP 700 hPa wind.

    In NWC SAF RDT-CW (ATBD Section 3.2.1.2.2), the first-guess motion field combines:
    1. Satellite Atmospheric Motion Vectors (GEO-HRW) derived from cross-correlation of
       IR10.8 and WV6.2 templates in cloudy upper/mid-tropospheric regions.
    2. NWP 700 hPa horizontal wind (U700, V700) as the background steering flow,
       especially for newly triggered warm cells or low-correlation regions.

    Returns
    -------
    guess_dy_px, guess_dx_px : tuple[np.ndarray, np.ndarray]
        2D arrays (H, W) of pixel displacements over ``dt_seconds`` (north-up, west-left:
        positive ``dy_px`` = southward row index increase, positive ``dx_px`` = eastward col increase).
    """
    H, W = curr_ir108.shape
    dt_s = max(float(dt_seconds), 60.0)

    # Convert NWP 700 hPa wind (m/s, meteorological U=eastward, V=northward) to pixel displacement
    if u700_ms is not None and v700_ms is not None:
        u_arr = np.broadcast_to(np.asarray(u700_ms, dtype=np.float64), (H, W))
        v_arr = np.broadcast_to(np.asarray(v700_ms, dtype=np.float64), (H, W))
        nwp_dx_px = (u_arr * dt_s) / (pixel_dx_km * 1000.0)
        # North-up grid: northward wind V > 0 corresponds to decreasing row index (dy_px < 0)
        nwp_dy_px = -(v_arr * dt_s) / (pixel_dy_km * 1000.0)
    else:
        nwp_dx_px = np.zeros((H, W), dtype=np.float64)
        nwp_dy_px = np.zeros((H, W), dtype=np.float64)

    if prev_ir108 is None or prev_ir108.shape != curr_ir108.shape:
        return nwp_dy_px.astype(np.float32), nwp_dx_px.astype(np.float32)

    # Coarse grid of AMV blocks
    rows = list(range(max_shift_px, H - block_size - max_shift_px + 1, stride))
    cols = list(range(max_shift_px, W - block_size - max_shift_px + 1, stride))
    if not rows or not cols:
        return nwp_dy_px.astype(np.float32), nwp_dx_px.astype(np.float32)

    nr, nc = len(rows), len(cols)
    amv_dy = np.zeros((nr, nc), dtype=np.float64)
    amv_dx = np.zeros((nr, nc), dtype=np.float64)
    amv_weight = np.zeros((nr, nc), dtype=np.float64)

    # Combine IR10.8 and WV6.2 if available for richer upper-level texture
    if prev_wv062 is not None and curr_wv062 is not None:
        p_field = 0.7 * np.nan_to_num(prev_ir108, nan=290.0) + 0.3 * np.nan_to_num(prev_wv062, nan=240.0)
        c_field = 0.7 * np.nan_to_num(curr_ir108, nan=290.0) + 0.3 * np.nan_to_num(curr_wv062, nan=240.0)
    else:
        p_field = np.nan_to_num(prev_ir108, nan=290.0)
        c_field = np.nan_to_num(curr_ir108, nan=290.0)

    for ir, r0 in enumerate(rows):
        for ic, c0 in enumerate(cols):
            # Only compute AMV where clouds colder than 268 K exist
            sub_ir = prev_ir108[r0 : r0 + block_size, c0 : c0 + block_size]
            if np.nanmin(sub_ir) > 268.15:
                continue
            tpl = p_field[r0 : r0 + block_size, c0 : c0 + block_size]
            tpl_std = float(np.std(tpl))
            if tpl_std < 1.2:
                continue
            tpl_norm = tpl - float(np.mean(tpl))

            best_corr = -1.0
            best_dy = 0.0
            best_dx = 0.0
            for dy in range(-max_shift_px, max_shift_px + 1):
                for dx in range(-max_shift_px, max_shift_px + 1):
                    cand = c_field[
                        r0 + dy : r0 + dy + block_size,
                        c0 + dx : c0 + dx + block_size,
                    ]
                    cand_std = float(np.std(cand))
                    if cand_std < 1.2:
                        continue
                    corr = float(
                        np.mean(tpl_norm * (cand - float(np.mean(cand))))
                        / (tpl_std * cand_std)
                    )
                    if corr > best_corr:
                        best_corr = corr
                        best_dy = float(dy)
                        best_dx = float(dx)

            if best_corr >= 0.65:
                amv_dy[ir, ic] = best_dy
                amv_dx[ir, ic] = best_dx
                amv_weight[ir, ic] = np.clip((best_corr - 0.60) / 0.35, 0.0, 1.0)

    # Smooth and upsample AMV field to full resolution
    amv_dy_s = gaussian_filter(amv_dy * amv_weight, sigma=1.0)
    amv_dx_s = gaussian_filter(amv_dx * amv_weight, sigma=1.0)
    w_s = gaussian_filter(amv_weight, sigma=1.0)

    valid_w = w_s > 0.05
    amv_dy_coarse = np.where(valid_w, amv_dy_s / np.maximum(w_s, 1e-6), 0.0)
    amv_dx_coarse = np.where(valid_w, amv_dx_s / np.maximum(w_s, 1e-6), 0.0)
    w_coarse = np.clip(w_s, 0.0, 0.85)

    zoom_y = H / float(nr)
    zoom_x = W / float(nc)
    amv_dy_full = zoom(amv_dy_coarse, (zoom_y, zoom_x), order=1)[:H, :W]
    amv_dx_full = zoom(amv_dx_coarse, (zoom_y, zoom_x), order=1)[:H, :W]
    w_full = np.clip(zoom(w_coarse, (zoom_y, zoom_x), order=1)[:H, :W], 0.0, 0.85)

    # Warm clouds (> 253 K) rely more on 700 hPa NWP steering flow; cold clouds (< 235 K)
    # rely predominantly on upper-tropospheric HRW AMVs.
    cold_factor = np.clip((258.15 - np.nan_to_num(curr_ir108, nan=290.0)) / 25.0, 0.0, 1.0)
    alpha = w_full * (0.4 + 0.6 * cold_factor)

    guess_dy_px = alpha * amv_dy_full + (1.0 - alpha) * nwp_dy_px
    guess_dx_px = alpha * amv_dx_full + (1.0 - alpha) * nwp_dx_px

    return guess_dy_px.astype(np.float32), guess_dx_px.astype(np.float32)
