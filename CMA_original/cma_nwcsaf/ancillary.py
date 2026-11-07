"""Ancillary data handling and autonomous Pure-SEVIRI 7-channel IR estimation for GEO-CMA.

Supports dual operation:
1. Full NWC SAF mode: consumes external NWP fields (`T_surf`, `T_950`, `T_850`, `T_700`,
   `T_500`, `T_tropo`, `ciwv`), land-sea-coast mask, elevation, and snow/ice map.
2. Autonomous Pure-SEVIRI 7-channel IR mode: estimates viewing geometry from lat/lon,
   derives causal clear-sky surface skin temperature (`T_surf`) and tropospheric reference
   levels directly from the 7 SEVIRI thermal/absorption channels (`WV_062`, `WV_073`,
   `IR_087`, `IR_097`, `IR_108`, `IR_120`, `IR_134`).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter, uniform_filter

from .config import NwcSafCmaConfig

_NATURAL_EARTH_LAND_GEOM = None
_LAND_MASK_CACHE: dict[tuple, np.ndarray] = {}


def resolve_geostationary_land_mask(
    lat_deg: np.ndarray | None,
    lon_deg: np.ndarray | None,
    shape: tuple[int, int],
    default_land: bool = True,
) -> np.ndarray:
    """Resolve boolean land mask on the fixed geostationary grid from `(lat_deg, lon_deg)`.

    Uses cached 1:50m Natural Earth continental/island polygons (`shapely.contains_xy`).
    Falls back to `default_land` when coordinates are absent or synthetic.
    """
    if lat_deg is None or lon_deg is None:
        return np.full(shape, default_land, dtype=bool)

    lat_arr = np.asarray(lat_deg, dtype=np.float32)
    lon_arr = np.asarray(lon_deg, dtype=np.float32)
    if lat_arr.shape != shape or lon_arr.shape != shape:
        return np.full(shape, default_land, dtype=bool)

    valid_ll = np.isfinite(lat_arr) & np.isfinite(lon_arr)
    if not np.any(valid_ll):
        return np.full(shape, default_land, dtype=bool)

    lat_span = float(np.nanmax(lat_arr[valid_ll]) - np.nanmin(lat_arr[valid_ll]))
    lon_span = float(np.nanmax(lon_arr[valid_ll]) - np.nanmin(lon_arr[valid_ll]))
    if lat_span < 1.5 and lon_span < 1.5:
        return np.full(shape, default_land, dtype=bool)

    # Fast deterministic cache key for fixed geostationary grids
    v_idx = np.flatnonzero(valid_ll)
    key = (
        shape,
        int(v_idx.size),
        round(float(lat_arr.ravel()[v_idx[0]]), 4),
        round(float(lon_arr.ravel()[v_idx[0]]), 4),
        round(float(lat_arr.ravel()[v_idx[-1]]), 4),
        round(float(lon_arr.ravel()[v_idx[-1]]), 4),
    )
    if key in _LAND_MASK_CACHE:
        return _LAND_MASK_CACHE[key].copy()

    global _NATURAL_EARTH_LAND_GEOM
    try:
        import cartopy.io.shapereader as shpreader
        import shapely
        from shapely.ops import unary_union

        if _NATURAL_EARTH_LAND_GEOM is None:
            shp_path = shpreader.natural_earth(resolution="50m", category="physical", name="land")
            _NATURAL_EARTH_LAND_GEOM = unary_union(list(shpreader.Reader(shp_path).geometries()))
            shapely.prepare(_NATURAL_EARTH_LAND_GEOM)

        safe_lon = np.where(valid_ll, lon_arr, 0.0).astype(np.float64)
        safe_lat = np.where(valid_ll, lat_arr, 0.0).astype(np.float64)
        land_mask = np.asarray(
            shapely.contains_xy(_NATURAL_EARTH_LAND_GEOM, safe_lon, safe_lat) & valid_ll,
            dtype=bool,
        )
        if len(_LAND_MASK_CACHE) >= 8:
            _LAND_MASK_CACHE.clear()
        _LAND_MASK_CACHE[key] = land_mask
        return land_mask.copy()
    except Exception:
        return np.full(shape, default_land, dtype=bool)


@dataclass
class AncillaryData:
    """Resolved ancillary fields on the satellite pixel grid `(H, W)`."""

    sec_zenith: np.ndarray          # Viewing zenith angle secant >= 1.0, shape (H, W)
    sat_zenith_deg: np.ndarray      # Satellite viewing zenith angle (deg), shape (H, W)
    is_land: np.ndarray             # Boolean land mask, shape (H, W)
    is_coast: np.ndarray            # Boolean coastal zone mask, shape (H, W)
    is_sea: np.ndarray              # Boolean open-sea mask, shape (H, W)
    snow_ice_mask: np.ndarray       # 0 = none, 1 = sea ice, 2 = continental snow, shape (H, W)
    elevation_m: np.ndarray         # Terrain elevation (m), shape (H, W)
    t_surf_k: np.ndarray            # Clear-sky surface skin / SST reference (K), shape (H, W)
    t_950hpa_k: np.ndarray          # 950 hPa air temperature (K), shape (H, W)
    t_850hpa_k: np.ndarray          # 850 hPa air temperature (K), shape (H, W)
    t_700hpa_k: np.ndarray          # 700 hPa air temperature (K), shape (H, W)
    t_500hpa_k: np.ndarray          # 500 hPa air temperature (K), shape (H, W)
    t_tropo_k: np.ndarray          # Tropopause temperature reference (K), shape (H, W)
    thermal_inversion: np.ndarray   # Boolean low-level thermal inversion indicator, shape (H, W)
    nwp_available: bool             # True if external NWP T_surf was provided
    solar_zenith_deg: np.ndarray    # Solar zenith angle (deg), default 110.0 (night/IR-only)
    sunglint_mask: np.ndarray       # Boolean sunglint flag, default False in IR-only mode


def compute_satellite_zenith_secant(
    lat_deg: np.ndarray | None,
    lon_deg: np.ndarray | None,
    shape: tuple[int, int],
    sub_lon_deg: float = 9.5,
    max_zenith_deg: float = 78.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute geostationary satellite viewing zenith angle (deg) and secant `sec(theta)`.

    Uses exact spherical/geostationary orbit geometry (orbit radius `R_geo = 42164 km`,
    Earth radius `R_e = 6371 km`). If `lat_deg` or `lon_deg` are omitted, assumes a typical
    mid-latitude European viewing secant of 1.35 (`~42 deg` zenith).
    """
    if lat_deg is None or lon_deg is None:
        zen = np.full(shape, 42.0, dtype=np.float32)
        sec = np.full(shape, 1.0 / np.cos(np.deg2rad(42.0)), dtype=np.float32)
        return zen, sec

    lat_arr = np.asarray(lat_deg, dtype=np.float64)
    lon_arr = np.asarray(lon_deg, dtype=np.float64)
    valid_geo = np.isfinite(lat_arr) & np.isfinite(lon_arr)
    lat_rad = np.deg2rad(np.where(valid_geo, lat_arr, 45.0))
    dlon_rad = np.deg2rad(np.where(valid_geo, lon_arr, sub_lon_deg) - sub_lon_deg)

    cos_beta = np.clip(np.cos(lat_rad) * np.cos(dlon_rad), -1.0, 1.0)
    sin_beta = np.sqrt(np.maximum(0.0, 1.0 - cos_beta * cos_beta))

    r_ratio = 42164.0 / 6371.0  # ~6.618
    denom = np.sqrt(r_ratio * r_ratio + 1.0 - 2.0 * r_ratio * cos_beta)
    sin_zen = np.clip((r_ratio * sin_beta) / np.maximum(denom, 1e-6), 0.0, 1.0)
    zen_deg = np.rad2deg(np.arcsin(sin_zen)).astype(np.float32)
    zen_deg = np.where(valid_geo, zen_deg, max_zenith_deg).astype(np.float32)
    zen_clipped_rad = np.deg2rad(np.minimum(zen_deg, max_zenith_deg))
    sec_zen = (1.0 / np.cos(zen_clipped_rad)).astype(np.float32)
    return zen_deg, sec_zen


def estimate_autonomous_ir_ancillary(
    bt_7ch: dict[str, np.ndarray],
    cfg: NwcSafCmaConfig,
    *,
    lat_deg: np.ndarray | None = None,
    lon_deg: np.ndarray | None = None,
    is_land: np.ndarray | None = None,
    elevation_m: np.ndarray | None = None,
    snow_ice_mask: np.ndarray | None = None,
    t_surf_k: np.ndarray | None = None,
    t_950hpa_k: np.ndarray | None = None,
    t_850hpa_k: np.ndarray | None = None,
    t_700hpa_k: np.ndarray | None = None,
    t_500hpa_k: np.ndarray | None = None,
    t_tropo_k: np.ndarray | None = None,
    prev_t_surf_k: np.ndarray | None = None,
    solar_zenith_deg: np.ndarray | None = None,
    sunglint_mask: np.ndarray | None = None,
) -> AncillaryData:
    """Build complete `AncillaryData`, filling any missing NWP fields via 7-channel IR physics."""
    ir_087 = np.asarray(bt_7ch["IR_087"], dtype=np.float32)
    ir_097 = np.asarray(bt_7ch["IR_097"], dtype=np.float32)
    ir_108 = np.asarray(bt_7ch["IR_108"], dtype=np.float32)
    ir_120 = np.asarray(bt_7ch["IR_120"], dtype=np.float32)
    ir_134 = np.asarray(bt_7ch["IR_134"], dtype=np.float32)
    wv_062 = np.asarray(bt_7ch["WV_062"], dtype=np.float32)
    wv_073 = np.asarray(bt_7ch["WV_073"], dtype=np.float32)

    shape = ir_108.shape
    sat_zen_deg, sec_zen = compute_satellite_zenith_secant(
        lat_deg, lon_deg, shape, cfg.sub_lon_deg, cfg.max_viewing_zenith_deg
    )

    # 1. Surface type (land / coast / sea) — automatically resolved from geostationary
    # (lat_deg, lon_deg) against 1:50m Natural Earth coastline polygons when is_land is omitted.
    if is_land is None:
        land_mask = resolve_geostationary_land_mask(lat_deg, lon_deg, shape, default_land=True)
    else:
        land_mask = np.asarray(is_land, dtype=bool)
    sea_raw = ~land_mask
    if np.any(land_mask) and np.any(sea_raw):
        coast_mask = (binary_dilation(land_mask, iterations=1) & sea_raw) | (
            binary_dilation(sea_raw, iterations=1) & land_mask
        )
    else:
        coast_mask = np.zeros(shape, dtype=bool)
    sea_mask = sea_raw & (~coast_mask)

    elev = (
        np.zeros(shape, dtype=np.float32)
        if elevation_m is None
        else np.asarray(elevation_m, dtype=np.float32)
    )

    # 2. Snow / Ice map (0 = none, 1 = sea ice, 2 = continental snow)
    # Note: As stated in ATBD Section 2.2.2.7 (p. 65), snow is not newly detected by IR alone at
    # night-time; cold winter land inversions without snow must be handled by the GEO-CMA Step 9a
    # spatial cold-inversion filter (`Bit 26`) rather than pre-labeled as snow.
    if snow_ice_mask is not None:
        snow_mask = np.asarray(snow_ice_mask, dtype=np.uint8)
    else:
        snow_mask = np.zeros(shape, dtype=np.uint8)

    # 3. Clear-sky Surface Skin Temperature / SST estimation
    nwp_available = t_surf_k is not None
    if t_surf_k is not None:
        t_surf = np.asarray(t_surf_k, dtype=np.float32).copy()
    else:
        # Split-window water-vapor absorption correction (clamped to clear-sky window scale
        # so humid summer boundary-layer continuum does not inflate T_surf relative to IR_108):
        btd_108_120_pos = np.clip(ir_108 - ir_120, 0.0, 1.8)
        t_skin_inst = ir_108 + 0.65 * btd_108_120_pos * np.minimum(sec_zen, 1.8)

        has_broad_lat = False
        if lat_deg is not None:
            lat_f = np.asarray(lat_deg, dtype=np.float32)
            valid_lat = np.isfinite(lat_f) & np.isfinite(ir_108)
            if np.any(valid_lat) and (float(np.nanmax(lat_f[valid_lat]) - np.nanmin(lat_f[valid_lat])) >= 4.0):
                has_broad_lat = True

        if has_broad_lat:
            lat_f = np.asarray(lat_deg, dtype=np.float32)
            lon_f = (
                np.asarray(lon_deg, dtype=np.float32)
                if lon_deg is not None
                else np.zeros(shape, dtype=np.float32)
            )
            valid_px = np.isfinite(ir_108) & np.isfinite(ir_120) & np.isfinite(lat_f)
            t_scale = np.clip((ir_108 - 268.0) / 22.0, 0.45, 1.0)
            snd_votes = (
                ((ir_108 - wv_062) > (22.0 * t_scale)).astype(np.int32)
                + ((ir_108 - wv_073) > (10.0 * t_scale)).astype(np.int32)
                + ((ir_108 - ir_134) > (11.5 * t_scale)).astype(np.int32)
                + ((ir_108 - ir_097) > (11.0 * t_scale)).astype(np.int32)
            )
            swd_max = np.where(
                land_mask,
                2.1 + 0.16 * np.clip(ir_108 - 278.0, 0.0, 35.0),
                1.9 + 0.08 * np.clip(ir_108 - 278.0, 0.0, 25.0),
            )
            stratus_guard = (
                np.where(land_mask, (ir_108 - ir_087) < 3.6, (ir_108 - ir_087) < 2.2)
                | (ir_108 > 289.0)
            )
            non_cirrus = (
                valid_px
                & (snd_votes >= 3)
                & ((ir_108 - ir_120) < swd_max)
                & ((ir_087 - ir_108) < 0.15)
                & stratus_guard
                & np.where(land_mask, ir_108 > 270.0, ir_108 > 271.5)
            )

            # Fit latitudinal baseline separately over Land, Atlantic/North/Baltic Sea,
            # and Mediterranean/Black/Azov/Caspian Sea basin (separated by Iberia/France)
            t_lat_model = np.full(shape, 288.0, dtype=np.float32)
            sea_med = (
                (~land_mask)
                & (lat_f < 48.0)
                & (lon_f > -5.5)
                & (~((lat_f > 42.5) & (lon_f < 3.5)))
            )
            sea_atl = (~land_mask) & (~sea_med)
            for mask_type, q_pct in [(land_mask, 75.0), (sea_atl, 80.0), (sea_med, 80.0)]:
                m_valid = non_cirrus & mask_type
                if np.any(m_valid):
                    lats_v = lat_f[m_valid]
                    ts_v = t_skin_inst[m_valid]
                    bins = np.arange(np.floor(float(lats_v.min())), np.ceil(float(lats_v.max())) + 2.0, 2.0)
                    bin_centers, bin_p = [], []
                    for b0, b1 in zip(bins[:-1], bins[1:]):
                        in_b = (lats_v >= b0) & (lats_v < b1)
                        if int(in_b.sum()) > 60:
                            bin_centers.append(0.5 * (b0 + b1))
                            bin_p.append(float(np.percentile(ts_v[in_b], q_pct)))
                    if len(bin_centers) >= 2:
                        bin_p_mono = np.minimum.accumulate(np.asarray(bin_p, dtype=np.float64))
                        lat_query = lat_f[mask_type & valid_px].astype(np.float64)
                        interp_vals = np.interp(
                            lat_query,
                            np.asarray(bin_centers, dtype=np.float64),
                            bin_p_mono,
                        )
                        top_lat = float(bin_centers[-1])
                        top_val = float(bin_p_mono[-1])
                        above_top = lat_query > top_lat
                        if np.any(above_top):
                            interp_vals[above_top] = np.maximum(
                                272.5,
                                top_val - 0.35 * (lat_query[above_top] - top_lat),
                            )
                        t_lat_model[mask_type & valid_px] = interp_vals.astype(np.float32)
                    else:
                        t_lat_model[mask_type & valid_px] = float(np.percentile(ts_v, q_pct))

            clear_anchor = non_cirrus & np.where(
                land_mask,
                (t_skin_inst >= t_lat_model - 7.5) | (ir_108 > 284.0),
                (t_skin_inst >= t_lat_model - 3.2),
            )

            t_surf = t_lat_model.copy()
            for is_l, mask_type in [(True, land_mask), (False, ~land_mask)]:
                if not np.any(mask_type):
                    continue
                m_anc = clear_anchor & mask_type
                w = m_anc.astype(np.float32)
                val_w = np.where(m_anc, t_skin_inst, 0.0).astype(np.float32)

                c25 = uniform_filter(w, size=25, mode="nearest")
                s25 = uniform_filter(val_w, size=25, mode="nearest")
                mean25 = s25 / np.maximum(c25, 1e-5)

                c65 = uniform_filter(w, size=65, mode="nearest")
                s65 = uniform_filter(val_w, size=65, mode="nearest")
                mean65 = s65 / np.maximum(c65, 1e-5)

                local_env = np.where(
                    c25 > 0.06,
                    mean25,
                    np.where(c65 > 0.02, mean65, t_lat_model),
                )
                if is_l:
                    surf_est = np.where(
                        m_anc & ((ir_108 > 283.0) | (t_skin_inst >= local_env - 4.5)),
                        np.maximum(t_skin_inst, local_env - 2.5),
                        0.75 * local_env + 0.25 * t_lat_model,
                    )
                else:
                    surf_est = np.where(
                        m_anc & (t_skin_inst >= local_env - 2.2),
                        np.maximum(t_skin_inst, local_env - 1.2),
                        0.70 * local_env + 0.30 * t_lat_model,
                    )
                t_surf[mask_type] = surf_est[mask_type].astype(np.float32)
            likely_clear = clear_anchor & (t_skin_inst >= t_surf - 4.0)
        else:
            # Local patch / synthetic grid mode without broad latitudinal span:
            abs_clear_votes = (
                ((ir_108 - wv_062) > 22.0).astype(np.int32)
                + ((ir_108 - wv_073) > 11.0).astype(np.int32)
                + ((ir_108 - ir_134) > 13.0).astype(np.int32)
                + ((ir_108 - ir_097) > 12.0).astype(np.int32)
            )
            likely_clear = (
                np.isfinite(ir_108)
                & (ir_108 > 262.0)
                & (abs_clear_votes >= 3)
                & ((ir_108 - ir_120) < 1.75)
                & ((ir_087 - ir_108) < 0.35)
            )

            finite_mask = np.isfinite(t_skin_inst)
            if np.any(likely_clear):
                global_ref = float(np.nanpercentile(t_skin_inst[likely_clear], 75.0))
            elif np.any(finite_mask):
                global_ref = float(np.nanpercentile(t_skin_inst[finite_mask], cfg.surface_warm_percentile))
            else:
                global_ref = 288.0
            global_ref = max(global_ref, 275.0)

            clear_f = likely_clear.astype(np.float32)
            weighted_skin = np.where(likely_clear, t_skin_inst, 0.0).astype(np.float32)
            local_sum = uniform_filter(weighted_skin, size=31, mode="nearest")
            local_cnt = uniform_filter(clear_f, size=31, mode="nearest")

            broad_sum = uniform_filter(weighted_skin, size=91, mode="nearest")
            broad_cnt = uniform_filter(clear_f, size=91, mode="nearest")

            t_surf_local = np.where(
                local_cnt > 0.05,
                local_sum / np.maximum(local_cnt, 1e-5),
                np.where(
                    broad_cnt > 0.02,
                    broad_sum / np.maximum(broad_cnt, 1e-5),
                    global_ref,
                ),
            ).astype(np.float32)

            t_surf = np.where(
                likely_clear,
                np.maximum(t_skin_inst, t_surf_local - 3.5),
                np.maximum(t_surf_local, t_skin_inst),
            ).astype(np.float32)

        if prev_t_surf_k is not None:
            prev_s = np.asarray(prev_t_surf_k, dtype=np.float32)
            alpha = cfg.surface_ema_alpha
            t_surf = np.where(
                likely_clear,
                (1.0 - alpha) * prev_s + alpha * np.maximum(t_skin_inst, t_surf),
                0.5 * prev_s + 0.5 * t_surf,
            ).astype(np.float32)

    # 4. Tropospheric temperature profile levels (950, 850, 700, 500 hPa, Tropopause)
    gamma = cfg.standard_lapse_rate_k_per_km  # 6.5 K/km
    t950 = (
        np.asarray(t_950hpa_k, dtype=np.float32)
        if t_950hpa_k is not None
        else (t_surf - 0.5 * gamma).astype(np.float32)
    )
    t850 = (
        np.asarray(t_850hpa_k, dtype=np.float32)
        if t_850hpa_k is not None
        else (t_surf - 1.5 * gamma).astype(np.float32)
    )
    t700 = (
        np.asarray(t_700hpa_k, dtype=np.float32)
        if t_700hpa_k is not None
        else (t_surf - 3.0 * gamma).astype(np.float32)
    )
    t500 = (
        np.asarray(t_500hpa_k, dtype=np.float32)
        if t_500hpa_k is not None
        else (t_surf - 5.6 * gamma).astype(np.float32)
    )

    if t_tropo_k is not None:
        ttropo = np.asarray(t_tropo_k, dtype=np.float32)
    else:
        # Combine WV_062, IR_134, and IR_097 to estimate upper-tropospheric / tropopause floor
        finite_wv = wv_062[np.isfinite(wv_062)]
        p10_wv = float(np.percentile(finite_wv, 10.0)) if finite_wv.size > 0 else 218.0
        ttropo_scalar = float(np.clip(p10_wv - 6.0, 203.0, 224.0))
        ttropo = np.full(shape, ttropo_scalar, dtype=np.float32)

    # 5. Low-level thermal inversion indicator
    # Over land, when T_950 > T_surf or WV_073 - IR_108 is unusually high in clear-like texture
    sd_108 = uniform_filter(ir_108 * ir_108, size=3) - uniform_filter(ir_108, size=3) ** 2
    sd_108 = np.sqrt(np.maximum(0.0, sd_108)).astype(np.float32)
    inv_flag = land_mask & (
        (t950 > t_surf + 0.5)
        | (
            ((wv_073 - ir_108) > cfg.inversion_btd_073_108_thr_k)
            & (sd_108 < cfg.inversion_max_sd_t108_k)
            & ((ir_108 - ir_120) < cfg.inversion_max_btd_108_120_k)
            & (ir_108 > cfg.inversion_min_t108_k)
        )
    )

    sza = (
        np.full(shape, 110.0, dtype=np.float32)
        if solar_zenith_deg is None
        else np.asarray(solar_zenith_deg, dtype=np.float32)
    )
    glint = (
        np.zeros(shape, dtype=bool)
        if sunglint_mask is None
        else np.asarray(sunglint_mask, dtype=bool)
    )

    return AncillaryData(
        sec_zenith=sec_zen,
        sat_zenith_deg=sat_zen_deg,
        is_land=land_mask,
        is_coast=coast_mask,
        is_sea=sea_mask,
        snow_ice_mask=snow_mask,
        elevation_m=elev,
        t_surf_k=t_surf,
        t_950hpa_k=t950,
        t_850hpa_k=t850,
        t_700hpa_k=t700,
        t_500hpa_k=t500,
        t_tropo_k=ttropo,
        thermal_inversion=inv_flag,
        nwp_available=nwp_available,
        solar_zenith_deg=sza,
        sunglint_mask=glint,
    )
