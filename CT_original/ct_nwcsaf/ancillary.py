"""Viewing geometry, surface masks, and NWP / Pure-SEVIRI 7-channel atmospheric profiles.

Provides both:
1. Full NWP/ancillary mode (accepting ERA5/ECMWF grids for T_surf, T_950, T_850, T_700, T_500,
   T_tropo, TCWV, land/sea mask, snow/ice mask, and lat/lon viewing geometry).
2. Autonomous Pure-SEVIRI 7-channel IR mode (`WV_062, WV_073, IR_087, IR_097, IR_108, IR_120, IR_134`),
   estimating causal surface skin temperature, UTLS/tropopause reference temperature, intermediate
   tropospheric pressure-level temperatures (950, 850, 700, 500 hPa), and total column water vapour
   (TCWV) directly from the 7 infrared bands.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage

from .config import NwcSafCmaCtConfig

# Canonical 7-channel SEVIRI ordering mandated by AGENTS.md
SEVIRI_7CH_NAMES: tuple[str, ...] = (
    "WV_062",
    "WV_073",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
)


@dataclass
class AtmosphericProfileFields:
    """2D atmospheric and surface fields on the satellite image grid `(H, W)`."""

    secant: np.ndarray  # Satellite viewing zenith angle secant 1 / cos(theta_sat)
    land_mask: np.ndarray  # Boolean (True = Land, False = Sea)
    coast_mask: np.ndarray  # Boolean (True = Coastal zone)
    snow_ice_mask: np.ndarray  # Boolean (True = Snow over land or Ice over sea)
    t_surf_k: np.ndarray  # Clear-sky surface skin / SST reference temperature [K]
    t_950_k: np.ndarray  # 950 hPa temperature [K]
    t_850_k: np.ndarray  # 850 hPa temperature [K]
    t_700_k: np.ndarray  # 700 hPa temperature [K]
    t_500_k: np.ndarray  # 500 hPa temperature [K]
    t_tropo_k: np.ndarray  # Tropopause / UTLS reference temperature [K]
    tcwv_kg_m2: np.ndarray  # Total column water vapour [kg/m^2]
    low_level_inversion: np.ndarray  # Boolean flag for low-level thermal inversion
    nwp_available: bool  # True if external NWP fields were supplied
    tropo_available: bool  # True if tropopause temperature is available


def compute_satellite_secant(
    lat_deg: np.ndarray | None,
    lon_deg: np.ndarray | None,
    shape: tuple[int, int],
    cfg: NwcSafCmaCtConfig,
) -> np.ndarray:
    """Compute satellite viewing zenith angle secant `1 / cos(theta_sat)` for geostationary orbit.

    Uses exact spherical/geostationary viewing geometry (orbit radius r_sat = 42164 km,
    Earth radius r_e = 6371 km) relative to `(0 deg N, sub_satellite_lon_deg)`.
    """
    if lat_deg is None or lon_deg is None:
        return np.full(shape, float(cfg.default_secant), dtype=np.float32)

    lat_arr = np.asarray(lat_deg, dtype=np.float64)
    lon_arr = np.asarray(lon_deg, dtype=np.float64)
    valid_geo = np.isfinite(lat_arr) & np.isfinite(lon_arr)
    lat_rad = np.deg2rad(np.where(valid_geo, lat_arr, 45.0))
    dlon_rad = np.deg2rad(
        np.where(valid_geo, lon_arr, float(cfg.sub_satellite_lon_deg)) - float(cfg.sub_satellite_lon_deg)
    )

    cos_gamma = np.clip(np.cos(lat_rad) * np.cos(dlon_rad), -1.0, 1.0)
    sin_gamma = np.sqrt(np.maximum(0.0, 1.0 - cos_gamma * cos_gamma))

    # Ratio of geostationary orbital radius to Earth radius: H_sat / R_e = 42164.0 / 6371.0
    ratio = 42164.0 / 6371.0
    distance = np.sqrt(ratio * ratio + 1.0 - 2.0 * ratio * cos_gamma)
    cos_theta_sat = np.clip((ratio * cos_gamma - 1.0) / np.maximum(distance, 1e-6), 0.1, 1.0)
    _ = sin_gamma
    secant = np.clip(1.0 / cos_theta_sat, 1.0, float(cfg.max_secant))
    secant = np.where(valid_geo, secant, float(cfg.max_secant))
    return secant.astype(np.float32)


def compute_coastal_mask(land_mask: np.ndarray) -> np.ndarray:
    """Identify coastal pixels following ATBD Section 2.2.1.2.7 (morphological dilation/erosion)."""
    land = np.asarray(land_mask, dtype=bool)
    if not np.any(land) or np.all(land):
        return np.zeros_like(land, dtype=bool)
    struct = np.ones((3, 3), dtype=bool)
    dilated = ndimage.binary_dilation(land, structure=struct, iterations=2)
    eroded = ndimage.binary_erosion(land, structure=struct, iterations=2)
    return np.logical_xor(dilated, eroded)


_NATURAL_EARTH_LAND_GEOM = None
_LAND_MASK_CACHE: dict[tuple, np.ndarray] = {}


def resolve_geostationary_land_mask(
    lat_deg: np.ndarray | None,
    lon_deg: np.ndarray | None,
    shape: tuple[int, int],
    default_land: bool = False,
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
        land = np.asarray(
            shapely.contains_xy(_NATURAL_EARTH_LAND_GEOM, safe_lon, safe_lat) & valid_ll,
            dtype=bool,
        )
        if len(_LAND_MASK_CACHE) >= 8:
            _LAND_MASK_CACHE.clear()
        _LAND_MASK_CACHE[key] = land
        return land.copy()
    except Exception:
        return np.full(shape, default_land, dtype=bool)


def extract_seviri_7ch(cube_7ch: np.ndarray) -> dict[str, np.ndarray]:
    """Validate and unpack a `(7, H, W)` SEVIRI brightness temperature tensor in Kelvin."""
    arr = np.asarray(cube_7ch, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 7:
        raise ValueError(
            f"Expected SEVIRI 7-channel tensor of shape (7, H, W), got {arr.shape}."
        )
    return {name: arr[idx] for idx, name in enumerate(SEVIRI_7CH_NAMES)}


def build_atmospheric_profile_fields(
    cube_7ch: np.ndarray,
    cfg: NwcSafCmaCtConfig,
    *,
    running_t_surf_k: np.ndarray | None = None,
    lat_deg: np.ndarray | None = None,
    lon_deg: np.ndarray | None = None,
    land_mask: np.ndarray | None = None,
    snow_ice_mask: np.ndarray | None = None,
    nwp_fields: dict[str, np.ndarray | float] | None = None,
) -> tuple[AtmosphericProfileFields, np.ndarray]:
    """Build full 2D atmospheric profile and surface fields from NWP or Pure-SEVIRI 7-channel IR.

    Returns `(profile_fields, updated_running_t_surf_k)`.
    """
    ch = extract_seviri_7ch(cube_7ch)
    wv062 = ch["WV_062"]
    wv073 = ch["WV_073"]
    ir087 = ch["IR_087"]
    ir097 = ch["IR_097"]
    ir108 = ch["IR_108"]
    ir120 = ch["IR_120"]
    ir134 = ch["IR_134"]

    shape = ir108.shape
    finite_mask = np.isfinite(ir108) & np.isfinite(ir120) & np.isfinite(wv073)

    secant = compute_satellite_secant(lat_deg, lon_deg, shape, cfg)

    if land_mask is not None:
        land = np.asarray(land_mask, dtype=bool)
    else:
        # Automatically resolve fixed geostationary land-sea mask from (lat_deg, lon_deg)
        # against 1:50m Natural Earth coastline polygons (falling back to sea=True on synthetic grids).
        land = resolve_geostationary_land_mask(lat_deg, lon_deg, shape, default_land=False)

    coast = compute_coastal_mask(land)

    if snow_ice_mask is not None:
        snow_ice = np.asarray(snow_ice_mask, dtype=bool)
    else:
        # Per ATBD Section 2.2.1.2.2.15 & Section 2.3 (p. 65), snow/ice is only
        # flagged when an external snow/ice mask or solar/NIR channels are provided.
        # In pure IR mode, cold ground with nocturnal inversion (T7.3 - T10.8 > 0.5 K)
        # is handled by the final spatial filter (ATBD Section 2.2.1.2.7, p. 50).
        snow_ice = np.zeros(shape, dtype=bool)

    # 1. Surface Skin / SST Reference Temperature T_surf
    swd_clear_clamped = np.clip(ir108 - ir120, 0.0, 1.8)
    t_skin_inst = ir108 + 0.65 * swd_clear_clamped * np.minimum(secant, 1.8)

    has_broad_lat = False
    if lat_deg is not None:
        lat_f = np.asarray(lat_deg, dtype=np.float32)
        valid_lat = np.isfinite(lat_f) & finite_mask
        if np.any(valid_lat) and (float(np.nanmax(lat_f[valid_lat]) - np.nanmin(lat_f[valid_lat])) >= 4.0):
            has_broad_lat = True

    if has_broad_lat:
        lat_f = np.asarray(lat_deg, dtype=np.float32)
        lon_f = (
            np.asarray(lon_deg, dtype=np.float32)
            if lon_deg is not None
            else np.zeros(shape, dtype=np.float32)
        )
        valid_px = finite_mask & np.isfinite(lat_f)
        t_scale = np.clip((ir108 - 268.0) / 22.0, 0.45, 1.0)
        snd_votes = (
            ((ir108 - wv062) > (22.0 * t_scale)).astype(np.int32)
            + ((ir108 - wv073) > (10.0 * t_scale)).astype(np.int32)
            + ((ir108 - ir134) > (11.5 * t_scale)).astype(np.int32)
            + ((ir108 - ir097) > (11.0 * t_scale)).astype(np.int32)
        )
        swd_max = np.where(
            land,
            2.1 + 0.16 * np.clip(ir108 - 278.0, 0.0, 35.0),
            1.9 + 0.08 * np.clip(ir108 - 278.0, 0.0, 25.0),
        )
        stratus_guard = (
            np.where(land, (ir108 - ir087) < 3.6, (ir108 - ir087) < 2.2)
            | (ir108 > 289.0)
        )
        non_cirrus = (
            valid_px
            & (snd_votes >= 3)
            & ((ir108 - ir120) < swd_max)
            & ((ir087 - ir108) < 0.15)
            & stratus_guard
            & np.where(land, ir108 > 270.0, ir108 > 271.5)
        )

        # Fit latitudinal baseline separately over Land, Atlantic/North/Baltic Sea,
        # and Mediterranean/Black/Azov/Caspian Sea basin (separated by Iberia/France)
        t_lat_model = np.full(shape, 288.0, dtype=np.float32)
        sea_med = (
            (~land)
            & (lat_f < 48.0)
            & (lon_f > -5.5)
            & (~((lat_f > 42.5) & (lon_f < 3.5)))
        )
        sea_atl = (~land) & (~sea_med)
        for mask_type, q_pct in [(land, 75.0), (sea_atl, 80.0), (sea_med, 80.0)]:
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
                    # Enforce physical monotonic cooling with latitude so high-latitude
                    # extrapolation never curves upward toward the pole
                    bin_p_mono = np.minimum.accumulate(np.asarray(bin_p, dtype=np.float64))
                    lat_query = lat_f[mask_type & valid_px].astype(np.float64)
                    interp_vals = np.interp(
                        lat_query,
                        np.asarray(bin_centers, dtype=np.float64),
                        bin_p_mono,
                    )
                    # Gentle high-latitude climatological lapse (-0.35 K/deg) north of last anchor bin
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
            land,
            (t_skin_inst >= t_lat_model - 7.5) | (ir108 > 284.0),
            (t_skin_inst >= t_lat_model - 3.2),
        )

        t_surf_env = t_lat_model.copy()
        for is_l, mask_type in [(True, land), (False, ~land)]:
            if not np.any(mask_type):
                continue
            m_anc = clear_anchor & mask_type
            w = m_anc.astype(np.float32)
            val_w = np.where(m_anc, t_skin_inst, 0.0).astype(np.float32)

            c25 = ndimage.uniform_filter(w, size=25, mode="nearest")
            s25 = ndimage.uniform_filter(val_w, size=25, mode="nearest")
            mean25 = s25 / np.maximum(c25, 1e-5)

            c65 = ndimage.uniform_filter(w, size=65, mode="nearest")
            s65 = ndimage.uniform_filter(val_w, size=65, mode="nearest")
            mean65 = s65 / np.maximum(c65, 1e-5)

            local_env = np.where(
                c25 > 0.06,
                mean25,
                np.where(c65 > 0.02, mean65, t_lat_model),
            )
            if is_l:
                surf_est = np.where(
                    m_anc & ((ir108 > 283.0) | (t_skin_inst >= local_env - 4.5)),
                    np.maximum(t_skin_inst, local_env - 2.5),
                    0.75 * local_env + 0.25 * t_lat_model,
                )
            else:
                surf_est = np.where(
                    m_anc & (t_skin_inst >= local_env - 2.2),
                    np.maximum(t_skin_inst, local_env - 1.2),
                    0.70 * local_env + 0.30 * t_lat_model,
                )
            t_surf_env[mask_type] = surf_est[mask_type].astype(np.float32)

        if running_t_surf_k is None:
            updated_t_surf = t_surf_env
        else:
            prev_s = np.asarray(running_t_surf_k, dtype=np.float32)
            alpha = float(cfg.surface_ema_alpha)
            likely_clear = clear_anchor & (t_skin_inst >= t_surf_env - 4.0)
            updated_t_surf = np.where(
                likely_clear,
                (1.0 - alpha) * prev_s + alpha * np.maximum(t_skin_inst, t_surf_env),
                0.5 * prev_s + 0.5 * t_surf_env,
            ).astype(np.float32)
    else:
        swd_local_clamped = np.clip(ir108 - ir120, 0.0, 1.8)
        t_skin_local = ir108 + (0.55 + 0.15 * (secant - 1.0)) * swd_local_clamped

        clear_sounding_votes = (
            ((wv062 - ir108) <= -18.0).astype(np.int32)
            + ((wv073 - ir108) <= -8.0).astype(np.int32)
            + ((ir097 - ir108) <= -10.0).astype(np.int32)
            + ((ir134 - ir108) <= -8.0).astype(np.int32)
        )
        clear_sounding_mask = finite_mask & (clear_sounding_votes >= 3) & ((ir108 - ir120) < 2.2)

        if np.any(finite_mask):
            fallback_surf_scalar = float(
                np.nanpercentile(t_skin_local[finite_mask], cfg.surface_quantile_fallback)
            )
            fallback_surf_scalar = float(np.clip(fallback_surf_scalar, 272.0, 312.0))
        else:
            fallback_surf_scalar = 290.0

        clear_vals = np.where(clear_sounding_mask, t_skin_local, 0.0).astype(np.float64)
        clear_cnt = clear_sounding_mask.astype(np.float64)
        mean_clear_num = ndimage.uniform_filter(clear_vals, size=31, mode="nearest")
        mean_clear_den = ndimage.uniform_filter(clear_cnt, size=31, mode="nearest")
        local_clear_mean = np.where(
            mean_clear_den > 0.02,
            mean_clear_num / np.maximum(mean_clear_den, 1e-6),
            fallback_surf_scalar,
        ).astype(np.float32)

        if running_t_surf_k is None:
            t_surf_ref = np.where(
                clear_sounding_mask,
                t_skin_local,
                local_clear_mean,
            ).astype(np.float32)
        else:
            t_surf_ref = np.asarray(running_t_surf_k, dtype=np.float32).copy()

        likely_clear = clear_sounding_mask & (ir108 >= t_surf_ref - 5.0)
        alpha = float(cfg.surface_ema_alpha)
        updated_t_surf = np.where(
            likely_clear,
            (1.0 - alpha) * t_surf_ref + alpha * t_skin_local,
            t_surf_ref,
        ).astype(np.float32)

    # 2. Check if external NWP fields are supplied
    nwp_available = nwp_fields is not None and len(nwp_fields) > 0

    def _get_field(key: str, default: np.ndarray) -> np.ndarray:
        if nwp_fields is None or key not in nwp_fields:
            return default
        val = nwp_fields[key]
        if np.ndim(val) == 0:
            return np.full(shape, float(val), dtype=np.float32)
        return np.asarray(val, dtype=np.float32)

    t_surf_final = _get_field("T_surf", updated_t_surf)

    # 3. Pure-SEVIRI 7-channel IR Atmospheric Sounding Surrogates (when NWP is absent)
    # Standard tropospheric lapse relationships anchored by T_surf, CO2 (IR_134), and WV (WV_073, WV_062):
    # - 950 hPa (~0.5 km): T_surf - 3.5 K
    # - 850 hPa (~1.5 km): T_surf - 9.5 K (modulated by low-level CO2/window contrast)
    # - 700 hPa (~3.0 km): T_surf - 19.0 K
    # - 500 hPa (~5.5 km): T_surf - 35.0 K
    # - Tropopause (~11-12 km): derived from upper-tropospheric WV_062 / cold anvil regime (~214-220 K)
    co2_cooling_anomaly = np.clip((t_surf_final - ir134) - 24.0, -6.0, 6.0)
    default_t_950 = (t_surf_final - 3.5).astype(np.float32)
    default_t_850 = (t_surf_final - 9.5 - 0.15 * co2_cooling_anomaly).astype(np.float32)
    default_t_700 = (t_surf_final - 19.0 - 0.25 * co2_cooling_anomaly).astype(np.float32)
    default_t_500 = (t_surf_final - 35.0 - 0.30 * co2_cooling_anomaly).astype(np.float32)

    if np.any(finite_mask):
        wv062_cold = float(np.nanpercentile(wv062[finite_mask], 5.0))
        ir108_cold = float(np.nanpercentile(ir108[finite_mask], 2.0))
        tropo_est = float(np.clip(min(wv062_cold - 8.0, ir108_cold, 218.0), 202.0, 224.0))
    else:
        tropo_est = 216.0
    default_t_tropo = np.full(shape, tropo_est, dtype=np.float32)

    t_950 = _get_field("T_950", default_t_950)
    t_850 = _get_field("T_850", default_t_850)
    t_700 = _get_field("T_700", default_t_700)
    t_500 = _get_field("T_500", default_t_500)
    t_tropo = _get_field("T_tropo", default_t_tropo)
    tropo_available = True if (nwp_fields is None or "T_tropo" in nwp_fields) else False

    # 4. Total Column Water Vapour (TCWV in kg/m^2)
    # In Pure-IR mode, clear-sky split-window difference (T10.8 - T12.0) and warm T_surf
    # scale with boundary-layer moisture content (Clausius-Clapeyron + continuum absorption).
    swd = np.maximum(0.0, ir108 - ir120)
    default_tcwv = np.clip(
        18.0 + 0.65 * (t_surf_final - 285.0) + 4.5 * np.clip(swd / secant, 0.0, 3.0),
        4.0,
        65.0,
    ).astype(np.float32)
    tcwv = _get_field("TCWV", default_tcwv)

    # 5. Low-level thermal inversion diagnosis
    # NWP criterion: T_950 > T_surf (or T_850 > T_surf)
    # Pure-SEVIRI IR criterion: WV_073 > IR_108 + 0.5 K or IR_134 > IR_108 over cold surface,
    # or strong negative T8.7 - T10.8 in cold low-level air (ATBD Section 2.2.1.2.7 & 3.2.1.2.2.4)
    ir_inversion = (
        finite_mask
        & (
            ((wv073 - ir108) > cfg.cma_cold_inversion_t73_minus_t108_min_k)
            | ((ir108 < t_surf_final) & ((ir087 - ir108) < -1.2 - (secant - 1.0)))
        )
    )
    low_level_inversion = (t_950 > t_surf_final + 0.5) | (t_850 > t_surf_final + 0.5) | ir_inversion
    if nwp_fields is not None and "low_level_inversion" in nwp_fields:
        low_level_inversion = low_level_inversion | np.asarray(
            nwp_fields["low_level_inversion"], dtype=bool
        )

    _ = ir097  # Available in cube_7ch for CMA/CT spectral tests

    profile = AtmosphericProfileFields(
        secant=secant,
        land_mask=land,
        coast_mask=coast,
        snow_ice_mask=snow_ice,
        t_surf_k=t_surf_final,
        t_950_k=t_950,
        t_850_k=t_850,
        t_700_k=t_700,
        t_500_k=t_500,
        t_tropo_k=t_tropo,
        tcwv_kg_m2=tcwv,
        low_level_inversion=low_level_inversion,
        nwp_available=nwp_available,
        tropo_available=tropo_available,
    )
    return profile, updated_t_surf
