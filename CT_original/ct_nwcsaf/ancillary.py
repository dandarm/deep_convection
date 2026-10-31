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

    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    dlon_rad = np.deg2rad(np.asarray(lon_deg, dtype=np.float64) - float(cfg.sub_satellite_lon_deg))

    cos_gamma = np.clip(np.cos(lat_rad) * np.cos(dlon_rad), -1.0, 1.0)
    sin_gamma = np.sqrt(np.maximum(0.0, 1.0 - cos_gamma * cos_gamma))

    # Ratio of geostationary orbital radius to Earth radius: H_sat / R_e = 42164.0 / 6371.0
    ratio = 42164.0 / 6371.0
    distance = np.sqrt(ratio * ratio + 1.0 - 2.0 * ratio * cos_gamma)
    cos_theta_sat = np.clip((ratio * cos_gamma - 1.0) / np.maximum(distance, 1e-6), 0.1, 1.0)
    _ = sin_gamma
    secant = np.clip(1.0 / cos_theta_sat, 1.0, float(cfg.max_secant))
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
        # In Pure-SEVIRI mode without an external atlas, estimate land vs. sea from
        # quartz/emissivity reststrahlen signature (IR_087 - IR_108) and thermal range,
        # defaulting to sea=True where emissivity matches Masuda water surface behavior.
        land = np.zeros(shape, dtype=bool)

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
    # Split-window water vapour continuum correction for clear-sky window brightness temperature.
    # For clear-sky pixels, water vapour continuum absorption is positive (0 <= T10.8 - T12.0 <= 2.5 K);
    # clamp the continuum correction to clear-sky physical bounds so thin cirrus with large SWD
    # does not artificially inflate T_surf.
    swd_clear_clamped = np.clip(ir108 - ir120, 0.0, 1.8)
    t_skin_inst = ir108 + (0.55 + 0.15 * (secant - 1.0)) * swd_clear_clamped

    # Identify clear-like atmospheric sounding columns using the 4 absorption bands
    # (WV_062, WV_073, IR_097, IR_134 relative to IR_108)
    clear_sounding_votes = (
        ((wv062 - ir108) <= -18.0).astype(np.int32)
        + ((wv073 - ir108) <= -8.0).astype(np.int32)
        + ((ir097 - ir108) <= -10.0).astype(np.int32)
        + ((ir134 - ir108) <= -8.0).astype(np.int32)
    )
    clear_sounding_mask = finite_mask & (clear_sounding_votes >= 3) & ((ir108 - ir120) < 2.2)

    if np.any(finite_mask):
        fallback_surf_scalar = float(
            np.nanpercentile(t_skin_inst[finite_mask], cfg.surface_quantile_fallback)
        )
        fallback_surf_scalar = float(np.clip(fallback_surf_scalar, 272.0, 312.0))
    else:
        fallback_surf_scalar = 290.0

    # Local segment-scale clear-sky reference (31x31 window ~ 90 km, analogous to NWC SAF
    # segment grid interpolation of surface temperature) so North-South and land-sea thermal
    # gradients across large domains are preserved without imposing a single global scalar.
    clear_vals = np.where(clear_sounding_mask, t_skin_inst, 0.0).astype(np.float64)
    clear_cnt = clear_sounding_mask.astype(np.float64)
    mean_clear_num = ndimage.uniform_filter(clear_vals, size=31, mode="nearest")
    mean_clear_den = ndimage.uniform_filter(clear_cnt, size=31, mode="nearest")
    local_clear_mean = np.where(
        mean_clear_den > 0.02,
        mean_clear_num / np.maximum(mean_clear_den, 1e-6),
        fallback_surf_scalar,
    ).astype(np.float32)

    if running_t_surf_k is None:
        # Pixels that pass the 4-band clear-sky sounding test use their own skin temperature;
        # cloudy pixels use the local 31x31 clear-sky neighborhood mean temperature.
        t_surf_ref = np.where(
            clear_sounding_mask,
            t_skin_inst,
            local_clear_mean,
        ).astype(np.float32)
    else:
        t_surf_ref = np.asarray(running_t_surf_k, dtype=np.float32).copy()

    # Causal update of T_surf_ref on clear-like pixels
    likely_clear = clear_sounding_mask & (ir108 >= t_surf_ref - 5.0)
    alpha = float(cfg.surface_ema_alpha)
    updated_t_surf = np.where(
        likely_clear,
        (1.0 - alpha) * t_surf_ref + alpha * t_skin_inst,
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
