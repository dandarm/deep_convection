"""API-Free ERA5 NWP Integration for NWC SAF RDT-CW (via Google Cloud ARCO-ERA5 Zarr).

Fetches real hourly ERA5 reanalysis fields over the European/MSG domain directly from the
public Google Cloud ARCO-ERA5 Zarr v2/v3 bucket without API keys or registration, caches
them locally in ``RDT_original/results/era5_cache/``, and computes the exact NWC SAF
RDT-CW NWP auxiliary inputs:

1. WMO Thermal Tropopause Temperature (``T_tropo`` in K), Pressure (``P_tropo`` in hPa),
   and Geometric Altitude (``Z_tropo`` in km) via the WMO (1957) 2 K/km lapse-rate criterion.
2. Thermodynamic Instability Indices:
   - Lifted Index (``LI`` in K)
   - Showalter Index (``SHW`` in K)
   - K-Index (``KI`` in degC)
   - Convective Available Potential Energy (``CAPE`` in J/kg)
   - Binary NWC SAF Convective Mask (``NWPCONV`` in {0, 1})
3. 700 hPa Barotropic Steering Wind ``(U_700, V_700)`` in m/s for first-guess tracking.
4. Surface Skin / 2m Temperature (``T_surf`` in K) for Cloud Type (CT) radiative checks.
5. Column ``T(P)`` / ``Z(P)`` profile interpolation for Cloud-Top Pressure (``CTPress``)
   and satellite parallax correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import map_coordinates
import zarr

from emma_gpm.seviri import RSS_CRS


ARCO_ERA5_BASE_URL = (
    "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3"
)
G0 = 9.80665  # Standard gravity (m/s^2)


@dataclass(frozen=True)
class Era5RdtFields:
    """ERA5 NWP fields on the native 0.25-deg European grid or interpolated to a SEVIRI grid."""

    timestamp_utc: str
    lat: np.ndarray  # 1D (N_lat,) or 2D (H, W)
    lon: np.ndarray  # 1D (N_lon,) or 2D (H, W)
    t_tropo_k: np.ndarray
    p_tropo_hpa: np.ndarray
    z_tropo_km: np.ndarray
    u700_ms: np.ndarray
    v700_ms: np.ndarray
    li_k: np.ndarray
    shw_k: np.ndarray
    ki_c: np.ndarray
    cape_jkg: np.ndarray
    nwpconv_mask: np.ndarray
    t_surf_k: np.ndarray
    levels_hpa: np.ndarray | None = None
    temp_profile_k: np.ndarray | None = None
    z_profile_km: np.ndarray | None = None


def _dewpoint_from_q_p(q_kgkg: np.ndarray, p_hpa: float | np.ndarray) -> np.ndarray:
    """Compute dewpoint temperature (K) from specific humidity (kg/kg) and pressure (hPa)."""
    q = np.clip(np.asarray(q_kgkg, dtype=np.float64), 1e-7, 0.05)
    p = np.asarray(p_hpa, dtype=np.float64)
    # Vapor pressure e in hPa
    e_hpa = np.clip((q * p) / (0.622 + 0.378 * q), 1e-4, 100.0)
    # Bolton (1980) inverse formula for dewpoint in Celsius
    ln_ratio = np.log(e_hpa / 6.112)
    td_c = (243.5 * ln_ratio) / (17.67 - ln_ratio)
    return td_c + 273.15


def _lift_parcel_to_500hpa(
    t0_k: np.ndarray,
    td0_k: np.ndarray,
    p0_hpa: float | np.ndarray,
) -> np.ndarray:
    """Compute parcel temperature (K) lifted from ``p0_hpa`` to 500 hPa via Bolton (1980) moist adiabat."""
    t0 = np.asarray(t0_k, dtype=np.float64)
    td0 = np.minimum(np.asarray(td0_k, dtype=np.float64), t0)
    p0 = np.asarray(p0_hpa, dtype=np.float64)

    # Bolton (1980) LCL temperature (K)
    t_lcl = 56.0 + 1.0 / (1.0 / (td0 - 56.0) + np.log(t0 / td0) / 800.0)
    t_lcl = np.clip(t_lcl, 180.0, t0)

    # Saturation mixing ratio r_s (kg/kg) at (t0, td0)
    td_c = td0 - 273.15
    e_hpa = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    r_kgkg = np.clip(0.622 * e_hpa / np.maximum(p0 - e_hpa, 10.0), 1e-6, 0.04)
    r_gkg = r_kgkg * 1000.0

    # Equivalent potential temperature theta_e (K) via Bolton (1980) Eq. 43
    theta_dl = t0 * ((1000.0 / (p0 - e_hpa)) ** 0.2854) * ((t0 / t_lcl) ** (0.00028 * r_gkg))
    theta_e = theta_dl * np.exp((3.376 / t_lcl - 0.00254) * r_gkg * (1.0 + 0.00081 * r_gkg))

    # Invert theta_e(T_500, p=500 hPa) iteratively (4 Newton/fixed-point steps)
    # Start from dry adiabat at 500 hPa
    t_parcel = np.clip(t0 * ((500.0 / p0) ** 0.2854), 210.0, 290.0)
    for _ in range(6):
        tc = t_parcel - 273.15
        es_500 = 6.112 * np.exp((17.67 * tc) / (tc + 243.5))
        rs_gkg = np.clip(622.0 * es_500 / np.maximum(500.0 - es_500, 10.0), 0.0, 40.0)
        theta_500 = t_parcel * ((1000.0 / (500.0 - es_500)) ** 0.2854)
        theta_e_guess = theta_500 * np.exp(
            (3.376 / t_parcel - 0.00254) * rs_gkg * (1.0 + 0.00081 * rs_gkg)
        )
        # Damped update in temperature
        err = theta_e_guess - theta_e
        t_parcel = np.clip(t_parcel - 0.35 * err, 200.0, 290.0)

    return t_parcel


def compute_wmo_tropopause(
    levels_hpa: np.ndarray,
    temp_k: np.ndarray,
    geopotential_m2s2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute WMO (1957) thermal tropopause (T_tropo, P_tropo, Z_tropo).

    Parameters
    ----------
    levels_hpa : np.ndarray
        1D array of pressure levels (L,) in hPa.
    temp_k : np.ndarray
        3D array (L, H, W) of temperature in K.
    geopotential_m2s2 : np.ndarray
        3D array (L, H, W) of geopotential in m^2/s^2.

    Returns
    -------
    t_tropo_k, p_tropo_hpa, z_tropo_km : tuple of 2D arrays (H, W)
    """
    p_1d = np.asarray(levels_hpa, dtype=np.float64)
    # Sort from bottom (highest pressure, e.g. 500 hPa) to top (lowest pressure, e.g. 70 hPa)
    mask_utls = (p_1d >= 70.0) & (p_1d <= 500.0)
    idx_utls = np.where(mask_utls)[0]
    # Order by decreasing pressure (increasing altitude)
    order = idx_utls[np.argsort(-p_1d[idx_utls])]

    p_lev = p_1d[order]  # e.g. [500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 70]
    T = np.asarray(temp_k[order], dtype=np.float64)  # (K, H, W)
    Z_km = np.asarray(geopotential_m2s2[order], dtype=np.float64) / (G0 * 1000.0)  # (K, H, W)

    K, H, W = T.shape
    # Lapse rate Gamma = -dT/dZ in K/km for each layer k -> k+1
    dz = np.maximum(Z_km[1:] - Z_km[:-1], 0.05)  # (K-1, H, W)
    gamma = -(T[1:] - T[:-1]) / dz  # (K-1, H, W), positive when cooling with height

    # Fallback: cold-point tropopause between 300 hPa and 100 hPa
    cp_mask = (p_lev >= 100.0) & (p_lev <= 300.0)
    cp_indices = np.where(cp_mask)[0]
    cp_sub_idx = np.argmin(T[cp_indices], axis=0)
    cp_idx = cp_indices[cp_sub_idx]

    rows, cols = np.indices((H, W))
    t_tropo = T[cp_idx, rows, cols].copy()
    p_tropo = p_lev[cp_idx].astype(np.float64).copy()
    z_tropo = Z_km[cp_idx, rows, cols].copy()
    found = np.zeros((H, W), dtype=bool)

    # Scan upward from 450 hPa to 100 hPa for the lowest WMO lapse-rate crossing (Gamma <= 2.0 K/km)
    for k in range(K - 2):
        if p_lev[k] > 450.0 or p_lev[k] < 100.0:
            continue
        cond_layer = (~found) & (gamma[k] <= 2.0)
        if not np.any(cond_layer):
            continue
        # Verify average lapse rate over the next ~2 km does not exceed 2.0 K/km
        k_upper = min(k + 2, K - 1)
        dz_2km = np.maximum(Z_km[k_upper] - Z_km[k], 0.1)
        gamma_2km = -(T[k_upper] - T[k]) / dz_2km
        valid = cond_layer & (gamma_2km <= 2.0)
        if not np.any(valid):
            continue

        # Reichler et al. (2003) WMO tropopause interpolation between layer midpoints around knee k:
        if k > 0:
            g_prev = gamma[k - 1]
            g_curr = gamma[k]
            denom = np.maximum(g_prev - g_curr, 0.1)
            frac = np.clip((g_prev - 2.0) / denom, 0.0, 1.0)
            zm_prev = 0.5 * (Z_km[k - 1] + Z_km[k])
            zm_curr = 0.5 * (Z_km[k] + Z_km[k + 1])
            z_interp = zm_prev + frac * (zm_curr - zm_prev)
            # Interpolate temperature and pressure along the two half-layers meeting at level k
            in_upper_half = z_interp >= Z_km[k]
            dz_low = np.maximum(Z_km[k] - Z_km[k - 1], 0.05)
            dz_high = np.maximum(Z_km[k + 1] - Z_km[k], 0.05)
            w_low = np.clip((z_interp - Z_km[k - 1]) / dz_low, 0.0, 1.0)
            w_high = np.clip((z_interp - Z_km[k]) / dz_high, 0.0, 1.0)
            t_interp = np.where(
                in_upper_half,
                T[k] + w_high * (T[k + 1] - T[k]),
                T[k - 1] + w_low * (T[k] - T[k - 1]),
            )
            p_interp = np.where(
                in_upper_half,
                p_lev[k] + w_high * (p_lev[k + 1] - p_lev[k]),
                p_lev[k - 1] + w_low * (p_lev[k] - p_lev[k - 1]),
            )
        else:
            t_interp = T[k]
            p_interp = np.full((H, W), p_lev[k], dtype=np.float64)
            z_interp = Z_km[k]

        t_tropo = np.where(valid, t_interp, t_tropo)
        p_tropo = np.where(valid, p_interp, p_tropo)
        z_tropo = np.where(valid, z_interp, z_tropo)
        found |= valid

    return (
        t_tropo.astype(np.float32),
        p_tropo.astype(np.float32),
        z_tropo.astype(np.float32),
    )


def fetch_era5_europe_hour(
    timestamp: str | pd.Timestamp,
    cache_dir: str | Path,
    *,
    lat_min: float = 28.0,
    lat_max: float = 72.0,
    lon_min: float = -22.0,
    lon_max: float = 42.0,
) -> Era5RdtFields:
    """Fetch (or load from disk cache) hourly ERA5 NWP fields over Europe from ARCO-ERA5."""
    ts = pd.Timestamp(timestamp).tz_localize(None).round("h")
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"era5_rdt_{ts:%Y%m%dT%H%M}.npz"

    if cache_file.exists():
        data = np.load(cache_file)
        return Era5RdtFields(
            timestamp_utc=str(data["timestamp_utc"]),
            lat=data["lat"],
            lon=data["lon"],
            t_tropo_k=data["t_tropo_k"],
            p_tropo_hpa=data["p_tropo_hpa"],
            z_tropo_km=data["z_tropo_km"],
            u700_ms=data["u700_ms"],
            v700_ms=data["v700_ms"],
            li_k=data["li_k"],
            shw_k=data["shw_k"],
            ki_c=data["ki_c"],
            cape_jkg=data["cape_jkg"],
            nwpconv_mask=data["nwpconv_mask"],
            t_surf_k=data["t_surf_k"],
            levels_hpa=data["levels_hpa"] if "levels_hpa" in data else None,
            temp_profile_k=data["temp_profile_k"] if "temp_profile_k" in data else None,
            z_profile_km=data["z_profile_km"] if "z_profile_km" in data else None,
        )

    # Compute exact integer hour index since 1900-01-01 00:00:00
    t_idx = int((ts - pd.Timestamp("1900-01-01 00:00:00")).total_seconds() // 3600)

    lat_full = zarr.open_array(f"{ARCO_ERA5_BASE_URL}/latitude", mode="r")[:]
    lon_full = zarr.open_array(f"{ARCO_ERA5_BASE_URL}/longitude", mode="r")[:]
    levels_full = zarr.open_array(f"{ARCO_ERA5_BASE_URL}/level", mode="r")[:]

    # Subset latitude [lat_max .. lat_min] (ERA5 latitude is 90 to -90)
    lat_mask = (lat_full >= lat_min) & (lat_full <= lat_max)
    lat_idx = np.where(lat_mask)[0]
    lat_sub = lat_full[lat_idx].astype(np.float32)

    # ERA5 longitude is [0 .. 359.75]; reorder to [-180 .. 179.75] and subset [lon_min .. lon_max]
    lon_wrapped = ((lon_full + 180.0) % 360.0) - 180.0
    lon_order = np.argsort(lon_wrapped)
    lon_sorted = lon_wrapped[lon_order]
    lon_mask = (lon_sorted >= lon_min) & (lon_sorted <= lon_max)
    lon_sub_indices = lon_order[lon_mask]
    lon_sub = lon_sorted[lon_mask].astype(np.float32)

    def _subset_3d(arr_3d: np.ndarray) -> np.ndarray:
        return arr_3d[:, lat_idx, :][:, :, lon_sub_indices]

    def _subset_2d(arr_2d: np.ndarray) -> np.ndarray:
        return arr_2d[lat_idx, :][:, lon_sub_indices]

    # Fetch 1 chunk per variable at t_idx
    temp_3d = _subset_3d(
        np.asarray(zarr.open_array(f"{ARCO_ERA5_BASE_URL}/temperature", mode="r")[t_idx], dtype=np.float32)
    )
    geopot_3d = _subset_3d(
        np.asarray(zarr.open_array(f"{ARCO_ERA5_BASE_URL}/geopotential", mode="r")[t_idx], dtype=np.float32)
    )
    u_3d = _subset_3d(
        np.asarray(zarr.open_array(f"{ARCO_ERA5_BASE_URL}/u_component_of_wind", mode="r")[t_idx], dtype=np.float32)
    )
    v_3d = _subset_3d(
        np.asarray(zarr.open_array(f"{ARCO_ERA5_BASE_URL}/v_component_of_wind", mode="r")[t_idx], dtype=np.float32)
    )
    q_3d = _subset_3d(
        np.asarray(zarr.open_array(f"{ARCO_ERA5_BASE_URL}/specific_humidity", mode="r")[t_idx], dtype=np.float32)
    )
    skt_2d = _subset_2d(
        np.asarray(zarr.open_array(f"{ARCO_ERA5_BASE_URL}/skin_temperature", mode="r")[t_idx], dtype=np.float32)
    )
    cape_2d = _subset_2d(
        np.asarray(
            zarr.open_array(f"{ARCO_ERA5_BASE_URL}/convective_available_potential_energy", mode="r")[t_idx],
            dtype=np.float32,
        )
    )

    # 1. WMO Tropopause
    t_tropo_k, p_tropo_hpa, z_tropo_km = compute_wmo_tropopause(levels_full, temp_3d, geopot_3d)

    # 2. 700 hPa steering wind
    idx_700 = int(np.where(levels_full == 700)[0][0])
    u700_ms = u_3d[idx_700].astype(np.float32)
    v700_ms = v_3d[idx_700].astype(np.float32)

    # 3. Thermodynamic instability indices (KI, SHW, LI)
    idx_850 = int(np.where(levels_full == 850)[0][0])
    idx_500 = int(np.where(levels_full == 500)[0][0])
    idx_925 = int(np.where(levels_full == 925)[0][0])

    t850 = temp_3d[idx_850].astype(np.float64)
    t700 = temp_3d[idx_700].astype(np.float64)
    t500 = temp_3d[idx_500].astype(np.float64)
    t925 = temp_3d[idx_925].astype(np.float64)

    td850 = _dewpoint_from_q_p(q_3d[idx_850], 850.0)
    td700 = _dewpoint_from_q_p(q_3d[idx_700], 700.0)
    td925 = _dewpoint_from_q_p(q_3d[idx_925], 925.0)

    # K-Index in Celsius: (T850 - T500) + Td850_C - (T700 - Td700)
    ki_c = ((t850 - t500) + (td850 - 273.15) - (t700 - td700)).astype(np.float32)

    # Showalter Index (SHW): T500 - T_parcel(850 -> 500 hPa)
    t_parcel_850 = _lift_parcel_to_500hpa(t850, td850, 850.0)
    shw_k = (t500 - t_parcel_850).astype(np.float32)

    # Lifted Index (LI): T500 - min(T_parcel(925 -> 500), T_parcel(850 -> 500))
    t_parcel_925 = _lift_parcel_to_500hpa(t925, td925, 925.0)
    li_k = (t500 - np.maximum(t_parcel_925, t_parcel_850)).astype(np.float32)

    # NWC SAF Convective Mask (ATBD Section 3.2.1.2.3 & UM Table 5):
    # Potentially convective airmass if LI <= +2 K, or SHW <= +4 K, or KI >= 25 C, or CAPE >= 80 J/kg
    nwpconv_mask = (
        (li_k <= 2.0) | (shw_k <= 4.0) | (ki_c >= 25.0) | (np.nan_to_num(cape_2d, nan=0.0) >= 80.0)
    ).astype(np.uint8)

    # Keep subset of pressure levels [100..1000 hPa] for CTPress / parallax interpolation
    keep_lev_mask = (levels_full >= 70) & (levels_full <= 1000)
    levels_sub = levels_full[keep_lev_mask].astype(np.float32)
    temp_prof = temp_3d[keep_lev_mask].astype(np.float32)
    z_prof = (geopot_3d[keep_lev_mask] / (G0 * 1000.0)).astype(np.float32)

    np.savez_compressed(
        cache_file,
        timestamp_utc=ts.isoformat(),
        lat=lat_sub,
        lon=lon_sub,
        t_tropo_k=t_tropo_k,
        p_tropo_hpa=p_tropo_hpa,
        z_tropo_km=z_tropo_km,
        u700_ms=u700_ms,
        v700_ms=v700_ms,
        li_k=li_k,
        shw_k=shw_k,
        ki_c=ki_c,
        cape_jkg=cape_2d,
        nwpconv_mask=nwpconv_mask,
        t_surf_k=skt_2d,
        levels_hpa=levels_sub,
        temp_profile_k=temp_prof,
        z_profile_km=z_prof,
    )

    return Era5RdtFields(
        timestamp_utc=ts.isoformat(),
        lat=lat_sub,
        lon=lon_sub,
        t_tropo_k=t_tropo_k,
        p_tropo_hpa=p_tropo_hpa,
        z_tropo_km=z_tropo_km,
        u700_ms=u700_ms,
        v700_ms=v700_ms,
        li_k=li_k,
        shw_k=shw_k,
        ki_c=ki_c,
        cape_jkg=cape_2d,
        nwpconv_mask=nwpconv_mask,
        t_surf_k=skt_2d,
        levels_hpa=levels_sub,
        temp_profile_k=temp_prof,
        z_profile_km=z_prof,
    )


def interpolate_era5_to_seviri_grid(
    era5: Era5RdtFields,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> Era5RdtFields:
    """Interpolate 0.25-deg ERA5 fields onto a 2D SEVIRI geostationary grid (x_m, y_m)."""
    transformer = Transformer.from_crs(RSS_CRS, "EPSG:4326", always_xy=True)
    gx, gy = np.meshgrid(np.asarray(x_m, dtype=np.float64), np.asarray(y_m, dtype=np.float64))
    lon_2d, lat_2d = transformer.transform(gx, gy)

    # Map (lat_2d, lon_2d) to fractional indices in (era5.lat, era5.lon)
    # Note: era5.lat is descending (e.g. 72.0 down to 28.0), era5.lon is ascending (-22.0 to 42.0)
    lat_0 = float(era5.lat[0])
    dlat = float(era5.lat[1] - era5.lat[0])  # negative (-0.25)
    lon_0 = float(era5.lon[0])
    dlon = float(era5.lon[1] - era5.lon[0])  # positive (+0.25)

    row_f = np.clip((lat_2d - lat_0) / dlat, 0.0, len(era5.lat) - 1.0)
    col_f = np.clip((lon_2d - lon_0) / dlon, 0.0, len(era5.lon) - 1.0)
    coords = np.array([row_f, col_f], dtype=np.float64)

    def _interp2d(field: np.ndarray, order: int = 1) -> np.ndarray:
        return map_coordinates(
            np.nan_to_num(field, nan=float(np.nanmedian(field))),
            coords,
            order=order,
            mode="nearest",
        ).astype(np.float32)

    nwpconv_2d = (_interp2d(era5.nwpconv_mask.astype(np.float32), order=1) >= 0.5).astype(np.uint8)

    return Era5RdtFields(
        timestamp_utc=era5.timestamp_utc,
        lat=lat_2d.astype(np.float32),
        lon=lon_2d.astype(np.float32),
        t_tropo_k=_interp2d(era5.t_tropo_k),
        p_tropo_hpa=_interp2d(era5.p_tropo_hpa),
        z_tropo_km=_interp2d(era5.z_tropo_km),
        u700_ms=_interp2d(era5.u700_ms),
        v700_ms=_interp2d(era5.v700_ms),
        li_k=_interp2d(era5.li_k),
        shw_k=_interp2d(era5.shw_k),
        ki_c=_interp2d(era5.ki_c),
        cape_jkg=_interp2d(era5.cape_jkg),
        nwpconv_mask=nwpconv_2d,
        t_surf_k=_interp2d(era5.t_surf_k),
        levels_hpa=era5.levels_hpa,
        temp_profile_k=era5.temp_profile_k,
        z_profile_km=era5.z_profile_km,
    )
