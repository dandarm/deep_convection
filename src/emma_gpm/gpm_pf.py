from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


GMI_FIELDS = [
    "year", "month", "day", "hour", "orbit", "lat", "lon", "npixels",
    "arearain", "volrain", "maxrainrate", "minpct10", "minpct19",
    "minpct37", "minpct85", "minv166", "minv1833", "minv1837", "maxiwp",
    "r_lon", "r_lat", "r_major", "r_minor", "r_orientation", "r_solid",
    "landocean",
]

DPR_FIELD_ALIASES = {
    "orbit": ("ORBIT", "orbit"),
    "group_number": ("GRPNUM", "grpnum"),
    "year": ("YEAR", "year"),
    "month": ("MONTH", "month"),
    "day": ("DAY", "day"),
    "hour": ("HOUR", "hour"),
    "lat": ("LAT", "lat"),
    "lon": ("LON", "lon"),
    "npixels": ("NPIXELS", "npixels"),
    "nstrat_dpr": ("NSTRAT_DPR", "NSTRATDPR", "nstrat_dpr"),
    "nconv_dpr": ("NCONV_DPR", "NCONVDPR", "nconv_dpr"),
    "rainstrat_dpr": ("RAINSTRAT_DPR", "RAINSTRATDPR", "rainstrat_dpr"),
    "rainconv_dpr": ("RAINCONV_DPR", "RAINCONVDPR", "rainconv_dpr"),
    "max_nsz": ("MAXNSZ", "maxnsz"),
    "max_ns_precip": ("MAXNSPRECIP", "maxnsprecip"),
    "max_height": ("MAXHT", "maxht"),
    "max_height_20dbz": ("MAXHT20", "maxht20"),
    "max_height_30dbz": ("MAXHT30", "maxht30"),
    "max_height_40dbz": ("MAXHT40", "maxht40"),
    "min_pct37": ("MIN37PCT", "min37pct"),
    "min_pct85": ("MIN85PCT", "min85pct"),
    "r_lon": ("R_LON", "r_lon"),
    "r_lat": ("R_LAT", "r_lat"),
    "r_major": ("R_MAJOR", "r_major"),
    "r_minor": ("R_MINOR", "r_minor"),
    "r_orientation": ("R_ORIENTATION", "r_orientation"),
    "r_solid": ("R_SOLID", "r_solid"),
    "landocean": ("LANDOCEAN", "landocean"),
}


def _timestamp_from_float_hour(frame: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(
        {"year": frame.year.astype(int), "month": frame.month.astype(int), "day": frame.day.astype(int)},
        utc=True,
    )
    return date + pd.to_timedelta(frame.hour.astype(float), unit="h")


def _domain_mask(lat: np.ndarray, lon: np.ndarray, domain: dict[str, float]) -> np.ndarray:
    return (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= domain["lat_min"])
        & (lat <= domain["lat_max"])
        & (lon >= domain["lon_min"])
        & (lon <= domain["lon_max"])
    )


def read_gmi_pf(files: Iterable[str | Path], domain: dict[str, float]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(map(Path, files)):
        with h5py.File(path) as handle:
            lat = handle["lat"][:]
            lon = handle["lon"][:]
            keep = _domain_mask(lat, lon, domain)
            idx = np.flatnonzero(keep)
            values = {name: handle[name][idx] for name in GMI_FIELDS}
        frame = pd.DataFrame(values)
        frame["source_file"] = path.name
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    out["observation_time"] = _timestamp_from_float_hour(out)
    out["sensor"] = "GMI"
    out["observation_kind"] = "ggpf_precipitation_feature"
    out["feature_uid"] = (
        "GMI:" + out.orbit.astype(int).astype(str) + ":" + out.index.astype(str)
    )
    return out


def _select_hdf4_name(names: set[str], aliases: tuple[str, ...]) -> str | None:
    lower = {name.lower(): name for name in names}
    for alias in aliases:
        if alias in names:
            return alias
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def read_dpr_pf(files: Iterable[str | Path], domain: dict[str, float]) -> pd.DataFrame:
    """Read monthly DPRrpf HDF4 tables.

    pyhdf is intentionally imported lazily so GMI-only preparation remains usable
    when the optional HDF4 runtime is not installed.
    """
    from pyhdf.SD import SD, SDC  # type: ignore

    rows: list[pd.DataFrame] = []
    for path in sorted(map(Path, files)):
        handle = SD(str(path), SDC.READ)
        names = set(handle.datasets())
        resolved: dict[str, str] = {}
        for canonical, aliases in DPR_FIELD_ALIASES.items():
            name = _select_hdf4_name(names, aliases)
            if name is not None:
                resolved[canonical] = name
        for required in ("year", "month", "day", "hour", "orbit", "lat", "lon"):
            if required not in resolved:
                raise KeyError(f"{path}: missing required DPR-PF field {required}; available={sorted(names)}")
        lat = np.asarray(handle.select(resolved["lat"])[:]).squeeze()
        lon = np.asarray(handle.select(resolved["lon"])[:]).squeeze()
        idx = np.flatnonzero(_domain_mask(lat, lon, domain))
        values = {
            canonical: np.asarray(handle.select(source)[:]).squeeze()[idx]
            for canonical, source in resolved.items()
        }
        handle.end()
        frame = pd.DataFrame(values)
        frame["source_file"] = path.name
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    out["observation_time"] = _timestamp_from_float_hour(out)
    out["sensor"] = "DPR"
    out["observation_kind"] = "dprrpf_precipitation_feature"
    out["feature_uid"] = (
        "DPR:" + out.orbit.astype(int).astype(str) + ":" + out.index.astype(str)
    )
    return out

