"""Build DeepStorm-like GMI/CloudSat targets from 2B-CSATGPM files.

DEEPSTORM is an algorithm rather than a distributed training table.  The
CloudSat-GPM coincidence product provides the closest public, already-matched
source for reconstructing GMI inputs and CloudSat ice/deep-convection targets.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd


GMI_TB_COLUMNS = (
    "tb_10v_k",
    "tb_10h_k",
    "tb_18v_k",
    "tb_18h_k",
    "tb_23v_k",
    "tb_36v_k",
    "tb_36h_k",
    "tb_89v_k",
    "tb_89h_k",
    "tb_166v_k",
    "tb_166h_k",
    "tb_183_3v_k",
    "tb_183_7v_k",
)

# Variables retained by the published GMI DEEPSTORM models.  They are metadata
# for a later faithful baseline; the probe target builder does not use them as
# labels or as SEVIRI-probe inputs.
DEEPSTORM_GMI_IWP_INPUTS = (
    "elevation",
    "temperature_2m",
    "relative_humidity_400hpa",
    "specific_humidity_700hpa",
    "specific_humidity_975hpa",
    "tb_10v_k",
    "tb_166h_k",
    "tb_183_7v_k",
    "tb_18v_k",
    "tb_36h_k",
    "tb_89v_k",
)
DEEPSTORM_GMI_DC_INPUTS = (
    "relative_humidity_450hpa",
    "specific_humidity_925hpa",
    "specific_humidity_1000hpa",
    "tb_10v_k",
    "tb_166h_k",
    "tb_183_3v_k",
    "tb_183_7v_k",
    "tb_36h_k",
    "tb_36v_k",
    "tb_89v_k",
)


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def dataset_inventory(handle: h5py.File) -> pd.DataFrame:
    """Return paths, shapes and dtypes for every dataset in an HDF5/netCDF4 file."""
    rows: list[dict[str, object]] = []

    def collect(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            rows.append({"path": name, "shape": tuple(obj.shape), "dtype": str(obj.dtype)})

    handle.visititems(collect)
    return pd.DataFrame(rows)


def _find_dataset(
    handle: h5py.File,
    aliases: Sequence[str],
    *,
    group_hints: Sequence[str] = (),
    required: bool = True,
) -> h5py.Dataset | None:
    inventory = dataset_inventory(handle)
    aliases_norm = tuple(_normalise_name(alias) for alias in aliases)
    hints_norm = tuple(_normalise_name(hint) for hint in group_hints)
    candidates: list[tuple[int, str]] = []
    for path in inventory.get("path", []):
        path_norm = _normalise_name(str(path))
        leaf_norm = _normalise_name(str(path).split("/")[-1])
        matched = [alias for alias in aliases_norm if leaf_norm == alias or path_norm.endswith(alias)]
        if not matched:
            continue
        score = 100 * max(len(alias) for alias in matched)
        score += 10 * sum(hint in path_norm for hint in hints_norm)
        score -= str(path).count("/")
        candidates.append((score, str(path)))
    if not candidates:
        if required:
            available = ", ".join(inventory.get("path", []).astype(str).tolist())
            raise KeyError(f"none of {list(aliases)!r} found; available datasets: {available}")
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return handle[candidates[0][1]]


def _attribute_scalar(dataset: h5py.Dataset, names: Sequence[str]) -> object | None:
    lower = {str(key).lower(): key for key in dataset.attrs}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            value = np.asarray(dataset.attrs[key]).squeeze()
            return value.item() if value.ndim == 0 else value
    return None


def _read_numeric(dataset: h5py.Dataset, *, apply_scale: bool = True) -> np.ndarray:
    raw = np.asarray(dataset[()])
    values = raw.astype(float)
    for name in ("_FillValue", "missing_value"):
        fill = _attribute_scalar(dataset, (name,))
        if fill is not None:
            values[raw == fill] = np.nan
    if apply_scale:
        scale = _attribute_scalar(dataset, ("scale_factor",))
        offset = _attribute_scalar(dataset, ("add_offset",))
        if scale is not None:
            values *= float(scale)
        if offset is not None:
            values += float(offset)
    return np.squeeze(values)


def _orient_rays(values: np.ndarray, n_rays: int, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 0:
        raise ValueError(f"{name} is scalar, expected {n_rays} ray values")
    if values.shape[0] == n_rays:
        return values
    if values.ndim >= 2 and values.shape[-1] == n_rays:
        return np.moveaxis(values, -1, 0)
    raise ValueError(f"{name} shape {values.shape} does not contain {n_rays} rays")


def decode_cloud_type(cloud_scenario: np.ndarray) -> np.ndarray:
    """Decode CloudScenario bits 1--4, conditional on the bit-0 valid flag."""
    scenario = np.asarray(cloud_scenario)
    valid = np.isfinite(scenario) & (scenario >= 0)
    encoded = np.where(valid, scenario, 0).astype(np.int64)
    determined = np.bitwise_and(encoded, 0x01) == 1
    decoded = np.bitwise_and(np.right_shift(encoded, 1), 0x0F).astype(float)
    decoded[~valid | ~determined] = np.nan
    return decoded


def _parse_orbit(source_file: str) -> int | pd.NA:
    match = re.search(r"\.(\d{5,6})\.V\d", source_file, flags=re.IGNORECASE)
    return int(match.group(1)) if match else pd.NA


def _to_iwp_kg_m2(values: np.ndarray, units: object | None) -> np.ndarray:
    text = "" if units is None else str(units).lower().replace(" ", "")
    values = values.astype(float)
    values[values < 0] = np.nan
    if "kg" in text:
        return values
    if "g" in text:
        return values / 1000.0
    # 2C-ICE ice_water_path is documented in g m-2.  Keep this explicit
    # fallback for files whose netCDF conversion dropped the units attribute.
    return values / 1000.0


def aggregate_csatgpm_arrays(
    *,
    source_file: str,
    scan_index: np.ndarray,
    pixel_index: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    brightness_temperature: np.ndarray,
    ice_water_path_kg_m2: np.ndarray | None,
    cloud_scenario: np.ndarray | None,
    observation_time: Sequence[object] | None = None,
    domain: dict[str, float] | None = None,
    deep_convection_code: int = 8,
    iwp_threshold_kg_m2: float = 0.5,
) -> pd.DataFrame:
    """Aggregate repeated CloudSat rays to independent GMI S1 footprints."""
    latitude = np.asarray(latitude, dtype=float).reshape(-1)
    n_rays = len(latitude)
    longitude = _orient_rays(np.asarray(longitude, dtype=float), n_rays, "longitude").reshape(-1)
    scan_index = _orient_rays(np.asarray(scan_index), n_rays, "scan_index").reshape(-1)
    pixel_index = _orient_rays(np.asarray(pixel_index), n_rays, "pixel_index").reshape(-1)
    tb = _orient_rays(np.asarray(brightness_temperature, dtype=float), n_rays, "brightness_temperature")
    if tb.ndim != 2 or tb.shape[1] != len(GMI_TB_COLUMNS):
        raise ValueError(f"brightness_temperature must be (ray, 13), got {tb.shape}")

    raw = pd.DataFrame(
        {
            "gmi_scan": scan_index,
            "gmi_pixel": pixel_index,
            "latitude": latitude,
            "longitude": longitude,
        }
    )
    for index, column in enumerate(GMI_TB_COLUMNS):
        raw[column] = tb[:, index]

    if observation_time is None:
        raw["observation_time"] = pd.NaT
    else:
        raw["observation_time"] = pd.to_datetime(observation_time, utc=True, errors="coerce")

    if ice_water_path_kg_m2 is not None:
        iwp = _orient_rays(
            np.asarray(ice_water_path_kg_m2, dtype=float), n_rays, "ice_water_path"
        ).reshape(-1)
        raw["iwp_kg_m2"] = iwp
    else:
        raw["iwp_kg_m2"] = np.nan

    if cloud_scenario is not None:
        cloud_type = decode_cloud_type(
            _orient_rays(np.asarray(cloud_scenario), n_rays, "cloud_scenario")
        )
        if cloud_type.ndim == 1:
            cloud_type = cloud_type[:, None]
        raw["dc_ray"] = np.any(cloud_type == deep_convection_code, axis=1).astype(float)
        raw.loc[np.all(~np.isfinite(cloud_type), axis=1), "dc_ray"] = np.nan
    else:
        raw["dc_ray"] = np.nan

    valid_key = np.isfinite(raw["gmi_scan"]) & np.isfinite(raw["gmi_pixel"])
    raw = raw.loc[valid_key].copy()
    keys = ["gmi_scan", "gmi_pixel"]
    grouped = raw.groupby(keys, sort=False, dropna=False)
    result = grouped.agg(
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        observation_time=("observation_time", "median"),
        n_cloudsat_rays=("latitude", "size"),
        target_iwp_max_kg_m2=("iwp_kg_m2", "max"),
        target_iwp_mean_kg_m2=("iwp_kg_m2", "mean"),
        target_iwp_median_kg_m2=("iwp_kg_m2", "median"),
        target_iwp_valid_rays=("iwp_kg_m2", "count"),
        target_dc_fraction=("dc_ray", "mean"),
        target_dc_valid_rays=("dc_ray", "count"),
    )
    tb_median = grouped[list(GMI_TB_COLUMNS)].median()
    tb_spread = (grouped[list(GMI_TB_COLUMNS)].max() - grouped[list(GMI_TB_COLUMNS)].min()).max(axis=1)
    result = result.join(tb_median)
    result["tb_repeat_max_abs_diff_k"] = tb_spread
    result["target_dc_any"] = (result["target_dc_fraction"] > 0).astype("Int8")
    result.loc[result["target_dc_valid_rays"] == 0, "target_dc_any"] = pd.NA
    result["target_iwp_gt_0p5"] = (
        result["target_iwp_max_kg_m2"] > iwp_threshold_kg_m2
    ).astype("Int8")
    result.loc[result["target_iwp_valid_rays"] == 0, "target_iwp_gt_0p5"] = pd.NA
    result = result.reset_index()

    if domain is not None:
        inside = (
            result.latitude.between(domain["lat_min"], domain["lat_max"])
            & result.longitude.between(domain["lon_min"], domain["lon_max"])
        )
        result = result.loc[inside].copy()

    stem = Path(source_file).stem
    result.insert(0, "source_file", Path(source_file).name)
    result.insert(1, "coincidence_uid", stem)
    result.insert(2, "orbit", _parse_orbit(Path(source_file).name))
    result.insert(3, "split_group", stem)
    result.insert(
        4,
        "sample_uid",
        stem + ":" + result.gmi_scan.astype(int).astype(str) + ":" + result.gmi_pixel.astype(int).astype(str),
    )
    return result.reset_index(drop=True)


def _decode_time(dataset: h5py.Dataset | None, source_file: str, n_rays: int) -> pd.DatetimeIndex:
    if dataset is None:
        return pd.DatetimeIndex([pd.NaT] * n_rays)
    raw = np.asarray(dataset[()]).squeeze()
    raw = _orient_rays(raw, n_rays, "time").reshape(-1)
    if raw.dtype.kind in "SUO":
        return pd.DatetimeIndex(pd.to_datetime(raw.astype(str), utc=True, errors="coerce"))
    units = _attribute_scalar(dataset, ("units",))
    units_text = "" if units is None else str(units)
    match = re.search(r"(seconds|minutes|hours)\s+since\s+(.+)", units_text, re.IGNORECASE)
    if match:
        unit = {"seconds": "s", "minutes": "m", "hours": "h"}[match.group(1).lower()]
        origin = pd.to_datetime(match.group(2), utc=True, errors="coerce")
        return pd.DatetimeIndex(origin + pd.to_timedelta(raw.astype(float), unit=unit))
    if np.nanmedian(raw.astype(float)) > 100_000_000:
        return pd.DatetimeIndex(pd.to_datetime(raw.astype(float), unit="s", utc=True, errors="coerce"))
    file_match = re.search(r"(\d{8})-S(\d{6})", source_file)
    if file_match:
        origin = pd.to_datetime("".join(file_match.groups()), format="%Y%m%d%H%M%S", utc=True)
        return pd.DatetimeIndex(origin + pd.to_timedelta(raw.astype(float), unit="s"))
    return pd.DatetimeIndex([pd.NaT] * n_rays)


def read_csatgpm_file(
    path: str | Path,
    *,
    domain: dict[str, float] | None = None,
    deep_convection_code: int = 8,
    iwp_threshold_kg_m2: float = 0.5,
) -> pd.DataFrame:
    """Read one V04-style 2B-CSATGPM coincidence file."""
    path = Path(path)
    with h5py.File(path) as handle:
        lat_ds = _find_dataset(handle, ("Latitude", "lat"), group_hints=("GMI", "S1"))
        lon_ds = _find_dataset(handle, ("Longitude", "lon"), group_hints=("GMI", "S1"))
        latitude = _read_numeric(lat_ds)  # type: ignore[arg-type]
        n_rays = latitude.reshape(-1).shape[0]
        longitude = _orient_rays(_read_numeric(lon_ds), n_rays, "longitude")  # type: ignore[arg-type]
        combined_indices_ds = _find_dataset(
            handle, ("scan_indices",), group_hints=("1CRGPMGMI",), required=False
        )
        if combined_indices_ds is not None:
            combined_indices = _orient_rays(
                _read_numeric(combined_indices_ds, apply_scale=False), n_rays, "scan_indices"
            )
            if combined_indices.ndim != 2 or combined_indices.shape[1] != 2:
                raise ValueError(f"scan_indices must be (ray, 2), got {combined_indices.shape}")
            scan_index = combined_indices[:, 0]
            pixel_index = combined_indices[:, 1]
        else:
            scan_ds = _find_dataset(handle, ("scan_index_S1", "scanindexs1"), group_hints=("CS", "GMI"))
            pixel_ds = _find_dataset(handle, ("pix_index_S1", "pixel_index_S1", "pixindexs1"), group_hints=("CS", "GMI"))
            scan_index = _read_numeric(scan_ds, apply_scale=False)  # type: ignore[arg-type]
            pixel_index = _read_numeric(pixel_ds, apply_scale=False)  # type: ignore[arg-type]
        tb_ds = _find_dataset(handle, ("Tb", "Tc", "brightness_temperature"), group_hints=("CS", "GMI", "XCAL"))
        iwp_ds = _find_dataset(handle, ("ice_water_path", "IWP"), group_hints=("2CICE",), required=False)
        scenario_ds = _find_dataset(handle, ("cloud_scenario", "CloudScenario"), group_hints=("CLDCLASS",), required=False)
        time_ds = _find_dataset(
            handle, ("scan_time",), group_hints=("GMI",), required=False
        )
        if time_ds is None:
            time_ds = _find_dataset(
                handle,
                ("Profile_time", "profiletime", "time"),
                group_hints=("CS", "GEOPROF"),
                required=False,
            )

        iwp = None
        if iwp_ds is not None:
            iwp = _to_iwp_kg_m2(
                _orient_rays(_read_numeric(iwp_ds), n_rays, "ice_water_path").reshape(-1),
                _attribute_scalar(iwp_ds, ("units",)),
            )
        scenario = None if scenario_ds is None else _orient_rays(_read_numeric(scenario_ds, apply_scale=False), n_rays, "cloud_scenario")
        times = _decode_time(time_ds, path.name, n_rays)

        return aggregate_csatgpm_arrays(
            source_file=path.name,
            scan_index=scan_index,
            pixel_index=pixel_index,
            latitude=latitude,
            longitude=longitude,
            brightness_temperature=_read_numeric(tb_ds),  # type: ignore[arg-type]
            ice_water_path_kg_m2=iwp,
            cloud_scenario=scenario,
            observation_time=times,
            domain=domain,
            deep_convection_code=deep_convection_code,
            iwp_threshold_kg_m2=iwp_threshold_kg_m2,
        )


def read_csatgpm_files(
    files: Iterable[str | Path],
    **kwargs: object,
) -> pd.DataFrame:
    """Read several coincidence files and enforce globally unique samples."""
    frames = [read_csatgpm_file(path, **kwargs) for path in sorted(map(Path, files))]
    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True)
    if output.sample_uid.duplicated().any():
        duplicates = output.loc[output.sample_uid.duplicated(), "sample_uid"].head().tolist()
        raise ValueError(f"duplicate sample_uid values across files: {duplicates}")
    return output
