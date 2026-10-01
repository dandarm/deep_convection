"""Local processing helpers for native EUMETSAT SEVIRI RSS products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import numpy as np
from pyproj import CRS, Transformer
from satpy import Scene


THERMAL_CHANNELS = (
    "WV_062",
    "WV_073",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
)

RSS_CRS = CRS.from_proj4(
    "+proj=geos +lon_0=9.5 +h=35785831 +a=6378169 "
    "+rf=295.488065897014 +units=m +no_defs"
)
NATIVE_END_TIME_RE = re.compile(r"-(\d{14})\.\d+Z-NA\.nat$")


@dataclass(frozen=True)
class GeosBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def xy_bbox(self) -> tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max


@dataclass(frozen=True)
class NativeSeviriCrop:
    data: np.ndarray
    x: np.ndarray
    y: np.ndarray
    channels: tuple[str, ...]
    start_time: datetime
    end_time: datetime
    calibration: str
    units: str
    source_path: Path


def geographic_bbox_to_geos_bounds(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    *,
    perimeter_samples: int = 1001,
) -> GeosBounds:
    """Project a geographic box and retain extrema over its whole perimeter."""
    if not lon_min < lon_max or not lat_min < lat_max:
        raise ValueError("Expected lon_min < lon_max and lat_min < lat_max")
    if perimeter_samples < 2:
        raise ValueError("perimeter_samples must be at least 2")

    fraction = np.linspace(0.0, 1.0, perimeter_samples)
    horizontal_lons = lon_min + (lon_max - lon_min) * fraction
    vertical_lats = lat_min + (lat_max - lat_min) * fraction
    lons = np.concatenate(
        [
            horizontal_lons,
            horizontal_lons,
            np.full_like(fraction, lon_min),
            np.full_like(fraction, lon_max),
        ]
    )
    lats = np.concatenate(
        [
            np.full_like(fraction, lat_min),
            np.full_like(fraction, lat_max),
            vertical_lats,
            vertical_lats,
        ]
    )
    transformer = Transformer.from_crs("EPSG:4326", RSS_CRS, always_xy=True)
    x, y = transformer.transform(lons, lats)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise ValueError("The requested geographic box is outside the RSS view")
    return GeosBounds(
        x_min=float(np.min(x[finite])),
        x_max=float(np.max(x[finite])),
        y_min=float(np.min(y[finite])),
        y_max=float(np.max(y[finite])),
    )


def native_slot_start_from_name(path: str | Path) -> datetime:
    """Infer the 5-minute slot start from the product end time in its name."""
    match = NATIVE_END_TIME_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Unrecognized native RSS filename: {path}")
    end_time = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    return end_time.replace(minute=(end_time.minute // 5) * 5, second=0)


def load_native_crop(
    path: str | Path,
    bounds: GeosBounds,
    *,
    channels: tuple[str, ...] = THERMAL_CHANNELS,
    calibration: str = "brightness_temperature",
) -> NativeSeviriCrop:
    """Read seven thermal channels and crop them in the native grid.

    DeMeTra requested channels by name and therefore received Satpy's default
    highest calibration (brightness temperature for SEVIRI IR/WV).  Keeping the
    request explicit here preserves that behaviour independently of Satpy's
    dataset-selection defaults.
    """
    if calibration not in {"counts", "radiance", "brightness_temperature"}:
        raise ValueError(f"Unsupported calibration: {calibration}")
    source_path = Path(path)
    scene = Scene(reader="seviri_l1b_native", filenames=[str(source_path)])
    scene.load(list(channels), calibration=calibration)
    cropped = scene.crop(xy_bbox=bounds.xy_bbox)
    arrays = [np.asarray(cropped[channel].values) for channel in channels]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise RuntimeError(f"Channel shape mismatch in {source_path}: {shapes}")
    area = cropped[channels[0]].attrs["area"]
    x, y = area.get_proj_vectors()
    units = str(cropped[channels[0]].attrs.get("units", ""))
    return NativeSeviriCrop(
        data=np.stack(arrays),
        x=np.asarray(x),
        y=np.asarray(y),
        channels=channels,
        start_time=scene.start_time,
        end_time=scene.end_time,
        calibration=calibration,
        units=units,
        source_path=source_path,
    )
