"""NWC SAF RDT-CW GIF archive access and comparison helpers.

The public archive contains rendered contours, not the numerical NetCDF
product.  This module therefore keeps three operations explicit:

* deterministic, resumable discovery of the 15-minute GIF products;
* extraction of the four requested lifecycle contours while masking the
  ``triggering from split`` category;
* reprojection from the local MSG RSS view at 9.5E to the reference MSG view
  at 0E, optionally using a cloud-height hypothesis for differential
  parallax.

No interpolation or image registration is performed per frame.  The archive
rendering is tied to one independently calibrated, fixed affine transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from PIL import Image
from pyproj import CRS, Transformer
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


ARCHIVE_BASE_URL = "https://www.nwcsaf.org"
ARCHIVE_LAYOUTS = {
    "v2021": ("/ARCHIVO_v2021/GIF", "Europe-VISIR"),
    "v2025_msg": ("/ARCHIVO/v2025_MSG/GIF", "REFSYS-VISIR"),
}
SATELLITE_PRIORITY = ("MSG4", "MSG3", "MSG2", "MSG1")

# RGB values used by the official NWCPY RDT visualizer.  The orange phase is
# deliberately retained only as an exclusion mask.
PHASE_RGB = {
    "triggering": (255, 255, 0),
    "triggering_from_split": (255, 165, 0),
    "growing": (255, 0, 0),
    "mature": (128, 0, 128),
    "decay": (0, 0, 255),
}
PHASE_TO_TEACHER_CLASS = {
    "triggering": 2,
    "growing": 3,
    "mature": 4,
    "decay": 5,
}
TARGET_CLASSES = (2, 3, 4, 5)
TARGET_CLASS_NAMES = {2: "triggering", 3: "growing", 4: "mature", 5: "decay"}


def _utc_naive(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("A finite timestamp is required")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def archive_candidate_urls(
    timestamp: object,
    *,
    archive: str = "v2025_msg",
    base_url: str = ARCHIVE_BASE_URL,
) -> tuple[str, ...]:
    """Return deterministic MSG4-to-MSG1 fallbacks for one archive slot."""

    stamp = _utc_naive(timestamp)
    if stamp.second or stamp.microsecond or stamp.minute % 15:
        raise ValueError("NWC SAF MSG archive timestamps must be exact 15-minute slots")
    if archive not in ARCHIVE_LAYOUTS:
        raise ValueError(f"Unknown archive layout {archive!r}")
    prefix, region = ARCHIVE_LAYOUTS[archive]
    day_folder = f"{stamp.year}{stamp.dayofyear:03d}"
    suffix = f"{region}_{stamp:%Y%m%dT%H%M00Z}.rdt.gif"
    return tuple(
        f"{base_url.rstrip('/')}{prefix}/{day_folder}/S_NWC_RDT-CW_{satellite}_{suffix}"
        for satellite in SATELLITE_PRIORITY
    )


def archive_slots(start: object, end: object) -> pd.DatetimeIndex:
    """Return exact quarter-hour slots in the half-open interval [start, end)."""

    first, stop = _utc_naive(start), _utc_naive(end)
    if first >= stop or first.second or first.microsecond or first.minute % 15:
        raise ValueError("Require an aligned, increasing half-open interval")
    return pd.date_range(first, stop, freq="15min", inclusive="left")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_archive_gif(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        if image.format != "GIF":
            raise ValueError(f"Not a GIF product: {path}")
        width, height = image.size
        if width < 1000 or height < 700:
            raise ValueError(f"Unexpected archive rendering size {image.size}: {path}")
        image.seek(0)
        image.convert("RGB").getpixel((width // 2, height // 2))
    return width, height


@dataclass(frozen=True)
class GifGeoReference:
    """Fixed affine transform from MSG-0 geostationary metres to GIF pixels."""

    archive: str
    image_width: int
    image_height: int
    map_bbox_xyxy: tuple[int, int, int, int]
    geos_crs: CRS
    geos_to_pixel: np.ndarray
    calibration: dict

    @classmethod
    def from_json(cls, path: str | Path) -> "GifGeoReference":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        matrix = np.asarray(payload["geos_to_pixel"], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.allclose(matrix[2], [0.0, 0.0, 1.0]):
            raise ValueError("geos_to_pixel must be a 3x3 affine matrix")
        bbox = tuple(map(int, payload["map_bbox_xyxy"]))
        if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("Invalid map bounding box")
        return cls(
            archive=str(payload["archive"]),
            image_width=int(payload["image_size"][0]),
            image_height=int(payload["image_size"][1]),
            map_bbox_xyxy=bbox,
            geos_crs=CRS.from_user_input(payload["geos_crs"]),
            geos_to_pixel=matrix,
            calibration=dict(payload["calibration"]),
        )

    def validate_image(self, image: np.ndarray) -> None:
        if image.shape[:2] != (self.image_height, self.image_width):
            raise ValueError(
                f"GIF shape {image.shape[:2]} differs from calibrated "
                f"{(self.image_height, self.image_width)}"
            )

    def projected_to_pixel(
        self, x_m: np.ndarray, y_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        column = self.geos_to_pixel[0, 0] * x + self.geos_to_pixel[0, 2]
        row = self.geos_to_pixel[1, 1] * y + self.geos_to_pixel[1, 2]
        if self.geos_to_pixel[0, 1] != 0.0:
            column = column + self.geos_to_pixel[0, 1] * y
        if self.geos_to_pixel[1, 0] != 0.0:
            row = row + self.geos_to_pixel[1, 0] * x
        return row, column

    def pixel_to_projected(
        self, row: np.ndarray, column: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        inverse = np.linalg.inv(self.geos_to_pixel)
        column = np.asarray(column, dtype=np.float64)
        row = np.asarray(row, dtype=np.float64)
        x = inverse[0, 0] * column + inverse[0, 1] * row + inverse[0, 2]
        y = inverse[1, 0] * column + inverse[1, 1] * row + inverse[1, 2]
        return x, y


@dataclass(frozen=True)
class ReferencePhaseRaster:
    labels: np.ndarray
    excluded: np.ndarray
    ambiguous: np.ndarray
    line_masks: dict[str, np.ndarray]


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _filled_contours(line: np.ndarray, close_radius: int, min_line_pixels: int) -> np.ndarray:
    closed = ndimage.binary_closing(line, structure=_disk(close_radius))
    filled = ndimage.binary_fill_holes(closed)
    components, count = ndimage.label(filled)
    if not count:
        return np.zeros_like(line, dtype=bool)
    line_counts = np.bincount(components[line].ravel(), minlength=count + 1)
    keep = line_counts >= min_line_pixels
    keep[0] = False
    return keep[components]


def extract_reference_phases(
    image_or_path: np.ndarray | str | Path,
    georef: GifGeoReference,
    *,
    color_tolerance: float = 70.0,
    close_radius: int = 2,
    min_line_pixels: int = 4,
) -> ReferencePhaseRaster:
    """Extract and fill official lifecycle contours from one archive GIF.

    Anti-aliased contour pixels are assigned to their nearest official RGB
    colour.  The legend and title are excluded by the calibrated map bounds.
    Orange split-trigger objects remain an explicit invalid mask.
    """

    if isinstance(image_or_path, (str, Path)):
        with Image.open(image_or_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image_or_path, dtype=np.uint8)
    georef.validate_image(rgb)
    x0, y0, x1, y1 = georef.map_bbox_xyxy
    inside = np.zeros(rgb.shape[:2], dtype=bool)
    inside[y0:y1, x0:x1] = True

    palette_names = tuple(PHASE_RGB)
    palette = np.asarray([PHASE_RGB[name] for name in palette_names], dtype=np.int16)
    delta = rgb.astype(np.int16)[..., None, :] - palette[None, None, :, :]
    squared = np.sum(delta.astype(np.int32) ** 2, axis=-1)
    nearest = np.argmin(squared, axis=-1)
    accepted = np.take_along_axis(squared, nearest[..., None], axis=-1)[..., 0] <= color_tolerance**2

    line_masks: dict[str, np.ndarray] = {}
    filled_masks: dict[str, np.ndarray] = {}
    for index, name in enumerate(palette_names):
        line = inside & accepted & (nearest == index)
        line_masks[name] = line
        filled_masks[name] = _filled_contours(line, close_radius, min_line_pixels)

    requested = tuple(PHASE_TO_TEACHER_CLASS)
    stack = np.stack([filled_masks[name] for name in requested])
    ambiguous = stack.sum(axis=0) > 1
    labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for name in requested:
        labels[filled_masks[name] & ~ambiguous] = PHASE_TO_TEACHER_CLASS[name]
    excluded = filled_masks["triggering_from_split"] | ambiguous | ~inside
    labels[excluded] = 0
    return ReferencePhaseRaster(labels, excluded, ambiguous, line_masks)


def _satellite_ecef(longitude_deg: float, semi_major_m: float, height_m: float) -> np.ndarray:
    longitude = np.radians(longitude_deg)
    radius = semi_major_m + height_m
    return np.asarray([radius * np.cos(longitude), radius * np.sin(longitude), 0.0])


def _ray_ellipsoid_first_intersection(
    origin: np.ndarray,
    target_xyz: np.ndarray,
    semi_major_m: float,
    semi_minor_m: float,
    *,
    minimum_t: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    direction = target_xyz - origin[:, None]
    scale = np.asarray([semi_major_m, semi_major_m, semi_minor_m], dtype=np.float64)[:, None]
    o = origin[:, None] / scale
    d = direction / scale
    aa = np.sum(d * d, axis=0)
    bb = 2.0 * np.sum(o * d, axis=0)
    cc = np.sum(o * o, axis=0) - 1.0
    discriminant = bb * bb - 4.0 * aa * cc
    valid = discriminant >= 0.0
    root = np.sqrt(np.maximum(discriminant, 0.0))
    t0 = (-bb - root) / (2.0 * aa)
    t1 = (-bb + root) / (2.0 * aa)
    candidates = np.stack([t0, t1])
    candidates[candidates < minimum_t] = np.inf
    t = np.min(candidates, axis=0)
    valid &= np.isfinite(t)
    xyz = origin[:, None] + direction * t[None, :]
    xyz[:, ~valid] = np.nan
    return xyz, valid


def differential_view_parallax(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    *,
    source_longitude_deg: float = 9.5,
    target_longitude_deg: float = 0.0,
    cloud_height_km: float = 12.0,
    satellite_height_m: float = 35_785_831.0,
    semi_major_m: float = 6_378_169.0,
    inverse_flattening: float = 295.488065897014,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map apparent surface coordinates between two geostationary views.

    The source pixel defines a satellite ray.  Its intersection with an
    ellipsoid expanded by ``cloud_height_km`` defines a cloud point.  The ray
    from the target satellite through that point is then intersected with the
    reference ellipsoid, yielding the apparent target-view surface position.
    """

    lon = np.asarray(longitude_deg, dtype=np.float64)
    lat = np.asarray(latitude_deg, dtype=np.float64)
    shape = np.broadcast_shapes(lon.shape, lat.shape)
    lon, lat = np.broadcast_to(lon, shape), np.broadcast_to(lat, shape)
    finite = np.isfinite(lon) & np.isfinite(lat)
    if cloud_height_km < 0.0:
        raise ValueError("cloud_height_km must be nonnegative")
    if cloud_height_km == 0.0:
        return lon.copy(), lat.copy(), finite

    semi_minor_m = semi_major_m * (1.0 - 1.0 / inverse_flattening)
    geodetic = CRS.from_proj4(
        f"+proj=longlat +a={semi_major_m} +rf={inverse_flattening} +no_defs"
    )
    geocentric = CRS.from_proj4(
        f"+proj=geocent +a={semi_major_m} +rf={inverse_flattening} +units=m +no_defs"
    )
    to_xyz = Transformer.from_crs(geodetic, geocentric, always_xy=True)
    to_lonlat = Transformer.from_crs(geocentric, geodetic, always_xy=True)
    flat_lon, flat_lat = lon.ravel(), lat.ravel()
    selected = np.flatnonzero(finite.ravel())
    output_lon = np.full(lon.size, np.nan, dtype=np.float64)
    output_lat = np.full(lat.size, np.nan, dtype=np.float64)
    output_valid = np.zeros(lon.size, dtype=bool)
    if not selected.size:
        return output_lon.reshape(shape), output_lat.reshape(shape), output_valid.reshape(shape)
    gx, gy, gz = to_xyz.transform(
        flat_lon[selected], flat_lat[selected], np.zeros(selected.size)
    )
    ground = np.vstack([gx, gy, gz])

    source = _satellite_ecef(source_longitude_deg, semi_major_m, satellite_height_m)
    cloud, source_valid = _ray_ellipsoid_first_intersection(
        source,
        ground,
        semi_major_m + cloud_height_km * 1000.0,
        semi_minor_m + cloud_height_km * 1000.0,
    )
    target = _satellite_ecef(target_longitude_deg, semi_major_m, satellite_height_m)
    apparent, target_valid = _ray_ellipsoid_first_intersection(
        target,
        cloud,
        semi_major_m,
        semi_minor_m,
        minimum_t=1.0,
    )
    out_lon, out_lat, _ = to_lonlat.transform(apparent[0], apparent[1], apparent[2])
    subset_valid = source_valid & target_valid & np.isfinite(out_lon) & np.isfinite(out_lat)
    output_lon[selected[subset_valid]] = np.asarray(out_lon)[subset_valid]
    output_lat[selected[subset_valid]] = np.asarray(out_lat)[subset_valid]
    output_valid[selected[subset_valid]] = True
    return output_lon.reshape(shape), output_lat.reshape(shape), output_valid.reshape(shape)


def rss_grid_to_gif_pixels(
    x_m: np.ndarray,
    y_m: np.ndarray,
    rss_crs: CRS | str,
    georef: GifGeoReference,
    *,
    cloud_height_km: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return GIF row/column coordinates for a north-up RSS grid."""

    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x_m and y_m must be one-dimensional coordinate vectors")
    xx, yy = np.meshgrid(x, y)
    rss_projection = CRS.from_user_input(rss_crs)
    source_longitude = float(rss_projection.to_dict().get("lon_0", 9.5))
    target_longitude = float(georef.geos_crs.to_dict().get("lon_0", 0.0))
    to_lonlat = Transformer.from_crs(rss_projection, "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(xx, yy)
    lon_ref, lat_ref, valid = differential_view_parallax(
        lon,
        lat,
        source_longitude_deg=source_longitude,
        target_longitude_deg=target_longitude,
        cloud_height_km=cloud_height_km,
    )
    to_reference = Transformer.from_crs("EPSG:4326", georef.geos_crs, always_xy=True)
    gx, gy = to_reference.transform(lon_ref, lat_ref)
    row, column = georef.projected_to_pixel(gx, gy)
    x0, y0, x1, y1 = georef.map_bbox_xyxy
    valid &= (
        np.isfinite(row)
        & np.isfinite(column)
        & (column >= x0)
        & (column < x1)
        & (row >= y0)
        & (row < y1)
    )
    return row, column, valid


def sample_reference_to_grid(
    reference: ReferencePhaseRaster,
    row: np.ndarray,
    column: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour sampling preserves categorical contour-derived labels."""

    finite = valid & np.isfinite(row) & np.isfinite(column)
    rr = np.zeros(row.shape, dtype=np.int64)
    cc = np.zeros(column.shape, dtype=np.int64)
    rr[finite] = np.rint(row[finite]).astype(np.int64)
    cc[finite] = np.rint(column[finite]).astype(np.int64)
    inside = (
        finite
        & (rr >= 0)
        & (rr < reference.labels.shape[0])
        & (cc >= 0)
        & (cc < reference.labels.shape[1])
    )
    labels = np.zeros(row.shape, dtype=np.uint8)
    excluded = np.ones(row.shape, dtype=bool)
    labels[inside] = reference.labels[rr[inside], cc[inside]]
    excluded[inside] = reference.excluded[rr[inside], cc[inside]]
    return labels, excluded


def _weighted_sum(mask: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(np.where(mask, weights, 0.0), dtype=np.float64))


def binary_spatial_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    area_km2: np.ndarray,
    *,
    spacing_km: tuple[float, float],
    buffer_km: Iterable[float] = (5.0, 10.0, 20.0),
) -> dict[str, float | int]:
    """Area-weighted binary scores plus symmetric buffered IoU/F1."""

    ref = np.asarray(reference, dtype=bool) & valid
    pred = np.asarray(prediction, dtype=bool) & valid
    area = np.asarray(area_km2, dtype=np.float64)
    intersection = _weighted_sum(ref & pred, area)
    union = _weighted_sum(ref | pred, area)
    pred_area = _weighted_sum(pred, area)
    ref_area = _weighted_sum(ref, area)
    result: dict[str, float | int] = {
        "reference_pixels": int(ref.sum()),
        "prediction_pixels": int(pred.sum()),
        "reference_area_km2": ref_area,
        "prediction_area_km2": pred_area,
        "intersection_area_km2": intersection,
        "iou": intersection / union if union else np.nan,
        "precision": intersection / pred_area if pred_area else np.nan,
        "recall": intersection / ref_area if ref_area else np.nan,
        "area_ratio": pred_area / ref_area if ref_area else np.nan,
    }
    ref_distance = (
        ndimage.distance_transform_edt(~ref, sampling=spacing_km)
        if ref.any()
        else np.full(ref.shape, np.inf)
    )
    pred_distance = (
        ndimage.distance_transform_edt(~pred, sampling=spacing_km)
        if pred.any()
        else np.full(pred.shape, np.inf)
    )
    for radius in buffer_km:
        ref_buffer = valid & (ref_distance <= radius)
        pred_buffer = valid & (pred_distance <= radius)
        buffered_intersection = _weighted_sum(ref_buffer & pred_buffer, area)
        buffered_union = _weighted_sum(ref_buffer | pred_buffer, area)
        tolerant_precision = _weighted_sum(pred & (ref_distance <= radius), area) / pred_area if pred_area else np.nan
        tolerant_recall = _weighted_sum(ref & (pred_distance <= radius), area) / ref_area if ref_area else np.nan
        suffix = f"{int(radius)}km"
        result[f"buffered_iou_{suffix}"] = buffered_intersection / buffered_union if buffered_union else np.nan
        result[f"precision_{suffix}"] = tolerant_precision
        result[f"recall_{suffix}"] = tolerant_recall
        result[f"f1_{suffix}"] = (
            2.0 * tolerant_precision * tolerant_recall / (tolerant_precision + tolerant_recall)
            if np.isfinite(tolerant_precision + tolerant_recall) and tolerant_precision + tolerant_recall > 0
            else np.nan
        )
    return result


def _component_records(mask: np.ndarray, area: np.ndarray, x: np.ndarray, y: np.ndarray) -> list[dict]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    records: list[dict] = []
    for identifier in range(1, count + 1):
        component = labels == identifier
        if not component.any():
            continue
        weights = np.where(component, area, 0.0)
        total = float(weights.sum())
        yy, xx = np.where(component)
        records.append(
            {
                "id": identifier,
                "mask": component,
                "area_km2": total,
                "centroid_x_km": float(np.average(x[xx] / 1000.0, weights=weights[yy, xx])),
                "centroid_y_km": float(np.average(y[yy] / 1000.0, weights=weights[yy, xx])),
            }
        )
    return records


def match_objects(
    reference: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    area_km2: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    spacing_km: tuple[float, float],
    maximum_centroid_km: float = 150.0,
) -> tuple[list[dict], dict[str, int]]:
    """One-to-one component matching with centroid and boundary diagnostics."""

    ref_objects = _component_records(np.asarray(reference, bool) & valid, area_km2, x_m, y_m)
    pred_objects = _component_records(np.asarray(prediction, bool) & valid, area_km2, x_m, y_m)
    if not ref_objects or not pred_objects:
        return [], {
            "reference_objects": len(ref_objects),
            "prediction_objects": len(pred_objects),
            "matched_objects": 0,
        }
    cost = np.full((len(ref_objects), len(pred_objects)), 1e6, dtype=np.float64)
    pair_data: dict[tuple[int, int], tuple[float, float]] = {}
    for i, ref in enumerate(ref_objects):
        for j, pred in enumerate(pred_objects):
            dx = ref["centroid_x_km"] - pred["centroid_x_km"]
            dy = ref["centroid_y_km"] - pred["centroid_y_km"]
            distance = float(np.hypot(dx, dy))
            intersection = _weighted_sum(ref["mask"] & pred["mask"], area_km2)
            union = _weighted_sum(ref["mask"] | pred["mask"], area_km2)
            iou = intersection / union if union else 0.0
            if distance <= maximum_centroid_km or iou > 0.0:
                cost[i, j] = distance / maximum_centroid_km + (1.0 - iou)
                pair_data[i, j] = (distance, iou)
    ref_index, pred_index = linear_sum_assignment(cost)
    rows: list[dict] = []
    for i, j in zip(ref_index, pred_index):
        if cost[i, j] >= 1e6:
            continue
        ref, pred = ref_objects[i], pred_objects[j]
        distance, iou = pair_data[i, j]
        ref_boundary = ref["mask"] & ~ndimage.binary_erosion(ref["mask"])
        pred_boundary = pred["mask"] & ~ndimage.binary_erosion(pred["mask"])
        distance_to_ref = ndimage.distance_transform_edt(~ref_boundary, sampling=spacing_km)
        distance_to_pred = ndimage.distance_transform_edt(~pred_boundary, sampling=spacing_km)
        directed_ref = distance_to_pred[ref_boundary]
        directed_pred = distance_to_ref[pred_boundary]
        combined = np.concatenate([directed_ref, directed_pred])
        rows.append(
            {
                "reference_object": ref["id"],
                "prediction_object": pred["id"],
                "centroid_distance_km": distance,
                "iou": iou,
                "reference_area_km2": ref["area_km2"],
                "prediction_area_km2": pred["area_km2"],
                "area_ratio": pred["area_km2"] / ref["area_km2"] if ref["area_km2"] else np.nan,
                "mean_boundary_distance_km": float(np.mean(combined)) if combined.size else np.nan,
                "hausdorff95_km": float(
                    max(np.percentile(directed_ref, 95), np.percentile(directed_pred, 95))
                ) if directed_ref.size and directed_pred.size else np.nan,
            }
        )
    return rows, {
        "reference_objects": len(ref_objects),
        "prediction_objects": len(pred_objects),
        "matched_objects": len(rows),
    }


__all__ = [
    "ARCHIVE_BASE_URL",
    "PHASE_RGB",
    "PHASE_TO_TEACHER_CLASS",
    "TARGET_CLASSES",
    "TARGET_CLASS_NAMES",
    "GifGeoReference",
    "ReferencePhaseRaster",
    "archive_candidate_urls",
    "archive_slots",
    "binary_spatial_metrics",
    "differential_view_parallax",
    "extract_reference_phases",
    "match_objects",
    "rss_grid_to_gif_pixels",
    "sample_reference_to_grid",
    "sha256_file",
    "validate_archive_gif",
]
