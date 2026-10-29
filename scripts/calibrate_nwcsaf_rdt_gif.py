#!/usr/bin/env python3
"""Calibrate one fixed NWC SAF GIF canvas against same-slot MSG IR 12.0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from satpy import Scene
from scipy import ndimage
from skimage import exposure
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from skimage.restoration import inpaint_biharmonic
from skimage.transform import SimilarityTransform, warp


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.nwcsaf_rdt_archive import PHASE_RGB  # noqa: E402
from scripts.scan_nwcsaf_rdt_activity import EXPECTED_LAYOUTS  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ir_gray(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    gray = np.clip((300.0 - np.where(finite, values, 300.0)) / 110.0, 0.0, 1.0) * 255.0
    gray[~finite] = 0.0
    return gray.astype(np.uint8)


def _gif_registration_gray(rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = bbox
    crop = rgb[y0:y1, x0:x1]
    palette = np.asarray(list(PHASE_RGB.values()), dtype=np.int16)
    delta = crop.astype(np.int16)[..., None, :] - palette[None, None, :, :]
    colored = np.min(np.sum(delta.astype(np.int32) ** 2, axis=-1), axis=-1) <= 70**2
    colored = ndimage.binary_dilation(colored, structure=np.ones((5, 5), dtype=bool))
    gray = np.dot(crop[..., :3], np.asarray([0.299, 0.587, 0.114])).astype(np.uint8)
    # Lifecycle contours are explicitly removed before feature registration.
    cleaned = np.clip(
        inpaint_biharmonic(gray.astype(np.float32) / 255.0, colored, channel_axis=None) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    return cleaned, colored.astype(bool)


def calibrate(
    native: Path,
    gif: Path,
    archive: str,
    *,
    channel: str = "IR_120",
) -> tuple[dict, np.ndarray]:
    layout = EXPECTED_LAYOUTS[archive]
    with Image.open(gif) as image:
        if image.size != layout["size"]:
            raise ValueError(f"GIF size {image.size} differs from expected {layout['size']}")
        gif_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gif_gray, contour_mask = _gif_registration_gray(gif_rgb, layout["map_bbox"])
    x0, y0, x1, y1 = layout["map_bbox"]

    scene = Scene(filenames=[str(native)], reader="seviri_l1b_native")
    scene.load([channel], calibration="brightness_temperature")
    data = scene[channel]
    native_values = np.asarray(data.values, dtype=np.float32)
    native_gray = _ir_gray(native_values)
    area = data.attrs["area"]
    source_x, source_y = (np.asarray(values, dtype=np.float64) for values in area.get_proj_vectors())
    if native_gray.shape != (len(source_y), len(source_x)):
        raise ValueError("Native raster and projected vectors disagree")

    native_features = exposure.equalize_adapthist(native_gray, clip_limit=0.02, kernel_size=64)
    gif_features = exposure.equalize_adapthist(gif_gray, clip_limit=0.02, kernel_size=64)
    native_sift = SIFT(upsampling=1, n_octaves=6, c_dog=0.01, c_edge=12)
    gif_sift = SIFT(upsampling=1, n_octaves=6, c_dog=0.01, c_edge=12)
    native_sift.detect_and_extract(native_features)
    gif_sift.detect_and_extract(gif_features)
    matches = match_descriptors(
        native_sift.descriptors,
        gif_sift.descriptors,
        metric="euclidean",
        max_ratio=0.72,
        cross_check=True,
    )
    if len(matches) < 50:
        raise RuntimeError(f"Too few feature matches: {len(matches)}")
    source_points = native_sift.keypoints[matches[:, 0]][:, ::-1].astype(np.float64)
    target_points = gif_sift.keypoints[matches[:, 1]][:, ::-1].astype(np.float64)
    target_points[:, 0] += x0
    target_points[:, 1] += y0
    model, inliers = ransac(
        (source_points, target_points),
        SimilarityTransform,
        min_samples=3,
        residual_threshold=4.0,
        max_trials=20_000,
        stop_probability=0.999,
        rng=0,
    )
    if model is None or inliers is None:
        raise RuntimeError("Robust affine registration failed")
    if int(inliers.sum()) < 40:
        raise RuntimeError(f"Too few affine inliers: {int(inliers.sum())}")
    predicted = model(source_points)
    residual = np.linalg.norm(predicted - target_points, axis=1)[inliers]

    dx = (source_x[-1] - source_x[0]) / (len(source_x) - 1)
    dy = (source_y[-1] - source_y[0]) / (len(source_y) - 1)
    projected_to_index = np.asarray(
        [[1.0 / dx, 0.0, -source_x[0] / dx], [0.0, 1.0 / dy, -source_y[0] / dy], [0, 0, 1]],
        dtype=np.float64,
    )
    index_to_gif = model.params
    geos_to_pixel = index_to_gif @ projected_to_index

    warped = warp(
        native_gray,
        inverse_map=model.inverse,
        output_shape=gif_rgb.shape[:2],
        order=1,
        mode="constant",
        cval=0.0,
        preserve_range=True,
    ).astype(np.uint8)
    gif_full_gray = np.dot(gif_rgb[..., :3], np.asarray([0.299, 0.587, 0.114])).astype(np.uint8)
    yy, xx = np.mgrid[: gif_rgb.shape[0], : gif_rgb.shape[1]]
    valid = (
        (xx >= x0)
        & (xx < x1)
        & (yy >= y0)
        & (yy < y1)
        & (warped > 0)
    )
    local_contours = np.zeros(valid.shape, dtype=bool)
    local_contours[y0:y1, x0:x1] = contour_mask
    valid &= ~local_contours
    correlation = float(np.corrcoef(warped[valid].astype(float), gif_full_gray[valid].astype(float))[0, 1])
    if not np.isfinite(correlation) or correlation < 0.75:
        raise RuntimeError(f"Registration correlation is too low: {correlation}")

    payload = {
        "archive": archive,
        "image_size": list(layout["size"]),
        "map_bbox_xyxy": list(layout["map_bbox"]),
        "geos_crs": area.crs.to_wkt(),
        "geos_to_pixel": geos_to_pixel.tolist(),
        "calibration": {
            "method": "fixed same-slot IR_120 background SIFT registration; lifecycle contours inpainted before matching",
            "timestamp_utc": data.attrs["start_time"].isoformat(),
            "reference_gif": gif.name,
            "reference_gif_sha256": _sha256(gif),
            "primary_source_collection": "EO:EUM:DAT:MSG:HRSEVIRI",
            "primary_product": native.parent.name,
            "background_channel": "IR_120 brightness temperature",
            "pearson_correlation": correlation,
            "sift_candidate_matches": len(matches),
            "sift_inliers": int(inliers.sum()),
            "sift_median_residual_pixels": float(np.median(residual)),
            "sift_p95_residual_pixels": float(np.percentile(residual, 95.0)),
        },
    }
    preview = np.concatenate(
        [
            gif_rgb[y0:y1, x0:x1],
            np.repeat(warped[y0:y1, x0:x1, None], 3, axis=2),
        ],
        axis=1,
    )
    return payload, preview


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--gif", type=Path, required=True)
    parser.add_argument("--archive", choices=sorted(EXPECTED_LAYOUTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--channel", default="IR_120")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload, preview = calibrate(args.native, args.gif, args.archive, channel=args.channel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview).save(args.preview)
    print(json.dumps(payload["calibration"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
