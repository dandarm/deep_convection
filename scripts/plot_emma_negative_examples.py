#!/usr/bin/env python3
"""Visualise hard and matched-background negative crop candidates for EMMA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import binary_dilation, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_emma_heatmap_examples import load_ir108, lookup_for_scene, utc_naive  # noqa: E402
from emma_gpm.seviri import RSS_CRS  # noqa: E402

CROP = 224
BUFFER = 16


def all_mcs_mask(emma_root: Path, time: pd.Timestamp, x, y, lookup_cache):
    path = emma_root / f"{time:%Y}" / f"{time:%m}" / f"tracking_{time:%Y%m%dT%H}.nc"
    with h5py.File(path, "r") as handle:
        ids = np.asarray(handle["mcs_id"][0])
        lat = np.asarray(handle["lat"][:])
        lon = np.asarray(handle["lon"][:])
    key = (len(x), len(y), float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    if key not in lookup_cache:
        lookup_cache[key] = lookup_for_scene(x, y, lat, lon)
    valid, row, col = lookup_cache[key]
    result = np.zeros(valid.shape, dtype=bool)
    result[valid] = ids[row[valid], col[valid]] > 0
    return result


def window_sums(values: np.ndarray, origins_y: np.ndarray, origins_x: np.ndarray) -> np.ndarray:
    integral = np.pad(values.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    y1, x1 = origins_y + CROP, origins_x + CROP
    return integral[y1, x1] - integral[origins_y, x1] - integral[y1, origins_x] + integral[origins_y, origins_x]


def candidate_origins(mask: np.ndarray, image: np.ndarray, reference_y: int, kind: str):
    safe = ~binary_dilation(mask, iterations=BUFFER)
    step = 8
    yy, xx = np.meshgrid(
        np.arange(0, mask.shape[0] - CROP + 1, step),
        np.arange(0, mask.shape[1] - CROP + 1, step),
        indexing="ij",
    )
    origins_y, origins_x = yy.ravel(), xx.ravel()
    clear = window_sums(~safe, origins_y, origins_x) == 0
    finite = window_sums(np.isfinite(image), origins_y, origins_x) >= int(CROP * CROP * 0.98)
    centre_y, centre_x = origins_y + CROP // 2, origins_x + CROP // 2
    distance = distance_transform_edt(~mask)[centre_y, centre_x]
    latitude_match = np.abs(centre_y - reference_y) <= 112
    if kind == "hard_same_scene":
        distance_match = (distance >= CROP // 2 + BUFFER) & (distance <= 300)
    else:
        distance_match = distance >= 400
    selected = clear & finite & latitude_match & distance_match
    if not selected.any():
        # Preserve the no-MCS and validity conditions; only relax latitude.
        selected = clear & finite & distance_match
    return origins_y[selected], origins_x[selected], distance[selected]


def plot_sheet(cases, out: Path) -> None:
    figure, axes = plt.subplots(6, 5, figsize=(15, 18), constrained_layout=True)
    for axis, case in zip(axes.flat, cases):
        axis.imshow(case["image"], cmap="gray_r", vmin=190, vmax=300)
        axis.set_title(
            f"{case['target_time']:%m-%d %H} · {case['kind_short']}\n"
            f"MCS-to-crop-edge {case['distance_km']:.0f} km",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Thirty EMMA-negative crop candidates: IR10.8, no accepted MCS in crop + 16 px buffer", fontsize=13)
    figure.savefig(out, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-manifest", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--emma-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    positives = pd.read_csv(args.positive_manifest)
    if len(positives) != 30:
        raise ValueError("The visual positive manifest must contain 30 rows")

    rng = np.random.default_rng(20260913)
    latlon_to_rss = Transformer.from_crs("EPSG:4326", RSS_CRS, always_xy=True)
    datasets = {}
    lookup_cache = {}
    cases = []
    # The first half are difficult same-scene negatives; the second half are
    # matched-latitude backgrounds from the same seasonal/diurnal strata.
    for number, row in enumerate(positives.itertuples(index=False)):
        kind = "hard_same_scene" if number < 15 else "matched_background"
        target_time = utc_naive(row.target_time)
        seviri_time = utc_naive(row.seviri_time)
        image, x, y = load_ir108(args.seviri_root, seviri_time, datasets)
        mcs_mask = all_mcs_mask(args.emma_root, target_time, x, y, lookup_cache)
        _, reference_y_m = latlon_to_rss.transform(float(row.centroid_lon_gridmean), float(row.centroid_lat_gridmean))
        reference_y = int(np.argmin(np.abs(y - reference_y_m)))
        origins_y, origins_x, distances = candidate_origins(mcs_mask, image, reference_y, kind)
        if not len(origins_y):
            raise RuntimeError(f"No valid {kind} candidate for {target_time}")
        # The centre-to-MCS distance is not interpretable for a 224-pixel crop.
        # Measure the actual gap from any crop pixel to any MCS pixel, then keep
        # hard negatives close and backgrounds genuinely remote.
        distance_field = distance_transform_edt(~mcs_mask)
        edge_distances = np.asarray(
            [distance_field[y0:y0 + CROP, x0:x0 + CROP].min() for y0, x0 in zip(origins_y, origins_x)]
        )
        if kind == "hard_same_scene":
            close = edge_distances <= 96  # 48-km buffer to ~288 km from crop edge
        else:
            close = edge_distances >= 320
        if close.any():
            origins_y, origins_x, distances, edge_distances = (
                origins_y[close], origins_x[close], distances[close], edge_distances[close]
            )
        pick = int(rng.integers(len(origins_y)))
        y0, x0 = int(origins_y[pick]), int(origins_x[pick])
        cases.append(
            {
                "image": image[y0:y0 + CROP, x0:x0 + CROP],
                "target_time": target_time,
                "seviri_time": seviri_time,
                "kind": kind,
                "kind_short": "hard" if kind == "hard_same_scene" else "background",
                "origin_y": y0,
                "origin_x": x0,
                "centre_distance_pixels": float(distances[pick]),
                "distance_pixels": float(edge_distances[pick]),
                "distance_km": float(edge_distances[pick] * 3.0004),
                "valid_fraction": float(np.isfinite(image[y0:y0 + CROP, x0:x0 + CROP]).mean()),
            }
        )
    for dataset in datasets.values():
        dataset.close()
    plot_sheet(cases, args.out_dir / "emma_negative_examples_30.png")
    pd.DataFrame([{key: value for key, value in case.items() if key != "image"} for case in cases]).to_csv(
        args.out_dir / "emma_negative_examples_30.csv", index=False
    )
    print(f"hard={sum(case['kind'] == 'hard_same_scene' for case in cases)} background={sum(case['kind'] == 'matched_background' for case in cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
