#!/usr/bin/env python3
"""Mine cold-cloud crops without an EMMA MCS for borderline inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, distance_transform_edt, label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_emma_heatmap_examples import load_ir108, utc_naive  # noqa: E402
from plot_emma_negative_examples import all_mcs_mask  # noqa: E402

CROP = 224


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-manifest", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--emma-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold-k", type=float, default=241.0)
    parser.add_argument("--buffer-pixels", type=int, default=16)
    parser.add_argument("--per-scene", type=int, default=5)
    args = parser.parse_args()

    manifest = pd.read_csv(args.positive_manifest)
    datasets = {}
    lookup_cache = {}
    rows = []
    seen = set()
    for source in manifest.itertuples(index=False):
        target = utc_naive(source.target_time)
        seviri = utc_naive(source.seviri_time)
        image, x, y = load_ir108(args.seviri_root, seviri, datasets)
        mcs = all_mcs_mask(args.emma_root, target, x, y, lookup_cache)
        blocked = binary_dilation(mcs, iterations=args.buffer_pixels)
        cold = np.isfinite(image) & (image < args.threshold_k) & ~blocked
        components, count = label(cold, structure=np.ones((3, 3), dtype=np.uint8))
        distance = distance_transform_edt(~mcs)
        candidates = []
        for component_id in range(1, count + 1):
            yy, xx = np.where(components == component_id)
            if len(xx) < 24:
                continue
            centre_y = int(round((int(yy.min()) + int(yy.max())) / 2))
            centre_x = int(round((int(xx.min()) + int(xx.max())) / 2))
            y0 = int(np.clip(centre_y - CROP // 2, 0, image.shape[0] - CROP))
            x0 = int(np.clip(centre_x - CROP // 2, 0, image.shape[1] - CROP))
            key = (seviri, y0, x0)
            if key in seen:
                continue
            window_blocked = blocked[y0:y0 + CROP, x0:x0 + CROP]
            window = image[y0:y0 + CROP, x0:x0 + CROP]
            if window_blocked.any() or np.isfinite(window).mean() < 0.98:
                continue
            edge_distance = float(distance[y0:y0 + CROP, x0:x0 + CROP].min())
            cold_fraction = float(np.mean(window < args.threshold_k))
            candidates.append(
                {
                    "target_time": target,
                    "seviri_time": seviri,
                    "kind": "cold_cloud_no_emma",
                    "kind_short": "cold-cloud hard",
                    "origin_y": y0,
                    "origin_x": x0,
                    "centre_distance_pixels": float(distance[centre_y, centre_x]),
                    "distance_pixels": edge_distance,
                    "distance_km": edge_distance * 3.0004,
                    "valid_fraction": float(np.isfinite(window).mean()),
                    "cold_component_pixels": int(len(xx)),
                    "cold_fraction_lt241k": cold_fraction,
                    "ir108_min_k": float(np.nanmin(window)),
                    "ir108_p10_k": float(np.nanquantile(window, 0.10)),
                }
            )
            seen.add(key)
        # Large, cold, coherent non-EMMA regions are the most challenging.
        candidates.sort(
            key=lambda row: (row["cold_component_pixels"], row["cold_fraction_lt241k"]),
            reverse=True,
        )
        rows.extend(candidates[: args.per_scene])

    for dataset in datasets.values():
        dataset.close()
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No cold-cloud negatives found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(
        f"candidates={len(result)} scenes={result.seviri_time.nunique()} "
        f"cold_fraction_median={result.cold_fraction_lt241k.median():.3f} path={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
