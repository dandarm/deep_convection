#!/usr/bin/env python3
"""Render a review sheet of OPERA-verified N1 cold-cloud candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_emma_heatmap_examples import load_ir108, utc_naive  # noqa: E402


CROP = 224


def balanced_selection(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["central_time"] = pd.to_datetime(frame.central_time)
    frame["month"] = frame.central_time.dt.month
    frame["hour_bin"] = (frame.central_time.dt.hour // 6).astype(int)
    frame["review_score"] = (
        np.log1p(frame.central_shield_pixels)
        + 0.5 * frame.opera_good_fraction_median
        + 0.15 * frame.video_matched_frames
    )
    ordered = frame.sort_values("review_score", ascending=False)
    groups = {
        key: group.reset_index(drop=True)
        for key, group in ordered.groupby(["month", "hour_bin"], sort=True)
    }
    selected: list[pd.Series] = []
    depth = 0
    while len(selected) < min(count, len(frame)):
        added = False
        for key in sorted(groups):
            group = groups[key]
            if depth < len(group):
                selected.append(group.iloc[depth])
                added = True
                if len(selected) == min(count, len(frame)):
                    break
        if not added:
            break
        depth += 1
    return pd.DataFrame(selected).sort_values("central_time").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--threshold-k", type=float, default=241.0)
    parser.add_argument("--shield-threshold-k", type=float, default=253.0)
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog)
    n1 = catalog[catalog["class"] == "N1_candidate"].copy()
    if n1.empty:
        raise RuntimeError("No N1 candidates in catalog")
    selected = balanced_selection(n1, args.count)
    datasets = {}
    cases = []
    for row in selected.itertuples(index=False):
        timestamp = utc_naive(row.central_time)
        image, _, _ = load_ir108(args.seviri_root, timestamp, datasets)
        labels, _ = label(
            np.isfinite(image) & (image < args.shield_threshold_k),
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        y0, x0 = int(row.origin_y), int(row.origin_x)
        shield = labels[y0 : y0 + CROP, x0 : x0 + CROP] == int(row.shield_id)
        cases.append(
            (
                row,
                image[y0 : y0 + CROP, x0 : x0 + CROP],
                shield,
            )
        )

    figure, axes = plt.subplots(6, 5, figsize=(16, 18), constrained_layout=True)
    for axis in axes.flat:
        axis.set_visible(False)
    for axis, (row, image, shield) in zip(axes.flat, cases):
        axis.set_visible(True)
        axis.imshow(image, cmap="gray_r", vmin=190, vmax=300)
        axis.contour(shield, levels=[0.5], colors=["#00a6ff"], linewidths=1.2)
        axis.contour(
            shield & (image < args.threshold_k),
            levels=[0.5],
            colors=["#004cff"],
            linewidths=1.0,
        )
        cooling = float(row.video_precentral_min_bt_slope_k_per_min)
        axis.set_title(
            f"{pd.Timestamp(row.central_time):%m-%d %H:%M} · "
            f"scudo {int(row.central_shield_pixels)} px · nucleo {int(row.central_cold_component_pixels)} px\n"
            f"Tmin {row.ir108_min_k:.1f} K · radar max {row.opera_max_dbzh_5slots:.1f} dBZ\n"
            f"trend Tmin pre-centro {cooling:+.3f} K/min",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        "Candidati N1: nube fredda, crop EMMA-vuoto, radar osservabile e quieto\n"
        "azzurro = scudo <253 K · blu = nucleo <241 K · OPERA verificato sullo scudo",
        fontsize=15,
        weight="bold",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_dir / "opera_verified_n1_overview_30.png", dpi=180)
    plt.close(figure)
    selected.to_csv(args.out_dir / "opera_verified_n1_overview_30.csv", index=False)
    for dataset in datasets.values():
        dataset.close()
    print(f"available_n1={len(n1)} plotted={len(selected)} out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
