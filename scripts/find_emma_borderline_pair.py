#!/usr/bin/env python3
"""Rank visually/spectrally similar positive and negative EMMA crops."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_emma_heatmap_examples import (  # noqa: E402
    centered_crop,
    load_case_mask,
    load_ir108,
    utc_naive,
)

CROP = 224
POOL = 8


def load_cube(
    zarr_root: Path,
    timestamp: pd.Timestamp,
    cache,
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
):
    # Reuse the coordinate/time validation from the IR10.8 loader, then retain
    # all seven channels from the same indexed frame.
    _, x, y = load_ir108(zarr_root, timestamp, cache)
    store = zarr_root / f"{timestamp:%Y}" / f"{timestamp:%Y%m%d}.zarr"
    dataset = cache[store]
    indices = np.flatnonzero(
        pd.DatetimeIndex(dataset.time.values) == np.datetime64(timestamp.to_datetime64(), "ns")
    )
    cube = np.asarray(
        dataset.seviri.isel(time=int(indices[0]), y=y_slice, x=x_slice).values,
        dtype=np.float32,
    )
    return cube, x, y, [str(value) for value in dataset.channel.values]


def average_pool(cube: np.ndarray) -> np.ndarray:
    blocks = cube.reshape(cube.shape[0], 28, POOL, 28, POOL)
    finite = np.isfinite(blocks)
    count = finite.sum(axis=(2, 4))
    total = np.nansum(blocks, axis=(2, 4))
    result = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=result, where=count > 0)
    return result


def overlap_slices(size: int, shift: int):
    if shift >= 0:
        return slice(shift, size), slice(0, size - shift)
    return slice(0, size + shift), slice(-shift, size)


def translation_tolerant_rmse(a: np.ndarray, b: np.ndarray, max_shift: int = 3):
    best = (np.inf, 0, 0)
    for dy in range(-max_shift, max_shift + 1):
        ay, by = overlap_slices(a.shape[1], dy)
        for dx in range(-max_shift, max_shift + 1):
            ax, bx = overlap_slices(a.shape[2], dx)
            error = float(np.sqrt(np.nanmean((a[:, ay, ax] - b[:, by, bx]) ** 2)))
            if error < best[0]:
                best = (error, dy, dx)
    return best


def distribution_features(cube: np.ndarray) -> np.ndarray:
    values = []
    for channel in cube:
        finite = channel[np.isfinite(channel)]
        values.extend(np.quantile(finite, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]))
        gy, gx = np.gradient(channel)
        gradient = np.hypot(gx, gy)
        values.extend([np.nanmean(gradient), np.nanquantile(gradient, 0.90)])
    return np.asarray(values, dtype=np.float64)


def load_samples(args):
    positives = pd.read_csv(args.positive_manifest)
    negatives = pd.read_csv(args.negative_manifest)
    datasets = {}
    lookup_cache = {}
    positive_samples = []
    negative_samples = []
    channel_names = None

    for row in positives.itertuples(index=False):
        target = utc_naive(row.target_time)
        timestamp = utc_naive(row.seviri_time)
        _, x, y = load_ir108(args.seviri_root, timestamp, datasets)
        mask = load_case_mask(args.emma_root, target, int(row.mcs_id), x, y, lookup_cache)
        ys, xs, contained = centered_crop(mask)
        cube, _, _, channel_names = load_cube(
            args.seviri_root, timestamp, datasets, ys, xs
        )
        positive_samples.append(
            {
                "sample_id": f"positive_{row.track_uid}_{target:%Y%m%dT%H}",
                "kind": "positive",
                "target_time": target,
                "seviri_time": timestamp,
                "track_uid": row.track_uid,
                "mcs_id": int(row.mcs_id),
                "cube": cube,
                "mask": mask[ys, xs],
                "contained": contained,
            }
        )
    for number, row in enumerate(negatives.itertuples(index=False)):
        target = utc_naive(row.target_time)
        timestamp = utc_naive(row.seviri_time)
        y0, x0 = int(row.origin_y), int(row.origin_x)
        cube, _, _, channel_names = load_cube(
            args.seviri_root,
            timestamp,
            datasets,
            slice(y0, y0 + CROP),
            slice(x0, x0 + CROP),
        )
        negative_samples.append(
            {
                "sample_id": f"negative_{number:02d}_{row.kind}_{target:%Y%m%dT%H}",
                "kind": row.kind,
                "target_time": target,
                "seviri_time": timestamp,
                "track_uid": "",
                "mcs_id": 0,
                "cube": cube,
                "mask": np.zeros((CROP, CROP), dtype=bool),
                "contained": True,
                "origin_y": y0,
                "origin_x": x0,
                "distance_km": float(row.distance_km),
            }
        )
    for dataset in datasets.values():
        dataset.close()
    return positive_samples, negative_samples, channel_names


def rank_pairs(positives, negatives):
    samples = positives + negatives
    pooled = np.stack([average_pool(sample["cube"]) for sample in samples])
    channel_median = np.nanmedian(pooled, axis=(0, 2, 3), keepdims=True)
    channel_scale = np.nanquantile(pooled, 0.90, axis=(0, 2, 3), keepdims=True) - np.nanquantile(
        pooled, 0.10, axis=(0, 2, 3), keepdims=True
    )
    normalized = (pooled - channel_median) / np.maximum(channel_scale, 1e-3)
    descriptors = np.stack([distribution_features(sample["cube"]) for sample in samples])
    descriptor_median = np.median(descriptors, axis=0)
    descriptor_scale = np.quantile(descriptors, 0.90, axis=0) - np.quantile(descriptors, 0.10, axis=0)
    descriptors = (descriptors - descriptor_median) / np.maximum(descriptor_scale, 1e-3)

    rows = []
    for p_index, positive in enumerate(positives):
        for n_index, negative in enumerate(negatives):
            spatial = float(
                np.sqrt(
                    np.nanmean(
                        (normalized[p_index] - normalized[len(positives) + n_index]) ** 2
                    )
                )
            )
            distribution = float(
                np.sqrt(np.mean((descriptors[p_index] - descriptors[len(positives) + n_index]) ** 2))
            )
            rows.append(
                {
                    "positive_index": p_index,
                    "negative_index": n_index,
                    "positive_id": positive["sample_id"],
                    "negative_id": negative["sample_id"],
                    "negative_kind": negative["kind"],
                    "spatial_rmse": spatial,
                    "distribution_rmse": distribution,
                    "best_shift_y_28px": 0,
                    "best_shift_x_28px": 0,
                }
            )
    ranked = pd.DataFrame(rows)
    # Use the unshifted distance as a fast prefilter, then do the expensive
    # translation-tolerant comparison only for the strongest candidates.
    for column in ("spatial_rmse", "distribution_rmse"):
        low, high = ranked[column].quantile([0.05, 0.95])
        ranked[column + "_normalized"] = (ranked[column] - low) / max(high - low, 1e-6)
    ranked["combined_distance"] = (
        0.70 * ranked.spatial_rmse_normalized + 0.30 * ranked.distribution_rmse_normalized
    )
    ranked = ranked.nsmallest(min(500, len(ranked)), "combined_distance").copy()
    for index, row in ranked.iterrows():
        spatial, dy, dx = translation_tolerant_rmse(
            normalized[int(row.positive_index)],
            normalized[len(positives) + int(row.negative_index)],
        )
        ranked.at[index, "spatial_rmse"] = spatial
        ranked.at[index, "best_shift_y_28px"] = dy
        ranked.at[index, "best_shift_x_28px"] = dx
    for column in ("spatial_rmse", "distribution_rmse"):
        low, high = ranked[column].quantile([0.05, 0.95])
        ranked[column + "_normalized"] = (ranked[column] - low) / max(high - low, 1e-6)
    ranked["combined_distance"] = (
        0.70 * ranked.spatial_rmse_normalized + 0.30 * ranked.distribution_rmse_normalized
    )
    return ranked.sort_values("combined_distance").reset_index(drop=True)


def plot_candidates(ranked, positives, negatives, ir_index: int, out: Path):
    figure, axes = plt.subplots(8, 2, figsize=(7, 25), constrained_layout=True)
    for rank, row in ranked.head(8).iterrows():
        positive = positives[int(row.positive_index)]
        negative = negatives[int(row.negative_index)]
        for axis, sample, label in (
            (axes[rank, 0], positive, "positive"),
            (axes[rank, 1], negative, negative["kind"].replace("_", " ")),
        ):
            axis.imshow(sample["cube"][ir_index], cmap="gray_r", vmin=190, vmax=300)
            if label == "positive":
                axis.contour(sample["mask"], levels=[0.5], colors="#ec4899", linewidths=0.9)
            axis.set_title(f"rank {rank + 1} · {label}\n{sample['target_time']:%Y-%m-%d %H:%MZ}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[rank, 1].text(
            0.02, 0.03, f"distance {row.combined_distance:.3f}", transform=axes[rank, 1].transAxes,
            color="white", fontsize=8, bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
        )
    figure.suptitle("Most similar positive/negative candidates · IR10.8", fontsize=13)
    figure.savefig(out, dpi=180)
    plt.close(figure)


def plot_all_channels(positive, negative, channels, row, out: Path):
    figure, axes = plt.subplots(2, len(channels), figsize=(21, 6.4), constrained_layout=True)
    for index, channel in enumerate(channels):
        combined = np.concatenate(
            [positive["cube"][index].ravel(), negative["cube"][index].ravel()]
        )
        finite = combined[np.isfinite(combined)]
        vmin, vmax = np.quantile(finite, [0.01, 0.99])
        axes[0, index].imshow(positive["cube"][index], cmap="gray_r", vmin=vmin, vmax=vmax)
        axes[0, index].contour(positive["mask"], levels=[0.5], colors="#ec4899", linewidths=0.8)
        axes[1, index].imshow(negative["cube"][index], cmap="gray_r", vmin=vmin, vmax=vmax)
        axes[0, index].set_title(channel)
        for axis in axes[:, index]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_ylabel("POSITIVE\nEMMA contour", fontsize=11)
    axes[1, 0].set_ylabel(f"NEGATIVE\n{negative['kind'].replace('_', ' ')}", fontsize=11)
    figure.suptitle(
        f"Borderline pair · distance {row.combined_distance:.3f} · "
        f"positive {positive['target_time']:%Y-%m-%d %H:%MZ} / negative {negative['target_time']:%Y-%m-%d %H:%MZ}",
        fontsize=13,
    )
    figure.savefig(out, dpi=190)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-manifest", type=Path, required=True)
    parser.add_argument("--negative-manifest", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--emma-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=1)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    positives, negatives, channels = load_samples(args)
    ranked = rank_pairs(positives, negatives)
    ranked.to_csv(args.out_dir / "emma_borderline_pair_candidates.csv", index=False)
    ir_index = channels.index("IR_108")
    plot_candidates(
        ranked, positives, negatives, ir_index,
        args.out_dir / "emma_borderline_pair_candidates.png",
    )
    selected = ranked.iloc[args.rank - 1]
    positive = positives[int(selected.positive_index)]
    negative = negatives[int(selected.negative_index)]
    plot_all_channels(
        positive, negative, channels, selected,
        args.out_dir / "emma_borderline_pair_all_channels.png",
    )
    pd.DataFrame(
        [
            {
                **selected.to_dict(),
                "positive_target_time": positive["target_time"],
                "positive_seviri_time": positive["seviri_time"],
                "positive_track_uid": positive["track_uid"],
                "positive_mcs_id": positive["mcs_id"],
                "negative_target_time": negative["target_time"],
                "negative_seviri_time": negative["seviri_time"],
                "negative_origin_y": negative.get("origin_y"),
                "negative_origin_x": negative.get("origin_x"),
                "negative_distance_km": negative.get("distance_km"),
            }
        ]
    ).to_csv(args.out_dir / "emma_borderline_pair_selected.csv", index=False)
    print(pd.DataFrame([selected]).to_string(index=False))
    print(f"positive={positive['sample_id']}")
    print(f"negative={negative['sample_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
