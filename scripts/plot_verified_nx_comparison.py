#!/usr/bin/env python3
"""Render N1/N2 examples from the common cold-shield object catalogue."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import binary_dilation, label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_emma_heatmap_examples import load_ir108, utc_naive  # noqa: E402
from plot_opera_n2_presentation import (  # noqa: E402
    RADAR_BOUNDS,
    RADAR_CMAP,
    RADAR_NORM,
    qualified_radar_mask,
    radar_slot_metrics,
)
from validate_emma_negatives_opera import (  # noqa: E402
    CROP,
    OFFSETS_MINUTES,
    OPERA_CRS,
    download_opera,
    opera_indices_for_crop,
    read_opera,
)
from emma_gpm.seviri import RSS_CRS  # noqa: E402


BT_CMAP = "Blues_r"
BT_NORM = Normalize(vmin=205.0, vmax=295.0)
RADAR_GRADIENT = "YlOrRd"
RADAR_GRADIENT_NORM = Normalize(vmin=-5.0, vmax=65.0)
MISSING_CMAP = ListedColormap(["#7f7f7f"])
SHIELD_COLOR = "#57ff4f"
SHIELD_HALO = "#202020"


def comparison_rgb(
    shield: np.ndarray,
    core: np.ndarray,
    radar: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    rgb = np.full((*shield.shape, 3), 0.97, dtype=np.float32)
    rgb[~observed] = (0.72, 0.72, 0.72)
    rgb[shield] = (0.20, 0.75, 0.90)
    rgb[core] = (0.15, 0.35, 0.90)
    rgb[radar] = (0.93, 0.20, 0.16)
    rgb[shield & radar] = (0.98, 0.55, 0.12)
    rgb[core & radar] = (0.10, 0.10, 0.10)
    return rgb


def prepare_case(
    row,
    case_id: str,
    seviri_root: Path,
    opera_cache: Path,
    datasets,
    transformer: Transformer,
    qind_threshold: float,
    shield_threshold: float,
    core_threshold: float,
) -> dict[str, object]:
    timestamp = utc_naive(row.central_time)
    image, x, y = load_ir108(seviri_root, timestamp, datasets)
    y0, x0 = int(row.origin_y), int(row.origin_x)
    crop = image[y0 : y0 + CROP, x0 : x0 + CROP]
    labels, _ = label(
        np.isfinite(image) & (image < shield_threshold),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    shield = labels[y0 : y0 + CROP, x0 : x0 + CROP] == int(row.shield_id)
    core = shield & (crop < core_threshold)
    footprint = binary_dilation(shield, iterations=4)
    radar_row, radar_col, inside = opera_indices_for_crop(
        x, y, x0, y0, transformer
    )
    slots = []
    for offset in OFFSETS_MINUTES:
        radar_time = timestamp + pd.Timedelta(int(offset), unit="min")
        dbzh, qind = read_opera(download_opera(radar_time, opera_cache))
        slots.append(
            {
                "offset": int(offset),
                "radar_time": radar_time,
                **radar_slot_metrics(
                    footprint,
                    radar_row,
                    radar_col,
                    inside,
                    dbzh,
                    qind,
                    qind_threshold,
                ),
            }
        )
    areas40 = [slot["area40"] for slot in slots]
    # For radar-quiet cases every area is zero, so argmax would arbitrarily
    # select the first (-30 min) slot.  Use the central slot instead.
    peak_index = int(np.argmax(areas40)) if max(areas40) > 0 else list(OFFSETS_MINUTES).index(0)
    peak = slots[peak_index]
    radar = qualified_radar_mask(peak["z"], peak["q"], shield, qind_threshold)
    observed = inside & np.isfinite(peak["z"]) & (
        np.isclose(peak["z"], -32.0)
        | (np.isfinite(peak["q"]) & (peak["q"] >= qind_threshold))
    )
    coverage = inside & np.isfinite(peak["z"])
    qualified_echo = (
        coverage
        & ~np.isclose(peak["z"], -32.0)
        & np.isfinite(peak["q"])
        & (peak["q"] >= qind_threshold)
    )
    return {
        "case_id": case_id,
        "row": row,
        "crop": crop,
        "shield": shield,
        "core": core,
        "radar": radar,
        "observed": observed,
        "coverage": coverage,
        "qualified_echo": qualified_echo,
        "slots": slots,
        "peak": peak,
    }


def plot_page(cases: list[dict[str, object]], out: Path, label_text: str) -> None:
    fig, axes = plt.subplots(
        len(cases), 2, figsize=(8.6, 3.45 * len(cases)), constrained_layout=True
    )
    if len(cases) == 1:
        axes = axes[np.newaxis, :]
    bt_image = None
    radar_image = None
    for row_axes, case in zip(axes, cases):
        case_id = case["case_id"]
        ax = row_axes[0]
        bt_image = ax.imshow(case["crop"], cmap=BT_CMAP, norm=BT_NORM)
        ax.set_title(f"{case_id} · temperatura IR10.8", fontsize=9)

        ax = row_axes[1]
        ax.imshow(case["crop"], cmap=BT_CMAP, norm=BT_NORM)
        missing = np.where(~case["coverage"], 1.0, np.nan)
        ax.imshow(missing, cmap=MISSING_CMAP, vmin=0.0, vmax=1.0, alpha=0.48)
        z = np.where(case["qualified_echo"], case["peak"]["z"], np.nan)
        radar_image = ax.imshow(
            z,
            cmap=RADAR_GRADIENT,
            norm=RADAR_GRADIENT_NORM,
            alpha=np.where(np.isfinite(z), 0.72, 0.0),
        )
        # A dark halo plus a solid lime line remains legible over both the blue
        # temperature field and the warm radar overlay.
        ax.contour(
            case["shield"], [0.5], colors=SHIELD_HALO, linewidths=2.5,
            linestyles="-"
        )
        ax.contour(
            case["shield"], [0.5], colors=SHIELD_COLOR, linewidths=1.25,
            linestyles="-"
        )
        peak_offset = int(case["peak"]["offset"])
        ax.set_title(f"{case_id} · IR + radar, Δt={peak_offset:+d} min", fontsize=9)
        for panel in row_axes:
            panel.set_xticks([])
            panel.set_yticks([])

    fig.suptitle(label_text, fontsize=14, weight="bold")
    if bt_image is not None:
        colorbar = fig.colorbar(
            bt_image, ax=axes[:, 0].tolist(), fraction=0.028, pad=0.012
        )
        colorbar.set_label("temperatura di brillanza [K]")
    if radar_image is not None:
        colorbar = fig.colorbar(
            radar_image,
            ax=axes[:, 1].tolist(), fraction=0.028, pad=0.012,
        )
        colorbar.set_label("DBZH [dBZ]")
    fig.legend(
        handles=[
            Line2D([0], [0], color=SHIELD_COLOR, linestyle="-", linewidth=2.0,
                   label="scudo candidato (<253 K)"),
            Patch(facecolor="#7f7f7f", alpha=0.48, label="OPERA nodata"),
        ],
        loc="lower center", ncol=2, frameon=False,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--opera-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qind-threshold", type=float, default=0.2)
    parser.add_argument("--shield-threshold-k", type=float, default=253.0)
    parser.add_argument("--core-threshold-k", type=float, default=241.0)
    parser.add_argument("--rows-per-page", type=int, default=6)
    parser.add_argument("--max-per-class", type=int, default=30)
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog)
    transformer = Transformer.from_crs(RSS_CRS, OPERA_CRS, always_xy=True)
    datasets = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for category, short in (("N1_candidate", "N1"), ("N2_candidate", "N2")):
        subset = catalog[catalog["class"] == category].copy()
        if category == "N1_candidate":
            subset = subset.sort_values(
                ["central_shield_pixels", "opera_good_fraction_median"],
                ascending=False,
            )
        else:
            subset = subset.sort_values(
                ["opera_max_area40_km2_5slots", "central_shield_pixels"],
                ascending=False,
            )
        subset = subset.head(args.max_per_class).reset_index(drop=True)
        subset.to_csv(args.out_dir / f"{short.lower()}_figure_selection.csv", index=False)
        cases = [
            prepare_case(
                row,
                f"{short}-{index + 1:03d}",
                args.seviri_root,
                args.opera_cache,
                datasets,
                transformer,
                args.qind_threshold,
                args.shield_threshold_k,
                args.core_threshold_k,
            )
            for index, row in enumerate(subset.itertuples(index=False))
        ]
        if not cases:
            continue
        plot_page(cases[: args.rows_per_page], args.out_dir / f"{short.lower()}_overview.png", short)
        pages = math.ceil(len(cases) / args.rows_per_page)
        for page in range(pages):
            start = page * args.rows_per_page
            stop = min(start + args.rows_per_page, len(cases))
            plot_page(
                cases[start:stop],
                args.out_dir / f"{short.lower()}_page_{page + 1:02d}.png",
                f"{short} · {page + 1}/{pages}",
            )
    for dataset in datasets.values():
        dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
