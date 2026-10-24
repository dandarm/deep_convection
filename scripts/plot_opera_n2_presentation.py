#!/usr/bin/env python3
"""Create presentation-oriented N2 panels from the OPERA validation audit."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import binary_closing, label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_emma_heatmap_examples import load_ir108, utc_naive  # noqa: E402
from plot_emma_negative_examples import all_mcs_mask  # noqa: E402
from validate_emma_negatives_opera import (  # noqa: E402
    COLD_THRESHOLD_K,
    CROP,
    OFFSETS_MINUTES,
    OPERA_CRS,
    download_opera,
    largest_cold_component,
    mapped_field,
    opera_indices_for_crop,
    read_opera,
)
from emma_gpm.seviri import RSS_CRS  # noqa: E402


RADAR_BOUNDS = [-5, 20, 30, 35, 40, 45, 55, 70]
RADAR_CMAP = ListedColormap(
    ["#dbeafe", "#67e8f9", "#22d3ee", "#facc15", "#fb923c", "#ef4444", "#7f1d1d"]
)
RADAR_NORM = BoundaryNorm(RADAR_BOUNDS, RADAR_CMAP.N)


def radar_slot_metrics(
    component: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    inside: np.ndarray,
    dbzh: np.ndarray,
    qind: np.ndarray,
    qind_threshold: float,
) -> dict[str, object]:
    z = mapped_field(dbzh, row, col, inside)
    q = mapped_field(qind, row, col, inside)
    valid = component & inside & np.isfinite(z)
    no_echo = valid & np.isclose(z, -32.0)
    quality_echo = valid & np.isfinite(q) & (q >= qind_threshold)
    good = no_echo | quality_echo
    if good.any():
        max_dbzh = float(np.nanmax(z[good]))
        unique = np.unique(np.column_stack((row[good], col[good])), axis=0)
        values = dbzh[unique[:, 0], unique[:, 1]]
        area35 = float(np.count_nonzero(values >= 35.0) * 4.0)
        area40 = float(np.count_nonzero(values >= 40.0) * 4.0)
    else:
        max_dbzh = np.nan
        area35 = 0.0
        area40 = 0.0
    return {
        "z": z,
        "q": q,
        "max_dbzh": max_dbzh,
        "area35": area35,
        "area40": area40,
        "good_fraction": float(good.sum() / max(int(component.sum()), 1)),
    }


def qualified_radar_mask(
    z: np.ndarray,
    q: np.ndarray,
    cold_component: np.ndarray,
    qind_threshold: float,
) -> np.ndarray:
    raw = np.isfinite(q) & (q >= qind_threshold) & np.isfinite(z) & (z >= 40.0)
    raw = binary_closing(raw, structure=np.ones((3, 3), dtype=bool))
    objects, count = label(raw, structure=np.ones((3, 3), dtype=np.uint8))
    keep = np.zeros_like(raw)
    for object_id in range(1, count + 1):
        obj = objects == object_id
        if obj.sum() >= 4 and np.any(obj & cold_component):
            keep |= obj
    return keep


def mask_rgb(cold: np.ndarray, radar: np.ndarray, emma: np.ndarray, observed: np.ndarray) -> np.ndarray:
    image = np.full((*cold.shape, 3), 0.96, dtype=np.float32)
    image[~observed] = (0.72, 0.72, 0.72)
    image[cold] = (0.55, 0.78, 0.98)
    image[radar] = (0.96, 0.32, 0.28)
    image[cold & radar] = (0.58, 0.25, 0.72)
    image[emma] = (0.93, 0.25, 0.58)
    return image


def prepare_case(
    item,
    seviri_root: Path,
    emma_root: Path,
    opera_cache: Path,
    qind_threshold: float,
    datasets,
    lookup_cache,
    transformer: Transformer,
) -> dict[str, object]:
    timestamp = utc_naive(item.seviri_time)
    target_time = utc_naive(item.target_time)
    image, x, y = load_ir108(seviri_root, timestamp, datasets)
    y0, x0 = int(item.origin_y), int(item.origin_x)
    crop = image[y0 : y0 + CROP, x0 : x0 + CROP]
    cold = largest_cold_component(crop)
    emma = all_mcs_mask(emma_root, target_time, x, y, lookup_cache)[
        y0 : y0 + CROP, x0 : x0 + CROP
    ]
    row, col, inside = opera_indices_for_crop(x, y, x0, y0, transformer)
    slots = []
    for offset in OFFSETS_MINUTES:
        radar_time = pd.Timestamp(timestamp) + pd.Timedelta(int(offset), unit="min")
        dbzh, qind = read_opera(download_opera(radar_time, opera_cache))
        slots.append(
            {
                "offset": int(offset),
                "radar_time": radar_time,
                **radar_slot_metrics(cold, row, col, inside, dbzh, qind, qind_threshold),
            }
        )
    peak_index = int(np.argmax([slot["area40"] for slot in slots]))
    peak = slots[peak_index]
    radar = qualified_radar_mask(peak["z"], peak["q"], cold, qind_threshold)
    observed = inside & np.isfinite(peak["z"]) & (
        np.isclose(peak["z"], -32.0)
        | (np.isfinite(peak["q"]) & (peak["q"] >= qind_threshold))
    )
    return {
        "timestamp": timestamp,
        "target_time": target_time,
        "crop": crop,
        "cold": cold,
        "emma": emma,
        "slots": slots,
        "peak": peak,
        "radar": radar,
        "observed": observed,
        "item": item,
    }


def plot_page(cases: list[dict[str, object]], out: Path, title: str) -> None:
    figure, axes = plt.subplots(len(cases), 4, figsize=(17, 3.6 * len(cases)), constrained_layout=True)
    if len(cases) == 1:
        axes = axes[np.newaxis, :]
    for row_axes, case in zip(axes, cases):
        item = case["item"]
        case_id = case["case_id"]
        crop = case["crop"]
        cold = case["cold"]
        emma = case["emma"]
        peak = case["peak"]

        # A. Satellite cloud-top view.
        axis = row_axes[0]
        axis.imshow(crop, cmap="gray_r", vmin=205, vmax=295)
        axis.contour(cold, levels=[0.5], colors="#2563eb", linewidths=1.2)
        if emma.any():
            axis.contour(emma, levels=[0.5], colors="#ec4899", linewidths=1.3)
        axis.set_title(f"{case_id} · IR10.8", fontsize=9)

        # B. Continuous radar field at the most active slot, with low-quality
        # pixels muted rather than outlined into many small fragments.
        axis = row_axes[1]
        z = np.where(case["observed"], peak["z"], np.nan)
        radar_image = axis.imshow(z, cmap=RADAR_CMAP, norm=RADAR_NORM)
        axis.contour(cold, levels=[0.5], colors="#2563eb", linewidths=1.0)
        axis.set_facecolor("#b8b8b8")
        axis.set_title(f"{case_id} · OPERA", fontsize=9)

        # C. Three comparable object masks.
        axis = row_axes[2]
        axis.imshow(mask_rgb(cold, case["radar"], emma, case["observed"]))
        axis.set_title(f"{case_id} · confronto", fontsize=9)
        axis.text(
            0.02,
            0.03,
            "EMMA: vuota" if not emma.any() else "EMMA: presente",
            transform=axis.transAxes,
            fontsize=8,
            color="#9d174d",
            bbox={"facecolor": "white", "alpha": 0.85, "pad": 2},
        )

        # D. Radar evolution across the five historical 15-minute composites.
        axis = row_axes[3]
        offsets = [slot["offset"] for slot in case["slots"]]
        max_dbzh = [slot["max_dbzh"] for slot in case["slots"]]
        area35 = [slot["area35"] for slot in case["slots"]]
        area40 = [slot["area40"] for slot in case["slots"]]
        axis.plot(offsets, max_dbzh, color="#b91c1c", marker="o", linewidth=1.8, label="max DBZH")
        axis.axhline(40, color="#ef4444", linestyle="--", linewidth=0.8)
        axis.set_ylabel("massimo DBZH [dBZ]", color="#991b1b", fontsize=8)
        axis.tick_params(axis="y", labelcolor="#991b1b", labelsize=8)
        axis.set_xticks(offsets)
        axis.set_xlabel("minuti rispetto a SEVIRI", fontsize=8)
        second = axis.twinx()
        second.plot(offsets, area35, color="#f59e0b", marker="s", linewidth=1.3, label="area ≥35")
        second.plot(offsets, area40, color="#7f1d1d", marker="^", linewidth=1.3, label="area ≥40")
        second.set_ylabel("area radar nel core freddo [km²]", fontsize=8)
        second.tick_params(axis="y", labelsize=8)
        axis.grid(alpha=0.2)
        axis.set_title(f"{case_id} · evoluzione", fontsize=9)
        axis.legend(loc="upper left", fontsize=6, framealpha=0.8)
        second.legend(loc="upper right", fontsize=6, framealpha=0.8)

        for panel in row_axes[:3]:
            panel.set_xticks([])
            panel.set_yticks([])

    figure.suptitle(title, fontsize=15, weight="bold")
    colorbar = figure.colorbar(
        radar_image,
        ax=axes[:, 1].tolist(),
        orientation="vertical",
        fraction=0.018,
        pad=0.01,
        ticks=RADAR_BOUNDS,
    )
    colorbar.set_label("OPERA DBZH [dBZ]")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--emma-root", type=Path, required=True)
    parser.add_argument("--opera-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qind-threshold", type=float, default=0.2)
    parser.add_argument("--rows-per-page", type=int, default=6)
    args = parser.parse_args()

    frame = pd.read_csv(args.validation)
    n2 = frame[frame.provisional_class.eq("N2_candidate")].copy()
    n2 = n2.sort_values(
        ["opera_max_area40_km2_5slots", "opera_good_fraction_median"],
        ascending=False,
    ).reset_index(drop=True)
    if n2.empty:
        raise RuntimeError("No N2 candidates in validation table")

    transformer = Transformer.from_crs(RSS_CRS, OPERA_CRS, always_xy=True)
    datasets = {}
    lookup_cache = {}
    cases = []
    for case_number, item in enumerate(n2.itertuples(index=False), start=1):
        case = prepare_case(
                item,
                args.seviri_root,
                args.emma_root,
                args.opera_cache,
                args.qind_threshold,
                datasets,
                lookup_cache,
                transformer,
            )
        case["case_id"] = f"N2-{case_number:03d}"
        cases.append(case)
    for dataset in datasets.values():
        dataset.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # First page is the concise presentation overview; the atlas preserves all
    # 24 N2 candidates in equally structured pages.
    plot_page(
        cases[: args.rows_per_page],
        args.out_dir / "opera_n2_presentation_overview.png",
        "N2: convezione radar reale senza maschera MCS EMMA",
    )
    page_count = math.ceil(len(cases) / args.rows_per_page)
    for page in range(page_count):
        start = page * args.rows_per_page
        stop = min(start + args.rows_per_page, len(cases))
        plot_page(
            cases[start:stop],
            args.out_dir / f"opera_n2_atlas_page_{page + 1:02d}.png",
            f"Atlante N2 OPERA–SEVIRI · pagina {page + 1}/{page_count}",
        )
    print(f"n2={len(cases)} pages={page_count} out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
