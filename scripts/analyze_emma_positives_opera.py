#!/usr/bin/env python3
"""Compare EMMA-positive masks with cold-cloud and OPERA evidence."""

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

from mine_opera_verified_n1 import radar_class  # noqa: E402
from plot_emma_heatmap_examples import (  # noqa: E402
    centered_crop,
    load_case_mask,
    load_ir108,
    utc_naive,
)
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
    component_metrics,
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
EMMA_COLOR = "#00a65a"


def overlapping_shield(
    image: np.ndarray,
    emma: np.ndarray,
    shield_threshold: float,
    core_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels, count = label(
        np.isfinite(image) & (image < shield_threshold),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    best_id = 0
    best_overlap = 0
    for object_id in range(1, count + 1):
        overlap = int(np.count_nonzero((labels == object_id) & emma))
        if overlap > best_overlap:
            best_id, best_overlap = object_id, overlap
    shield = labels == best_id if best_id else np.zeros_like(emma)
    core = shield & (image < core_threshold)
    return shield, core


def positive_mask_rgb(
    shield: np.ndarray,
    radar: np.ndarray,
    emma: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    rgb = np.full((*shield.shape, 3), 0.97, dtype=np.float32)
    rgb[~observed] = (0.72, 0.72, 0.72)
    rgb[shield] = (0.20, 0.75, 0.90)
    rgb[emma] = (0.93, 0.25, 0.58)
    rgb[radar] = (0.93, 0.20, 0.16)
    rgb[shield & emma] = (0.50, 0.28, 0.78)
    rgb[shield & radar] = (0.98, 0.55, 0.12)
    rgb[emma & radar] = (0.80, 0.10, 0.38)
    rgb[shield & emma & radar] = (0.10, 0.10, 0.10)
    return rgb


def fraction(numerator: np.ndarray, denominator: np.ndarray) -> float:
    size = int(denominator.sum())
    return float(np.count_nonzero(numerator & denominator) / size) if size else np.nan


def prepare_case(
    item,
    seviri_root: Path,
    emma_root: Path,
    opera_cache: Path,
    datasets,
    lookup_cache,
    transformer: Transformer,
    qind_threshold: float,
    shield_threshold: float,
    core_threshold: float,
) -> tuple[dict[str, object], dict[str, object]]:
    target = utc_naive(item.target_time)
    timestamp = utc_naive(item.seviri_time)
    image, x, y = load_ir108(seviri_root, timestamp, datasets)
    full_emma = load_case_mask(
        emma_root, target, int(item.mcs_id), x, y, lookup_cache
    )
    ys, xs, contained = centered_crop(full_emma, CROP)
    y0, x0 = int(ys.start), int(xs.start)
    crop = image[ys, xs]
    emma = full_emma[ys, xs]
    shield, core = overlapping_shield(
        crop, emma, shield_threshold, core_threshold
    )
    footprint = binary_dilation(shield, iterations=4)
    row, col, inside = opera_indices_for_crop(x, y, x0, y0, transformer)
    opera_frames = []
    slots = []
    for offset in OFFSETS_MINUTES:
        radar_time = timestamp + pd.Timedelta(int(offset), unit="min")
        dbzh, qind = read_opera(download_opera(radar_time, opera_cache))
        opera_frames.append((dbzh, qind))
        slots.append(
            {
                "offset": int(offset),
                "radar_time": radar_time,
                **radar_slot_metrics(
                    footprint, row, col, inside, dbzh, qind, qind_threshold
                ),
            }
        )
    metrics = component_metrics(
        footprint, row, col, inside, opera_frames, qind_threshold
    )
    branch = radar_class(metrics)
    positive_class = {
        "N2_candidate": "P_radar_confirmed",
        "N1_candidate": "P_radar_quiet",
        "NU": "P_unresolved",
    }[branch]
    areas40 = [slot["area40"] for slot in slots]
    peak_index = int(np.argmax(areas40)) if max(areas40) > 0 else list(OFFSETS_MINUTES).index(0)
    peak = slots[peak_index]
    radar = qualified_radar_mask(
        peak["z"], peak["q"], shield, qind_threshold
    )
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
    record = {
        "case_id": f"P{int(item.Index) + 1:02d}",
        "target_time": target,
        "seviri_time": timestamp,
        "track_uid": item.track_uid,
        "mcs_id": int(item.mcs_id),
        "origin_y": y0,
        "origin_x": x0,
        "emma_contained": bool(contained),
        "emma_pixels": int(emma.sum()),
        "shield_pixels": int(shield.sum()),
        "core_pixels": int(core.sum()),
        "emma_fraction_in_shield": fraction(shield, emma),
        "emma_fraction_in_core": fraction(core, emma),
        "shield_fraction_in_emma": fraction(emma, shield),
        "emma_fraction_radar40_peak": fraction(radar, emma),
        "shield_fraction_radar40_peak": fraction(radar, shield),
        "positive_evidence_class": positive_class,
        "n2_radar_branch": branch,
        **metrics,
    }
    case = {
        "record": record,
        "crop": crop,
        "emma": emma,
        "shield": shield,
        "core": core,
        "radar": radar,
        "observed": observed,
        "coverage": coverage,
        "qualified_echo": qualified_echo,
        "slots": slots,
        "peak": peak,
    }
    return record, case


def plot_page(cases: list[dict[str, object]], out: Path, page_label: str) -> None:
    fig, axes = plt.subplots(
        len(cases), 2, figsize=(8.6, 3.45 * len(cases)), constrained_layout=True
    )
    if len(cases) == 1:
        axes = axes[np.newaxis, :]
    bt_image = None
    radar_image = None
    for row_axes, case in zip(axes, cases):
        record = case["record"]
        case_id = record["case_id"]

        ax = row_axes[0]
        bt_image = ax.imshow(case["crop"], cmap=BT_CMAP, norm=BT_NORM)
        ax.contour(case["emma"], [0.5], colors="white", linewidths=2.5)
        ax.contour(case["emma"], [0.5], colors=EMMA_COLOR, linewidths=1.4)
        ax.set_title(f"{case_id} · temperatura IR10.8 + EMMA", fontsize=9)

        ax = row_axes[1]
        ax.imshow(case["crop"], cmap=BT_CMAP, norm=BT_NORM)
        missing = np.where(~case["coverage"], 1.0, np.nan)
        ax.imshow(missing, cmap=MISSING_CMAP, vmin=0.0, vmax=1.0, alpha=0.48)
        z = np.where(case["qualified_echo"], case["peak"]["z"], np.nan)
        radar_image = ax.imshow(
            z, cmap=RADAR_GRADIENT, norm=RADAR_GRADIENT_NORM,
            alpha=np.where(np.isfinite(z), 0.72, 0.0),
        )
        ax.contour(case["emma"], [0.5], colors="white", linewidths=2.5)
        ax.contour(case["emma"], [0.5], colors=EMMA_COLOR, linewidths=1.4)
        peak_offset = int(case["peak"]["offset"])
        ax.set_title(f"{case_id} · IR + EMMA + radar, Δt={peak_offset:+d} min", fontsize=9)

        for panel in row_axes:
            panel.set_xticks([])
            panel.set_yticks([])

    fig.suptitle(f"EMMA positivi · {page_label}", fontsize=14, weight="bold")
    if bt_image is not None:
        colorbar = fig.colorbar(
            bt_image, ax=axes[:, 0].tolist(), fraction=0.028, pad=0.012
        )
        colorbar.set_label("temperatura di brillanza [K]")
    if radar_image is not None:
        colorbar = fig.colorbar(
            radar_image, ax=axes[:, 1].tolist(), fraction=0.028, pad=0.012,
        )
        colorbar.set_label("DBZH [dBZ]")
    fig.legend(
        handles=[
            Line2D([0], [0], color=EMMA_COLOR, linewidth=2.0,
                   label="maschera EMMA"),
            Patch(facecolor="#7f7f7f", alpha=0.48, label="OPERA nodata"),
        ],
        loc="lower center", ncol=2, frameon=False,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--emma-root", type=Path, required=True)
    parser.add_argument("--opera-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qind-threshold", type=float, default=0.2)
    parser.add_argument("--shield-threshold-k", type=float, default=253.0)
    parser.add_argument("--core-threshold-k", type=float, default=241.0)
    parser.add_argument("--rows-per-page", type=int, default=6)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    # Accept both the dedicated visual-selection manifest and the catalogue of
    # EMMA objects that happen to fall inside already downloaded sequences.
    manifest = manifest.rename(
        columns={
            "emma_target_time": "target_time",
            "central_time": "seviri_time",
        }
    )
    required = {"target_time", "seviri_time", "track_uid", "mcs_id"}
    missing = required.difference(manifest.columns)
    if missing:
        raise SystemExit(f"Manifest missing columns: {sorted(missing)}")
    manifest = manifest.reset_index().rename(columns={"index": "Index"})
    transformer = Transformer.from_crs(RSS_CRS, OPERA_CRS, always_xy=True)
    datasets = {}
    lookup_cache = {}
    records = []
    cases = []
    for item in manifest.itertuples(index=False):
        record, case = prepare_case(
            item,
            args.seviri_root,
            args.emma_root,
            args.opera_cache,
            datasets,
            lookup_cache,
            transformer,
            args.qind_threshold,
            args.shield_threshold_k,
            args.core_threshold_k,
        )
        records.append(record)
        cases.append(case)
        print(f"{record['case_id']} {record['positive_evidence_class']}", flush=True)
    for dataset in datasets.values():
        dataset.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(
        args.out_dir / "emma_positive_opera_validation.csv", index=False
    )
    plot_page(
        cases[: args.rows_per_page],
        args.out_dir / "emma_positive_comparison_overview.png",
        "overview",
    )
    pages = math.ceil(len(cases) / args.rows_per_page)
    for page in range(pages):
        start = page * args.rows_per_page
        stop = min(start + args.rows_per_page, len(cases))
        plot_page(
            cases[start:stop],
            args.out_dir / f"emma_positive_comparison_page_{page + 1:02d}.png",
            f"pagina {page + 1}/{pages}",
        )
    counts = pd.Series([r["positive_evidence_class"] for r in records]).value_counts()
    print(counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
