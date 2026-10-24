#!/usr/bin/env python3
"""Render compact IR/radar videos for representative N1, N2 and EMMA cases."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from analyze_emma_positives_opera import EMMA_COLOR  # noqa: E402
from plot_emma_heatmap_examples import (  # noqa: E402
    centered_crop,
    load_case_mask,
    load_ir108,
    utc_naive,
)
from validate_emma_negatives_opera import (  # noqa: E402
    CROP,
    OFFSETS_MINUTES,
    OPERA_CRS,
    download_opera,
    mapped_field,
    opera_indices_for_crop,
    read_opera,
)
from emma_gpm.seviri import RSS_CRS  # noqa: E402


BT_CMAP = "Blues_r"
BT_NORM = Normalize(205.0, 295.0)
RADAR_CMAP = "YlOrRd"
RADAR_NORM = Normalize(-5.0, 65.0)
MISSING_CMAP = ListedColormap(["#7f7f7f"])


def load_radar_slots(center, cache, row, col, inside, qind_threshold=0.2):
    slots = {}
    for offset in OFFSETS_MINUTES:
        dbzh, qind = read_opera(
            download_opera(center + pd.Timedelta(int(offset), unit="min"), cache)
        )
        z = mapped_field(dbzh, row, col, inside)
        q = mapped_field(qind, row, col, inside)
        coverage = inside & np.isfinite(z)
        echo = (
            coverage & ~np.isclose(z, -32.0) & np.isfinite(q)
            & (q >= qind_threshold)
        )
        slots[int(offset)] = (z, coverage, echo)
    return slots


def render_video(
    label: str,
    center: pd.Timestamp,
    y0: int,
    x0: int,
    seviri_root: Path,
    opera_cache: Path,
    output: Path,
    ffmpeg: str,
    emma: np.ndarray | None = None,
) -> None:
    datasets = {}
    _, x, y = load_ir108(seviri_root, center, datasets)
    transformer = Transformer.from_crs(RSS_CRS, OPERA_CRS, always_xy=True)
    row, col, inside = opera_indices_for_crop(x, y, x0, y0, transformer)
    radar_slots = load_radar_slots(center, opera_cache, row, col, inside)
    offsets = list(range(-30, 31, 5))
    frames = []
    for offset in offsets:
        image, _, _ = load_ir108(
            seviri_root, center + pd.Timedelta(offset, unit="min"), datasets
        )
        frames.append(image[y0 : y0 + CROP, x0 : x0 + CROP])
    for dataset in datasets.values():
        dataset.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nx_video_") as temporary:
        temporary_path = Path(temporary)
        for number, (offset, crop) in enumerate(zip(offsets, frames)):
            radar_offset = min(OFFSETS_MINUTES, key=lambda value: abs(value - offset))
            z, coverage, echo = radar_slots[int(radar_offset)]
            fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9), constrained_layout=True)
            bt = axes[0].imshow(crop, cmap=BT_CMAP, norm=BT_NORM)
            axes[1].imshow(crop, cmap=BT_CMAP, norm=BT_NORM)
            missing = np.where(~coverage, 1.0, np.nan)
            axes[1].imshow(missing, cmap=MISSING_CMAP, vmin=0, vmax=1, alpha=0.48)
            radar_values = np.where(echo, z, np.nan)
            radar = axes[1].imshow(
                radar_values, cmap=RADAR_CMAP, norm=RADAR_NORM,
                alpha=np.where(np.isfinite(radar_values), 0.72, 0.0),
            )
            if emma is not None:
                for axis in axes:
                    axis.contour(emma, [0.5], colors="white", linewidths=2.7)
                    axis.contour(emma, [0.5], colors=EMMA_COLOR, linewidths=1.5)
            axes[0].set_title("temperatura IR10.8")
            axes[1].set_title(f"IR + radar OPERA ({radar_offset:+d} min)")
            for axis in axes:
                axis.set_xticks([])
                axis.set_yticks([])
            fig.colorbar(bt, ax=axes[0], fraction=0.046, pad=0.02).set_label("K")
            fig.colorbar(radar, ax=axes[1], fraction=0.046, pad=0.02).set_label("dBZ")
            timestamp = center + pd.Timedelta(offset, unit="min")
            fig.suptitle(f"{label} · {timestamp:%Y-%m-%d %H:%M} UTC · Δt={offset:+d} min",
                         fontsize=13, weight="bold")
            fig.savefig(temporary_path / f"frame_{number:03d}.png", dpi=125)
            plt.close(fig)
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-framerate", "2",
                "-i", str(temporary_path / "frame_%03d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(output),
            ],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n1-selection", type=Path, required=True)
    parser.add_argument("--n2-selection", type=Path, required=True)
    parser.add_argument("--positive-manifest", type=Path, required=True)
    parser.add_argument("--positive-index", type=int, default=5,
                        help="Zero-based positive row; default 5 is P06.")
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--emma-root", type=Path, required=True)
    parser.add_argument("--opera-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    n1 = pd.read_csv(args.n1_selection).iloc[0]
    n2 = pd.read_csv(args.n2_selection).iloc[0]
    for label, row in (("N1", n1), ("N2", n2)):
        render_video(
            label, utc_naive(row.central_time), int(row.origin_y), int(row.origin_x),
            args.seviri_root, args.opera_cache, args.out_dir / f"{label.lower()}_example.mp4",
            args.ffmpeg,
        )

    positives = pd.read_csv(args.positive_manifest).reset_index().rename(columns={"index": "Index"})
    item = positives.iloc[args.positive_index]
    center = utc_naive(item.seviri_time)
    datasets = {}
    _, x, y = load_ir108(args.seviri_root, center, datasets)
    for dataset in datasets.values():
        dataset.close()
    full_emma = load_case_mask(
        args.emma_root, utc_naive(item.target_time), int(item.mcs_id), x, y, {}
    )
    ys, xs, _ = centered_crop(full_emma, CROP)
    render_video(
        f"EMMA P{args.positive_index + 1:02d}", center, int(ys.start), int(xs.start),
        args.seviri_root, args.opera_cache, args.out_dir / "emma_positive_example.mp4",
        args.ffmpeg, full_emma[ys, xs],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
