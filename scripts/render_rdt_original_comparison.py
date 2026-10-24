#!/usr/bin/env python3
"""Render an observational comparison of UTLS v0 and pure RDT outputs.

The UTLS-v0 labels are read from the immutable full-domain teacher archive and
cropped only for display.  The pure labels are read from the cached full-domain
September 2020 run.  The latter already include the experimental parallax
remapping performed by ``run_september_week_full_domain.py``; the figure states
this explicitly because boundary offsets are therefore not an accuracy metric.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.rdt.storage import load_domain_crop  # noqa: E402


CLASS_NAMES = {
    0: "sfondo",
    1: "nube non convettiva",
    2: "sviluppo",
    3: "maturità/decadimento",
    4: "overshooting top",
}
CLASS_COLORS = ["#00000000", "#94a3b8", "#f59e0b", "#ef4444", "#d946ef"]
CLASS_CMAP = ListedColormap(CLASS_COLORS)
CLASS_NORM = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), CLASS_CMAP.N)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path(
            os.environ.get(
                "SEVIRI_ZARR_ROOT",
                r"E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N",
            )
        ),
    )
    parser.add_argument(
        "--utls-label-root",
        type=Path,
        default=ROOT / "results" / "rdt_preview" / "atbd_local_20200917",
    )
    parser.add_argument(
        "--pure-cache",
        type=Path,
        default=(
            ROOT
            / "RDT_original"
            / "results"
            / "comparison_seviri_2020"
            / "_cache_20200917.npz"
        ),
    )
    parser.add_argument("--start-time", default="2020-09-17T13:30:00")
    parser.add_argument("--origin-y", type=int, default=302)
    parser.add_argument("--origin-x", type=int, default=1223)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures" / "videomae_pretraining",
    )
    return parser.parse_args()


def _map_pure_to_project(labels: np.ndarray) -> np.ndarray:
    """Collapse the pure 0..6 phase schema to the canonical project classes."""
    if np.any(labels > 6):
        raise ValueError("Pure cache contains a class outside the documented 0..6 schema")
    lookup = np.asarray([0, 1, 2, 2, 3, 3, 4], dtype=np.uint8)
    return lookup[labels]


def _class_count(labels: np.ndarray, class_id: int) -> np.ndarray:
    return np.count_nonzero(labels == class_id, axis=(1, 2))


def _boundary(labels: np.ndarray) -> np.ndarray:
    foreground = labels > 0
    return foreground & ~ndimage.binary_erosion(foreground, structure=np.ones((3, 3)))


def _load_inputs(args: argparse.Namespace) -> dict:
    example = load_domain_crop(
        args.zarr_root,
        args.utls_label_root,
        args.start_time,
        args.origin_y,
        args.origin_x,
    )
    current = example["result"]["classes"].astype(np.uint8, copy=False)
    if current.shape != (16, 224, 224):
        raise ValueError(f"Unexpected UTLS crop shape: {current.shape}")

    with np.load(args.pure_cache, allow_pickle=False) as archive:
        pure_full = archive["sem_pure"]
        if pure_full.shape[0] != 288:
            raise ValueError(f"Expected a complete 5-min day, got {pure_full.shape}")
        height, width = pure_full.shape[1:]
        crop_size = current.shape[-1]
        # The VideoMAE reader receives native row/column origins.  Its output is
        # north-up/west-left; the full-domain cache is already in that orientation.
        crop_y = height - args.origin_y - crop_size
        crop_x = args.origin_x
        if crop_y < 0 or crop_x < 0 or crop_y + crop_size > height or crop_x + crop_size > width:
            raise ValueError("Requested crop falls outside the pure full-domain cache")
        start = pd.Timestamp(args.start_time)
        frame0 = int((start - start.normalize()).total_seconds() // (5 * 60))
        pure_native = pure_full[
            frame0 : frame0 + 16,
            crop_y : crop_y + crop_size,
            crop_x : crop_x + crop_size,
        ].copy()

    pure_mapped = _map_pure_to_project(pure_native)
    if pure_mapped.shape != current.shape:
        raise ValueError("The two products do not cover the same displayed tensor shape")
    return {
        "example": example,
        "current": current,
        "pure_native": pure_native,
        "pure": pure_mapped,
        "cache_crop_origin": [crop_y, crop_x],
        "cache_start_frame": frame0,
    }


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#07111f")
    axis.tick_params(colors="#a9b7ca")
    for spine in axis.spines.values():
        spine.set_color("#334155")


def render_video(data: dict, output: Path, fps: float) -> None:
    example = data["example"]
    current = data["current"]
    pure = data["pure"]
    pure_native = data["pure_native"]
    times = pd.DatetimeIndex(example["times"])
    frames = np.arange(len(times))

    current_conv = np.count_nonzero(np.isin(current, [2, 3, 4]), axis=(1, 2))
    pure_conv = np.count_nonzero(np.isin(pure, [2, 3, 4]), axis=(1, 2))
    current_cloud = np.count_nonzero(current > 0, axis=(1, 2))
    pure_cloud = np.count_nonzero(pure > 0, axis=(1, 2))

    fig = plt.figure(figsize=(14.4, 8.0), facecolor="#07111f")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.14, 0.86), hspace=0.25, wspace=0.18)
    ax_ir = fig.add_subplot(grid[0, 0])
    ax_current = fig.add_subplot(grid[0, 1])
    ax_pure = fig.add_subplot(grid[0, 2])
    ax_timeline = fig.add_subplot(grid[1, :2])
    ax_note = fig.add_subplot(grid[1, 2])
    for axis in (ax_ir, ax_current, ax_pure, ax_timeline, ax_note):
        _style_axis(axis)

    ir0 = example["clip"][0, 4]
    ir_artist = ax_ir.imshow(ir0, origin="upper", cmap="turbo_r", vmin=205, vmax=285)
    ax_ir.set_title("Osservazione · IR 10.8 µm", color="white")
    current_ir = ax_current.imshow(ir0, origin="upper", cmap="gray_r", vmin=205, vmax=285)
    current_artist = ax_current.imshow(
        np.ma.masked_where(current[0] == 0, current[0]),
        origin="upper",
        cmap=CLASS_CMAP,
        norm=CLASS_NORM,
        alpha=0.72,
        interpolation="nearest",
    )
    current_edge = ax_current.imshow(
        np.ma.masked_where(~_boundary(current[0]), _boundary(current[0])),
        origin="upper",
        cmap=ListedColormap(["#ffffff"]),
        alpha=0.92,
    )
    ax_current.set_title("Teacher attuale · UTLS v0", color="white")
    pure_ir = ax_pure.imshow(ir0, origin="upper", cmap="gray_r", vmin=205, vmax=285)
    pure_artist = ax_pure.imshow(
        np.ma.masked_where(pure[0] == 0, pure[0]),
        origin="upper",
        cmap=CLASS_CMAP,
        norm=CLASS_NORM,
        alpha=0.72,
        interpolation="nearest",
    )
    pure_edge = ax_pure.imshow(
        np.ma.masked_where(~_boundary(pure[0]), _boundary(pure[0])),
        origin="upper",
        cmap=ListedColormap(["#ffffff"]),
        alpha=0.92,
    )
    ax_pure.set_title("RDT_original · puro, rimappato 0–4", color="white")
    for axis in (ax_ir, ax_current, ax_pure):
        axis.set(xlabel="ovest → est", ylabel="nord → sud")
    colorbar = fig.colorbar(ir_artist, ax=ax_ir, fraction=0.046, pad=0.03)
    colorbar.set_label("brightness temperature [K]", color="white")
    colorbar.ax.tick_params(colors="white")

    ax_timeline.plot(frames, current_cloud, "--", color="#7dd3fc", alpha=0.7, label="UTLS · nube")
    ax_timeline.plot(frames, pure_cloud, "--", color="#c4b5fd", alpha=0.7, label="puro · nube")
    ax_timeline.plot(frames, current_conv, color="#22d3ee", linewidth=2.0, label="UTLS · convettivo")
    ax_timeline.plot(frames, pure_conv, color="#d946ef", linewidth=2.0, label="puro · convettivo")
    cursor = ax_timeline.axvline(0, color="#f8fafc", linewidth=1.4)
    ax_timeline.set(
        xlim=(0, 15),
        xlabel="frame a 5 minuti",
        ylabel="pixel nel crop",
        title="Copertura rilevata: nube e classi convettive 2–4",
    )
    ax_timeline.title.set_color("white")
    ax_timeline.grid(color="#334155", alpha=0.65, linewidth=0.6)
    legend = ax_timeline.legend(ncol=2, loc="upper left", facecolor="#111827", edgecolor="#334155")
    for label in legend.get_texts():
        label.set_color("white")

    ax_note.axis("off")
    note = ax_note.text(
        0.02,
        0.98,
        "",
        transform=ax_note.transAxes,
        va="top",
        color="white",
        family="monospace",
        fontsize=9.3,
        linespacing=1.43,
    )
    title = fig.suptitle("", color="white", fontsize=15.5, y=0.986)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=2600,
        metadata={
            "title": "UTLS v0 vs pure RDT-CW-inspired comparison",
            "comment": "Same SEVIRI observations; pure cache includes experimental parallax",
        },
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(output), dpi=100):
        for frame, stamp in enumerate(times):
            ir = example["clip"][frame, 4]
            ir_artist.set_data(ir)
            current_ir.set_data(ir)
            pure_ir.set_data(ir)
            current_artist.set_data(np.ma.masked_where(current[frame] == 0, current[frame]))
            pure_artist.set_data(np.ma.masked_where(pure[frame] == 0, pure[frame]))
            c_edge = _boundary(current[frame])
            p_edge = _boundary(pure[frame])
            current_edge.set_data(np.ma.masked_where(~c_edge, c_edge))
            pure_edge.set_data(np.ma.masked_where(~p_edge, p_edge))
            cursor.set_xdata([frame, frame])

            exact_agreement = np.count_nonzero(current[frame] == pure[frame]) / current[frame].size
            fg_union = (current[frame] > 0) | (pure[frame] > 0)
            foreground_agreement = (
                np.count_nonzero((current[frame] > 0) == (pure[frame] > 0)) / current[frame].size
            )
            current_counts = [_class_count(current[frame : frame + 1], c)[0] for c in range(1, 5)]
            pure_counts = [_class_count(pure[frame : frame + 1], c)[0] for c in range(1, 5)]
            pure_phase_counts = {
                int(cls): int(count)
                for cls, count in zip(*np.unique(pure_native[frame], return_counts=True))
                if cls > 0
            }
            note.set_text(
                "Mappatura pura → progetto\n"
                "  1→1; 2/3→2; 4/5→3; 6→4\n\n"
                f"pixel per classe 1/2/3/4\n"
                f"  UTLS : {current_counts}\n"
                f"  puro : {pure_counts}\n\n"
                f"fasi pure native : {pure_phase_counts}\n"
                f"accordo classe    : {exact_agreement:6.1%}\n"
                f"accordo nube/sf.  : {foreground_agreement:6.1%}\n"
                f"unione foreground : {int(fg_union.sum())} px\n\n"
                "CAUTELA GEOMETRICA\n"
                "La cache pura include la parallasse\n"
                "sperimentale; UTLS è sulla griglia\n"
                "satellitare nativa. Gli offset di\n"
                "bordo non misurano accuratezza."
            )
            title.set_text(f"{stamp:%Y-%m-%d %H:%M} UTC · frame {frame + 1}/16 · stesso cubo SEVIRI")
            for _ in range(3):
                writer.grab_frame()
    plt.close(fig)


def render_preview(data: dict, output: Path, frame: int = 14) -> None:
    example = data["example"]
    current = data["current"][frame]
    pure = data["pure"][frame]
    ir = example["clip"][frame, 4]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    axes[0].imshow(ir, origin="upper", cmap="turbo_r", vmin=205, vmax=285)
    axes[0].set_title("IR 10.8 µm")
    for axis, labels, title in (
        (axes[1], current, "UTLS v0 · classi 0–4"),
        (axes[2], pure, "Puro · rimappato 0–4"),
    ):
        axis.imshow(ir, origin="upper", cmap="gray_r", vmin=205, vmax=285)
        axis.imshow(
            np.ma.masked_where(labels == 0, labels),
            origin="upper",
            cmap=CLASS_CMAP,
            norm=CLASS_NORM,
            alpha=0.72,
            interpolation="nearest",
        )
        axis.contour(labels > 0, levels=[0.5], colors="white", linewidths=0.65)
        axis.set_title(title)
    for axis in axes:
        axis.set(xlabel="ovest → est", ylabel="nord → sud")
    fig.suptitle(
        f"Confronto qualitativo · {example['times'][frame]:%Y-%m-%d %H:%M} UTC\n"
        "La variante pura include la parallasse sperimentale; UTLS usa la griglia satellitare nativa"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    data = _load_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "rdt_original_pure_vs_utls_track411"
    video_path = args.output_dir / f"{stem}.mp4"
    preview_path = args.output_dir / f"{stem}_preview.png"
    summary_path = args.output_dir / f"{stem}_summary.json"
    render_video(data, video_path, args.fps)
    render_preview(data, preview_path)

    current = data["current"]
    pure = data["pure"]
    summary = {
        "video": str(video_path.resolve()),
        "preview": str(preview_path.resolve()),
        "times": [pd.Timestamp(t).isoformat() for t in data["example"]["times"]],
        "source_root": str(args.zarr_root.resolve()),
        "utls_label_root": str(args.utls_label_root.resolve()),
        "pure_cache": str(args.pure_cache.resolve()),
        "native_crop_origin": [args.origin_y, args.origin_x],
        "pure_cache_crop_origin_north_up": data["cache_crop_origin"],
        "pure_cache_start_frame": data["cache_start_frame"],
        "pure_class_mapping": {"0": 0, "1": 1, "2": 2, "3": 2, "4": 3, "5": 3, "6": 4},
        "geometry_warning": (
            "Pure cached labels include experimental parallax correction; UTLS-v0 labels and "
            "SEVIRI imagery remain on the native satellite grid. Pixelwise offsets are not an "
            "accuracy measurement."
        ),
        "per_frame": [
            {
                "time": pd.Timestamp(data["example"]["times"][i]).isoformat(),
                "utls_counts_0_to_4": np.bincount(current[i].ravel(), minlength=5).tolist(),
                "pure_mapped_counts_0_to_4": np.bincount(pure[i].ravel(), minlength=5).tolist(),
                "exact_class_agreement": float(np.mean(current[i] == pure[i])),
            }
            for i in range(16)
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
