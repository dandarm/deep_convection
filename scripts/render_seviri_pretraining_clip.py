"""Render one seven-channel SEVIRI training clip as an MP4 contact sheet."""

from __future__ import annotations

import argparse
from datetime import timedelta
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emma_gpm.seviri_clips import (  # noqa: E402
    VIDEO_MAE_CHANNELS,
    SeviriZarrClipReader,
)


BT_MIN_K = 180.0
BT_MAX_K = 330.0
SPECTRAL_DIFFERENCES = (
    ("WV 6.2 − IR 10.8", 0, 4, -80.0, 20.0),
    ("IR 8.7 − IR 10.8", 2, 4, -15.0, 15.0),
    ("IR 10.8 − IR 12.0", 4, 5, -10.0, 10.0),
    ("IR 10.8 − IR 13.4", 4, 6, -20.0, 40.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "catalogs" / "seviri_pretraining_smoke.csv",
    )
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path(r"E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N"),
    )
    parser.add_argument("--clip-id", default="smoke_000")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "figures"
            / "videomae_pretraining"
            / "seviri_smoke_000_7channels_physical.mp4"
        ),
    )
    parser.add_argument("--fps", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    selected = manifest.loc[manifest["clip_id"] == args.clip_id]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one row for clip_id={args.clip_id!r}; found {len(selected)}."
        )
    row = selected.iloc[0]
    start_time = pd.Timestamp(row["start_time"])

    with SeviriZarrClipReader(args.zarr_root) as reader:
        clip = reader.load(
            start_time,
            origin_y=int(row["origin_y"]),
            origin_x=int(row["origin_x"]),
        )

    # The same fixed Kelvin scale is used for every raw channel and every clip.
    # This mirrors the dataset rather than contrast-normalizing each band.
    fig = plt.figure(figsize=(21, 8.5))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.0), hspace=0.28)
    top = outer[0].subgridspec(1, len(VIDEO_MAE_CHANNELS), wspace=0.08)
    bottom = outer[1].subgridspec(1, len(SPECTRAL_DIFFERENCES), wspace=0.10)
    axes = [fig.add_subplot(top[0, index]) for index in range(len(VIDEO_MAE_CHANNELS))]
    difference_axes = [
        fig.add_subplot(bottom[0, index]) for index in range(len(SPECTRAL_DIFFERENCES))
    ]
    images = []
    for channel, (axis, name) in enumerate(zip(axes, VIDEO_MAE_CHANNELS, strict=True)):
        image = axis.imshow(
            clip[0, channel],
            cmap="turbo_r",
            vmin=BT_MIN_K,
            vmax=BT_MAX_K,
            interpolation="nearest",
        )
        axis.set_title(name.replace("_", " "), fontsize=12)
        axis.set_xticks([])
        axis.set_yticks([])
        colorbar = fig.colorbar(image, ax=axis, orientation="horizontal", pad=0.035)
        colorbar.set_label("Brightness temperature [K]", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
        images.append(image)

    difference_images = []
    for axis, (label, minuend, subtrahend, vmin, vmax) in zip(
        difference_axes, SPECTRAL_DIFFERENCES, strict=True
    ):
        difference = clip[0, minuend] - clip[0, subtrahend]
        image = axis.imshow(
            difference,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
            interpolation="nearest",
        )
        axis.set_title(label, fontsize=12)
        axis.set_xticks([])
        axis.set_yticks([])
        colorbar = fig.colorbar(image, ax=axis, orientation="horizontal", pad=0.035)
        colorbar.set_label("Brightness-temperature difference [K]", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
        difference_images.append(image)

    timestamp_text = fig.suptitle("", fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.06, top=0.92)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "title": f"SEVIRI {args.clip_id}: seven-channel VideoMAE input",
        "artist": "deep_convection",
        "comment": "16 frames at the native 5-minute RSS cadence",
    }
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=5000,
        metadata=metadata,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(args.output), dpi=100):
        for frame in range(clip.shape[0]):
            timestamp = start_time.to_pydatetime() + timedelta(minutes=5 * frame)
            timestamp_text.set_text(
                f"SEVIRI {args.clip_id} — {timestamp:%Y-%m-%d %H:%M} UTC "
                f"— frame {frame + 1}/{clip.shape[0]}"
            )
            for channel, image in enumerate(images):
                image.set_data(clip[frame, channel])
            for difference_image, (_, minuend, subtrahend, _, _) in zip(
                difference_images, SPECTRAL_DIFFERENCES, strict=True
            ):
                difference_image.set_data(
                    clip[frame, minuend] - clip[frame, subtrahend]
                )
            writer.grab_frame()

    plt.close(fig)
    print(f"Wrote {args.output.resolve()} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
