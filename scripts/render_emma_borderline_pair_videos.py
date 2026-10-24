#!/usr/bin/env python3
"""Render fixed-domain, seven-channel videos for the selected borderline pair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_emma_heatmap_examples import (  # noqa: E402
    centered_crop,
    load_case_mask,
    utc_naive,
)


CROP = 224


def load_day(root: Path, timestamp: pd.Timestamp) -> xr.Dataset:
    path = root / f"{timestamp:%Y}" / f"{timestamp:%Y%m%d}.zarr"
    return xr.open_zarr(path, consolidated=True).sortby("time")


def render_video(
    *,
    cube: np.ndarray,
    times: pd.DatetimeIndex,
    channels: list[str],
    case_label: str,
    reference_time: pd.Timestamp,
    output: Path,
    mask: np.ndarray | None,
) -> pd.DataFrame:
    ir_index = channels.index("IR_108")
    finite = np.isfinite(cube)
    vmins = np.array(
        [np.quantile(cube[:, c][finite[:, c]], 0.01) for c in range(len(channels))]
    )
    vmaxs = np.array(
        [np.quantile(cube[:, c][finite[:, c]], 0.99) for c in range(len(channels))]
    )
    ir = cube[:, ir_index]
    p10 = np.nanquantile(ir, 0.10, axis=(1, 2))
    p25 = np.nanquantile(ir, 0.25, axis=(1, 2))
    cold_fraction = np.nanmean(ir < 241.0, axis=(1, 2))

    figure, axes = plt.subplots(2, 4, figsize=(14.4, 7.4), constrained_layout=True)
    image_axes = axes.ravel()[:7]
    artists = []
    for channel_index, (axis, channel) in enumerate(zip(image_axes, channels)):
        artist = axis.imshow(
            cube[0, channel_index],
            cmap="gray_r",
            vmin=float(vmins[channel_index]),
            vmax=float(vmaxs[channel_index]),
            interpolation="nearest",
        )
        artists.append(artist)
        axis.set_title(channel)
        axis.set_xticks([])
        axis.set_yticks([])
        if mask is not None:
            axis.contour(mask, levels=[0.5], colors="#ec4899", linewidths=0.65)

    curve_axis = axes.ravel()[7]
    relative_minutes = (times - reference_time).total_seconds() / 60.0
    curve_axis.plot(relative_minutes, p10, color="#2563eb", marker="o", ms=2.5, label="IR10.8 p10")
    curve_axis.plot(relative_minutes, p25, color="#7c3aed", marker="o", ms=2.5, label="IR10.8 p25")
    curve_axis.set_xlabel("minuti dal frame selezionato")
    curve_axis.set_ylabel("temperatura [K]")
    curve_axis.grid(alpha=0.25)
    curve_axis.legend(loc="upper left", fontsize=8)
    cold_axis = curve_axis.twinx()
    cold_axis.plot(relative_minutes, cold_fraction, color="#dc2626", alpha=0.75, label="frazione <241 K")
    cold_axis.set_ylabel("frazione crop <241 K", color="#dc2626")
    cold_axis.tick_params(axis="y", labelcolor="#dc2626")
    time_marker = curve_axis.axvline(relative_minutes[0], color="black", lw=1.3)

    title = figure.suptitle("")
    writer = FFMpegWriter(fps=2, codec="libx264", bitrate=3200)
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(figure, output, dpi=125):
        for frame_index, timestamp in enumerate(times):
            for channel_index, artist in enumerate(artists):
                artist.set_data(cube[frame_index, channel_index])
            time_marker.set_xdata([relative_minutes[frame_index], relative_minutes[frame_index]])
            contour_note = " | contorno EMMA statico @ target" if mask is not None else ""
            title.set_text(
                f"{case_label} | {timestamp:%Y-%m-%d %H:%M UTC} "
                f"({relative_minutes[frame_index]:+g} min){contour_note}"
            )
            writer.grab_frame()
    plt.close(figure)
    return pd.DataFrame(
        {
            "case": case_label,
            "time": times,
            "minutes_from_reference": relative_minutes,
            "ir108_p10_k": p10,
            "ir108_p25_k": p25,
            "ir108_fraction_lt241k": cold_fraction,
            "ir108_fraction_lt235k": np.nanmean(ir < 235.0, axis=(1, 2)),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--emma-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    selected = pd.read_csv(args.selected).iloc[0]
    positive_target = utc_naive(selected.positive_target_time)
    positive_reference = utc_naive(selected.positive_seviri_time)
    negative_reference = utc_naive(selected.negative_seviri_time)

    positive_day = load_day(args.seviri_root, positive_reference)
    negative_day = load_day(args.seviri_root, negative_reference)
    try:
        # Recover the exact positive crop used to select the borderline pair.
        x = np.asarray(positive_day.x.values)
        y = np.asarray(positive_day.y.values)
        lookup_cache: dict = {}
        positive_mask_full = load_case_mask(
            args.emma_root,
            positive_target,
            int(selected.positive_mcs_id),
            x,
            y,
            lookup_cache,
        )
        positive_ys, positive_xs, contained = centered_crop(positive_mask_full)
        negative_y0 = int(selected.negative_origin_y)
        negative_x0 = int(selected.negative_origin_x)

        positive_cube = np.asarray(
            positive_day.seviri.isel(y=positive_ys, x=positive_xs).values,
            dtype=np.float32,
        )
        negative_cube = np.asarray(
            negative_day.seviri.isel(
                y=slice(negative_y0, negative_y0 + CROP),
                x=slice(negative_x0, negative_x0 + CROP),
            ).values,
            dtype=np.float32,
        )
        positive_times = pd.DatetimeIndex(positive_day.time.values)
        negative_times = pd.DatetimeIndex(negative_day.time.values)
        channels = [str(value) for value in positive_day.channel.values]
        if channels != [str(value) for value in negative_day.channel.values]:
            raise RuntimeError("Channel order differs between the two cases")
        if len(positive_times) != 19 or len(negative_times) != 19:
            raise RuntimeError(
                f"Expected 19 frames per case, got {len(positive_times)} and {len(negative_times)}"
            )

        positive_stats = render_video(
            cube=positive_cube,
            times=positive_times,
            channels=channels,
            case_label=f"POSITIVO EMMA | track 2020:{int(selected.positive_mcs_id)}",
            reference_time=positive_reference,
            output=args.out_dir / "borderline_positive_2020-08-15_0130_all_channels.mp4",
            mask=positive_mask_full[positive_ys, positive_xs],
        )
        negative_stats = render_video(
            cube=negative_cube,
            times=negative_times,
            channels=channels,
            case_label="SENZA EMMA | candidato ambiguo, non negativo confermato",
            reference_time=negative_reference,
            output=args.out_dir / "borderline_no_emma_2020-05-31_1530_all_channels.mp4",
            mask=None,
        )
        metadata = pd.DataFrame(
            [
                {
                    "case": "positive_emma",
                    "reference_time": positive_reference,
                    "first_time": positive_times.min(),
                    "last_time": positive_times.max(),
                    "frames": len(positive_times),
                    "origin_y": positive_ys.start,
                    "origin_x": positive_xs.start,
                    "emma_object_fully_contained": contained,
                },
                {
                    "case": "no_emma_ambiguous",
                    "reference_time": negative_reference,
                    "first_time": negative_times.min(),
                    "last_time": negative_times.max(),
                    "frames": len(negative_times),
                    "origin_y": negative_y0,
                    "origin_x": negative_x0,
                    "emma_object_fully_contained": pd.NA,
                },
            ]
        )
        metadata.to_csv(args.out_dir / "borderline_pair_video_metadata.csv", index=False)
        pd.concat([positive_stats, negative_stats], ignore_index=True).to_csv(
            args.out_dir / "borderline_pair_video_ir108_timeseries.csv", index=False
        )
    finally:
        positive_day.close()
        negative_day.close()
    print(f"rendered videos in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
