#!/usr/bin/env python3
"""Render one week of SEVIRI IR10.8 with OPERA DBZH overlays."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from pyproj import Transformer
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from emma_gpm.seviri import RSS_CRS  # noqa: E402
from report_opera_archive import scan_archive  # noqa: E402
from validate_emma_negatives_opera import (  # noqa: E402
    OPERA_CRS,
    OPERA_PIXEL_M,
    OPERA_SHAPE,
    read_opera,
)


SEVIRI_GLOBE = ccrs.Globe(
    semimajor_axis=6378169, inverse_flattening=295.488065897014
)
SEVIRI_MAP_CRS = ccrs.Geostationary(
    central_longitude=9.5,
    satellite_height=35785831,
    globe=SEVIRI_GLOBE,
)


def common_times(
    inventory: pd.DataFrame,
    seviri_root: Path,
    start: pd.Timestamp,
    days: int,
) -> pd.DatetimeIndex:
    stop = start + pd.Timedelta(int(days), unit="D")
    radar = pd.DatetimeIndex(
        inventory.loc[
            inventory.timestamp_utc.ge(start) & inventory.timestamp_utc.lt(stop),
            "timestamp_utc",
        ]
    )
    satellite: list[pd.Timestamp] = []
    for day in pd.date_range(
        start.normalize(), (stop - pd.Timedelta(1, unit="D")).normalize()
    ):
        store = seviri_root / f"{day:%Y}" / f"{day:%Y%m%d}.zarr"
        if not store.exists():
            continue
        with xr.open_zarr(store, consolidated=True) as dataset:
            satellite.extend(pd.DatetimeIndex(dataset.time.values).tolist())
    return radar.intersection(pd.DatetimeIndex(satellite)).sort_values()


def minimum_gap_sample(
    timestamps: pd.DatetimeIndex, minutes: int
) -> pd.DatetimeIndex:
    if minutes <= 15 or len(timestamps) < 2:
        return timestamps
    selected = [timestamps[0]]
    gap = pd.Timedelta(int(minutes), unit="min")
    for timestamp in timestamps[1:]:
        if timestamp - selected[-1] >= gap:
            selected.append(timestamp)
    return pd.DatetimeIndex(selected)


def radar_lookup(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    transformer = Transformer.from_crs(RSS_CRS, OPERA_CRS, always_xy=True)
    grid_x, grid_y = np.meshgrid(x, y)
    opera_x, opera_y = transformer.transform(grid_x, grid_y)
    finite = np.isfinite(opera_x) & np.isfinite(opera_y)
    col = np.full(grid_x.shape, -1, dtype=np.int32)
    row = np.full(grid_y.shape, -1, dtype=np.int32)
    col[finite] = np.floor(opera_x[finite] / OPERA_PIXEL_M).astype(np.int32)
    row[finite] = np.floor(-opera_y[finite] / OPERA_PIXEL_M).astype(np.int32)
    inside = (
        finite
        & (row >= 0)
        & (row < OPERA_SHAPE[0])
        & (col >= 0)
        & (col < OPERA_SHAPE[1])
    )
    return row, col, inside


def orient(
    field: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = field
    out_x, out_y = x, y
    if out_y[0] < out_y[-1]:
        result, out_y = result[::-1], out_y[::-1]
    if out_x[0] > out_x[-1]:
        result, out_x = result[:, ::-1], out_x[::-1]
    return result, out_x, out_y


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=pd.Timestamp, default=pd.Timestamp("2020-05-02"))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument(
        "--opera-root",
        type=Path,
        default=Path(r"E:\Datasets\Deep_convection\raw\opera"),
    )
    parser.add_argument(
        "--seviri-root",
        type=Path,
        default=Path(
            r"E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--sampling-minutes", type=int, default=15)
    parser.add_argument("--qind-min", type=float, default=0.2)
    parser.add_argument("--dbzh-min", type=float, default=-5.0)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    if args.days < 1 or args.fps < 1 or args.sampling_minutes < 1:
        raise SystemExit("days, fps and sampling-minutes must be positive")

    inventory = scan_archive(args.opera_root, args.year)
    timestamps = minimum_gap_sample(
        common_times(inventory, args.seviri_root, args.start, args.days),
        args.sampling_minutes,
    )
    if timestamps.empty:
        raise SystemExit("No exact OPERA-SEVIRI timestamps in the requested interval")
    paths = inventory.set_index("timestamp_utc").path
    datasets: dict[Path, xr.Dataset] = {}
    first_store = (
        args.seviri_root / f"{timestamps[0]:%Y}" / f"{timestamps[0]:%Y%m%d}.zarr"
    )
    first = xr.open_zarr(first_store, consolidated=True)
    x, y = np.asarray(first.x), np.asarray(first.y)
    channels = list(map(str, first.channel.values))
    channel = channels.index("IR_108")
    datasets[first_store] = first
    row, col, inside = radar_lookup(x, y)
    _, plot_x, plot_y = orient(np.zeros((len(y), len(x))), x, y)
    extent = (
        float(plot_x.min()), float(plot_x.max()),
        float(plot_y.min()), float(plot_y.max()),
    )

    def frame(timestamp: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        store = args.seviri_root / f"{timestamp:%Y}" / f"{timestamp:%Y%m%d}.zarr"
        if store not in datasets:
            datasets[store] = xr.open_zarr(store, consolidated=True)
        dataset = datasets[store]
        index = int(
            np.flatnonzero(pd.DatetimeIndex(dataset.time.values) == timestamp)[0]
        )
        ir = np.asarray(
            dataset.seviri.isel(time=index, channel=channel), dtype=np.float32
        )
        dbzh, qind = read_opera(Path(paths.loc[timestamp]))
        radar = np.full(ir.shape, np.nan, dtype=np.float32)
        quality = np.full(ir.shape, np.nan, dtype=np.float32)
        radar[inside] = dbzh[row[inside], col[inside]]
        quality[inside] = qind[row[inside], col[inside]]
        echo = (
            np.isfinite(radar)
            & ~np.isclose(radar, -32.0)
            & np.isfinite(quality)
            & (quality >= args.qind_min)
            & (radar >= args.dbzh_min)
        )
        radar = np.where(echo, radar, np.nan)
        ir, _, _ = orient(ir, x, y)
        radar, _, _ = orient(radar, x, y)
        return ir, radar

    ir, radar = frame(timestamps[0])
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6), constrained_layout=True,
        subplot_kw={"projection": SEVIRI_MAP_CRS},
    )
    bt = axes[0].imshow(
        ir, cmap="Blues_r", norm=Normalize(205, 295), extent=extent,
        origin="upper", transform=SEVIRI_MAP_CRS,
    )
    axes[0].set_title("SEVIRI IR 10.8 µm")
    axes[1].imshow(
        ir, cmap="Blues_r", norm=Normalize(205, 295), extent=extent,
        origin="upper", transform=SEVIRI_MAP_CRS,
    )
    echo_image = axes[1].imshow(
        radar, cmap="turbo", norm=Normalize(-5, 65),
        alpha=np.where(np.isfinite(radar), 0.72, 0.0), extent=extent,
        origin="upper", transform=SEVIRI_MAP_CRS,
    )
    axes[1].set_title(f"IR 10.8 + OPERA DBZH (QIND ≥ {args.qind_min:.2f})")
    for axis in axes:
        axis.set_extent(extent, crs=SEVIRI_MAP_CRS)
        axis.coastlines(resolution="50m", color="#111827", linewidth=0.65)
        axis.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            edgecolor="#374151", linewidth=0.3,
        )
    fig.colorbar(bt, ax=axes[0], shrink=0.82, label="brightness temperature [K]")
    fig.colorbar(echo_image, ax=axes[1], shrink=0.82, label="riflettività [dBZ]")
    title = fig.suptitle("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=args.fps,
        metadata={"title": "SEVIRI IR10.8 + OPERA DBZH weekly sequence"},
        bitrate=2800,
    )
    records = []
    with writer.saving(fig, str(args.output), dpi=args.dpi):
        for number, timestamp in enumerate(timestamps, start=1):
            ir, radar = frame(timestamp)
            bt.set_data(ir)
            axes[1].images[0].set_data(ir)
            echo_image.set_data(radar)
            echo_image.set_alpha(np.where(np.isfinite(radar), 0.72, 0.0))
            title.set_text(
                f"{timestamp:%Y-%m-%d %H:%M UTC} · frame {number}/{len(timestamps)}"
            )
            writer.grab_frame()
            records.append(
                {
                    "frame": number,
                    "timestamp_utc": timestamp.isoformat(),
                    "radar_echo_fraction": float(np.isfinite(radar).mean()),
                }
            )
            print(f"frame={number}/{len(timestamps)} time={timestamp}", flush=True)
    plt.close(fig)
    for dataset in datasets.values():
        dataset.close()
    pd.DataFrame(records).to_csv(args.output.with_suffix(".csv"), index=False)
    print(f"video={args.output.resolve()} frames={len(timestamps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
