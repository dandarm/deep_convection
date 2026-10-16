#!/usr/bin/env python3
"""Screen a season of OPERA composites before requesting MSG/SEVIRI.

The script samples one radar-centred window per day, downloads five OPERA
DBZH/QIND composites (-30, -15, 0, +15, +30 minutes), and summarizes the
European radar mosaic in non-overlapping spatial tiles.  The result is a
cheap OPERA-first catalogue of places and times where a cold SEVIRI cloud may
later be tested as a true N1 negative (cold cloud, adequate radar coverage,
no persistent convective echo).

This is only a *radar pre-screen*.  N1 is assigned later, after checking that
the SEVIRI tile contains a sufficiently cold cloud and that EMMA is empty.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import timedelta
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_emma_negatives_opera import (  # noqa: E402
    OFFSETS_MINUTES,
    OPERA_CRS,
    OPERA_PIXEL_M,
    download_opera,
    read_opera,
)


def sampled_times(start: str, end: str) -> list[pd.Timestamp]:
    """Return one diurnally rotating 15-minute-aligned UTC time per day."""
    days = pd.date_range(start, end, freq="D")
    result: list[pd.Timestamp] = []
    for index, day in enumerate(days):
        # Seven is coprime with 24, so all UTC hours are covered before the
        # sequence repeats.  :30 leaves both +/-30 minute slots in the day for
        # all but the harmless midnight boundary, which download_opera handles.
        hour = int((index * 7) % 24)
        result.append(day + timedelta(hours=hour, minutes=30))
    return result


def download_all(
    central_times: list[pd.Timestamp], cache: Path, workers: int
) -> tuple[dict[pd.Timestamp, Path], dict[pd.Timestamp, str]]:
    timestamps = sorted(
        {
            center + timedelta(minutes=int(offset))
            for center in central_times
            for offset in OFFSETS_MINUTES
        }
    )
    paths: dict[pd.Timestamp, Path] = {}
    errors: dict[pd.Timestamp, str] = {}

    def fetch(timestamp: pd.Timestamp) -> tuple[pd.Timestamp, Path]:
        return timestamp, download_opera(timestamp, cache)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, timestamp): timestamp for timestamp in timestamps}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            timestamp = futures[future]
            try:
                downloaded_time, path = future.result()
                paths[downloaded_time] = path
            except Exception as exc:  # keep the seasonal run resumable
                errors[timestamp] = f"{type(exc).__name__}: {exc}"
            done += 1
            if done % 50 == 0 or done == len(timestamps):
                print(
                    f"downloaded_or_cached={len(paths)}/{len(timestamps)} "
                    f"failed={len(errors)} checked={done}",
                    flush=True,
                )
    return paths, errors


def tile_reduce(field: np.ndarray, tile: int, operation: str) -> np.ndarray:
    ny = field.shape[0] // tile
    nx = field.shape[1] // tile
    blocks = field[: ny * tile, : nx * tile].reshape(ny, tile, nx, tile)
    if operation == "mean":
        return blocks.mean(axis=(1, 3))
    if operation == "max":
        return blocks.max(axis=(1, 3))
    if operation == "sum":
        return blocks.sum(axis=(1, 3))
    raise ValueError(operation)


def summarize_center(
    center: pd.Timestamp,
    paths: dict[pd.Timestamp, Path],
    tile: int,
    qind_threshold: float,
) -> tuple[pd.DataFrame | None, str | None]:
    timestamps = [
        center + timedelta(minutes=int(value)) for value in OFFSETS_MINUTES
    ]
    missing = [timestamp for timestamp in timestamps if timestamp not in paths]
    if missing:
        return None, "missing_opera:" + ",".join(value.isoformat() for value in missing)

    slot_metrics: list[dict[str, np.ndarray]] = []
    for timestamp in timestamps:
        dbzh, qind = read_opera(paths[timestamp])
        valid = np.isfinite(dbzh)
        no_echo = valid & np.isclose(dbzh, -32.0)
        quality_echo = valid & np.isfinite(qind) & (qind >= qind_threshold)
        good = no_echo | quality_echo
        usable_dbzh = np.where(good, dbzh, -99.0)
        slot_metrics.append(
            {
                "valid_fraction": tile_reduce(valid.astype(np.float32), tile, "mean"),
                "good_fraction": tile_reduce(good.astype(np.float32), tile, "mean"),
                "max_dbzh": tile_reduce(usable_dbzh, tile, "max"),
                "area35": tile_reduce((good & (dbzh >= 35.0)).astype(np.int32), tile, "sum")
                * (OPERA_PIXEL_M / 1000.0) ** 2,
                "area40": tile_reduce((good & (dbzh >= 40.0)).astype(np.int32), tile, "sum")
                * (OPERA_PIXEL_M / 1000.0) ** 2,
            }
        )

    valid = np.stack([item["valid_fraction"] for item in slot_metrics])
    good = np.stack([item["good_fraction"] for item in slot_metrics])
    max_dbzh = np.stack([item["max_dbzh"] for item in slot_metrics])
    area35 = np.stack([item["area35"] for item in slot_metrics])
    area40 = np.stack([item["area40"] for item in slot_metrics])
    ny, nx = valid.shape[1:]
    tile_y, tile_x = np.indices((ny, nx))
    row_center = tile_y * tile + tile / 2.0
    col_center = tile_x * tile + tile / 2.0
    x = col_center * OPERA_PIXEL_M
    y = -row_center * OPERA_PIXEL_M
    to_lonlat = Transformer.from_crs(OPERA_CRS, "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(x, y)

    result = pd.DataFrame(
        {
            "central_time": center,
            "tile_y": tile_y.ravel(),
            "tile_x": tile_x.ravel(),
            "center_lon": lon.ravel(),
            "center_lat": lat.ravel(),
            "valid_fraction_median": np.median(valid, axis=0).ravel(),
            "good_fraction_median": np.median(good, axis=0).ravel(),
            "max_dbzh_5slots": np.max(max_dbzh, axis=0).ravel(),
            "max_area35_km2_5slots": np.max(area35, axis=0).ravel(),
            "max_area40_km2_5slots": np.max(area40, axis=0).ravel(),
            "slots_area35_ge_16km2": np.sum(area35 >= 16.0, axis=0).ravel(),
            "tile_size_radar_pixels": tile,
            "tile_size_km": tile * OPERA_PIXEL_M / 1000.0,
        }
    )
    # Use the common EMMA/SEVIRI European domain, not the entire OPERA mosaic.
    result = result[
        result.center_lon.between(-20.0, 40.0)
        & result.center_lat.between(30.0, 70.0)
    ].copy()
    observable = (
        (result.valid_fraction_median >= 0.70)
        & (result.good_fraction_median >= 0.05)
    )
    # Deliberately conservative: no >=35 dBZ area in any of the five slots
    # and no isolated usable radar pixel reaching 30 dBZ.
    quiet = observable & (result.max_dbzh_5slots < 30.0) & (
        result.max_area35_km2_5slots == 0.0
    )
    result["radar_observable"] = observable
    result["radar_quiet_candidate"] = quiet
    return result, None


def select_balanced_times(summary: pd.DataFrame, count: int) -> pd.DataFrame:
    eligible = summary[
        (summary.status == "ok") & (summary.quiet_tiles >= 1)
    ].copy()
    if len(eligible) <= count:
        eligible["selected_for_seviri"] = True
        return eligible

    eligible["month"] = eligible.central_time.dt.month
    eligible["hour"] = eligible.central_time.dt.hour
    eligible["hour_bin"] = (eligible["hour"] // 6).astype(int)
    eligible["quality_score"] = (
        eligible.quiet_tiles
        + 0.25 * eligible.observable_tiles
        + 10.0 * eligible.quiet_fraction
    )
    # Allocate an equal quota to each month, then split each monthly quota
    # across four six-hour UTC bins.  This avoids a lexicographic round-robin
    # accidentally exhausting the budget in the first months/hours.
    months = sorted(eligible.month.unique())
    base_month, month_remainder = divmod(count, len(months))
    chosen_indices: list[int] = []
    for month_position, month in enumerate(months):
        month_quota = base_month + (month_position < month_remainder)
        month_rows = eligible[eligible.month == month]
        base_bin, bin_remainder = divmod(month_quota, 4)
        month_indices: list[int] = []
        for hour_bin in range(4):
            quota = base_bin + (hour_bin < bin_remainder)
            candidates = month_rows[month_rows.hour_bin == hour_bin].nlargest(
                quota, "quality_score"
            )
            month_indices.extend(candidates.index.tolist())
        if len(month_indices) < month_quota:
            remaining = month_rows.drop(index=month_indices).nlargest(
                month_quota - len(month_indices), "quality_score"
            )
            month_indices.extend(remaining.index.tolist())
        chosen_indices.extend(month_indices[:month_quota])
    result = eligible.loc[chosen_indices].sort_values("central_time").reset_index(drop=True)
    result["selected_for_seviri"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-05-01")
    parser.add_argument("--end", default="2020-09-30")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-tiles", type=Path, required=True)
    parser.add_argument("--out-times", type=Path, required=True)
    parser.add_argument("--out-seviri-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--qind-threshold", type=float, default=0.20)
    parser.add_argument("--seviri-centers", type=int, default=60)
    parser.add_argument("--sequence-minutes", type=int, default=30)
    parser.add_argument("--sequence-step-minutes", type=int, default=5)
    args = parser.parse_args()

    centers = sampled_times(args.start, args.end)
    print(
        f"central_times={len(centers)} opera_slots={len(centers) * len(OFFSETS_MINUTES)}",
        flush=True,
    )
    paths, errors = download_all(centers, args.cache, args.workers)

    tile_frames: list[pd.DataFrame] = []
    time_rows: list[dict[str, object]] = []
    for index, center in enumerate(centers, start=1):
        tiles, error = summarize_center(center, paths, args.tile, args.qind_threshold)
        if tiles is None:
            time_rows.append(
                {"central_time": center, "status": "failed", "reason": error}
            )
        else:
            tile_frames.append(tiles)
            observable = int(tiles.radar_observable.sum())
            quiet = int(tiles.radar_quiet_candidate.sum())
            time_rows.append(
                {
                    "central_time": center,
                    "status": "ok",
                    "reason": "",
                    "domain_tiles": len(tiles),
                    "observable_tiles": observable,
                    "quiet_tiles": quiet,
                    "quiet_fraction": quiet / observable if observable else 0.0,
                }
            )
        if index % 10 == 0 or index == len(centers):
            print(f"screened={index}/{len(centers)}", flush=True)

    args.out_tiles.parent.mkdir(parents=True, exist_ok=True)
    all_tiles = pd.concat(tile_frames, ignore_index=True) if tile_frames else pd.DataFrame()
    all_tiles.to_csv(args.out_tiles, index=False)
    times = pd.DataFrame(time_rows)
    times["central_time"] = pd.to_datetime(times.central_time)
    times.to_csv(args.out_times, index=False)

    selected = select_balanced_times(times, args.seviri_centers)
    manifest_rows = []
    for row in selected.itertuples(index=False):
        center = pd.Timestamp(row.central_time)
        sequence_start = center - timedelta(minutes=args.sequence_minutes)
        sequence_end = center + timedelta(
            minutes=args.sequence_minutes + args.sequence_step_minutes
        )
        manifest_rows.append(
            {
                "central_time": center.isoformat(),
                # EMMA hourly masks align best with SEVIRI at -30 minutes.
                "emma_target_time": (center + timedelta(minutes=30)).isoformat(),
                "slot_start": sequence_start.isoformat(),
                "slot_end": sequence_end.isoformat(),
                "expected_frames": (
                    2 * args.sequence_minutes // args.sequence_step_minutes + 1
                ),
                "frame_step_minutes": args.sequence_step_minutes,
                "quiet_tiles": int(row.quiet_tiles),
                "observable_tiles": int(row.observable_tiles),
            }
        )
    manifest = pd.DataFrame(manifest_rows).drop_duplicates("slot_start")
    args.out_seviri_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_seviri_manifest, index=False)
    print(
        f"tile_rows={len(all_tiles)} selected_centers={len(selected)} "
        f"seviri_sequences={len(manifest)} "
        f"expected_frames={int(manifest.expected_frames.sum()) if len(manifest) else 0} "
        f"download_errors={len(errors)}",
        flush=True,
    )
    if errors:
        error_path = args.out_times.with_name(args.out_times.stem + "_download_errors.csv")
        pd.DataFrame(
            [{"timestamp": key, "error": value} for key, value in errors.items()]
        ).to_csv(error_path, index=False)
        print(f"download_errors_csv={error_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
