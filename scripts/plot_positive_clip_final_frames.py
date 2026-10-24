#!/usr/bin/env python3
"""Plot the final IR10.8 frame of every complete EMMA-positive clip."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_emma_heatmap_examples import (  # noqa: E402
    centered_crop,
    load_case_mask,
    load_ir108,
    utc_naive,
)


FRAME_OFFSETS_MINUTES = tuple(range(-30, 31, 5))
BT_CMAP = "Blues_r"
BT_MIN = 205.0
BT_MAX = 295.0
EMMA_COLOR = "#57ff4f"
EMMA_HALO = "#202020"


def has_frame(root: Path, timestamp: pd.Timestamp, cache: dict[Path, xr.Dataset]) -> bool:
    store = root / f"{timestamp:%Y}" / f"{timestamp:%Y%m%d}.zarr"
    if not store.exists():
        return False
    if store not in cache:
        cache[store] = xr.open_zarr(store, consolidated=True)
    values = pd.DatetimeIndex(cache[store].time.values)
    return bool(np.any(values == np.datetime64(timestamp.to_datetime64(), "ns")))


def select_confirmed(evidence: pd.DataFrame, count: int = 30) -> pd.DataFrame:
    confirmed = evidence[
        evidence.positive_evidence_class.eq("P_radar_confirmed")
    ].copy()
    confirmed["target_time"] = pd.to_datetime(confirmed.target_time)
    confirmed = confirmed.sort_values(
        ["target_time", "opera_valid_fraction_median"],
        ascending=[True, False],
    )
    # First retain one object from every distinct observed time.  If fewer
    # than 30 times exist, fill the remainder with other distinct tracks.
    selected = confirmed.drop_duplicates("target_time", keep="first")
    if len(selected) < count:
        remaining = confirmed[~confirmed.case_id.isin(selected.case_id)].sort_values(
            ["opera_valid_fraction_median", "opera_max_dbzh_5slots"],
            ascending=False,
        )
        selected = pd.concat([selected, remaining.head(count - len(selected))])
    return selected.sort_values("target_time").head(count).reset_index(drop=True)


def plot_page(cases: list[dict[str, object]], path: Path, title: str) -> None:
    fig, axes = plt.subplots(5, 5, figsize=(14, 14), constrained_layout=True)
    image = None
    for axis, case in zip(axes.flat, cases):
        image = axis.imshow(
            case["image"], cmap=BT_CMAP, vmin=BT_MIN, vmax=BT_MAX
        )
        axis.contour(
            case["mask"], [0.5], colors=EMMA_HALO, linewidths=2.2
        )
        axis.contour(
            case["mask"], [0.5], colors=EMMA_COLOR, linewidths=1.05
        )
        timestamp = case["target_time"]
        axis.set_title(
            f"{case['clip_id']} · {timestamp:%d %b %H:%M}", fontsize=8
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(cases):]:
        axis.axis("off")
    fig.suptitle(title, fontsize=15, weight="bold")
    if image is not None:
        colorbar = fig.colorbar(
            image, ax=axes.ravel().tolist(), orientation="horizontal",
            fraction=0.022, pad=0.018,
        )
        colorbar.set_label("temperatura di brillanza IR10.8 [K]")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--emma-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs)
    pairs["central_time"] = pd.to_datetime(pairs.central_time)
    pairs["emma_target_time"] = pd.to_datetime(pairs.emma_target_time)
    evidence = pd.read_csv(args.evidence)
    evidence["target_time"] = pd.to_datetime(evidence.target_time)
    evidence["seviri_time"] = pd.to_datetime(evidence.seviri_time)

    stores: dict[Path, xr.Dataset] = {}
    lookup_cache: dict[object, object] = {}
    cases: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    for index, row in enumerate(pairs.itertuples(index=False), start=1):
        center = utc_naive(row.central_time)
        target = utc_naive(row.emma_target_time)
        times = [center + pd.Timedelta(offset, unit="min")
                 for offset in FRAME_OFFSETS_MINUTES]
        complete = all(has_frame(args.seviri_root, time, stores) for time in times)
        availability_rows.append(
            {
                "clip_id": f"P{index:03d}",
                "central_time": center,
                "target_time": target,
                "track_uid": row.track_uid,
                "mcs_id": int(row.mcs_id),
                "complete_13_frames": complete,
            }
        )
        if not complete:
            continue
        final_image, x, y = load_ir108(args.seviri_root, times[-1], stores)
        full_mask = load_case_mask(
            args.emma_root, target, int(row.mcs_id), x, y, lookup_cache
        )
        ys, xs, contained = centered_crop(full_mask)
        cases.append(
            {
                "clip_id": f"P{index:03d}",
                "target_time": target,
                "track_uid": row.track_uid,
                "mcs_id": int(row.mcs_id),
                "image": final_image[ys, xs],
                "mask": full_mask[ys, xs],
                "contained": contained,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    availability = pd.DataFrame(availability_rows)
    availability.to_csv(args.out_dir / "positive_clip_availability.csv", index=False)

    pages = math.ceil(len(cases) / 25)
    for page in range(pages):
        start, stop = page * 25, min((page + 1) * 25, len(cases))
        plot_page(
            cases[start:stop],
            args.out_dir / f"positive_clips_final_frame_page_{page + 1:02d}.png",
            f"Clip positive EMMA complete · frame finale · {start + 1}–{stop}",
        )

    selected = select_confirmed(evidence, count=30)
    selected.to_csv(args.out_dir / "opera_confirmed_positive_30.csv", index=False)
    case_by_key = {
        (case["track_uid"], case["target_time"]): case for case in cases
    }
    confirmed_cases = [
        case_by_key[(row.track_uid, utc_naive(row.target_time))]
        for row in selected.itertuples(index=False)
    ]
    for page in range(math.ceil(len(confirmed_cases) / 25)):
        start, stop = page * 25, min((page + 1) * 25, len(confirmed_cases))
        plot_page(
            confirmed_cases[start:stop],
            args.out_dir / f"opera_confirmed_positive_final_frame_page_{page + 1:02d}.png",
            f"Positivi EMMA confermati da OPERA · {start + 1}–{stop}",
        )

    for dataset in stores.values():
        dataset.close()
    print(f"complete_positive_clips={len(cases)}")
    print(f"opera_confirmed_available={len(evidence[evidence.positive_evidence_class.eq('P_radar_confirmed')])}")
    print(f"selected_confirmed={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
