#!/usr/bin/env python3
"""Count EMMA-positive object-hours for which a downloaded SEVIRI sequence exists."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--emma-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    rows = []
    for item in manifest.itertuples(index=False):
        target = pd.Timestamp(item.emma_target_time)
        path = (
            args.emma_root / f"{target:%Y}" / f"{target:%m}"
            / f"tracking_{target:%Y%m%dT%H}.nc"
        )
        with h5py.File(path) as handle:
            lat = np.asarray(handle["lat"])
            lon = np.asarray(handle["lon"])
            ys = np.flatnonzero((lat >= 30.0) & (lat <= 70.0))
            xs = np.flatnonzero((lon >= -20.0) & (lon <= 40.0))
            ids = np.asarray(
                handle["mcs_id"][0, ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]
            )
        unique = np.unique(ids[ids > 0]).astype(int)
        rows.extend(
            {
                "central_time": item.central_time,
                "emma_target_time": item.emma_target_time,
                "mcs_id": int(value),
                "track_uid": f"{target.year}:{int(value)}",
            }
            for value in unique
        )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"paired_positive_object_hours={len(output)}")
    print(f"paired_positive_tracks={output.track_uid.nunique() if len(output) else 0}")
    print(f"sequences_with_positive={output.emma_target_time.nunique() if len(output) else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
