#!/usr/bin/env python3
"""Summarize 15-minute NWC SAF activity scans for candidate windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


PHASES = ("triggering", "growing", "mature", "decay")
PHASE_WEIGHTS = {"triggering": 1.0, "growing": 1.25, "mature": 1.25, "decay": 0.75}


def _longest_run(values: pd.Series) -> int:
    longest = current = 0
    for value in values.astype(bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def summarize(slots: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for period, group in slots.groupby("period", sort=False):
        group = group.sort_values("time").copy()
        available = group.status.eq("ok")
        active = available & group.target_components.fillna(0).gt(0)
        weighted = sum(
            group[f"{phase}_components"].fillna(0) * weight
            for phase, weight in PHASE_WEIGHTS.items()
        )
        longest_active = _longest_run(active)
        phase_presence = sum(
            group[f"{phase}_components"].fillna(0).gt(0).any() for phase in PHASES
        )
        rows.append(
            {
                "period": period,
                "archive": group.archive.dropna().iloc[0],
                "start": group.time.min(),
                "end": group.time.max(),
                "expected_slots": len(group),
                "available_slots": int(available.sum()),
                "coverage_fraction": float(available.mean()),
                "active_slots": int(active.sum()),
                "active_fraction_available": float(active.sum() / max(1, available.sum())),
                "longest_active_run_slots": longest_active,
                "integrated_components": int(group.target_components.fillna(0).sum()),
                "peak_components": int(group.target_components.fillna(0).max()),
                "integrated_line_pixels": int(group.target_line_pixels.fillna(0).sum()),
                **{
                    f"{phase}_components": int(group[f"{phase}_components"].fillna(0).sum())
                    for phase in PHASES
                },
                "phase_presence": int(phase_presence),
                "refined_activity_score": float(
                    weighted.sum()
                    + 2.0 * group.target_components.fillna(0).max()
                    + 3.0 * longest_active
                    + 2.0 * phase_presence
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["refined_activity_score", "active_fraction_available", "start"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        result.insert(0, "refined_rank", np.arange(1, len(result) + 1))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with sqlite3.connect(args.database) as connection:
        slots = pd.read_sql_query("SELECT * FROM slots ORDER BY period,time", connection)
    if slots.empty:
        raise ValueError("The activity database is empty")
    slots["time"] = pd.to_datetime(slots.time)
    result = summarize(slots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    result.to_csv(temporary, index=False)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "candidate_windows": len(result),
                "fully_covered": int(result.coverage_fraction.eq(1.0).sum()),
                "output": str(args.output.resolve()),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
