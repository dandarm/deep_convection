#!/usr/bin/env python3
"""Rank continuous convective windows from an hourly NWC SAF GIF index."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


PHASES = ("triggering", "growing", "mature", "decay")
PHASE_WEIGHTS = {"triggering": 1.0, "growing": 1.25, "mature": 1.25, "decay": 0.75}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _slot_score(frame: pd.DataFrame) -> pd.Series:
    weighted = sum(frame[f"{phase}_components"] * weight for phase, weight in PHASE_WEIGHTS.items())
    return weighted + 0.5 * frame.phase_richness + 0.1 * np.sqrt(frame.target_line_pixels)


def _dominant_zone(objects: list[dict]) -> str:
    targets = [item for item in objects if item.get("phase") in PHASES]
    if not targets:
        return "none"
    weights = np.asarray([max(1, int(item.get("line_pixels", 1))) for item in targets], dtype=float)
    row = float(np.average([item["row_fraction"] for item in targets], weights=weights))
    column = float(np.average([item["column_fraction"] for item in targets], weights=weights))
    vertical = "north" if row < 1 / 3 else "middle" if row < 2 / 3 else "south"
    horizontal = "west" if column < 1 / 3 else "central" if column < 2 / 3 else "east"
    return f"{vertical}_{horizontal}"


def build_windows(slots: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    rows: list[dict] = []
    for period, raw_group in slots.groupby("period", sort=False):
        group = raw_group.copy().sort_values("time").set_index("time")
        expected = pd.date_range(group.index.min(), group.index.max(), freq="1h")
        group = group.reindex(expected)
        group["available"] = group.status.eq("ok")
        numeric = [
            "target_line_pixels",
            "target_components",
            "phase_richness",
            *(f"{phase}_components" for phase in PHASES),
        ]
        group[numeric] = group[numeric].fillna(0)
        group["slot_score"] = _slot_score(group)
        group["objects_json"] = group.objects_json.fillna("[]")
        archive_values = raw_group.archive.dropna().unique()
        archive = str(archive_values[0]) if len(archive_values) == 1 else "mixed"
        for start_index in range(0, len(group) - window_hours + 1):
            window = group.iloc[start_index : start_index + window_hours]
            if float(window.available.mean()) < (window_hours - 1) / window_hours:
                continue
            objects = []
            for value in window.objects_json:
                objects.extend(json.loads(value))
            phase_presence = sum((window[f"{phase}_components"] > 0).any() for phase in PHASES)
            rows.append(
                {
                    "period": period,
                    "archive": archive,
                    "start": window.index[0],
                    "end_exclusive": pd.Timestamp(window.index[-1]) + timedelta(hours=1),
                    "year_month": window.index[0].strftime("%Y-%m"),
                    "season": {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                               6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}[window.index[0].month],
                    "available_slots": int(window.available.sum()),
                    "active_slots": int((window.target_components > 0).sum()),
                    "integrated_components": int(window.target_components.sum()),
                    "peak_components": int(window.target_components.max()),
                    "integrated_line_pixels": int(window.target_line_pixels.sum()),
                    "phase_presence": int(phase_presence),
                    "triggering_components": int(window.triggering_components.sum()),
                    "growing_components": int(window.growing_components.sum()),
                    "mature_components": int(window.mature_components.sum()),
                    "decay_components": int(window.decay_components.sum()),
                    "dominant_canvas_zone": _dominant_zone(objects),
                    "activity_score": float(
                        window.slot_score.sum()
                        + 2.0 * window.target_components.max()
                        + 2.0 * phase_presence
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["activity_score", "integrated_components", "start"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        result.insert(0, "global_rank", np.arange(1, len(result) + 1))
    return result


def _separated(candidate: pd.Series, selected: list[pd.Series], separation_hours: float) -> bool:
    start = pd.Timestamp(candidate.start)
    return all(
        candidate.period != row.period
        or abs(start - pd.Timestamp(row.start)) >= timedelta(hours=separation_hours)
        for row in selected
    )


def balanced_candidates(
    windows: pd.DataFrame,
    *,
    count: int,
    separation_hours: float,
) -> pd.DataFrame:
    selected: list[pd.Series] = []
    # First retain one active window per month and period, preventing a single
    # summer outbreak from dominating the discovery sample.
    for (_, _), group in windows.groupby(["period", "year_month"], sort=True):
        for _, row in group.sort_values("activity_score", ascending=False).iterrows():
            if row.active_slots and _separated(row, selected, separation_hours):
                selected.append(row)
                break
    if len(selected) > count:
        selected = sorted(selected, key=lambda row: row.activity_score, reverse=True)[:count]
    for _, row in windows.iterrows():
        if len(selected) >= count:
            break
        if row.active_slots and _separated(row, selected, separation_hours):
            selected.append(row)
    if not selected:
        return windows.iloc[:0].copy()
    result = pd.DataFrame(selected).sort_values("activity_score", ascending=False, kind="stable")
    result.insert(0, "selection_rank", np.arange(1, len(result) + 1))
    return result.reset_index(drop=True)


def quiet_controls(
    windows: pd.DataFrame,
    *,
    count: int,
    separation_hours: float,
) -> pd.DataFrame:
    quiet = windows[(windows.active_slots == 0) & (windows.available_slots == windows.available_slots.max())]
    if quiet.empty:
        return quiet.copy()
    quiet = quiet.assign(quarter=pd.to_datetime(quiet.start).dt.to_period("Q").astype(str))
    selected: list[pd.Series] = []
    for (_, _), group in quiet.groupby(["period", "quarter"], sort=True):
        row = group.iloc[len(group) // 2]
        if _separated(row, selected, separation_hours):
            selected.append(row)
        if len(selected) >= count:
            break
    result = pd.DataFrame(selected)
    if not result.empty:
        result.insert(0, "control_rank", np.arange(1, len(result) + 1))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-hours", type=int, default=6)
    parser.add_argument("--candidate-count", type=int, default=36)
    parser.add_argument("--control-count", type=int, default=8)
    parser.add_argument("--separation-hours", type=float, default=12.0)
    args = parser.parse_args()
    if args.window_hours < 2 or args.candidate_count < 1 or args.control_count < 0:
        parser.error("Invalid window or sample counts")
    with sqlite3.connect(args.database) as connection:
        slots = pd.read_sql_query("SELECT * FROM slots ORDER BY period,time", connection)
    if slots.empty:
        raise ValueError("The activity database is empty")
    slots["time"] = pd.to_datetime(slots.time)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = build_windows(slots, args.window_hours)
    candidates = balanced_candidates(
        windows,
        count=args.candidate_count,
        separation_hours=args.separation_hours,
    )
    controls = quiet_controls(
        windows,
        count=args.control_count,
        separation_hours=args.separation_hours,
    )
    _atomic_csv(windows, args.output_dir / "ranked_windows.csv")
    _atomic_csv(candidates, args.output_dir / "balanced_candidates.csv")
    _atomic_csv(controls, args.output_dir / "quiet_controls.csv")
    summary = {
        "status": "complete",
        "source_database": str(args.database.resolve()),
        "catalogued_slots": len(slots),
        "available_slots": int((slots.status == "ok").sum()),
        "missing_slots": int((slots.status != "ok").sum()),
        "window_hours": args.window_hours,
        "ranked_windows": len(windows),
        "balanced_candidates": len(candidates),
        "quiet_controls": len(controls),
        "score_definition": (
            "sum of phase-weighted connected components plus phase richness, "
            "sqrt contour support, peak component count and phase coverage"
        ),
        "selection_warning": (
            "GIF-derived discovery proxy only; version cohorts are ranked separately "
            "and require independent georeferencing before physical-area comparison"
        ),
    }
    temporary = (args.output_dir / "summary.json.part")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output_dir / "summary.json")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
