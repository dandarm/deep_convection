#!/usr/bin/env python3
"""Create leakage-safe probing views from the DeepStorm-like catalog.

Two tables are written beside the source catalog:

* ``deepstorm_probe_9ch_iwp.parquet`` uses the nine GMI channels available
  across the broadest set of coincident footprints.
* ``deepstorm_probe_13ch_iwp.parquet`` additionally requires 166 and 183 GHz
  channels, making its high-frequency feature vector complete.

Rows without a valid IWP target are excluded.  A full CloudSat--GPM
coincidence file is assigned to exactly one split, preventing repeated or
nearby GMI footprints from leaking between train, validation, and test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path
LOW_FREQUENCY_CHANNELS = [
    "tb_10v_k",
    "tb_10h_k",
    "tb_18v_k",
    "tb_18h_k",
    "tb_23v_k",
    "tb_36v_k",
    "tb_36h_k",
    "tb_89v_k",
    "tb_89h_k",
]
HIGH_FREQUENCY_CHANNELS = [
    "tb_166v_k",
    "tb_166h_k",
    "tb_183_3v_k",
    "tb_183_7v_k",
]


def assign_group_splits(
    frame: pd.DataFrame,
    fractions: dict[str, float],
    target: str,
) -> dict[str, str]:
    """Assign whole coincidence groups while matching rows and positives.

    The greedy objective is deterministic and minimizes normalized deviations
    from the requested row and positive-label totals.  It has no dependency on
    scikit-learn and is suitable for the small number of coincidence groups.
    """
    names = list(fractions)
    if not names or abs(sum(fractions.values()) - 1.0) > 1e-8:
        raise ValueError("Split fractions must be non-empty and sum to one")

    stats = (
        frame.groupby("split_group", sort=True)
        .agg(
            rows=(target, "size"),
            positives=(target, "sum"),
            dc_positives=("target_dc_any", "sum"),
        )
        .reset_index()
        .sort_values(
            ["positives", "dc_positives", "rows", "split_group"],
            ascending=[False, False, False, True],
        )
    )
    total_rows = float(stats["rows"].sum())
    total_positive = float(stats["positives"].sum())
    total_dc_positive = float(stats["dc_positives"].sum())
    target_rows = {name: total_rows * fraction for name, fraction in fractions.items()}
    target_positive = {name: total_positive * fraction for name, fraction in fractions.items()}
    target_dc_positive = {name: total_dc_positive * fraction for name, fraction in fractions.items()}
    current = {name: {"rows": 0.0, "positives": 0.0, "dc_positives": 0.0, "groups": 0} for name in names}
    assignment: dict[str, str] = {}

    for position, group in enumerate(stats.itertuples(index=False)):
        remaining_groups = len(stats) - position
        scores: dict[str, float] = {}
        for name in names:
            score = 0.0
            for candidate_name in names:
                candidate_rows = current[candidate_name]["rows"] + (
                    float(group.rows) if candidate_name == name else 0.0
                )
                candidate_positive = current[candidate_name]["positives"] + (
                    float(group.positives) if candidate_name == name else 0.0
                )
                candidate_dc_positive = current[candidate_name]["dc_positives"] + (
                    float(group.dc_positives) if candidate_name == name else 0.0
                )
                score += (candidate_rows - target_rows[candidate_name]) ** 2 / max(
                    1.0, target_rows[candidate_name]
                )
                score += 4.0 * (candidate_positive - target_positive[candidate_name]) ** 2 / max(
                    1.0, target_positive[candidate_name]
                )
                score += 1.5 * (candidate_dc_positive - target_dc_positive[candidate_name]) ** 2 / max(
                    1.0, target_dc_positive[candidate_name]
                )
            # Before the final placements, make an empty split unattractive.
            empty_splits = sum(state["groups"] == 0 for state in current.values())
            if current[name]["groups"] > 0 and empty_splits >= remaining_groups:
                score += 1e12
            scores[name] = score
        chosen = min(names, key=lambda name: (scores[name], name))
        assignment[str(group.split_group)] = chosen
        current[chosen]["rows"] += float(group.rows)
        current[chosen]["positives"] += float(group.positives)
        current[chosen]["dc_positives"] += float(group.dc_positives)
        current[chosen]["groups"] += 1
    return assignment


def prepare_view(
    frame: pd.DataFrame,
    feature_columns: list[str],
    fractions: dict[str, float],
    stratify_on: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.loc[frame[feature_columns].notna().all(axis=1)].copy()
    assignment = assign_group_splits(work, fractions, stratify_on)
    work["split"] = work["split_group"].map(assignment).astype("string")
    work = work.sort_values(["split", "observation_time", "sample_uid"]).reset_index(drop=True)
    summary = (
        work.groupby("split", observed=True)
        .agg(
            rows=("sample_uid", "size"),
            groups=("split_group", "nunique"),
            iwp_positive=("target_iwp_gt_0p5", "sum"),
            dc_positive=("target_dc_any", "sum"),
        )
        .reset_index()
    )
    summary["iwp_positive_rate"] = summary["iwp_positive"] / summary["rows"]
    summary["dc_positive_rate"] = summary["dc_positive"] / summary["rows"]
    return work, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/deepstorm_probe.json")
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    catalog = resolve_project_path(ROOT, config["outputs"]["catalog"])
    output_dir = catalog.parent
    frame = pd.read_parquet(catalog)
    target = str(training["stratify_on"])
    if target not in frame:
        raise KeyError(f"Stratification target is absent from catalog: {target}")
    frame = frame.loc[frame["target_iwp_max_kg_m2"].notna()].copy()

    views = {
        "9ch_iwp": LOW_FREQUENCY_CHANNELS,
        "13ch_iwp": LOW_FREQUENCY_CHANNELS + HIGH_FREQUENCY_CHANNELS,
    }
    summaries: list[pd.DataFrame] = []
    for name, features in views.items():
        prepared, summary = prepare_view(frame, features, training["splits"], target)
        prepared.to_parquet(output_dir / f"deepstorm_probe_{name}.parquet", index=False)
        summary.insert(0, "view", name)
        summaries.append(summary)
        print(f"{name}: {len(prepared)} rows, {prepared['split_group'].nunique()} groups", flush=True)

    pd.concat(summaries, ignore_index=True).to_csv(output_dir / "probing_split_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
