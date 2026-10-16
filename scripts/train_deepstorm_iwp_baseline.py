#!/usr/bin/env python3
"""Fit a dependency-free GMI brightness-temperature baseline for IWP.

This is a catalogue sanity check and diagnostic baseline, not a reproduction
of the published DEEPSTORM retrieval and not a SEVIRI probing model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path


FEATURES = [
    "tb_10v_k", "tb_10h_k", "tb_18v_k", "tb_18h_k", "tb_23v_k",
    "tb_36v_k", "tb_36h_k", "tb_89v_k", "tb_89h_k",
]
TARGET = "target_iwp_gt_0p5"


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def fit_logistic(x: np.ndarray, y: np.ndarray, epochs: int = 2500, learning_rate: float = 0.08) -> tuple[np.ndarray, float]:
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    positive_weight = (len(y) - y.sum()) / max(1.0, y.sum())
    sample_weight = np.where(y == 1.0, positive_weight, 1.0)
    normalizer = sample_weight.sum()
    for _ in range(epochs):
        residual = (sigmoid(x @ weights + bias) - y) * sample_weight
        weights -= learning_rate * ((x.T @ residual) / normalizer + 1e-3 * weights)
        bias -= learning_rate * residual.sum() / normalizer
    return weights, bias


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    positive = int(y.sum())
    negative = len(y) - positive
    if not positive or not negative:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    positive = int(y.sum())
    if not positive:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    ordered = y[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / positive)


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = score >= threshold
    tp = int(np.sum(prediction & (y == 1)))
    fp = int(np.sum(prediction & (y == 0)))
    fn = int(np.sum(~prediction & (y == 1)))
    tn = int(np.sum(~prediction & (y == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "balanced_accuracy": 0.5 * (recall + tn / max(1, tn + fp)),
        "roc_auc": roc_auc(y, score), "average_precision": average_precision(y, score),
    }


def best_validation_threshold(y: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(score)
    best = max(
        (classification_metrics(y, score, float(threshold)) for threshold in candidates),
        key=lambda metrics: (metrics["f1"], metrics["recall"], -metrics["threshold"]),
    )
    return float(best["threshold"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/deepstorm_probe.json")
    args = parser.parse_args()
    config = load_config(args.config)
    catalog_dir = resolve_project_path(ROOT, config["outputs"]["catalog"]).parent
    table = pd.read_parquet(catalog_dir / "deepstorm_probe_9ch_iwp.parquet")
    if table.groupby("split")["split_group"].nunique().sum() != table["split_group"].nunique():
        raise RuntimeError("split_group leakage detected")

    train = table.loc[table["split"] == "train"]
    validation = table.loc[table["split"] == "validation"]
    test = table.loc[table["split"] == "test"]
    mean = train[FEATURES].mean().to_numpy(dtype=np.float64)
    std = train[FEATURES].std().replace(0, 1).to_numpy(dtype=np.float64)
    to_x = lambda frame: (frame[FEATURES].to_numpy(dtype=np.float64) - mean) / std
    train_y = train[TARGET].to_numpy(dtype=np.float64)
    validation_y = validation[TARGET].to_numpy(dtype=np.float64)
    test_y = test[TARGET].to_numpy(dtype=np.float64)

    weights, bias = fit_logistic(to_x(train), train_y)
    validation_score = sigmoid(to_x(validation) @ weights + bias)
    threshold = best_validation_threshold(validation_y, validation_score)
    test_score = sigmoid(to_x(test) @ weights + bias)
    result = {
        "dataset": "deepstorm_probe_9ch_iwp",
        "target": TARGET,
        "features": FEATURES,
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "positive_rows": {
            "train": int(train_y.sum()), "validation": int(validation_y.sum()), "test": int(test_y.sum()),
        },
        "validation": classification_metrics(validation_y, validation_score, threshold),
        "test": classification_metrics(test_y, test_score, threshold),
    }
    output = ROOT / "results" / "deepstorm_probe_9ch_iwp_baseline.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
