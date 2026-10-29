"""Causal windows and observations used by the SEVIRI SSL comparison.

Archive clips stay at 16 RSS frames. A separate history/target boundary keeps
future observations out of every encoder call, including joint training.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ForecastWindow:
    history_frames: int = 10
    horizons_minutes: tuple[int, ...] = (15, 30)
    clip_frames: int = 16
    cadence_minutes: int = 5

    def __post_init__(self):
        if self.clip_frames != 16 or self.cadence_minutes != 5:
            raise ValueError("SEVIRI archive clips require 16 frames at five minutes")
        if self.history_frames < 2 or self.history_frames % 2:
            raise ValueError("History must contain at least two frames and complete two-frame tubelets")
        horizons = self.horizons_minutes
        if (not horizons or tuple(sorted(set(horizons))) != tuple(horizons)
                or any(h <= 0 or h % self.cadence_minutes for h in horizons)):
            raise ValueError("Horizons must be increasing, unique positive multiples of five minutes")
        if self.target_indices[-1] >= self.clip_frames:
            raise ValueError("Requested history and future exceed the 16-frame clip")

    @property
    def steps(self):
        return tuple(h // self.cadence_minutes for h in self.horizons_minutes)

    @property
    def target_indices(self):
        return tuple(self.history_frames - 1 + step for step in self.steps)

    def split(self, pixels):
        if pixels.ndim != 5 or pixels.shape[1:3] != (self.clip_frames, 7):
            raise ValueError("Expected complete (B,16,7,H,W) SEVIRI clips")
        if not torch.isfinite(pixels).all():
            raise ValueError("Nonfinite observations cannot be used for forecasting")
        return (pixels[:, :self.history_frames].contiguous(),
                pixels[:, self.target_indices].contiguous())


def persistence(history, horizons):
    return history[:, -1:].expand(-1, horizons, -1, -1, -1)


def temporal_ablation(history, mode="chronological", *, seed=0, indices=None):
    """Explicit diagnostic transforms; never applied inside the archive loader.

The last observed frame stays fixed in the shuffle control, so persistence
and forecast reference time do not change. Seeds are tied to sample indices.
"""
    if mode == "chronological":
        return history
    if mode == "last_frame":
        return history[:, -1:].expand_as(history).contiguous()
    if mode != "shuffle":
        raise ValueError("Unknown temporal ablation")
    if indices is None or len(indices) != len(history):
        raise ValueError("Shuffle requires stable per-sample indices")
    rows = []
    for row, index in zip(history, indices):
        generator = torch.Generator().manual_seed(seed + int(index))
        order = torch.cat((torch.randperm(len(row) - 1, generator=generator),
                           torch.tensor([len(row) - 1])))
        rows.append(row[order.to(history.device)])
    return torch.stack(rows)


def variability_statistics(history_k, threshold_k=2.0):
    """Eulerian IR108 changes, per sample, measured only on historical frames."""
    change = history_k[:, 1:, 4] - history_k[:, :-1, 4]
    flat = change.abs().float().flatten(1)
    return {"mean_abs_change_k": flat.mean(1),
            "p90_abs_change_k": torch.quantile(flat, .9, dim=1),
            "fraction_above_threshold": (flat > threshold_k).float().mean(1)}
