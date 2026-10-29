"""Streaming forecast scores in physical units, per horizon and channel."""
import torch
from torch.nn import functional as F

from .seviri import THERMAL_CHANNELS


class ForecastMetrics:
    """Aggregate sufficient statistics; never average per-batch RMSEs.

Spatial gradients are forward finite differences in K/pixel. Tendency is
the change from the observed last frame divided by the horizon; its error
equals frame error divided by the horizon, so it is not independent evidence.
"""
    def __init__(self, horizons_minutes, std_k):
        if not horizons_minutes or std_k <= 0:
            raise ValueError("Positive horizons and scalar standard deviation required")
        self.horizons = tuple(horizons_minutes)
        self.std = float(std_k)
        self.sums = {}
        self.counts = {}
        self.samples = 0
        self.support_count = 0
        self.total_count = 0

    def _add(self, name, values, mask):
        mask = mask.expand_as(values)
        sums = torch.where(mask, values.double(), 0).sum((0, 3, 4)).cpu()
        counts = mask.sum((0, 3, 4)).cpu()
        self.sums[name] = self.sums.get(name, 0) + sums
        self.counts[name] = self.counts.get(name, 0) + counts

    @torch.no_grad()
    def update(self, prediction, target, support=None):
        if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[1:3] != (len(self.horizons), 7):
            raise ValueError("Metric inputs must match (B,horizons,7,H,W)")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("Nonfinite predictions or targets")
        prediction = prediction.float() * self.std
        target = target.float() * self.std
        error = prediction - target
        mask = torch.ones_like(error[:, :, :1], dtype=torch.bool)
        self.samples += len(error)
        if support is None:
            support = mask
        if support.shape != mask.shape or support.dtype != torch.bool:
            raise ValueError("Expected common support mask (B,horizons,1,H,W)")
        self.support_count += int(support.sum())
        self.total_count += support.numel()
        for prefix, eligible in (("", mask), ("supported_", support)):
            self._add(prefix + "mae_k", error.abs(), eligible)
            self._add(prefix + "mse_k2", error.square(), eligible)
        for axis in (-2, -1):
            diff = prediction.diff(dim=axis) - target.diff(dim=axis)
            self._add("gradient_mae_k_per_pixel", diff.abs(), torch.ones_like(diff, dtype=torch.bool))
        mins = error.new_tensor(self.horizons)[None, :, None, None, None]
        self._add("tendency_mae_k_per_min", error.abs() / mins, mask)
        intervals = torch.cat((mins[:, :1], mins.diff(dim=1)), dim=1)
        interval_error = torch.cat((error[:, :1], error.diff(dim=1)), dim=1)
        self._add("interval_tendency_mae_k_per_min", interval_error.abs() / intervals, mask)
        # Local minima are defined on the observed target, independently of the
        # prediction, including ties. Mean radiometry cancels in all differences.
        flat = target.reshape(-1, 1, *target.shape[-2:])
        local_minimum = (flat == -F.max_pool2d(-flat, 3, stride=1, padding=1)).reshape_as(target)
        self._add("local_minima_mae_k", error.abs(), local_minimum)
        minimum_error = (prediction.amin((-2, -1)) - target.amin((-2, -1))).abs()
        contrast = lambda x: torch.quantile(x.flatten(-2), .95, dim=-1) - torch.quantile(x.flatten(-2), .05, dim=-1)
        contrast_error = (contrast(prediction) - contrast(target)).abs()
        for name, values in (("minimum_temperature_mae_k", minimum_error), ("contrast_p95_p05_mae_k", contrast_error)):
            values = values[..., None, None]
            self._add(name, values, torch.ones_like(values, dtype=torch.bool))

    def report(self):
        if not self.samples:
            raise ValueError("Cannot report an empty validation split")
        records = []
        for horizon_index, horizon in enumerate(self.horizons):
            for channel_index, channel in enumerate(THERMAL_CHANNELS):
                row = dict(horizon_minutes=horizon, channel=channel)
                for name, sums in self.sums.items():
                    count = int(self.counts[name][horizon_index, channel_index])
                    value = float(sums[horizon_index, channel_index]) / count if count else None
                    row[name] = value
                    if name.endswith("mse_k2"):
                        row[name.replace("mse_k2", "rmse_k")] = value ** .5 if value is not None else None
                records.append(row)
        mse_k2 = float(self.sums["mse_k2"].sum() / self.counts["mse_k2"].sum())
        return dict(samples=self.samples, mse=mse_k2 / self.std ** 2, rmse_k=mse_k2 ** .5,
                    advection_support_fraction=self.support_count / self.total_count,
                    per_channel_horizon=records)
