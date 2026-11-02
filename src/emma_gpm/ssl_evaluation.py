"""Comparable forecasting evaluation on the same observations and support."""
from pathlib import Path

import torch

from .advection import advection_forecast
from .forecasting import persistence, temporal_ablation, variability_statistics
from .forecast_metrics import ForecastMetrics
from .ssl_masking import fixed_ssl_mask
from .videomae_v2 import fixed_running_cell_mask


@torch.no_grad()
def evaluate_ssl(model, loader, window, device, mean_k, std_k, *, mask_ratio=.9,
                 mask_strategy="tube", seed=314159, precision="fp32", max_shift=12,
                 ablation="chronological", variability_edges=None, example_path=None,
                 sample_days=None, progress_callback=None, distributed=False):
    if sample_days is not None and len(sample_days) != len(loader.dataset):
        raise ValueError('One ordered UTC day per validation sample required')
    mode = model.training
    model.eval()
    names = ["persistence", "advection"] + ([] if model.head is None else ["neural"])
    scores = {name: ForecastMetrics(window.horizons_minutes, std_k) for name in names}
    stratified = {level: {name: ForecastMetrics(window.horizons_minutes, std_k) for name in names}
                  for level in ("low", "medium", "high")} if variability_edges is not None else {}
    mvm_sum, count = 0., 0
    daily = {}
    variability = []
    try:
        for batch in loader:
            clip = batch["pixel_values"].to(device, non_blocking=loader.pin_memory)
            original_history, target = window.split(clip)
            indices = batch.get('sample_indices', list(range(count, count + len(clip))))
            batch_days = [sample_days[i] for i in indices] if sample_days is not None else []
            if model.mae is not None and len(set(batch_days)) > 1:
                raise ValueError('Per-day MVM metrics require batches contained within a UTC day')
            history = temporal_ablation(original_history, ablation, seed=seed, indices=indices)
            # Baselines are fixed on real chronological history, even when the
            # encoder is ablated. Residual decomposition uses this same baseline.
            adv, support, _ = advection_forecast(original_history, window.steps, max_shift)
            mask = decode = None
            if model.mae is not None:
                mask = fixed_ssl_mask(model.config, indices, mask_ratio, mask_strategy, seed, device)
                decode = fixed_running_cell_mask(model.config, indices, .5, seed + 1000000, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                output = model(history, mask=mask, decode_mask=decode, advection=adv)
            predictions = dict(persistence=persistence(original_history, len(window.steps)), advection=adv)
            if "prediction" in output:
                predictions["neural"] = output["prediction"]
            if "mvm_loss" in output:
                mvm_sum += float(output["mvm_loss"]) * len(clip)
            change = variability_statistics(original_history * std_k + mean_k)["mean_abs_change_k"]
            variability.extend(change.cpu().tolist())
            for name, prediction in predictions.items():
                scores[name].update(prediction, target, support)
                if stratified:
                    level = torch.bucketize(change, change.new_tensor(variability_edges), right=True)
                    for index, label in enumerate(stratified):
                        selected = level == index
                        if selected.any():
                            stratified[label][name].update(prediction[selected], target[selected], support[selected])
            for day in sorted(set(batch_days)):
                if day not in daily:
                    daily[day] = dict(scores={name:ForecastMetrics(window.horizons_minutes, std_k) for name in names},
                                      mvm_sum=0., samples=0)
                selected = torch.tensor([value == day for value in batch_days], device=device, dtype=torch.bool)
                size = batch_days.count(day)
                for name, prediction in predictions.items():
                    daily[day]['scores'][name].update(prediction[selected], target[selected], support[selected])
                daily[day]['samples'] += size
                if 'mvm_loss' in output:
                    daily[day]['mvm_sum'] += float(output['mvm_loss']) * size
            if count == 0 and example_path is not None:
                plot_forecasts(original_history[0], target[0], {name: values[0] for name, values in predictions.items()},
                               window, mean_k, std_k, example_path)
            count += len(clip)
            if progress_callback is not None:
                progress_callback(count)
        if distributed:
            import torch.distributed as dist
            payload = dict(scores=scores, stratified=stratified, daily=daily, count=count,
                           mvm_sum=mvm_sum, variability=variability)
            parts = [None] * dist.get_world_size()
            dist.all_gather_object(parts, payload)
            combined = parts[0]
            for other in parts[1:]:
                for name,metric in other['scores'].items():combined['scores'][name].merge(metric)
                for level,methods in other['stratified'].items():
                    for name,metric in methods.items():combined['stratified'][level][name].merge(metric)
                for day,values in other['daily'].items():
                    if day not in combined['daily']:
                        combined['daily'][day] = values
                    else:
                        target_day = combined['daily'][day]
                        for name,metric in values['scores'].items():target_day['scores'][name].merge(metric)
                        target_day['samples'] += values['samples']
                        target_day['mvm_sum'] += values['mvm_sum']
                combined['count'] += other['count']
                combined['mvm_sum'] += other['mvm_sum']
                combined['variability'].extend(other['variability'])
            scores, stratified, daily = (combined[k] for k in ('scores','stratified','daily'))
            count, mvm_sum, variability = (combined[k] for k in ('count','mvm_sum','variability'))
        report = {name: metric.report() for name, metric in scores.items()}
        report["mvm_mse"] = mvm_sum / count if model.mae is not None and count else None
        report["ablation"] = ablation
        report["historical_ir108_mean_abs_change_k"] = variability
        if sample_days is not None:
            report['by_day'] = {day:dict(
                samples=values['samples'],
                mvm_mse=values['mvm_sum']/values['samples'] if model.mae is not None else None,
                **{name:metric.report() for name,metric in values['scores'].items()})
                for day,values in sorted(daily.items())}
        if stratified:
            report["variability_edges_k"] = list(variability_edges)
            report["by_historical_variability"] = {
                level: {name: metric.report() if metric.samples else None for name, metric in metrics.items()}
                for level, metrics in stratified.items()}
        return report
    finally:
        model.train(mode)


def plot_forecasts(history, target, predictions, window, mean_k, std_k, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["last observed", "future observed", *predictions]
    figure, axes = plt.subplots(len(window.steps), len(columns), figsize=(3 * len(columns), 3 * len(window.steps)), squeeze=False)
    for index, horizon in enumerate(window.horizons_minutes):
        values = [history[-1, 4], target[index, 4], *(field[index, 4] for field in predictions.values())]
        for axis, name, field in zip(axes[index], columns, values):
            image = axis.imshow((field.detach().float().cpu() * std_k + mean_k).numpy(), vmin=180, vmax=330, cmap="inferno")
            axis.set_title(f"{name}\n" + ("t = 0" if name == "last observed" else f"+{horizon} min"))
            axis.set_xticks([])
            axis.set_yticks([])
    figure.colorbar(image, ax=axes.ravel().tolist(), label="IR108 brightness temperature (K)", shrink=.8)
    figure.suptitle("SEVIRI forecast comparison; north-up, west-left")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
