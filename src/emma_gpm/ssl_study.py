"""Saved-result comparisons at explicit, common budgets for the SSL notebook.

This reader uses committed epoch reports, never model inference or live latest
reports. It is independent of the source files sealed by the training supervisor.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FuncFormatter
import numpy as np
import pandas as pd

from .seviri import THERMAL_CHANNELS
from .ssl_dashboard import LABELS, COLORS, METRICS

FAMILIES = {
    'pretraining': ('future', 'joint', 'residual'),
    'frozen_common_head': tuple(f'probe_{name}' for name in ('random', 'mvm', 'future', 'joint', 'residual')),
    'reconstruction': ('mvm', 'joint'),
}


def read_json(path, required=False):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        if required:
            raise
        return {}


def load_study(root):
    root = Path(root).resolve()
    plan = read_json(root/'scaling_plan.json', required=True)
    study = dict(root=root, plan=plan, cases={}, runs={}, reports={},
                 captured_at=datetime.now(ZoneInfo('Europe/Paris')).isoformat())
    validation_hashes = set()
    for case in plan['cases']:
        name = case['name']
        queue = root/name
        local = read_json(queue/'comparison_plan.json', required=True)
        for filename in ('train.csv', 'validation.csv', 'channel_stats.json'):
            actual = hashlib.sha256((queue/'dataset'/filename).read_bytes()).hexdigest()
            if actual != local['dataset'][filename]:
                raise ValueError(f'Prepared dataset changed: {name}/{filename}')
            if filename == 'validation.csv':
                validation_hashes.add(actual)
        stats = read_json(queue/'dataset/channel_stats.json', required=True)
        study['cases'][name] = {**case, 'queue': queue, 'stats': stats,
            'encoder': plan.get('model_size', 'small').title(),
            'protocol': read_json(queue/'dataset/protocol.json'), 'plan': local}
        for stage in local['stages']:
            identity = read_json(queue/stage['name']/'run_config.json').get('identity', {})
            progress = read_json(queue/stage['name']/'progress.json')
            done = progress.get('completed_epochs', 0)
            epochs = [row for row in progress.get('epochs', []) if row['epoch'] <= done]
            if done and (not epochs or epochs[-1]['epoch'] != done):
                raise ValueError(f'Incomplete committed history: {name}/{stage["name"]}')
            study['runs'][name, stage['name']] = dict(case=name, name=stage['name'],
                directory=queue/stage['name'], stage=stage, progress=progress, epochs=epochs,
                loss=identity.get('loss', 'mse'),
                loss_weights={term: identity.get(f'lambda_{term}', 1.) for term in ('mvm', 'future')},
                completed_epochs=done, complete=done == stage['epochs'] and progress.get('status') == 'completed')
    if len(validation_hashes) != 1:
        raise ValueError('Dataset sizes must share the exact validation manifest')
    study['validation_sha256'] = validation_hashes.pop()
    return study


def merge_studies(*studies):
    """Collect registered runs, requiring common validation and nested training clips."""
    if not studies or len({item['validation_sha256'] for item in studies}) != 1:
        raise ValueError('Merged studies require the exact same validation manifest')
    cases, runs = {}, {}
    for item in studies:
        if cases.keys() & item['cases'].keys():
            raise ValueError('Dataset case identifiers must be unique across studies')
        cases.update(item['cases'])
        runs.update(item['runs'])
    cases = dict(sorted(cases.items(), key=lambda item: item[1]['train_samples']))
    previous = None
    for case in cases.values():
        frame = pd.read_csv(case['queue']/'dataset/train.csv')
        keys = set(zip(pd.to_datetime(frame.start_time, utc=True), frame.origin_y, frame.origin_x))
        if len(keys) != len(frame) or len(frame) != case['train_samples']:
            raise ValueError('Training manifests must contain the registered number of unique clips')
        if previous is not None and not previous < keys:
            raise ValueError('Each larger dataset must contain every smaller-set training clip')
        previous = keys
    return dict(root=studies[0]['root'], roots=[item['root'] for item in studies],
        plan={'cases': list(cases.values())}, cases=cases, runs=runs, reports={},
        captured_at=max(item['captured_at'] for item in studies),
        validation_sha256=studies[0]['validation_sha256'])


def extension_study(reference, extension, reference_case='n8192'):
    """A two-size view of the objectives registered in a nested extension."""
    if reference['validation_sha256'] != extension['validation_sha256']:
        raise ValueError('Extension comparison requires the exact same validation manifest')
    if len(extension['cases']) != 1 or reference_case not in reference['cases']:
        raise ValueError('Select one reference size and a single extension size')
    added_case = next(iter(extension['cases']))
    if added_case == reference_case:
        raise ValueError('Reference and extension must have different case names')
    cases = {reference_case: reference['cases'][reference_case], added_case: extension['cases'][added_case]}
    manifests = [pd.read_csv(case['queue']/'dataset/train.csv') for case in cases.values()]
    def clip_keys(frame):
        return set(zip(pd.to_datetime(frame.start_time, utc=True), frame.origin_y, frame.origin_x))
    small, large = map(clip_keys, manifests)
    if len(small) != len(manifests[0]) or len(large) != len(manifests[1]) or not small < large:
        raise ValueError('Extension must contain every reference clip and unique additions')
    names = [name for (case, name), run in extension['runs'].items() if run['stage']['kind'] == 'pretraining']
    if not names or any((reference_case, name) not in reference['runs'] for name in names):
        raise ValueError('Every extension objective requires a reference run')
    runs = {(case, name): source['runs'][case, name]
            for case, source in ((reference_case, reference), (added_case, extension)) for name in names}
    return dict(root=extension['root'], reference_root=reference['root'],
        plan={'cases': list(cases.values())}, cases=cases, runs=runs, reports={},
        captured_at=extension['captured_at'], validation_sha256=extension['validation_sha256'])


def epoch_report(study, case, name, epoch):
    key = (case, name, int(epoch))
    if key not in study['reports']:
        run = study['runs'][case, name]
        if epoch > run['completed_epochs']:
            raise ValueError('Requested validation epoch has not been committed')
        path = run['directory']/'validation_epochs'/f'epoch_{epoch:03d}.json'
        report = read_json(path, required=True)
        expected = {(h, c) for h in (15, 30) for c in THERMAL_CHANNELS}
        for method in ('persistence', 'advection', 'neural'):
            if method not in report:
                continue
            values = report[method]
            records = values['per_channel_horizon']
            if (values['samples'] != study['cases'][case]['validation_samples'] or len(records) != 14
                    or {(r['horizon_minutes'], r['channel']) for r in records} != expected):
                raise ValueError(f'Incomplete validation support: {case}/{name}/{epoch}')
            pooled = np.sqrt(np.mean([row['mse_k2'] for row in records]))
            if not np.isclose(pooled, values['rmse_k'], rtol=1e-6, atol=1e-6):
                raise ValueError('Global RMSE differs from the channel/horizon sufficient statistics')
        study['reports'][key] = report
    return study['reports'][key]


def progress_frame(study):
    records = []
    for run in study['runs'].values():
        progress = run['progress']
        records.append({'Training clips': study['cases'][run['case']]['train_samples'],
            'Encoder': study['cases'][run['case']].get('encoder', 'Small'),
            'Model': LABELS[run['name']], 'Saved epochs': run['completed_epochs'],
            'Target epochs': run['stage']['epochs'], 'State': progress.get('status', 'pending'),
            'Saved optimizer updates': run['epochs'][-1]['optimizer_steps'] if run['epochs'] else 0})
    return pd.DataFrame(records)


def select_budget(study, family='pretraining', budget='epoch', across_sizes=False):
    """Latest budget shared by available models, with explicit missing-model coverage."""
    if family not in FAMILIES or budget not in ('epoch', 'optimizer_steps'):
        raise ValueError('Unknown family or training budget')
    available = {case: [study['runs'][case, name] for name in FAMILIES[family]
                       if (case, name) in study['runs'] and study['runs'][case, name]['epochs']]
                 for case in study['cases']}
    groups = [sum(available.values(), [])] if across_sizes else list(available.values())
    selection = {}
    for runs in groups:
        if not runs or (across_sizes and any(not values for values in available.values())):
            continue
        common = set.intersection(*({row[budget] for row in run['epochs']} for run in runs))
        if not common:
            continue
        target = max(common)
        for run in runs:
            row = next(row for row in run['epochs'] if row[budget] == target)
            selection[run['case'], run['name']] = row['epoch']
    coverage = []
    for case, runs in available.items():
        planned = [name for name in FAMILIES[family] if (case, name) in study['runs']]
        selected = [run for run in runs if (case, run['name']) in selection]
        coverage.append({'Training clips': study['cases'][case]['train_samples'],
            'Encoder': study['cases'][case].get('encoder', 'Small'),
            'Available models': len(runs), 'Planned models': len(planned),
            'Models compared': ', '.join(LABELS[run['name']] for run in selected) or 'Awaiting a common budget',
            'Selected epochs': ', '.join(str(epoch) for epoch in sorted({selection[case, run['name']] for run in selected})),
            'Comparison': 'Final' if planned and len(selected) == len(planned) and all(run['complete'] for run in selected)
                          and all(selection[case, run['name']] == run['stage']['epochs'] for run in selected)
                          else 'Not scheduled' if not planned else 'Interim / partial'})
    return selection, pd.DataFrame(coverage)


def metrics_frame(study, selection, variability='all'):
    records = []
    baseline_reference = {}
    group_counts = {}
    group_edges = None
    for (case, name), epoch in selection.items():
        original = epoch_report(study, case, name, epoch)
        edges = original.get('variability_edges_k')
        if edges is not None:
            if group_edges is not None and not np.allclose(edges, group_edges, rtol=0, atol=1e-10):
                raise ValueError('Variability thresholds differ between compared models')
            group_edges = edges
        report = original if variability == 'all' else original.get('by_historical_variability', {}).get(variability, {})
        if not report or not report.get('advection'):
            continue
        samples = report['advection']['samples']
        previous = group_counts.setdefault(case, samples)
        if previous != samples:
            raise ValueError('Compared models have different validation group membership')
        updates = next(row['optimizer_steps'] for row in study['runs'][case, name]['epochs'] if row['epoch'] == epoch)
        for method in ('persistence', 'advection', 'neural'):
            values = report.get(method)
            if not values:
                continue
            if values['samples'] != samples:
                raise ValueError('Compared predictions have different sample counts')
            if method != 'neural':
                reference_key = (case, method)
                if reference_key in baseline_reference:
                    if not np.isclose(values['rmse_k'], baseline_reference[reference_key], atol=1e-4, rtol=0):
                        raise ValueError('Baselines differ between compared runs')
                    continue
                baseline_reference[reference_key] = values['rmse_k']
            records.extend(dict(case=case, training_clips=study['cases'][case]['train_samples'],
                encoder=study['cases'][case].get('encoder', 'Small'),
                model=name if method == 'neural' else method, epoch=epoch if method == 'neural' else None,
                optimizer_steps=updates if method == 'neural' else None,
                samples=samples, variability=variability, **row) for row in values['per_channel_horizon'])
    frame = pd.DataFrame(records)
    if not frame.empty and frame.duplicated(['case', 'model', 'horizon_minutes', 'channel']).any():
        raise ValueError('Duplicate channel/horizon score')
    return frame


def forecast_summary(frame):
    if frame.empty:
        return pd.DataFrame()
    records = []
    for (case, model), rows in frame.groupby(['case', 'model'], sort=False):
        first = rows.iloc[0]
        record = {'Training clips': int(first.training_clips), 'Model': LABELS[model],
            'Encoder': first.get('encoder', 'Small'),
            'Epoch': first.epoch, 'Optimizer updates': first.optimizer_steps,
            'Validation clips': int(first.samples), 'RMSE (K)': np.sqrt(rows.mse_k2.mean()),
            'MAE (K)': rows.mae_k.mean()}
        for horizon in (15, 30):
            record[f'RMSE +{horizon} min (K)'] = np.sqrt(rows[rows.horizon_minutes == horizon].mse_k2.mean())
        baseline = frame[(frame.case == case) & (frame.model == 'advection')]
        reference = np.sqrt(baseline.mse_k2.mean())
        record['RMSE reduction vs Advection (%)'] = 100*(1-record['RMSE (K)']/reference) if reference else np.nan
        records.append(record)
    return pd.DataFrame(records)


def variability_summary(study, selection):
    groups = []
    for name in ('low', 'medium', 'high'):
        table = forecast_summary(metrics_frame(study, selection, name))
        if not table.empty:
            table['Variability'] = name.capitalize()
            groups.append(table)
    return pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()


def size_comparison(study, family='pretraining'):
    cases = sorted(study['cases'], key=lambda case: study['cases'][case]['train_samples'])
    records = []
    for small, large in zip(cases, cases[1:]):
        names = [name for name in FAMILIES[family] if all((case, name) in study['runs'] for case in (small, large))]
        if not names:
            continue
        pair = {**study, 'cases': {case: study['cases'][case] for case in (small, large)},
                'runs': {(case, name): study['runs'][case, name] for case in (small, large) for name in names}}
        for record in _pair_size_comparison(pair, family):
            record.update({'Smaller training clips': study['cases'][small]['train_samples'],
                           'Larger training clips': study['cases'][large]['train_samples'],
                           'Encoder comparison': f"{study['cases'][small].get('encoder', 'Small')} → {study['cases'][large].get('encoder', 'Small')}"})
            records.append(record)
    return pd.DataFrame(records)


def _pair_size_comparison(study, family):
    cases = list(study['cases'])
    records = []
    for budget in ('epoch', 'optimizer_steps'):
        selection, _ = select_budget(study, family, budget, across_sizes=True)
        for name in FAMILIES[family]:
            keys = [(case, name) for case in cases]
            if not all(key in selection for key in keys):
                continue
            epochs = [selection[key] for key in keys]
            reports = [epoch_report(study, case, name, epoch) for case, epoch in zip(cases, epochs)]
            if family == 'reconstruction':
                errors = [np.sqrt(report['mvm_mse'])*study['cases'][case]['stats']['global_std_k']
                          for case, report in zip(cases, reports)]
            else:
                errors = [report['neural']['rmse_k'] for report in reports]
            value = next(row[budget] for row in study['runs'][cases[0], name]['epochs'] if row['epoch'] == epochs[0])
            records.append({'Model': LABELS[name], 'Matched budget': 'Epochs' if budget == 'epoch' else 'Optimizer updates',
                'Budget': value, 'Small-set epoch': epochs[0], 'Large-set epoch': epochs[1],
                'Small-set RMSE (K)': errors[0], 'Large-set RMSE (K)': errors[1],
                'RMSE reduction with more data (K)': errors[0]-errors[1]})
    return records


def trajectory_summary(study):
    records = []
    for run in study['runs'].values():
        if not run['epochs'] or run['stage']['kind'] != 'pretraining':
            continue
        std = study['cases'][run['case']]['stats']['global_std_k']
        task = 'Historical reconstruction' if run['name'] == 'mvm' else 'Future prediction'
        score = (lambda row: np.sqrt(row['validation_mvm'])*std) if run['name'] == 'mvm' else lambda row: row['validation_rmse_k']
        best = min(run['epochs'], key=score)
        last = run['epochs'][-1]
        records.append({'Training clips': study['cases'][run['case']]['train_samples'],
            'Encoder': study['cases'][run['case']].get('encoder', 'Small'),
            'Model': LABELS[run['name']], 'Task': task, 'Latest epoch': last['epoch'],
            'Latest RMSE (K)': score(last), 'Minimum recorded RMSE (K)': score(best),
            'Epoch of minimum': best['epoch'], 'Latest minus minimum (K)': score(last)-score(best)})
    return pd.DataFrame(records)


def reconstruction_summary(study):
    selection, coverage = select_budget(study, 'reconstruction')
    records = []
    for (case, name), epoch in selection.items():
        report = epoch_report(study, case, name, epoch)
        std = study['cases'][case]['stats']['global_std_k']
        records.append({'Training clips': study['cases'][case]['train_samples'], 'Model': LABELS[name],
            'Encoder': study['cases'][case].get('encoder', 'Small'),
            'Epoch': epoch, 'Historical reconstruction RMSE (K)': np.sqrt(report['mvm_mse'])*std})
    return pd.DataFrame(records), coverage


def method_order(frame):
    return [name for name in ('persistence', 'advection', *FAMILIES['pretraining'], *FAMILIES['frozen_common_head'])
            if name in set(frame.model)]


def _panels(count, width=7, **kwargs):
    columns = min(count, 2)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(width*columns, 4*rows),
                             squeeze=False, layout='constrained', **kwargs)
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    return fig, list(axes.flat[:count])


def _case_label(rows):
    return f"{rows.encoder.iloc[0] if 'encoder' in rows else 'Small'} · {int(rows.training_clips.iloc[0]):,} clips"


def channel_figure(frame, metric='rmse_k'):
    if frame.empty:
        return None
    cases = sorted(frame.case.unique(), key=lambda case: frame[frame.case == case].training_clips.iloc[0])
    methods = method_order(frame)
    figure, axes = plt.subplots(len(cases), 2, figsize=(15, 4*len(cases)), sharey=True, squeeze=False, layout='constrained')
    maximum = frame[metric].max()
    for i, case in enumerate(cases):
        for j, horizon in enumerate((15, 30)):
            rows = frame[(frame.case == case) & (frame.horizon_minutes == horizon)]
            table = rows.pivot(index='channel', columns='model', values=metric).reindex(index=THERMAL_CHANNELS, columns=methods)
            table.rename(columns=LABELS).plot.bar(ax=axes[i, j], color=[COLORS[name] for name in methods], legend=False)
            axes[i, j].set(title=f'{_case_label(rows)} · +{horizon} min', xlabel='Channel', ylabel=METRICS[metric],
                           ylim=(0, maximum*1.1 if maximum > 0 else 1))
            axes[i, j].tick_params(axis='x', rotation=30)
            axes[i, j].tick_params(axis='y', labelleft=True)
            axes[i, j].grid(axis='y', alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=len(methods))
    return figure


def aggregate_channels(frame):
    """Add a pooled-channel row per forecast; pool MSE before taking its root.

    Other metrics are equal-weight channel means. In particular, the mean of
    channel-specific local-minimum MAEs is a macro-average, since the number
    of observed minima can differ between channels.
    """
    pooled = []
    keys = ['case', 'model', 'horizon_minutes']
    metrics = set(METRICS) | {'mse_k2', 'supported_mse_k2'}
    for _, rows in frame.groupby(keys, sort=False):
        if len(rows) != len(THERMAL_CHANNELS) or set(rows.channel) != set(THERMAL_CHANNELS):
            raise ValueError('Channel aggregation requires each thermal channel exactly once')
        if rows.samples.nunique() != 1:
            raise ValueError('Channel aggregation requires identical clip support')
        record = rows.iloc[0].to_dict()
        record['channel'] = 'All channels'
        for metric in metrics & set(rows.columns):
            source = metric.replace('rmse_k', 'mse_k2') if metric.endswith('rmse_k') else metric
            if source not in rows:
                raise ValueError('Pooled RMSE requires saved squared errors')
            value = rows[source].mean()
            record[metric] = np.sqrt(value) if metric.endswith('rmse_k') else value
        pooled.append(record)
    return pd.concat([frame, pd.DataFrame(pooled)], ignore_index=True)


def channel_training_figure(frame, metric='rmse_k', channels=None):
    """Compare all model/dataset combinations inside each channel and horizon."""
    if frame.empty:
        return None
    if metric not in METRICS:
        raise ValueError('Unknown forecast metric')
    combined = aggregate_channels(frame)
    channels = ['All channels', *THERMAL_CHANNELS] if channels is None else list(channels)
    if not channels or not set(channels) <= set(combined.channel):
        raise ValueError('Select available channels or All channels')
    cases = sorted(frame.case.unique(), key=lambda case: frame[frame.case == case].training_clips.iloc[0])
    methods = [name for name in method_order(frame) if name not in ('persistence', 'advection')]
    fig, axes = plt.subplots(len(channels), 2, figsize=(15, 3.4*len(channels)),
                             squeeze=False, sharey='row', layout='constrained')
    width = .8 / max(1, len(methods))
    positions = np.arange(len(cases))
    labels = [_case_label(frame[frame.case == case]).replace(' · ', '\n').replace(' clips', '') for case in cases]
    for i, channel in enumerate(channels):
        maximum = combined.loc[combined.channel == channel, metric].max()
        for j, horizon in enumerate((15, 30)):
            axis = axes[i, j]
            rows = combined[(combined.channel == channel) & (combined.horizon_minutes == horizon)]
            for k, method in enumerate(methods):
                values = rows[rows.model == method].set_index('case')[metric]
                for index, case in enumerate(cases):
                    if case in values:
                        axis.bar(positions[index] + (k-(len(methods)-1)/2)*width,
                                 values.loc[case], width=width, color=COLORS[method])
            for baseline in ('persistence', 'advection'):
                values = rows.loc[rows.model == baseline, metric]
                if len(values):
                    if not np.allclose(values, values.iloc[0], rtol=1e-6, atol=1e-4):
                        raise ValueError('Reference forecasts differ across dataset sizes')
                    axis.axhline(values.iloc[0], color=COLORS[baseline], linestyle='--', linewidth=1.3)
            axis.set(title=f'{channel} · +{horizon} min', ylabel=METRICS[metric],
                     xticks=positions, xticklabels=labels, ylim=(0, maximum*1.08 if maximum > 0 else 1))
            axis.tick_params(axis='y', labelleft=True)
            axis.grid(axis='y', alpha=.2)
    from matplotlib.patches import Patch
    handles = [Patch(color=COLORS[method], label=LABELS[method]) for method in methods]
    handles += [Line2D([], [], color=COLORS[name], linestyle='--', label=LABELS[name])
                for name in ('persistence', 'advection') if name in set(frame.model)]
    fig.legend(handles=handles, loc='outside upper center', ncol=min(5, len(handles)))
    return fig


def size_comparison_figure(comparison):
    """Display the change in error at each explicitly matched pairwise budget."""
    if comparison.empty:
        return None
    pair_columns = ['Smaller training clips', 'Larger training clips', 'Encoder comparison']
    pairs = comparison[pair_columns].drop_duplicates()
    fig, axes = plt.subplots(len(pairs), 2, figsize=(14, 3.2*len(pairs)),
                             sharex=True, squeeze=False, layout='constrained')
    gain = 'RMSE reduction with more data (K)'
    limits = (min(0., comparison[gain].min()), max(0., comparison[gain].max()))
    padding = max((limits[1]-limits[0])*.12, .01)
    inverse = {value: key for key, value in LABELS.items()}
    for i, (small, large, encoders) in enumerate(pairs.itertuples(index=False, name=None)):
        selected = comparison[(comparison[pair_columns[0]] == small) & (comparison[pair_columns[1]] == large)]
        first, second = encoders.split(' → ')
        for j, budget in enumerate(('Epochs', 'Optimizer updates')):
            axis = axes[i, j]
            rows = selected[selected['Matched budget'] == budget]
            if rows.empty:
                axis.set_visible(False)
                continue
            if rows.Budget.nunique() != 1:
                raise ValueError('Each comparison panel requires one common training budget')
            axis.barh(rows.Model, rows[gain], color=[COLORS[inverse[name]] for name in rows.Model])
            axis.invert_yaxis()
            axis.axvline(0, color='black', linewidth=1)
            axis.set(title=f'{first} {small:,} → {second} {large:,}\n{int(rows.Budget.iloc[0]):,} {budget.lower()}',
                     xlabel='RMSE reduction (K)', xlim=(limits[0]-padding, limits[1]+padding))
            axis.tick_params(axis='x', labelbottom=True)
            axis.grid(axis='x', alpha=.2)
    return fig


def probe_reference_figure(comparison):
    if comparison.empty:
        return None
    rows = comparison[comparison.Model != LABELS['probe_random']]
    sizes = sorted(rows['Training clips'].unique())
    fig, axes = _panels(len(sizes), sharex=True)
    inverse = {value: key for key, value in LABELS.items()}
    for axis, size in zip(axes, sizes):
        selected = rows[rows['Training clips'] == size]
        axis.barh(selected.Model, selected['RMSE reduction vs Random (K)'],
                  color=[COLORS[inverse[name]] for name in selected.Model])
        axis.axvline(0, color='black', linewidth=1)
        axis.invert_yaxis()
        axis.set(title=f'{size:,} head-fitting clips', xlabel='RMSE reduction vs Random (K)')
        axis.grid(axis='x', alpha=.2)
    return fig


def overview_figure(frame):
    if frame.empty:
        return None
    summary = forecast_summary(frame)
    sizes = sorted(summary['Training clips'].unique())
    methods = method_order(frame)
    figure, axes = _panels(len(sizes), sharey=True)
    for axis, size in zip(axes, sizes):
        table = summary[summary['Training clips'] == size].set_index('Model')
        labels = [LABELS[name] for name in methods if LABELS[name] in table.index]
        axis.bar(labels, table.loc[labels, 'RMSE (K)'], color=[COLORS[name] for name in methods if LABELS[name] in table.index])
        epochs = sorted(set(table.Epoch.dropna().astype(int)))
        epoch_label = ', '.join(str(epoch) for epoch in epochs)
        axis.set(title=f'{_case_label(frame[frame.training_clips == size])} · epoch {epoch_label}', ylabel='Forecast RMSE (K)', ylim=(0, summary['RMSE (K)'].max()*1.1))
        axis.tick_params(axis='x', rotation=20)
        axis.grid(axis='y', alpha=.2)
    return figure


def variability_figure(table, metric='RMSE (K)'):
    if metric not in ('RMSE (K)', 'RMSE reduction vs Advection (%)'):
        raise ValueError('Unknown variability comparison metric')
    if metric != 'RMSE (K)' and not table.empty:
        table = table[~table.Model.isin([LABELS['persistence'], LABELS['advection']])]
    if table.empty:
        return None
    sizes = sorted(table['Training clips'].unique())
    labels = list(dict.fromkeys(table.Model))
    inverse = {value: key for key, value in LABELS.items()}
    figure, axes = _panels(len(sizes), sharey=True)
    for axis, size in zip(axes, sizes):
        rows = table[table['Training clips'] == size]
        values = rows.pivot(index='Variability', columns='Model', values=metric).reindex(index=['Low', 'Medium', 'High'], columns=labels)
        values.plot.bar(ax=axis, color=[COLORS[inverse[name]] for name in labels], legend=False)
        low, high = min(0., table[metric].min()), max(0., table[metric].max())
        padding = max((high-low)*.1, .01)
        axis.set(title=f'{rows.Encoder.iloc[0]} · {size:,} clips', ylabel=metric, xlabel='Historical variability',
                 ylim=(low-padding if low < 0 else 0, high+padding))
        axis.axhline(0, color='black', linewidth=.8)
        axis.tick_params(axis='x', rotation=0)
        axis.grid(axis='y', alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=len(labels))
    return figure


def structure_figure(frame, channel='IR_108'):
    rows = frame[frame.channel == channel]
    cases = sorted(rows.case.unique(), key=lambda case: rows[rows.case == case].training_clips.iloc[0])
    figure, axes = _panels(len(cases), sharex=True, sharey=True)
    for axis, case in zip(axes, cases):
        selected = rows[rows.case == case]
        for row in selected.itertuples():
            axis.scatter(row.rmse_k, row.gradient_mae_k_per_pixel, color=COLORS[row.model],
                         marker='o' if row.horizon_minutes == 15 else 's', s=60)
        axis.set(title=f'{_case_label(selected)} · {channel}',
                 xlabel='Forecast RMSE (K)', ylabel='Gradient MAE (K/pixel)')
        axis.grid(alpha=.2)
    handles = [Line2D([], [], color=COLORS[name], marker='o', linestyle='', label=LABELS[name]) for name in method_order(rows)]
    handles += [Line2D([], [], color='gray', marker=marker, linestyle='', label=f'+{horizon} min') for horizon, marker in ((15, 'o'), (30, 's'))]
    figure.legend(handles=handles, loc='outside lower center', ncol=4)
    return figure


def native_loss_history(study, run):
    """Express each run's actual weighted MSE objective in physical units."""
    if run.get('loss', 'mse') != 'mse':
        raise ValueError('Physical squared-error curves require an MSE training objective')
    history = pd.DataFrame(run['epochs']).copy()
    name = run['name']
    terms = ('mvm', 'future') if name == 'joint' else ('mvm',) if name == 'mvm' else ('future',)
    weights = run.get('loss_weights', {'mvm': 1., 'future': 1.})
    scale = study['cases'][run['case']]['stats']['global_std_k']**2
    for split in ('train', 'validation'):
        values = sum(history[f'{split}_{term}'] * weights[term] for term in terms)
        if split == 'train' and 'train_total' in history:
            if not np.allclose(values, history.train_total, rtol=1e-5, atol=1e-7):
                raise ValueError('Recorded total loss differs from the weighted objective components')
        history[f'{split}_objective_k2'] = values * scale
    return history


def learning_figure(study, family='pretraining'):
    names = ('mvm', 'future', 'joint', 'residual') if family == 'pretraining' else FAMILIES[family]
    cases = [case for case in study['cases'] if any(
        study['runs'].get((case, name), {}).get('epochs') for name in names)]
    if not cases:
        return None
    styles = (':', '--', '-.', '-')
    if len(cases) > len(styles):
        raise ValueError('Select at most four dataset sizes for the combined learning curves')
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True, layout='constrained')
    present = set()
    for case, style in zip(cases, styles):
        for name in names:
            run = study['runs'].get((case, name))
            if not run or not run['epochs']:
                continue
            present.add(name)
            history = native_loss_history(study, run)
            for i, split in enumerate(('train', 'validation')):
                for j, budget in enumerate(('epoch', 'optimizer_steps')):
                    if (history[budget] <= 0).any():
                        raise ValueError('Logarithmic training axes require positive recorded budgets')
                    axes[i, j].plot(history[budget], history[f'{split}_objective_k2'],
                        color=COLORS[name], linestyle=style, linewidth=1.3,
                        label=f"{LABELS[name]} · {study['cases'][case]['train_samples']:,}")
    for i, split in enumerate(('Training', 'Validation')):
        for j, budget in enumerate(('Epoch', 'Optimizer updates')):
            axis = axes[i, j]
            axis.set(title=f'{split} · {budget.lower()}', xlabel=budget,
                     ylabel='Objective loss (K²)', yscale='log', xscale='log')
            axis.xaxis.set_major_locator(LogLocator(base=10))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}' if value >= 1 else f'{value:g}'))
            axis.grid(alpha=.2, which='both')
    handles = [Line2D([], [], color=COLORS[name], label=LABELS[name]) for name in names if name in present]
    handles += [Line2D([], [], color='black', linestyle=style,
                label=f"{study['cases'][case].get('encoder', 'Small')} · {study['cases'][case]['train_samples']:,} clips")
                for case, style in zip(cases, styles)]
    figure.legend(handles=handles, loc='outside lower center', ncol=4, fontsize=10)
    return figure


def reconstruction_comparison_figure(study):
    table, _ = reconstruction_summary(study)
    if table.empty:
        return None
    sizes = table['Training clips'].unique()
    values = table.pivot(index='Training clips', columns='Model', values='Historical reconstruction RMSE (K)')
    values = values.reindex(index=sizes, columns=['MVM', 'Joint'])
    fig, axis = plt.subplots(figsize=(11, 4), layout='constrained')
    values.plot.bar(ax=axis, color=[COLORS['mvm'], COLORS['joint']])
    labels = []
    for size in sizes:
        rows = table[table['Training clips'] == size]
        epochs = ', '.join(map(str, sorted(rows.Epoch.unique())))
        labels.append(f'{rows.Encoder.iloc[0]} · {size:,}\nepoch {epochs}')
    axis.set_xticklabels(labels, rotation=0)
    axis.set(title='Reconstruction of hidden historical pixels', xlabel='Training set and selected checkpoint',
             ylabel='Reconstruction RMSE (K)', ylim=(0, table['Historical reconstruction RMSE (K)'].max()*1.1))
    axis.grid(axis='y', alpha=.2)
    return fig


def diagnostic_availability(study):
    records = []
    for case, data in study['cases'].items():
        analysis_status = read_json(data['queue']/'analysis/progress.json').get('status')
        for label, file in [('Temporal interventions', 'temporal_ablations.csv'),
                            ('Radiometric ridge probes', 'radiometric_probes.csv'),
                            ('Frozen-weight verification', 'frozen_encoder_checks.json'),
                            ('Reconstruction and forecast arrays', 'examples/catalog.json')]:
            path = data['queue']/'analysis'/file
            records.append({'Training clips': data['train_samples'], 'Diagnostic': label,
                            'Status': ('Available' if analysis_status == 'completed' else 'Available; analysis in progress')
                            if path.exists() else 'Not included in this series' if not any(
                                stage['kind'] == 'frozen_common_head' for stage in data['plan']['stages'])
                            else 'Scheduled after the training series'})
    return pd.DataFrame(records)


def probe_reference_summary(summary):
    """Compare each frozen probe with Random at the same size and epoch."""
    if summary.empty:
        return pd.DataFrame()
    probes = summary[summary.Model.str.startswith('Probe (')].copy()
    if probes.empty:
        return pd.DataFrame()
    keys = ['Training clips', 'Epoch']
    reference = probes.loc[probes.Model == LABELS['probe_random'], keys + ['RMSE (K)']]
    reference = reference.rename(columns={'RMSE (K)': 'Random reference RMSE (K)'})
    compared = probes.merge(reference, on=keys, how='left', validate='many_to_one')
    if compared['Random reference RMSE (K)'].isna().any():
        raise ValueError('Each probe requires a Random reference at the same size and epoch')
    compared['RMSE reduction vs Random (K)'] = compared['Random reference RMSE (K)'] - compared['RMSE (K)']
    compared['RMSE reduction vs Random (%)'] = 100 * compared['RMSE reduction vs Random (K)'] / compared['Random reference RMSE (K)']
    return compared[['Training clips', 'Model', 'Epoch', 'RMSE (K)',
                     'RMSE reduction vs Random (K)', 'RMSE reduction vs Random (%)']]


def interpretation(study, summary):
    sentences = []
    for size, rows in summary.groupby('Training clips'):
        predictions = rows[rows.Model.isin([LABELS[name] for name in FAMILIES['pretraining']])].sort_values('RMSE (K)')
        if predictions.empty:
            continue
        leader = predictions.iloc[0]
        values = ', '.join(f'{row.Model} {row["RMSE (K)"]:.2f} K' for _, row in predictions.iterrows())
        sentences.append(f'At the common epoch {int(leader.Epoch)}, the global forecast RMSE with {size:,} training clips is '
                         f'{values}. {leader.Model} has the lowest value among the available learned forecasts, '
                         f'with a {leader["RMSE reduction vs Advection (%)"]:.1f}% reduction relative to Advection.')
    probe_runs = [run for run in study['runs'].values() if run['stage']['kind'] == 'frozen_common_head']
    pretrained_complete = sum(run['complete'] for run in probe_runs if run['name'] != 'probe_random')
    pretrained_total = sum(run['name'] != 'probe_random' for run in probe_runs)
    if pretrained_total and pretrained_complete == pretrained_total:
        sentences.append('All frozen-encoder forecasting probes are complete. Their shared prediction task compares '
                         'MVM, Future, Joint and Residual representations with the randomly initialized encoder.')
    elif pretrained_total:
        sentences.append(f'The common forecasting probes using pretrained encoders have completed {pretrained_complete} of '
                         f'{pretrained_total} runs. Their completion supplies the common-task comparison of all four SSL objectives.')
    probe_selection, _ = select_budget(study, 'frozen_common_head')
    probe_summary = forecast_summary(metrics_frame(study, probe_selection))
    if not probe_summary.empty:
        for size, rows in probe_summary.groupby('Training clips'):
            predictions = rows[rows.Model.str.startswith('Probe (')].sort_values('RMSE (K)')
            if len(predictions) > 1:
                leader = predictions.iloc[0]
                sentences.append(f'At the common probe epoch {int(leader.Epoch)} with {size:,} training clips, '
                    f'{leader.Model} has the lowest RMSE among the {len(predictions)} available probes: '
                    f'{leader["RMSE (K)"]:.2f} K. The probe coverage table identifies the encoders included in this comparison.')
    sentences.append('These measurements describe the shared validation observations, the recorded training budgets and one initialization seed. '
                     'Channel, lead-time, variability and structural diagnostics below determine where the global ranking holds. '
                     'Additional observation periods and seeds will quantify its reproducibility.')
    return '\n\n'.join(sentences)
