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
            'encoder': case.get('model_size', plan.get('model_size', 'small')).title(),
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
    """Collect runs with common validation; architectures may share a dataset."""
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
        if previous is not None and not previous <= keys:
            raise ValueError('Each larger dataset must contain every smaller-set training clip')
        previous = keys
    return dict(root=studies[0]['root'], roots=[item['root'] for item in studies],
        plan={'cases': list(cases.values())}, cases=cases, runs=runs, reports={},
        captured_at=max(item['captured_at'] for item in studies),
        validation_sha256=studies[0]['validation_sha256'])


def add_completed_experiments(reference, addition):
    """Combine complementary objectives with exactly matched dataset manifests."""
    if reference['validation_sha256'] != addition['validation_sha256']:
        raise ValueError('Additional experiments require the same validation manifest')
    result = {**reference, 'cases': dict(reference['cases']),
              'runs': dict(reference['runs']), 'reports': {}}
    aliases = {}
    for name, case in addition['cases'].items():
        matches = [key for key, old in result['cases'].items()
                   if old['encoder'] == case['encoder'] and old['train_samples'] == case['train_samples']]
        if len(matches) > 1:
            raise ValueError('Ambiguous architecture and dataset size in reference study')
        canonical = matches[0] if matches else name
        aliases[name] = canonical
        if canonical in result['cases']:
            old = result['cases'][canonical]
            for filename in ('train.csv', 'validation.csv', 'channel_stats.json'):
                if (old['queue']/'dataset'/filename).read_bytes() != (case['queue']/'dataset'/filename).read_bytes():
                    raise ValueError('Shared case has different data: '+name)
            if old['encoder'] != case['encoder']:
                raise ValueError('Shared case has a different encoder architecture')
        else:
            result['cases'][name] = case
    for key, run in addition['runs'].items():
        key = (aliases[key[0]], key[1])
        if key in result['runs']:
            raise ValueError('An objective already exists for this case: '+str(key))
        result['runs'][key] = {**run, 'case': key[0]}
    result['captured_at'] = max(reference['captured_at'], addition['captured_at'])
    return result


def attach_distributed_training(reference, directory):
    """Read DDP histories in the existing geographic study without changing data."""
    directory = Path(directory).resolve()
    run_config = read_json(directory/'run_config.json', required=True)
    identity = run_config['identity']
    progress = read_json(directory/'progress.json', required=True)
    if identity['validation_sha256'] != reference['validation_sha256']:
        raise ValueError('DDP and geographic study use different validation clips')
    architecture = 'Base' if identity['model_config']['hidden_size'] == 768 else 'Small'
    matches = [name for name, case in reference['cases'].items()
               if case['train_samples'] == identity['stats']['samples'] and case['encoder'] == architecture]
    if len(matches) != 1:
        raise ValueError('Select exactly one matching geographic training case')
    case_name = matches[0]
    case = reference['cases'][case_name]
    if hashlib.sha256((case['queue']/'dataset/train.csv').read_bytes()).hexdigest() != identity['train_sha256']:
        raise ValueError('DDP training manifest differs from the geographic study')
    if any(case['stats'][key] != identity['stats'][key] for key in ('global_mean_k', 'global_std_k')):
        raise ValueError('DDP global normalization differs from the geographic study')
    objective = identity['objective']
    if objective != 'future':
        raise ValueError('This DDP history adapter currently expects future-frame MSE')
    key = (case_name, objective)
    if reference['runs'].get(key, {}).get('epochs'):
        raise ValueError('Refuse to replace an existing trained experiment')
    epochs = [{**row, 'train_future': row['train_loss'], 'train_total': row['train_loss'],
               'validation_future': row['validation_score']} for row in progress['epochs']]
    if [row['epoch'] for row in epochs] != list(range(1, progress['completed_epochs']+1)):
        raise ValueError('Incomplete distributed epoch history')
    result = {**reference, 'runs': dict(reference['runs']), 'reports': {}}
    stage = dict(name=objective, kind='pretraining', epochs=run_config['arguments']['epochs'])
    result['runs'][key] = dict(case=case_name, name=objective, directory=directory, stage=stage,
        progress=progress, epochs=epochs, loss='mse', loss_weights={'mvm': 1., 'future': 1.},
        completed_epochs=progress['completed_epochs'], complete=progress['status'] == 'completed'
        and progress['completed_epochs'] == stage['epochs'],
        execution=dict(site='Leonardo', global_batch=identity['batch_size'], world_size=identity['world_size']))
    return result


def distributed_validation_figure(directory):
    """Keep the two held-out populations visible with their own baselines."""
    directory = Path(directory)
    progress = read_json(directory/'progress.json', required=True)
    updates = {row['epoch']: row['optimizer_steps'] for row in progress['epochs']}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True, layout='constrained')
    for axis, folder, title in zip(axes, ('validation_epochs','validation_extended'),
                                  ('512 geographic-validation clips', '10,240 extended-validation clips')):
        rows = []
        for path in sorted((directory/folder).glob('epoch_*.json')):
            epoch = int(path.stem.rsplit('_',1)[1])
            if epoch in updates:
                report = read_json(path, required=True)
                rows.append(dict(updates=updates[epoch], **{name:report[name]['rmse_k']
                            for name in ('neural','persistence','advection')}))
        for name, label, color, style in [('neural','Base Future','#9467BD','-'),
            ('persistence','Persistence',COLORS['persistence'],'--'),('advection','Advection',COLORS['advection'],':')]:
            axis.plot([row['updates'] for row in rows], [row[name] for row in rows],
                      label=label, color=color, linestyle=style)
        axis.set(title=title, xlabel='Optimizer updates', ylabel='Forecast RMSE (K)', xscale='log')
        axis.grid(alpha=.2); axis.legend()
    return fig


def distributed_validation_history(directory, folder='validation_extended'):
    """Read the Future MSE on a held-out population at committed updates."""
    directory = Path(directory)
    config = read_json(directory/'run_config.json', required=True)
    if config['identity']['objective'] != 'future':
        raise ValueError('Additional forecast validation currently requires the Future objective')
    progress = read_json(directory/'progress.json', required=True)
    updates = {row['epoch']: row['optimizer_steps'] for row in progress['epochs']
               if row['epoch'] <= progress['completed_epochs']}
    scale = config['identity']['stats']['global_std_k']**2
    records = []
    for path in sorted((directory/folder).glob('epoch_*.json')):
        epoch = int(path.stem.rsplit('_', 1)[1])
        if epoch not in updates:
            continue
        metric = read_json(path, required=True)['neural']
        mse_k2 = metric['mse'] * scale
        if not np.isclose(mse_k2, metric['rmse_k']**2, rtol=1e-6, atol=1e-8):
            raise ValueError('Validation MSE and physical RMSE are inconsistent')
        records.append(dict(epoch=epoch, optimizer_steps=updates[epoch],
                            validation_objective_k2=mse_k2, validation_clips=metric['samples']))
    return pd.DataFrame(records, columns=['epoch', 'optimizer_steps', 'validation_objective_k2',
                                          'validation_clips']).sort_values('optimizer_steps')


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


def dataset_budget_comparison(study, family='pretraining'):
    """All sizes at a global common budget, relative to each architecture's smallest set."""
    if family == 'reconstruction':
        raise ValueError('Dataset budget comparison requires a common forecast target')
    records = []
    for budget in ('epoch', 'optimizer_steps'):
        selection, _ = select_budget(study, family, budget, across_sizes=True)
        selected = {}
        for (case, name), epoch in selection.items():
            data = study['cases'][case]
            history = next(row for row in study['runs'][case, name]['epochs'] if row['epoch'] == epoch)
            report = epoch_report(study, case, name, epoch)
            selected[case, name] = {'Encoder': data.get('encoder', 'Small'),
                'Training clips': data['train_samples'], 'Model': LABELS[name],
                'Matched budget': 'Epochs' if budget == 'epoch' else 'Optimizer updates',
                'Budget': history[budget], 'Epoch': epoch,
                'Optimizer updates': history['optimizer_steps'],
                'Clip presentations': epoch*data['train_samples'],
                'RMSE (K)': report['neural']['rmse_k']}
        reference_cases = {}
        for case, data in study['cases'].items():
            encoder = data.get('encoder', 'Small')
            previous = reference_cases.get(encoder)
            if previous is None or data['train_samples'] < study['cases'][previous]['train_samples']:
                reference_cases[encoder] = case
        for (case, name), row in selected.items():
            reference = selected.get((reference_cases[row['Encoder']], name))
            if reference is None:
                raise ValueError('Each objective requires a saved run at the architecture reference size')
            records.append({**row, 'Reference clips': reference['Training clips'],
                'Reference epoch': reference['Epoch'], 'Reference updates': reference['Optimizer updates'],
                'Reference RMSE (K)': reference['RMSE (K)'],
                'RMSE reduction (K)': reference['RMSE (K)']-row['RMSE (K)']})
    return pd.DataFrame(records)


def dataset_budget_figure(comparison):
    """Vertical bars for every available size; one reference per architecture."""
    if comparison.empty:
        return None
    from matplotlib.patches import Patch
    encoders = sorted(comparison.Encoder.unique(), key=lambda name: (name != 'Small', name))
    inverse = {value: key for key, value in LABELS.items()}
    methods = list(comparison.Model.unique())
    fig, axes = plt.subplots(len(encoders), 2, figsize=(15, 4*len(encoders)),
                             sharey=True, squeeze=False, layout='constrained')
    low, high = min(0., comparison['RMSE reduction (K)'].min()), max(0., comparison['RMSE reduction (K)'].max())
    padding = max((high-low)*.12, .02)
    for i, encoder in enumerate(encoders):
        for j, budget in enumerate(('Epochs', 'Optimizer updates')):
            axis = axes[i, j]
            rows = comparison[(comparison.Encoder == encoder) & (comparison['Matched budget'] == budget)]
            if rows.empty:
                axis.set_visible(False)
                continue
            if rows.Budget.nunique() != 1 or rows['Reference clips'].nunique() != 1:
                raise ValueError('Each panel requires one common budget and reference dataset')
            sizes = sorted(rows['Training clips'].unique())
            panel_methods = [model for model in methods if model in set(rows.Model)]
            width = .8/len(panel_methods)
            for k, model in enumerate(panel_methods):
                for _, row in rows[rows.Model == model].iterrows():
                    bars = axis.bar(sizes.index(row['Training clips'])+(k-(len(panel_methods)-1)/2)*width,
                                    row['RMSE reduction (K)'], width=width, color=COLORS[inverse[model]])
                    if row['Training clips'] != row['Reference clips']:
                        axis.bar_label(bars, labels=[f'{row["RMSE reduction (K)"]:.2f}'], padding=3, fontsize=9)
            other = 'Optimizer updates' if budget == 'Epochs' else 'Epoch'
            ticks = []
            for size in sizes:
                values = rows.loc[rows['Training clips'] == size, other].unique()
                if len(values) != 1:
                    raise ValueError('Objectives at the same size must share the displayed budget')
                ticks.append(f'{size:,}\n{int(values[0]):,}')
            axis.axhline(0, color='black', linewidth=1)
            axis.set(title=f'{encoder} · {int(rows.Budget.iloc[0]):,} {budget.lower()}\n'
                           f'Reference: {int(rows["Reference clips"].iloc[0]):,} clips',
                     ylabel='Forecast RMSE reduction (K)', xticks=range(len(sizes)), xticklabels=ticks,
                     xlabel=f'Training clips / {other.lower()}', ylim=(low-padding, high+padding))
            axis.tick_params(axis='y', labelleft=True)
            axis.grid(axis='y', alpha=.2)
    fig.legend(handles=[Patch(color=COLORS[inverse[model]], label=model) for model in methods],
               loc='outside upper center', ncol=len(methods))
    return fig


def budget_reversal_figure(pair_comparison):
    """Join the gains from two explicit budget criteria for a single dataset pair."""
    if pair_comparison.empty:
        return None
    pairs = pair_comparison[['Smaller training clips', 'Larger training clips', 'Encoder comparison']].drop_duplicates()
    if len(pairs) != 1:
        raise ValueError('Select exactly one dataset pair for the budget transition')
    small, large, encoders = pairs.iloc[0]
    budgets = ('Epochs', 'Optimizer updates')
    inverse = {value: key for key, value in LABELS.items()}
    fig, axis = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for model, rows in pair_comparison.groupby('Model', sort=False):
        indexed = rows.set_index('Matched budget')
        if not all(b in indexed.index for b in budgets):
            continue
        values = indexed.loc[list(budgets), 'RMSE reduction with more data (K)']
        axis.plot([0, 1], values, marker='o', linewidth=2, color=COLORS[inverse[model]], label=model)
    ticks = []
    for budget in budgets:
        rows = pair_comparison[pair_comparison['Matched budget'] == budget]
        if len(rows[['Budget', 'Small-set epoch', 'Large-set epoch']].drop_duplicates()) != 1:
            raise ValueError('Each transition endpoint requires one shared budget')
        row = rows.iloc[0]
        ticks.append(f'{int(row.Budget):,} {budget.lower()}\n'
                     f'Epochs: {int(row["Small-set epoch"])} / {int(row["Large-set epoch"])}')
    axis.set(title=f'{encoders} · {small:,} → {large:,} clips',
             xticks=[0, 1], xticklabels=ticks, xlim=(-.2, 1.2),
             ylabel='Forecast RMSE reduction (K)')
    axis.grid(axis='y', alpha=.2)
    axis.legend(loc='best')
    return fig


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
    groups = list(summary.groupby(['Training clips', 'Encoder'], sort=True))
    methods = method_order(frame)
    figure, axes = _panels(len(groups), sharey=True)
    for axis, ((size, encoder), selected) in zip(axes, groups):
        table = selected.set_index('Model')
        labels = [LABELS[name] for name in methods if LABELS[name] in table.index]
        axis.bar(labels, table.loc[labels, 'RMSE (K)'], color=[COLORS[name] for name in methods if LABELS[name] in table.index])
        epochs = sorted(set(table.Epoch.dropna().astype(int)))
        epoch_label = ', '.join(str(epoch) for epoch in epochs)
        axis.set(title=f'{encoder} · {size:,} clips · epoch {epoch_label}', ylabel='Forecast RMSE (K)', ylim=(0, summary['RMSE (K)'].max()*1.1))
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
    groups = list(table.groupby(['Training clips', 'Encoder'], sort=True))
    labels = list(dict.fromkeys(table.Model))
    inverse = {value: key for key, value in LABELS.items()}
    figure, axes = _panels(len(groups), sharey=True)
    for axis, ((size, encoder), rows) in zip(axes, groups):
        values = rows.pivot(index='Variability', columns='Model', values=metric).reindex(index=['Low', 'Medium', 'High'], columns=labels)
        values.plot.bar(ax=axis, color=[COLORS[inverse[name]] for name in labels], legend=False)
        low, high = min(0., table[metric].min()), max(0., table[metric].max())
        padding = max((high-low)*.1, .01)
        axis.set(title=f'{encoder} · {size:,} clips', ylabel=metric, xlabel='Historical variability',
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


def _size_colors(sizes):
    canonical_colors = {4096: '#0072B2', 8192: '#E69F00', 16384: '#009E73',
                        32768: '#CC79A7', 65536: '#D55E00', 131072: '#9467BD'}
    palette = plt.get_cmap('tab10')
    return {size: canonical_colors.get(size, palette(i % 10)) for i, size in enumerate(sorted(set(sizes)))}


def _validation_trajectory(study, run):
    history = native_loss_history(study, run)[['epoch', 'optimizer_steps', 'validation_objective_k2']].copy()
    if (not np.isfinite(history.to_numpy()).all() or (history.optimizer_steps <= 0).any()
            or (history.optimizer_steps.diff().dropna() <= 0).any()):
        raise ValueError('Validation trajectories require finite values and strictly increasing positive updates')
    history['best_validation_k2'] = history.validation_objective_k2.cummin()
    return history


def objective_minima(study):
    """Minimum of the native validation objective over every committed history."""
    records = []
    for run in study['runs'].values():
        if run['stage']['kind'] != 'pretraining' or not run['epochs']:
            continue
        history = _validation_trajectory(study, run)
        best = history.loc[history.validation_objective_k2.idxmin()]
        case = study['cases'][run['case']]
        records.append({'Objective': LABELS[run['name']], 'Encoder': case.get('encoder', 'Small'),
                        'Training clips': case['train_samples'],
                        'Minimum validation loss (K²)': best.validation_objective_k2,
                        'Epoch of minimum': int(best.epoch), 'Updates at minimum': int(best.optimizer_steps),
                        'Last saved updates': int(history.optimizer_steps.iloc[-1]), 'Complete': run['complete']})
    return pd.DataFrame(records)


def learning_differences(study, comparison, family='pretraining', reference_sizes=None):
    """Candidate minus reference at exact shared updates; cumulative minima use full past histories."""
    if comparison not in ('architecture', 'dataset'):
        raise ValueError('Choose an architecture or dataset comparison')
    names = ('mvm', 'future', 'joint', 'residual') if family == 'pretraining' else FAMILIES[family]
    runs = {}
    for run in study['runs'].values():
        if run['name'] not in names or not run['epochs']:
            continue
        data = study['cases'][run['case']]
        key = (data.get('encoder', 'Small'), data['train_samples'], run['name'])
        if key in runs:
            raise ValueError('Select one run per architecture, dataset size and objective')
        runs[key] = run
    references = (dict(reference_sizes) if reference_sizes is not None else
                  {encoder: min(size for candidate_encoder, size, _ in runs if candidate_encoder == encoder)
                   for encoder, _, _ in runs})
    frames = []
    histories = {}
    for (encoder, size, name), candidate in runs.items():
        if comparison == 'architecture':
            if encoder != 'Base':
                continue
            reference_key = ('Small', size, name)
        else:
            if encoder not in references or size <= references[encoder]:
                continue
            reference_key = (encoder, references[encoder], name)
        reference = runs.get(reference_key)
        if reference is None:
            continue
        if candidate.get('loss_weights', {}) != reference.get('loss_weights', {}):
            raise ValueError('Paired runs require the same objective weights')
        pair = []
        for run in (candidate, reference):
            key = (run['case'], name)
            if key not in histories:
                histories[key] = _validation_trajectory(study, run)
            pair.append(histories[key])
        joined = pair[0].merge(pair[1], on='optimizer_steps', how='inner',
                              suffixes=('_candidate', '_reference'), validate='one_to_one')
        if joined.empty:
            continue
        joined['validation_difference_k2'] = joined.validation_objective_k2_candidate - joined.validation_objective_k2_reference
        joined['best_difference_k2'] = joined.best_validation_k2_candidate - joined.best_validation_k2_reference
        joined['objective'], joined['comparison'] = name, comparison
        joined['encoder'], joined['training_clips'] = encoder, size
        joined['reference_encoder'], joined['reference_clips'] = reference_key[:2]
        joined['candidate_case'], joined['reference_case'] = candidate['case'], reference['case']
        frames.append(joined)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def learning_difference_figures(differences):
    """Signed instantaneous and best-so-far validation differences, separated by objective."""
    if differences.empty:
        return {}
    if differences.comparison.nunique() != 1:
        raise ValueError('Plot one comparison type at a time')
    comparison = differences.comparison.iloc[0]
    colors = _size_colors(differences.training_clips)
    figures = {}
    for name, objective in differences.groupby('objective', sort=False):
        encoders = sorted(objective.encoder.unique(), key=lambda name: (name != 'Small', name))
        fig, axes = plt.subplots(len(encoders), 2, figsize=(14, 4.2*len(encoders)),
                                 squeeze=False, layout='constrained')
        handles = {}
        for i, encoder in enumerate(encoders):
            rows = objective[objective.encoder == encoder]
            for (size, reference_size), curve in rows.groupby(['training_clips', 'reference_clips'], sort=True):
                curve = curve.sort_values('optimizer_steps')
                label = f'{size:,} clips' if comparison == 'architecture' else f'{size:,} − {reference_size:,} clips'
                for j, column in enumerate(('validation_difference_k2', 'best_difference_k2')):
                    line, = axes[i, j].plot(curve.optimizer_steps, curve[column], color=colors[size],
                                          marker='.', markersize=3, linewidth=1.4, label=label)
                    handles[label] = line
            for j, title in enumerate(('Current validation', 'Best validation reached so far')):
                axis = axes[i, j]
                contrast = 'Base − Small' if comparison == 'architecture' else encoder
                axis.axhline(0, color='black', linewidth=1)
                axis.set(title=f'{LABELS[name]} · {contrast} · {title}', xlabel='Optimizer updates',
                         ylabel='Validation loss difference (K²)', xscale='log')
                axis.xaxis.set_major_locator(LogLocator(base=10))
                axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
                axis.grid(alpha=.2, which='both')
        fig.legend(handles=list(handles.values()), loc='outside lower center',
                   ncol=min(3, len(handles)), fontsize=10)
        figures[name] = fig
    return figures


def objective_learning_figures(study, *, additional_studies=None, extra_validation=None,
                               validation_zoom=False,
                               validation_label='Historical validation · 512 clips'):
    """Overlay labelled validation populations without merging their comparisons.

    Each source retains its own normalization and histories. Different manifests
    receive different markers; merge_studies and paired metrics still require
    identical validation observations. Extra held-out curves appear only on
    validation axes. The optional lower panel enlarges the validation range;
    the upper panels retain the complete recorded history and its peaks.
    """
    sources = [(validation_label, study), *(additional_studies or {}).items()]
    extra_validation = extra_validation or []
    population_markers = {}
    marker_choices = (None, 'D', 's', '^', 'v')
    for _, source in sources:
        manifest = source['validation_sha256']
        if manifest not in population_markers:
            if len(population_markers) >= len(marker_choices):
                raise ValueError('Select at most five validation populations')
            population_markers[manifest] = marker_choices[len(population_markers)]
    objectives = ('mvm', 'future', 'joint', 'residual')
    colors = _size_colors(data['train_samples'] for _, source in sources
                          for data in source['cases'].values())
    styles = {'Small': '-', 'Base': '--'}
    figures = {}
    for name in objectives:
        runs = [(label, source, run) for label, source in sources for run in source['runs'].values()
                if run['name'] == name and run['stage']['kind'] == 'pretraining' and run['epochs']]
        if not runs:
            continue
        runs.sort(key=lambda item: (item[1]['cases'][item[2]['case']]['train_samples'],
                                   item[1]['cases'][item[2]['case']].get('encoder', 'Small')))
        if validation_zoom:
            figure = plt.figure(figsize=(15, 8.5), layout='constrained')
            grid = figure.add_gridspec(2, 2, height_ratios=(1, 1.15))
            axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
            detail = figure.add_subplot(grid[1, :], sharex=axes[1])
        else:
            figure, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True, layout='constrained')
            detail = None
        validation_axes = [axes[1]] + ([detail] if detail is not None else [])
        validation_values, settled_values, highlighted_handles = [], [], []
        plotted_sizes, architectures, unfinished = set(), set(), False
        plotted_populations = {}
        for population, source, run in runs:
            data = source['cases'][run['case']]
            size, encoder = data['train_samples'], data.get('encoder', 'Small')
            if encoder not in styles:
                raise ValueError(f'No registered line style for encoder {encoder}')
            history = native_loss_history(source, run)
            if (history[['epoch', 'optimizer_steps']] <= 0).any().any():
                raise ValueError('Logarithmic training axes require positive recorded budgets')
            plotted_sizes.add(size); architectures.add(encoder)
            ongoing = not run['complete']
            unfinished |= ongoing
            marker = population_markers[source['validation_sha256']]
            plotted_populations.setdefault(source['validation_sha256'], (population, marker))
            marks = sorted(set(np.linspace(0, len(history)-1, min(6, len(history)), dtype=int)))
            prominent = source is not study
            line_label = f'{encoder} · {size:,} clips' + (f' · {population}' if additional_studies else '')
            for axis, split in [(axes[0], 'train'), *[(a, 'validation') for a in validation_axes]]:
                axis.plot(history.optimizer_steps, history[f'{split}_objective_k2'],
                    color=colors[size], linestyle=styles[encoder], linewidth=2.8 if prominent else 1.3,
                    alpha=1. if prominent else .8, zorder=6 if prominent else 2,
                    marker=marker or ('o' if ongoing else None),
                    markevery=marks if marker else [len(history)-1],
                    markerfacecolor='white', markersize=5 if prominent else 4,
                    label=line_label)
                if ongoing and marker:
                    axis.scatter(history.optimizer_steps.iloc[-1], history[f'{split}_objective_k2'].iloc[-1],
                                 marker='o', facecolors='white', edgecolors=colors[size], s=36, zorder=8)
            if prominent and validation_zoom:
                highlighted_handles.append(Line2D([], [], color=colors[size], linestyle=styles[encoder],
                    linewidth=2.8, marker=marker, markerfacecolor='white',
                    label=f'{encoder} · {size:,} · {data["validation_samples"]:,} validation clips'))
            validation_values.extend(history.validation_objective_k2)
            settled_values.extend(history.validation_objective_k2.iloc[len(history)//4:])
            best = history.loc[history.validation_objective_k2.idxmin()]
            for axis in validation_axes:
                axis.scatter(best.optimizer_steps, best.validation_objective_k2,
                             marker='*', s=95, color=colors[size], edgecolors='black', linewidths=.5, zorder=7)
        extra_handles = []
        for extra in extra_validation:
            if extra['objective'] != name or extra['history'].empty:
                continue
            history = extra['history']
            values = history[['optimizer_steps', 'validation_objective_k2']].to_numpy()
            if not np.isfinite(values).all() or (values <= 0).any() or (np.diff(values[:, 0]) <= 0).any():
                raise ValueError('Additional validation requires positive losses and increasing updates')
            color, style = colors[extra['training_clips']], styles[extra['encoder']]
            for axis in validation_axes:
                axis.plot(history.optimizer_steps, history.validation_objective_k2,
                          color=color, linestyle=style, linewidth=2.8, marker='s', markersize=5,
                          markerfacecolor='white', label=extra['label'], zorder=6)
                best = history.loc[history.validation_objective_k2.idxmin()]
                axis.scatter(best.optimizer_steps, best.validation_objective_k2,
                             marker='*', s=95, color=color, edgecolors='black', linewidths=.5, zorder=7)
            validation_values.extend(history.validation_objective_k2)
            settled_values.extend(history.validation_objective_k2.iloc[len(history)//4:])
            extra_handles.append(Line2D([], [], color=color, linestyle=style, linewidth=2.8,
                                       marker='s', markerfacecolor='white', label=extra['label']))
        for i, split in enumerate(('Training', 'Validation')):
            axis = axes[i]
            axis.set(title=f'{LABELS[name]} · {split}', xlabel='Optimizer updates',
                     ylabel='Objective loss (K²)', yscale='log', xscale='log')
            axis.xaxis.set_major_locator(LogLocator(base=10))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 1 else f'{v:g}'))
            axis.grid(alpha=.2, which='both')
        if detail is not None:
            lower = min(validation_values)
            upper = max(lower, float(np.quantile(settled_values, .85)))
            padding = max((upper-lower)*.08, lower*.03, .01)
            detail.set(title=f'{LABELS[name]} · Validation detail', xlabel='Optimizer updates',
                       ylabel='Objective loss (K²)', xscale='log',
                       ylim=(max(0., lower-padding), upper+padding))
            detail.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.3g}'))
            detail.grid(alpha=.2, which='both')
            # Setting the scale on a shared axis resets its formatter.
            for axis in (*axes, detail):
                axis.xaxis.set_major_locator(LogLocator(base=10))
                axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 1 else f'{v:g}'))
        handles = [Line2D([], [], color=colors[size], linewidth=2, label=f'{size:,} clips')
                   for size in sorted(plotted_sizes)]
        handles += [Line2D([], [], color='black', linestyle=styles[encoder], label=encoder)
                    for encoder in styles if encoder in architectures]
        if len(plotted_populations) > 1:
            handles += [Line2D([], [], color='black', linestyle='-', marker=marker,
                               markerfacecolor='white', label=label)
                        for label, marker in plotted_populations.values()
                        if not validation_zoom or marker is None]
        handles += highlighted_handles + extra_handles
        handles.append(Line2D([], [], color='black', linestyle='', marker='*', markersize=10,
                              label='Minimum saved validation loss'))
        if unfinished:
            handles.append(Line2D([], [], color='black', linestyle='', marker='o',
                                  markerfacecolor='white', label='Latest saved epoch · incomplete run'))
        figure.legend(handles=handles, loc='outside lower center', ncol=4, fontsize=10)
        figures[name] = figure
    return figures


def learning_figure(study, family='pretraining'):
    names = ('mvm', 'future', 'joint', 'residual') if family == 'pretraining' else FAMILIES[family]
    cases = [case for case in study['cases'] if any(
        study['runs'].get((case, name), {}).get('epochs') for name in names)]
    if not cases:
        return None
    styles = (':', '--', '-.', '-')
    if len(cases) > len(styles):
        raise ValueError('Select at most four dataset sizes for the combined learning curves')
    figure, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True, layout='constrained')
    present = set()
    for case, style in zip(cases, styles):
        for name in names:
            run = study['runs'].get((case, name))
            if not run or not run['epochs']:
                continue
            present.add(name)
            history = native_loss_history(study, run)
            for i, split in enumerate(('train', 'validation')):
                if (history.optimizer_steps <= 0).any():
                    raise ValueError('Logarithmic training axes require positive recorded budgets')
                axes[i].plot(history.optimizer_steps, history[f'{split}_objective_k2'],
                    color=COLORS[name], linestyle=style, linewidth=1.3,
                    label=f"{LABELS[name]} · {study['cases'][case]['train_samples']:,}")
    for i, split in enumerate(('Training', 'Validation')):
        axis = axes[i]
        axis.set(title=split, xlabel='Optimizer updates',
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
    values = table.pivot(index=['Training clips', 'Encoder'], columns='Model', values='Historical reconstruction RMSE (K)')
    values = values.reindex(columns=['MVM', 'Joint'])
    fig, axis = plt.subplots(figsize=(max(11,len(values)*1.2), 4), layout='constrained')
    values.plot.bar(ax=axis, color=[COLORS['mvm'], COLORS['joint']])
    labels = []
    for size, encoder in values.index:
        rows = table[(table['Training clips'] == size) & (table.Encoder == encoder)]
        epochs = ', '.join(map(str, sorted(rows.Epoch.unique())))
        labels.append(f'{encoder} · {size:,}\nepoch {epochs}')
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
    for (size, encoder), rows in summary.groupby(['Training clips', 'Encoder']):
        predictions = rows[rows.Model.isin([LABELS[name] for name in FAMILIES['pretraining']])].sort_values('RMSE (K)')
        if predictions.empty:
            continue
        leader = predictions.iloc[0]
        values = ', '.join(f'{row.Model} {row["RMSE (K)"]:.2f} K' for _, row in predictions.iterrows())
        sentences.append(f'At the common epoch {int(leader.Epoch)}, the global forecast RMSE for {encoder} with {size:,} training clips is '
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
