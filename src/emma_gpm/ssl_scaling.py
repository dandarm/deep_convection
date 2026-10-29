"""Collect committed results of nested SEVIRI dataset-size experiments."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def collect_scaling(root):
    root = Path(root)
    plan = read_json(root/'scaling_plan.json')
    rows, states, channels = [], [], []
    for case in plan.get('cases', []):
        queue = root/case['name']
        encoder_size = case.get('model_size', plan.get('model_size', 'small'))
        stats = read_json(queue/'dataset/channel_stats.json')
        variance = stats['global_std_k']**2
        for stage in read_json(queue/'comparison_plan.json').get('stages', []):
            run = queue/stage['name']
            progress = read_json(run/'progress.json')
            completed = progress.get('completed_epochs', 0)
            states.append(dict(dataset=case['name'], encoder_size=encoder_size, training_clips=case['train_samples'],
                model=stage['name'], kind=stage['kind'], completed_epochs=completed,
                requested_epochs=stage['epochs'], status=progress.get('status', 'queued'),
                current_epoch=progress.get('current_epoch'), current_step=progress.get('current_step'),
                steps_per_epoch=progress.get('steps_per_epoch')))
            for epoch in progress.get('epochs', []):
                record = dict(dataset=case['name'], encoder_size=encoder_size, training_clips=case['train_samples'],
                    model=stage['name'], kind=stage['kind'], epoch=epoch['epoch'],
                    optimizer_steps=epoch['optimizer_steps'], clip_presentations=epoch['epoch']*case['train_samples'],
                    train_seconds=epoch['train_seconds'], validation_seconds=epoch['validation_seconds'],
                    validation_rmse_k=epoch.get('validation_rmse_k'),
                    persistence_rmse_k=epoch['persistence_rmse_k'], advection_rmse_k=epoch['advection_rmse_k'])
                for term in ('mvm', 'future'):
                    for split in ('train', 'validation'):
                        value = epoch.get(f'{split}_{term}')
                        record[f'{split}_{term}_mse_k2'] = None if value is None else value*variance
                rows.append(record)
            report = read_json(run/'validation_latest.json')
            for method in ('neural', 'persistence', 'advection'):
                if method != 'neural' and stage['name'] != 'future':
                    continue
                for item in report.get(method, {}).get('per_channel_horizon', []):
                    channels.append(dict(dataset=case['name'], encoder_size=encoder_size, training_clips=case['train_samples'],
                        model=stage['name'] if method=='neural' else method, epoch=completed,
                        kind=stage['kind'] if method=='neural' else 'baseline', **item))
    return pd.DataFrame(rows), pd.DataFrame(states), pd.DataFrame(channels)


def paired_comparisons(curves):
    """Retain both equal-epoch and equal-update contrasts; positive gain favours more data."""
    if curves.empty:
        return pd.DataFrame()
    sizes = sorted(curves.training_clips.unique())
    if len(sizes) != 2:
        return pd.DataFrame()
    small, large = (curves[curves.training_clips==size] for size in sizes)
    rows = []
    for budget in ('epoch', 'optimizer_steps'):
        paired = small.merge(large, on=['model','kind',budget], suffixes=('_small','_large'), validate='one_to_one')
        for item in paired.to_dict('records'):
            row = dict(model=item['model'], kind=item['kind'], matched_budget=budget,
                budget_value=item[budget], small_training_clips=sizes[0], large_training_clips=sizes[1],
                small_epoch=item['epoch'] if budget=='epoch' else item['epoch_small'],
                large_epoch=item['epoch'] if budget=='epoch' else item['epoch_large'])
            for metric in ('validation_rmse_k','validation_mvm_mse_k2'):
                a,b=item.get(metric+'_small'),item.get(metric+'_large')
                if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                    row[f'{metric}_small']=a
                    row[f'{metric}_large']=b
                    row[f'{metric}_gain']=a-b
            rows.append(row)
    return pd.DataFrame(rows)


def publish_summary(root):
    root = Path(root)
    curves, states, channels = collect_scaling(root)
    for name, frame in (('learning_curves.csv',curves),('training_status.csv',states),
                        ('latest_channel_metrics.csv',channels),('paired_comparisons.csv',paired_comparisons(curves))):
        path=root/name
        temporary=path.with_suffix('.csv.part')
        frame.to_csv(temporary,index=False)
        temporary.replace(path)
    return curves,states,channels


def learning_figure(root, kind='pretraining', horizontal='epoch'):
    import matplotlib.pyplot as plt
    from .ssl_dashboard import LABELS
    if horizontal not in ('epoch','optimizer_steps','clip_presentations'):
        raise ValueError('Unknown training budget')
    curves,_,_=collect_scaling(root)
    names=(['future','mvm','joint','residual'] if kind=='pretraining' else
           [f'probe_{name}' for name in ('random','mvm','future','joint','residual')])
    fig,axes=plt.subplots(1,len(names),figsize=(4*len(names),4),constrained_layout=True,squeeze=False)
    handles=[]
    labels=[]
    for axis,name in zip(axes.flat,names):
        metric='validation_mvm_mse_k2' if name=='mvm' else 'validation_rmse_k'
        axis.set(title=LABELS[name],xlabel={'epoch':'Epoch','optimizer_steps':'Optimizer updates',
            'clip_presentations':'Clip presentations'}[horizontal],
            ylabel='Reconstruction MSE (K²)' if name=='mvm' else 'Forecast RMSE (K)')
        if not curves.empty:
            for size,group in curves[curves.model==name].groupby('training_clips'):
                line,=axis.plot(group[horizontal],group[metric],label=f'{size:,} clips',marker='.' if len(group)<3 else None)
                if line.get_label() not in labels:
                    labels.append(line.get_label());handles.append(line)
        axis.grid(alpha=.2)
    if handles:
        fig.legend(handles,labels,loc='outside lower center',ncol=2)
    return fig
