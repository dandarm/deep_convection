"""Notebook views of pilot datasets, learning curves and saved evaluations."""
import asyncio
from contextlib import contextmanager
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .seviri import THERMAL_CHANNELS

LABELS={'mvm':'MVM','future':'Future','joint':'Joint','residual':'Residual',
    'probe_random':'Probe (Random)','probe_mvm':'Probe (MVM)','probe_future':'Probe (Future)',
    'probe_joint':'Probe (Joint)','probe_residual':'Probe (Residual)',
    'persistence':'Persistence','advection':'Advection'}
METRICS={'rmse_k':'RMSE (K)','mae_k':'MAE (K)',
    'supported_rmse_k':'Supported RMSE (K)',
    'gradient_mae_k_per_pixel':'Gradient MAE (K/pixel)',
    'local_minima_mae_k':'Local-minimum MAE (K)',
    'minimum_temperature_mae_k':'Minimum-temperature MAE (K)',
    'contrast_p95_p05_mae_k':'Contrast MAE (K)',
    'tendency_mae_k_per_min':'Reference-time tendency MAE (K/min)',
    'interval_tendency_mae_k_per_min':'Interval tendency MAE (K/min)'}
DESCRIPTOR_LABELS={
    'history_ir108_mean_abs_change_k':'Historical IR108 variability (K)',
    'history_ir108_net_change_k_per_15min':'Historical IR108 net change (K per 15 min)',
    'last_ir108_gradient_k_per_pixel':'Last-frame IR108 gradient (K/pixel)',
    'future_ir108_mean_change_15min_k':'Mean IR108 change at +15 min (K)',
    'future_ir108_mean_change_30min_k':'Mean IR108 change at +30 min (K)',
    'future_ir108_advection_residual_mae_15min_k':'IR108 residual amplitude at +15 min (K)',
    'future_ir108_advection_residual_mae_30min_k':'IR108 residual amplitude at +30 min (K)'}
COLORS=dict(zip(('mvm','future','joint','residual'),('tab:blue','tab:orange','tab:green','tab:red')))
COLORS.update({f'probe_{key}':value for key,value in list(COLORS.items())})
COLORS['probe_random']='tab:gray'
COLORS.update(persistence='black',advection='tab:purple')


@contextmanager
def widget_output(output):
    """Publish each refresh atomically, including in headless notebook runs.

    Explicit MIME bundles avoid relying on frontend interception of a live
    Output context and keep callback exceptions visible to the caller.
    """
    from IPython.utils.capture import capture_output
    with capture_output(display=True) as captured:
        yield
    records=[dict(output_type='display_data',data=item.data,metadata=item.metadata)
             for item in captured.outputs]
    for name,text in (('stdout',captured.stdout),('stderr',captured.stderr)):
        if text:
            records.append(dict(output_type='stream',name=name,text=text))
    output.outputs=tuple(records)


def read_json(path,default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return {} if default is None else default


def read_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError,pd.errors.EmptyDataError,pd.errors.ParserError):
        return pd.DataFrame()


def spatial_coverage(frame,height,width):
    difference=np.zeros((height+1,width+1),dtype=np.int64)
    if len(frame):
        y,x=frame.origin_y.to_numpy(int),frame.origin_x.to_numpy(int)
        for dy,dx,sign in ((0,0,1),(224,0,-1),(0,224,-1),(224,224,1)):
            np.add.at(difference,(y+dy,x+dx),sign)
    return difference.cumsum(0).cumsum(1)[:-1,:-1]


def dataset_overview(queue, show_tables=True):
    from IPython.display import display
    queue=Path(queue)
    protocol=read_json(queue/'dataset/protocol.json')
    if not protocol:
        print('Dataset preparation in progress.')
        return
    frames={key:read_csv(queue/f'dataset/{key}.csv') for key in ('train','validation')}
    rows=[]
    for name,frame in frames.items():
        times=pd.to_datetime(frame.start_time,utc=True)
        rows.append({'Subset':name,'Number of clips':len(frame),'Distinct start times':frame.start_time.nunique(),
            'Distinct crop origins':len(frame[['origin_y','origin_x']].drop_duplicates()),
            'First observation (UTC)':str(times.min()),'Last observation (UTC)':str(times.max()+pd.Timedelta(minutes=75))})
    if show_tables:
        display(pd.DataFrame(rows).set_index('Subset'))
    record=read_json(queue/'dataset/inventory.json')['stores'][0]
    height,width=record['shape'][2:]
    fig,axes=plt.subplots(2,2,figsize=(14,7),constrained_layout=True)
    for column,(name,frame) in enumerate(frames.items()):
        coverage=spatial_coverage(frame,height,width)
        if record['y_increasing']:
            coverage=coverage[::-1]
        if not record['x_increasing']:
            coverage=coverage[:,::-1]
        image=axes[0,column].imshow(coverage,cmap='viridis',vmin=0)
        axes[0,column].set_title(f'{name.capitalize()} spatial coverage')
        axes[0,column].set_xlabel('West–east pixel index')
        fig.colorbar(image,ax=axes[0,column],label='Number of clips')
        frame.groupby('day').size().plot.bar(ax=axes[1,column],color='tab:blue')
        axes[1,column].set_ylabel('Number of clips')
        axes[1,column].set_xlabel('Observation date (UTC)')
        axes[1,column].tick_params(axis='x',rotation=0)
    display(fig)
    plt.close(fig)


def learning_curves(queue):
    queue=Path(queue)
    std=read_json(queue/'dataset/channel_stats.json').get('global_std_k',1)
    fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
    groups=[(('mvm','joint'),'train_mvm','validation_mvm','Masked reconstruction'),
            (('future','joint','residual'),'train_future','validation_future','End-to-end prediction'),
            (tuple(f'probe_{x}' for x in ('random','mvm','future','joint','residual')),
             'train_future','validation_future','Frozen-encoder prediction')]
    for axis,(names,train_key,val_key,title) in zip(axes,groups):
        for name in names:
            frame=read_csv(queue/name/'loss_epochs.csv')
            if frame.empty or train_key not in frame:
                continue
            axis.plot(frame.epoch,frame[train_key]*std**2,'o-',color=COLORS[name],label=LABELS[name])
            if val_key in frame:
                axis.plot(frame.epoch,frame[val_key]*std**2,'o--',color=COLORS[name])
        axis.set(title=title,xlabel='Epoch',ylabel='MSE (K²)')
        axis.grid(alpha=.2)
        if axis.lines:
            from matplotlib.ticker import MaxNLocator
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.legend(fontsize=8)
        else:
            axis.set_title(f'{title} · pending')
            axis.set_xticks([])
            axis.set_yticks([])
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],color='black',linestyle='-',label='Training'),
                        Line2D([],[],color='black',linestyle='--',label='Validation')],
               loc='outside lower center',ncol=2)
    return fig


def training_table(queue):
    queue=Path(queue)
    rows=[]
    for stage in read_json(queue/'comparison_plan.json').get('stages',[]):
        progress=read_json(queue/stage['name']/'progress.json')
        rows.append({'Model':LABELS[stage['name']],'Status':progress.get('status','pending'),
            'Completed epochs':progress.get('completed_epochs',0),'Planned epochs':stage['epochs'],
            'Optimizer updates':progress.get('optimizer_steps',0)})
    return pd.DataFrame(rows).set_index('Model') if rows else pd.DataFrame()


def show_training(queue):
    from IPython.display import display
    display(training_table(queue))
    figure=learning_curves(queue)
    display(figure)
    plt.close(figure)


class TrainingDashboard:
    def __init__(self,queue):
        import ipywidgets as w
        self.queue=Path(queue)
        self.task=None
        self.button=w.Button(description='Refresh',icon='refresh')
        self.auto=w.ToggleButton(description='Auto (30 s)',value=False)
        self.output=w.Output()
        self.button.on_click(lambda _:self.refresh())
        self.auto.observe(self._toggle,names='value')
        self.widget=w.VBox([w.HBox([self.button,self.auto]),self.output])
        self.refresh()

    def refresh(self):
        from IPython.display import display
        with widget_output(self.output):
            display(training_table(self.queue))
            figure=learning_curves(self.queue)
            display(figure)
            plt.close(figure)
            queue_state=read_json(self.queue/'queue_status.json')
            print('Training:',queue_state.get('status','pending'),queue_state.get('stage',''))
            analysis_state=read_json(self.queue/'analysis/progress.json')
            if analysis_state:
                print('Analysis:',analysis_state.get('status'),analysis_state.get('run',''),analysis_state.get('phase',''),
                      f"{analysis_state['samples']}/{analysis_state['total_samples']}" if 'samples' in analysis_state else '')

    async def _loop(self):
        try:
            while self.auto.value:
                await asyncio.sleep(30)
                self.refresh()
        except asyncio.CancelledError:
            pass

    def _toggle(self,event):
        if self.task:
            self.task.cancel()
            self.task=None
        if event['new']:
            self.task=asyncio.get_running_loop().create_task(self._loop())

    def close(self):
        self.auto.value=False


def forecast_metrics_frame(queue,family='pretraining',variability='all'):
    queue=Path(queue)
    if family not in ('pretraining','frozen_common_head') or variability not in ('all','low','medium','high'):
        raise ValueError('Unknown model family or variability group')
    records=[]
    for stage in read_json(queue/'comparison_plan.json').get('stages',[]):
        if stage['kind']!=family and stage['name']!='future':
            continue
        report=read_json(queue/stage['name']/'validation_latest.json')
        if variability!='all':
            report=report.get('by_historical_variability',{}).get(variability,{})
        for method in ('persistence','advection','neural'):
            if method=='neural' and stage['kind']!=family:
                continue
            if method!='neural' and stage['name']!='future':
                continue
            values=report.get(method)
            if values:
                records.extend(dict(run=stage['name'] if method=='neural' else method,
                    samples=values['samples'],**row) for row in values['per_channel_horizon'])
    return pd.DataFrame(records)


def comparison_figure(queue,family='pretraining',metric='rmse_k',variability='all'):
    if metric not in METRICS:
        raise ValueError('Unknown forecast metric')
    frame=forecast_metrics_frame(queue,family,variability)
    if frame.empty:
        raise ValueError('Validation metrics are pending for this selection')
    fig,axes=plt.subplots(1,2,figsize=(15,4),sharey=True,constrained_layout=True)
    for axis,horizon in zip(axes,(15,30)):
        table=frame[frame.horizon_minutes==horizon].pivot(index='channel',columns='run',values=metric)
        table.reindex(THERMAL_CHANNELS).rename(columns=LABELS).plot.bar(ax=axis,
            color=[COLORS[name] for name in table.columns])
        axis.set(title=f'+{horizon} min',ylabel=METRICS[metric],xlabel='Channel')
        axis.tick_params(axis='x',rotation=35)
        axis.tick_params(axis='y',labelleft=True)
        axis.grid(axis='y',alpha=.2)
    # One scale across both lead times, including every selected model and channel.
    maximum=frame[metric].max()
    axes[0].set_ylim(0,maximum*1.1 if maximum>0 else 1)
    handles,labels=axes[0].get_legend_handles_labels()
    for axis in axes:axis.get_legend().remove()
    fig.legend(handles,labels,loc='outside lower center',ncol=4)
    summary=frame.groupby('run').mse_k2.mean().pow(.5).rename('Global RMSE (K)')
    summary=summary.to_frame().join(frame.groupby('run').samples.first().rename('Number of clips')).rename(index=LABELS)
    summary.index.name='Model'
    return fig,summary


def show_comparison(queue,family='pretraining',metric='rmse_k',variability='all'):
    from IPython.display import display
    figure,summary=comparison_figure(queue,family,metric,variability)
    display(figure)
    plt.close(figure)
    display(summary.style.format({'Global RMSE (K)':'{:.2f}'}))


def comparison_dashboard(queue):
    import ipywidgets as w
    family=w.Dropdown(options=[('End-to-end','pretraining'),('Frozen probes','frozen_common_head')],description='Models:')
    metric=w.Dropdown(options=[(label,key) for key,label in METRICS.items()],description='Metric:')
    variability=w.Dropdown(options=[('All','all'),('Low','low'),('Medium','medium'),('High','high')],description='Variability:')
    button=w.Button(description='Refresh')
    output=w.Output()
    def refresh(*_):
        with widget_output(output):
            show_comparison(queue,family.value,metric.value,variability.value)
    for control in (family,metric,variability):
        control.observe(refresh,names='value')
    button.on_click(refresh)
    refresh()
    return w.VBox([w.HBox([family,metric]),w.HBox([variability,button]),output])


def forecast_figure(queue,index,channel='IR_108',horizon=30,family='pretraining'):
    path=Path(queue)/'analysis/examples'/f'clip_{index:04d}.npz'
    band=THERMAL_CHANNELS.index(channel)
    with np.load(path) as archive:
        arrays={name:archive[name] for name in archive.files}
    h=(15,30).index(horizon)
    target=arrays['observed'][(12,15)[h],band]
    models=(['future','joint','residual'] if family=='pretraining' else
            [f'probe_{name}' for name in ('random','mvm','future','joint','residual')])
    names=['persistence','advection']+models
    fig,axes=plt.subplots(2,len(names)+1,figsize=(3*(len(names)+1),6),constrained_layout=True)
    axes[0,0].imshow(target,vmin=180,vmax=330,cmap='inferno')
    axes[0,0].set_title('Observed target')
    axes[1,0].imshow(arrays['observed'][9,band],vmin=180,vmax=330,cmap='inferno')
    axes[1,0].set_title('Last input')
    for column,name in enumerate(names,1):
        predicted=arrays[name][h,band].astype(np.float32)
        image=axes[0,column].imshow(predicted,vmin=180,vmax=330,cmap='inferno')
        axes[0,column].set_title(LABELS[name])
        error=axes[1,column].imshow(predicted-target.astype(np.float32),vmin=-20,vmax=20,cmap='RdBu_r')
        axes[1,column].set_title('Error')
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(image,ax=axes[0,:].tolist(),label='BT (K)',shrink=.7)
    fig.colorbar(error,ax=axes[1,1:].tolist(),label='Prediction − observation (K)',shrink=.7,extend='both')
    fig.suptitle(f'{channel} · validation clip {index} · +{horizon} min')
    return fig


def forecast_dashboard(queue):
    import ipywidgets as w
    from IPython.display import display
    queue=Path(queue)
    catalog=read_json(queue/'analysis/examples/catalog.json')
    if not catalog:
        return w.HTML('Prediction examples: analysis in progress.')
    clip=w.Dropdown(options=[(f"{c['index']} · {c['start_time']} · crop ({c['origin_y']},{c['origin_x']})",c['index']) for c in catalog['clips']],description='Clip:',layout=w.Layout(width='650px'))
    channel=w.Dropdown(options=THERMAL_CHANNELS,value='IR_108',description='Channel:')
    horizon=w.ToggleButtons(options=[15,30],value=30,description='Lead (min):')
    family=w.Dropdown(options=[('End-to-end','pretraining'),('Frozen probes','frozen_common_head')],description='Models:')
    output=w.Output()
    def refresh(*_):
        with widget_output(output):
            fig=forecast_figure(queue,clip.value,channel.value,horizon.value,family.value)
            display(fig)
            plt.close(fig)
    for control in (clip,channel,horizon,family):
        control.observe(refresh,names='value')
    refresh()
    return w.VBox([clip,w.HBox([channel,horizon,family]),output])


def radiometric_probe_figure(probes):
    """Show linear decoding of historical and future observables separately."""
    dynamic = probes[~probes.descriptor.str.startswith('history_mean_')]
    names = [name for name in LABELS if name in set(dynamic.run)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout='constrained')
    for axis, future, title in zip(axes, (False, True), ('Historical observables', 'Future observables')):
        rows = dynamic[dynamic.descriptor.str.startswith('future_') == future]
        descriptors = [name for name in DESCRIPTOR_LABELS if name in set(rows.descriptor)]
        for offset, name in zip(np.linspace(-.25, .25, len(names)), names):
            values = rows[rows.run == name].set_index('descriptor').r2.reindex(descriptors)
            axis.scatter(values, np.arange(len(descriptors))+offset, color=COLORS[name], label=LABELS[name], s=40)
        axis.set(title=title, xlabel='Validation R²', yticks=np.arange(len(descriptors)),
                 yticklabels=[DESCRIPTOR_LABELS[name] for name in descriptors])
        axis.invert_yaxis()
        axis.axvline(0, color='black', linestyle='--', linewidth=1)
        axis.grid(axis='x', alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3)
    return fig


def representation_diagnostics(queue, show_tables=False):
    from IPython.display import display
    queue=Path(queue)
    frame=read_csv(queue/'analysis/temporal_ablations.csv')
    probes=read_csv(queue/'analysis/radiometric_probes.csv')
    if frame.empty:
        print('Temporal analysis in progress.')
        return
    fig,axes=plt.subplots(1,2,figsize=(15,4),constrained_layout=True)
    for axis,kind in zip(axes,('pretraining','frozen_common_head')):
        subset=frame[frame.kind==kind]
        if subset.empty:
            axis.set_title('Analysis in progress')
            axis.set_axis_off()
            continue
        table=subset.pivot(index='run',columns='ablation',values='rmse_k')
        table=table.reindex(columns=['chronological','shuffle','last_frame'])
        table.rename(index=LABELS,columns={'chronological':'Chronological','shuffle':'Shuffled history','last_frame':'Last frame repeated'}).plot.bar(ax=axis)
        axis.set(ylabel='Forecast RMSE (K)',xlabel='',
                 title='End-to-end prediction' if kind=='pretraining' else 'Frozen-encoder prediction',
                 ylim=(0,table.to_numpy().max()*1.4))
        axis.legend(title='Input sequence',loc='upper left',fontsize=8)
        axis.tick_params(axis='x',rotation=20)
    display(fig)
    plt.close(fig)
    if not probes.empty:
        figure = radiometric_probe_figure(probes)
        display(figure)
        plt.close(figure)
        dynamic=probes[~probes.descriptor.str.startswith('history_mean_')]
        table=dynamic.pivot(index='descriptor',columns='run',values='r2').rename(index=DESCRIPTOR_LABELS,columns=LABELS)
        table.index.name='Observable'
        table.columns.name='Encoder'
        if show_tables:
            display(table.style.format('{:.2f}').set_caption('Radiometric ridge probes · validation R²'))
    controls=[read_csv(path) for path in (queue/'analysis').glob('probe_*_embedding_controls.csv')]
    if controls:
        table=pd.concat(controls).groupby(['run','intervention']).cosine_distance.mean().unstack().rename(index=LABELS,
            columns={'shuffle':'Shuffled history','last_frame':'Last frame repeated','advection_only':'Advection-only history'})
        fig,axis=plt.subplots(figsize=(10,4),constrained_layout=True)
        table.plot.bar(ax=axis)
        axis.set(ylabel='Mean cosine distance',xlabel='',title='Embedding sensitivity')
        axis.legend(title='Input sequence')
        axis.tick_params(axis='x',rotation=15)
        display(fig)
        plt.close(fig)


def reconstruction_figure(queue,index,channel='IR_108',frame=0):
    with np.load(Path(queue)/'analysis/examples'/f'clip_{index:04d}.npz') as data:
        band=THERMAL_CHANNELS.index(channel)
        observed=data['observed'][frame,band].astype(float)
        restored=data['mvm_reconstruction'][frame,band].astype(float)
        mask=data['mvm_mask'][frame,0]
    fig,axes=plt.subplots(1,4,figsize=(14,4),constrained_layout=True)
    cmap=plt.get_cmap('inferno').copy()
    cmap.set_bad('lightgray')
    for axis,values,title in zip(axes[:3],(observed,np.ma.array(observed,mask=mask),restored),
        ('Observation','Masked input','Reconstruction')):
        image=axis.imshow(values,cmap=cmap,vmin=180,vmax=330)
        axis.set_title(title)
    error=axes[3].imshow(np.ma.array(restored-observed,mask=~mask),cmap='RdBu_r',vmin=-20,vmax=20)
    axes[3].set_title('Masked-pixel error')
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(image,ax=axes[:3].tolist(),label='BT (K)',shrink=.7)
    fig.colorbar(error,ax=axes[3],label='Error (K)',shrink=.7,extend='both')
    fig.suptitle(f'MVM · {channel} · historical frame {frame+1}/10')
    return fig


def reconstruction_dashboard(queue):
    import ipywidgets as w
    from IPython.display import display
    queue=Path(queue)
    catalog=read_json(queue/'analysis/examples/catalog.json')
    if not catalog:
        return w.HTML('MVM reconstructions: analysis in progress.')
    clip=w.Dropdown(options=catalog['indices'],description='Clip:')
    channel=w.Dropdown(options=THERMAL_CHANNELS,value='IR_108',description='Channel:')
    frame=w.IntSlider(min=0,max=9,value=0,description='Frame:')
    play=w.Play(min=0,max=9,value=0,interval=400)
    w.jslink((play,'value'),(frame,'value'))
    output=w.Output()
    def refresh(*_):
        with widget_output(output):
            fig=reconstruction_figure(queue,clip.value,channel.value,frame.value)
            display(fig)
            plt.close(fig)
    for control in (clip,channel,frame):
        control.observe(refresh,names='value')
    refresh()
    return w.VBox([w.HBox([clip,channel]),w.HBox([play,frame]),output])
