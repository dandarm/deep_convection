"""Read saved probe reports and generate presentation-oriented comparisons."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd

from .three_head_data import CLASS_NAMES


COLORS = {'cma':['#334155','#f1f5f9'],
          'ct':['#334155','#f59e0b','#10b981','#ef4444','#06b6d4'],
          'rdt':['#334155','#718096','#fae014','#ff8800','#eb0055']}


def load_suite(root):
    root = Path(root)
    if not (root/'experiment_plan.json').exists():
        return dict(root=root, plan={}, runs={})
    plan=json.loads((root/'experiment_plan.json').read_text())
    runs={}
    for row in plan['experiments']:
        folder=root/row['name']
        def read(name):
            return json.loads((folder/name).read_text()) if (folder/name).exists() else None
        runs[row['name']]=dict(config={**plan['defaults'],**row},directory=folder,
            result=read('result.json'),history=read('history.json') or [],progress=read('progress.json') or {})
    return dict(root=root,plan=plan,runs=runs)


def status_text(suite):
    completed=[name for name,run in suite['runs'].items() if run['result']]
    active=[f"{name}: {run['progress'].get('status')}, epoch {run['progress'].get('epoch',run['progress'].get('completed_epochs',0))}"
            for name,run in suite['runs'].items() if run['progress'] and not run['result']]
    return f"Completed experiments: **{len(completed)} / {len(suite['runs'])}**. " + ('; '.join(active) if active else '')


def dataset_support_figure(suite):
    """Class frequencies from the published manifest, including rare OT support."""
    path = Path(suite['plan']['dataset']) / 'manifest.json'
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (head, names) in zip(axes, CLASS_NAMES.items()):
        for offset, (split, label, marker) in enumerate([
                ('train', 'Training', 'o'), ('validation', 'Validation', 's'), ('test', 'Test', '^')]):
            support = np.asarray(manifest['support'][split][head], dtype=float)
            fractions = np.where(support > 0, 100 * support / support.sum(), np.nan)
            ax.plot(np.arange(len(names))+(offset-1)*.13, fractions,
                    marker=marker, linestyle='none', label=label)
        ax.set(yscale='log' if head == 'rdt' else 'linear', ylabel='Valid target pixels (%)', title=head.upper(),
               xticks=np.arange(len(names)), xticklabels=names)
        if head != 'rdt':
            ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=35, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_ha('right')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def metric_frame(suite):
    rows=[]
    for name,run in suite['runs'].items():
        result=run['result']
        if result is None:
            continue
        for head,metrics in result['test']['metrics'].items():
            for i,label in enumerate(CLASS_NAMES[head]):
                rows.append(dict(experiment=name,head=head,class_name=label,class_id=i,
                    iou=metrics['iou'][i],f1=metrics['f1'][i],support=metrics['support'][i],
                    selected_epoch=result['best_epoch'],encoder_mode=run['config']['encoder_mode']))
            if head=='rdt':
                m=metrics['active_convection']
                rows.append(dict(experiment=name,head='active_convection',class_name='Active convection',class_id=1,
                    iou=m['csi'][1],f1=m['f1'][1],support=m['support'][1],selected_epoch=result['best_epoch'],
                    encoder_mode=run['config']['encoder_mode']))
    return pd.DataFrame(rows)


def result_summary_text(suite):
    """Describe completed held-out results with explicit comparison populations."""
    results = {name: run['result'] for name, run in suite['runs'].items() if run['result']}
    main = results.get('E1-Joint3')
    if main is None:
        return ''
    metrics = main['test']['metrics']
    active = metrics['rdt']['active_convection']
    paragraphs = [
        f"**Completed main probe:** E1-Joint3 selects epoch {main['best_epoch']} by validation loss. "
        f"On the held-out test period, cloud IoU is **{metrics['cma']['iou'][1]:.2f}**, "
        f"active-convection CSI is **{active['csi'][1]:.2f}**, and active-convection F1 is "
        f"**{active['f1'][1]:.2f}**. Its binary convection Brier score is {active['brier']:.3f}. "
        f"The encoder weights remain identical to the pretrained checkpoint."
    ]
    for name, description in [('E0-Rand', 'the frozen random encoder'),
                              ('E1-Single-rdt', 'the independent RDT decoder')]:
        if name in results:
            reference = results[name]['test']['metrics']['rdt']['active_convection']['csi'][1]
            paragraphs.append(f"Relative to {description} ({name}, CSI {reference:.2f}), "
                              f"E1-Joint3 changes active-convection CSI by {active['csi'][1]-reference:+.2f} "
                              "on the same test clips.")
    ot = metrics['rdt']
    if ot['support'][4]:
        paragraphs.append(f"The overshooting-top class has IoU {ot['iou'][4]:.2f} over "
                          f"{ot['support'][4]:,} labelled pixel observations. "
                          "This small support includes repeated observations and overlapping crops.")
    return '\n\n'.join(paragraphs)


def learning_figure(suite,names=None):
    names=names or ['E0-Rand','E0-RGB','E1-Single-cma','E1-Single-ct','E1-Single-rdt','E1-Joint3']
    if not any(suite['runs'].get(name,{}).get('history') for name in names):
        return None
    fig,axes=plt.subplots(1,3,figsize=(17,4.5))
    for ax,head in zip(axes,CLASS_NAMES):
        for name in names:
            run=suite['runs'].get(name)
            if not run or head not in run['config']['heads'] or not run['history']:
                continue
            history=run['history']
            ax.plot([row['updates'] for row in history],
                    [row['validation']['losses'][head] for row in history],label=name)
        ax.set(xscale='log',yscale='log',xlabel='Optimizer updates',ylabel='Validation loss',title=head.upper())
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def comparison_figure(suite,names,title,metric='iou'):
    data=metric_frame(suite)
    if data.empty:
        return None
    data=data[data.experiment.isin(names)]
    if data.empty:
        return None
    present=[name for name in names if name in set(data.experiment)]
    fig,axes=plt.subplots(1,3,figsize=(18,max(3.5,len(present)*.46)))
    for ax,head in zip(axes,('cma','ct','rdt')):
        for i,label in enumerate(CLASS_NAMES[head]):
            part=data[(data['head']==head)&(data.class_id==i)].set_index('experiment')
            values=[part.loc[name,metric] if name in part.index else np.nan for name in present]
            offset=(i-(len(CLASS_NAMES[head])-1)/2)*.12
            color=COLORS[head][i]
            if head=='cma' and i==1:color='#6495ed'
            ax.scatter(values,np.arange(len(present))+offset,label=label,color=color,edgecolor='#333333',linewidth=.3)
        ax.set(yticks=np.arange(len(present)),yticklabels=present if head=='cma' else [],xlim=(0,1),
               xlabel='Class IoU / CSI' if metric=='iou' else 'Class F1',title=head.upper())
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16))
    fig.suptitle(title);fig.tight_layout()
    return fig


def active_convection_figure(suite,names,title='Active-convection agreement'):
    data=metric_frame(suite)
    if data.empty:
        return None
    data=data[(data['head']=='active_convection')&data.experiment.isin(names)].set_index('experiment')
    present=[name for name in names if name in data.index]
    if not present:
        return None
    fig,ax=plt.subplots(figsize=(9,max(3,len(present)*.45)))
    ax.scatter([data.loc[name,'iou'] for name in present],np.arange(len(present)),color='#d97706')
    ax.set(yticks=np.arange(len(present)),yticklabels=present,xlim=(0,1),
           xlabel='Active-convection CSI',title=title)
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);fig.tight_layout()
    return fig


def brier_figure(suite,names):
    rows=[]
    for name in names:
        result=suite['runs'].get(name,{}).get('result')
        if result and 'rdt' in result['test']['metrics']:
            metric=result['test']['metrics']['rdt']
            rows.append((name,metric['brier_multiclass'],metric['active_convection']['brier']))
    if not rows:
        return None
    fig,axes=plt.subplots(1,2,figsize=(12,max(3.5,len(rows)*.45)),sharey=True)
    for ax,column,title in zip(axes,(1,2),('Five RDT classes','Active convection')):
        ax.scatter([row[column] for row in rows],np.arange(len(rows)))
        ax.set(yticks=np.arange(len(rows)),yticklabels=[row[0] for row in rows],
               xlabel='Brier score',title=title,xlim=(0,None))
        ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis();fig.tight_layout()
    return fig


def sequential_figure(suite):
    data=metric_frame(suite)
    if data.empty:
        return None
    data=data[data['head']=='active_convection'].set_index('experiment')
    fig,ax=plt.subplots(figsize=(8,4.5))
    any_line=False
    for variant,label in [('selective','Last four blocks + SSL weight anchor'),
                          ('cumulative','Last four blocks + retained auxiliary tasks')]:
        names=['E1-Single-rdt',f'E2-{variant}-rdt',f'E3-{variant}-rdt']
        points=[(i,float(data.loc[name,'iou'])) for i,name in enumerate(names) if name in data.index]
        if points:
            ax.plot(*zip(*points),marker='o',label=label);any_line=True
    if not any_line:
        plt.close(fig);return None
    ax.set(xticks=[0,1,2],xticklabels=['Frozen SSL','After CMA adaptation','After CMA → CT adaptation'],
           ylim=(0,1),ylabel='Active-convection CSI',title='Sequential transfer')
    ax.legend(fontsize=9);ax.grid(alpha=.2);fig.tight_layout()
    return fig


def ablation_figure(suite):
    rows=[]
    for name,run in suite['runs'].items():
        result=run['result']
        if not result or not result['interventions'] or 'rdt' not in result['test']['metrics']:
            continue
        baseline=result['test']['metrics']['rdt']['active_convection']['csi'][1]
        for mode,report in result['interventions'].items():
            value=report['metrics']['rdt']['active_convection']['csi'][1]
            if baseline is not None and value is not None:
                rows.append(dict(experiment=name,intervention=mode,change=value-baseline))
    if not rows:
        return None
    frame=pd.DataFrame(rows)
    labels={'shuffle_time':'Temporal permutation','repeat_last':'Repeated last frame',
            'teacher_two_channels':'WV 6.2 + IR 10.8','wv_and_ir108':'WV 6.2 + WV 7.3 + IR 10.8',
            'without_ir134':'All except IR 13.4'}
    fig,ax=plt.subplots(figsize=(12,5))
    names=frame.experiment.unique().tolist()
    for i,(mode,label) in enumerate(labels.items()):
        part=frame[frame.intervention==mode].set_index('experiment')
        ax.scatter(np.arange(len(names))+(i-2)*.13,[part.loc[n,'change'] for n in names],label=label)
    ax.axhline(0,color='black',lw=.7)
    ax.set(xticks=np.arange(len(names)),xticklabels=names,ylabel='CSI change from intact input',title='Temporal and spectral interventions')
    ax.tick_params(axis='x',rotation=30);ax.grid(axis='y',alpha=.2)
    ax.legend(fontsize=8,bbox_to_anchor=(1.01,1),loc='upper left');fig.tight_layout()
    return fig


def example_figure(suite,name,index=None,time_index=15,scene='convective'):
    run=suite['runs'].get(name)
    if not run or not run['result']:
        return None
    files=sorted(run['directory'].glob('example_*.npz'))
    if not files:
        return None
    if scene not in ('convective', 'background'):
        raise ValueError('Unknown example scene')
    file=run['directory']/f'example_{index:05d}.npz' if index is not None else None
    if file is None:
        # Select using held-out teacher labels alone, in stable sample order.
        largest_convective_area=0
        for candidate in files:
            with np.load(candidate) as archive:
                if 'target_rdt' not in archive:
                    continue
                target=archive['target_rdt'][time_index]
                valid=target!=255
                convective=int(np.count_nonzero((target>=2)&(target<=4)))
                background=valid.any() and np.all(target[valid]==0)
                if scene=='convective' and convective>largest_convective_area:
                    file=candidate;largest_convective_area=convective
                if scene=='background' and background:
                    file=candidate;break
    if file is None:
        return None
    with np.load(file) as archive:
        heads=[head for head in CLASS_NAMES if 'prediction_'+head in archive.files]
        fig,axes=plt.subplots(len(heads),2,figsize=(9,3.5*len(heads)),squeeze=False)
        for row,head in enumerate(heads):
            cmap=ListedColormap(COLORS[head]);cmap.set_bad('#aaaaaa')
            norm=BoundaryNorm(np.arange(len(CLASS_NAMES[head])+1)-.5,len(CLASS_NAMES[head]))
            valid=archive['target_'+head][time_index]!=255
            for col,(kind,title) in enumerate([('target','Teacher'),('prediction','Probe')]):
                values=np.ma.array(archive[kind+'_'+head][time_index],mask=~valid)
                im=axes[row,col].imshow(values,cmap=cmap,norm=norm,interpolation='nearest')
                axes[row,col].set_title(head.upper()+' · '+title);axes[row,col].axis('off')
            bar=fig.colorbar(im,ax=axes[row].tolist(),ticks=range(len(CLASS_NAMES[head])),fraction=.025,pad=.02)
            bar.ax.set_yticklabels(CLASS_NAMES[head]);bar.ax.tick_params(labelsize=8)
        fig.suptitle(f'{name} · clip {file.stem.removeprefix("example_")} · frame {time_index+1}/16')
        fig.subplots_adjust(top=.94,hspace=.2,right=.79)
        return fig


def teacher_gate_figure(roots):
    groups={}
    for title,root in roots.items():
        values=[]
        for split in ('train','validation','test'):
            path=Path(root)/split/'complete.json'
            value=None
            if path.exists():
                report=json.loads(path.read_text())
                value=report['quality_gate']['fractions']['inside_high_opaque']
            values.append(100*value if value is not None else np.nan)
        if np.isfinite(values).any():groups[title]=values
    if not groups:
        return None
    fig,ax=plt.subplots(figsize=(10,4.5))
    positions=np.arange(3);width=.8/len(groups)
    for i,(title,values) in enumerate(groups.items()):
        ax.bar(positions+(i-(len(groups)-1)/2)*width,values,width=width*.94,label=title)
    ax.axhline(90,color='#b23b3b',linestyle='--',label='Original agreement reference: 90%')
    ax.set(xticks=positions,xticklabels=['Training','Validation','Test'],
           ylabel='High-opaque CT agreement (%)',ylim=(0,100),title='Inter-teacher agreement')
    ax.legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.1),ncol=2);fig.tight_layout()
    return fig


def teacher_revision_figure(report):
    """Show changed target coverage and independent CT composition by split."""
    if report.get('status') != 'completed':
        return None
    splits=list(report['splits'])
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    positions=np.arange(len(splits))
    for i,(version,label) in enumerate([('legacy','Previous RDT v2'),('corrected','Corrected RDT v3')]):
        fractions=[report['splits'][split]['target_valid_cold_fraction'][version] for split in splits]
        axes[0].bar(positions+(i-.5)*.32,100*np.asarray(fractions),width=.3,label=label)
    axes[0].set(xticks=positions,xticklabels=splits,ylim=(0,100),
                ylabel='Cold-cell pixels retained as targets (%)',title='Available temporal targets')
    axes[0].legend(fontsize=9)
    composition=[]
    for split in splits:
        counts=np.asarray(report['splits'][split]['corrected_native_rdt_by_ct_confusion'])[2:].sum(0)
        composition.append(counts/max(1,counts.sum()))
    composition=100*np.asarray(composition)
    bottom=np.zeros(len(splits))
    for i,label in enumerate(CLASS_NAMES['ct']):
        axes[1].bar(positions,composition[:,i],bottom=bottom,color=COLORS['ct'][i],label=label)
        bottom+=composition[:,i]
    axes[1].set(xticks=positions,xticklabels=splits,ylim=(0,100),
                ylabel='RDT convective pixels (%)',title='CT regimes within corrected RDT convection')
    axes[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1))
    fig.tight_layout()
    return fig


def sampling_feasibility_figure(reports):
    rows=[]
    for label,path in reports.items():
        path=Path(path)
        if not path.exists():continue
        report=json.loads(path.read_text())
        if report['status']=='completed' and report.get('complete_seven_channel_inputs_checked'):
            rows.append((label,report['counts']['B']))
    if not rows:return None
    fig,ax=plt.subplots(figsize=(9,4))
    bars=ax.bar([r[0] for r in rows],[r[1] for r in rows],color='#456f99')
    ax.bar_label(bars,labels=[f'{r[1]:,}' for r in rows],padding=3)
    ax.axhline(154,color='#b23b3b',linestyle='--',label='Planned training allocation: 154 clips')
    ax.set(ylabel='Candidate clips',title='Available high-cloud hard negatives',ylim=(0,max(r[1] for r in rows)*1.15+1))
    ax.legend(fontsize=9);fig.tight_layout()
    return fig


def native_phase_agreement_figure(periods):
    columns=[];labels=[]
    for label,folder in periods.items():
        path=Path(folder)/'complete.json'
        if not path.exists():continue
        report=json.loads(path.read_text())
        confusion=np.asarray(report['native_rdt_by_ct_confusion'])[2:]
        support=confusion.sum(1)
        columns.append(np.divide(100*confusion[:,3],support,
                                 out=np.full(5,np.nan),where=support>0))
        labels.append(label)
    if not columns:return None
    values=np.asarray(columns).T
    fig,ax=plt.subplots(figsize=(max(8,len(columns)*1.35),4.5))
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#cccccc')
    im=ax.imshow(values,cmap=cmap,vmin=0,vmax=100,aspect='auto')
    for row in range(5):
        for col in range(len(columns)):
            if np.isfinite(values[row,col]):
                ax.text(col,row,f'{values[row,col]:.0f}',ha='center',va='center',
                        color='black' if values[row,col]>=60 else 'white')
    ax.set(xticks=range(len(columns)),xticklabels=labels,yticks=range(5),
           yticklabels=['Triggering','Growing','Mature','Decaying','Persistent OT'],
           title='High-opaque CT agreement by native RDT phase')
    ax.tick_params(axis='x',rotation=25)
    fig.colorbar(im,ax=ax,label='RDT phase pixels inside CT high opaque (%)')
    fig.tight_layout();return fig
