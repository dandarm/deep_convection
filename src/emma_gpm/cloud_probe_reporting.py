"""Read completed cloud probes; plots use frozen test-set measurements."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OBJECTIVE_NAMES = dict(mvm='MVM',future='Future',joint='Joint',residual='Residual',random='Random')
COLORS = dict(mvm='#7651a8',future='#2676b8',joint='#df842e',residual='#258f72',random='#777777')


def load_results(directory):
    records=[]
    for path in sorted(Path(directory).glob('*/result.json')):
        comparison=path.parent/'comparison.json'
        if not comparison.exists():
            continue  # The queue publishes the pairing after the final result.
        result=json.loads(path.read_text());identity=result['identity']
        record=dict(case=path.parent.name,path=str(path.parent),kind=identity['kind'],
            model=identity['model_size'].capitalize(),clips=identity['pretraining_clips'],
            pretraining_epochs=identity['pretraining_epochs'],best_epoch=result['best_epoch'],
            objective='random' if identity['kind']=='random' else identity['pretraining_objective'],**result['test'])
        record['paired_random_control']=json.loads(comparison.read_text())['paired_random_control']
        records.append(record)
    return pd.DataFrame(records)


def plot_comparison(frame):
    if frame.empty:
        print('I grafici saranno disponibili alla conclusione del primo probe.');return
    groups=sorted(set(zip(frame.model,frame.clips,frame.pretraining_epochs)),key=lambda x:(x[0]!='Small',x[1],x[2]))
    fig,axes=plt.subplots(1,2,figsize=(max(11,len(groups)*1.6),4.8),sharey=True,layout='constrained')
    objectives=['mvm','future','joint','residual','random']
    for ax,metric,title in zip(axes,['cloud_iou','cloud_dice'],['Cloud IoU','Cloud Dice']):
        for j,objective in enumerate(objectives):
            subset=frame[frame.objective==objective]
            xs,ys=[],[]
            for i,(model,clips,epochs) in enumerate(groups):
                rows=subset[(subset.model==model)&(subset.clips==clips)&(subset.pretraining_epochs==epochs)]
                if len(rows):
                    xs.append(i+(j-2)*.16);ys.append(100*rows.iloc[0][metric])
            ax.bar(xs,ys,width=.15,label=OBJECTIVE_NAMES[objective],color=COLORS[objective])
        ax.set_xticks(range(len(groups)),[f'{m}\n{n:,}\n{e} ep.' for m,n,e in groups],fontsize=9)
        ax.set_ylim(0,100);ax.set_ylabel('Test score (%)');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[-1].legend(ncol=3)
    return fig


def plot_paired_gain(frame):
    if frame.empty:return
    lookup=frame.set_index('case');pairs=[]
    for row in frame.itertuples():
        if row.kind=='pretrained' and row.paired_random_control in lookup.index:
            control=lookup.loc[row.paired_random_control]
            pairs.append((row,100*(row.cloud_iou-control.cloud_iou)))
    if not pairs:
        print('Il confronto appaiato apparirà quando sarà disponibile il controllo casuale.');return
    fig,ax=plt.subplots(figsize=(max(8,len(pairs)*.65),4),layout='constrained')
    ax.bar(range(len(pairs)),[d for _,d in pairs],color=[COLORS[r.objective] for r,_ in pairs])
    ax.set_xticks(range(len(pairs)),[f'{r.model}\n{r.clips:,}\n{OBJECTIVE_NAMES[r.objective]}' for r,_ in pairs],rotation=45,ha='right')
    ax.axhline(0,color='black',linewidth=.8);ax.set_ylabel('Cloud IoU difference (percentage points)')
    ax.set_title('Pretrained encoder versus paired random encoder');ax.grid(axis='y',alpha=.2)
    return fig


def plot_examples(directory,case):
    directory=Path(directory);result=json.loads((directory/case/'result.json').read_text())
    dataset=Path(result['identity']['dataset']);manifest=json.loads((dataset/'manifest.json').read_text())
    lookup={r['id']:r for r in manifest['samples']}
    examples=np.load(directory/case/'examples.npz',allow_pickle=False)
    comparison=json.loads((directory/case/'comparison.json').read_text())
    control_path=directory/comparison['paired_random_control']/'examples.npz'
    control=np.load(control_path,allow_pickle=False) if control_path.exists() else None
    if control is not None and not np.array_equal(control['ids'],examples['ids']):
        raise ValueError('Example IDs differ between paired models')
    positions=[0,2,3,5];columns=4 if control is not None else 3
    fig,axes=plt.subplots(len(positions),columns,figsize=(3.2*columns,3.1*len(positions)),layout='constrained')
    metadata=[]
    for line,index in enumerate(positions):
        row=lookup[str(examples['ids'][index])]
        ir=np.load(dataset/row['clip'],mmap_mode='r')[9,4]
        axes[line,0].imshow(ir,cmap='gray_r',vmin=210,vmax=300);axes[line,0].set_title('IR 10.8 µm (210–300 K)')
        axes[line,1].imshow(np.ma.masked_equal(examples['target'][index],255),cmap='Blues',vmin=0,vmax=1)
        axes[line,1].set_title('CMA target')
        axes[line,2].imshow(examples['probability'][index],cmap='Blues',vmin=0,vmax=1);axes[line,2].set_title('Pretrained: cloud probability')
        if control is not None:
            axes[line,3].imshow(control['probability'][index],cmap='Blues',vmin=0,vmax=1);axes[line,3].set_title('Random: cloud probability')
        for ax in axes[line]:ax.set_axis_off()
        metadata.append(f"{row['target_time']} UTC; crop ({row['canonical_y']}, {row['canonical_x']})")
    return fig,metadata
