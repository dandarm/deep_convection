"""Synchronized satellite/rule preview; no model loading or pretraining changes."""
import numpy as np
import pandas as pd
import json
from html import escape
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import ipywidgets as widgets
from IPython.display import display, clear_output

from .rdt_teacher import CLASS_NAMES, CLASS_COLORS
from .rdt_refined import RefinedRDTConfig
from dataclasses import asdict
from .rdt_preview import load_example
from .rdt_event_explorer import btd_norm


def domain_catalog_protocol(root, catalog):
    """Allow independent sequences only when scientific protocols agree."""
    if catalog.label_root.isna().any():
        raise ValueError('Missing domain label archive')
    roots=[Path(path) for path in catalog.label_root.unique()]
    if not roots:
        raise ValueError('Empty domain label catalog')
    # Time bounds and ID namespaces differ between independent sequences.
    sequence_keys={'start_date','warmup_minutes','track_id_offset','frame_index_offset'}
    common=None
    for directory in roots:
        protocol=json.loads((directory/'protocol.json').read_text())
        if protocol['root'] != str(Path(root).resolve()):
            raise ValueError('Domain labels belong to a different satellite archive')
        identity={key:value for key,value in protocol.items() if key not in sequence_keys}
        if common is not None and identity != common:
            raise ValueError('Incompatible domain teacher protocols in catalog')
        common=identity
    return roots, protocol


class RDTClipViewer:
    def __init__(self, root, catalog, config=None, warmup_frames=3):
        self.root,self.catalog,self.config,self.warmup = root,catalog,config,warmup_frames
        versions=catalog.teacher_version.dropna().unique() if 'teacher_version' in catalog else ['legacy']
        if len(versions)!=1:raise ValueError('Mixed teacher versions in catalog')
        self.teacher_version=versions[0]
        self.domain_teacher=self.teacher_version=='multispectral_utls_v0'
        if self.domain_teacher:
            from .rdt.config import RDTTeacherConfig
            self.label_roots,protocol=domain_catalog_protocol(root,catalog)
            self.label_root=self.label_roots[0]
            self.config=RDTTeacherConfig(**protocol['config'])
        self.halo=int(catalog.analysis_halo.dropna().iloc[0]) if 'analysis_halo' in catalog else 0
        if self.teacher_version=='refined_v2' and not isinstance(self.config,RefinedRDTConfig):
            self.config=RefinedRDTConfig(**asdict(config)) if config else RefinedRDTConfig()
        self.example=None;self.busy=False
        self.order=widgets.Dropdown(options=[('Occupazione media','convective_pixels')] +
            ([('Picco per frame','convective_peak_fraction'),('Picco OT','ot_peak_fraction')] if 'convective_peak_fraction' in catalog else []),
            description='Ordina')
        if 'convective_peak_fraction' in catalog:self.order.value='convective_peak_fraction'
        self.show_ids=widgets.Checkbox(value=False,description='Mostra ID')
        self.filter=widgets.Dropdown(options=['Tutte','Candidati convettivi','OT candidati','Solo nubi fredde','Senza nubi fredde'],description='Esempi')
        self.sample=widgets.Dropdown(options=[],description='Clip',layout=widgets.Layout(width='95%'))
        self.threshold=(widgets.Dropdown(options=[('Contorni adattivi','cell_id'),('Plateau','plateau_mask'),
            ('Proxy UTLS','t_utls_proxy_k'),('Profondità D','thermal_depth'),('Confidenza proxy','proxy_confidence'),
            ('Bordo T edge','t_edge_k')],description='Diagnostica') if self.domain_teacher else
            widgets.Dropdown(options=[240,230,220,210],value=240,description='Contorno K'))
        self.play=widgets.Play(min=0,max=15,interval=400)
        self.frame=widgets.IntSlider(min=0,max=15,description='Frame')
        self.link=widgets.jslink((self.play,'value'),(self.frame,'value'))
        self.track=widgets.Dropdown(options=[],description='Traccia')
        self.custom_time=widgets.Text(description='UTC',placeholder='2020-06-15T12:00:00')
        self.custom_y=widgets.BoundedIntText(value=200,min=0,max=100000,description='origin_y')
        self.custom_x=widgets.BoundedIntText(value=800,min=0,max=100000,description='origin_x')
        self.custom_button=widgets.Button(description='Carica coordinate')
        self.message,self.video,self.table,self.history,self.diagnostics=[widgets.Output() for _ in range(5)]
        table_panel=widgets.Accordion(children=[self.table],titles=('Tabella dei criteri',),selected_index=None)
        self.tabs=widgets.Tab(children=[self.video,widgets.VBox([widgets.HBox([self.threshold,self.show_ids]),self.diagnostics,table_panel]),widgets.VBox([self.track,self.history])])
        for i,title in enumerate(['Clip','Contorni e criteri','Evoluzione']):self.tabs.set_title(i,title)
        guide=widgets.Accordion(children=[widgets.HTML(
            '<p>Le percentuali descrivono l’occupazione dei pixel, non la probabilità che una nube sia convettiva. '
            'Il picco riguarda il frame più attivo; la media riguarda tutti i 16 frame.</p>'
            '<p>Il tratteggio segnala storia o fase non valutabile. Le classi sono candidati delle regole RDT-CW-inspired, '
            'non predizioni VideoMAE. Soglie, tracking e limiti sono descritti in RDT_plan.md.</p>')],
            titles=('Guida alla lettura',),selected_index=None)
        self.widget=widgets.VBox([widgets.HBox([self.filter,self.order]),self.sample,
            widgets.HBox([self.play,self.frame]),self.message,self.tabs,
            widgets.Accordion(children=[widgets.VBox([self.custom_time,widgets.HBox([self.custom_y,self.custom_x]),self.custom_button])],
                              titles=('Carica una clip per coordinate',),selected_index=None),guide])
        self.tabs.observe(self.show_frame,names='selected_index')
        self.order.observe(self.update_options,names='value')
        self.show_ids.observe(self.show_frame,names='value')
        self.filter.observe(self.update_options,names='value')
        self.sample.observe(self.load_sample,names='value')
        self.frame.observe(self.show_frame,names='value');self.threshold.observe(self.show_frame,names='value')
        self.track.observe(self.show_history,names='value')
        self.custom_button.on_click(self.load_custom)
        self.update_options()

    def update_options(self, change=None):
        good=self.catalog[self.catalog.status=='ok'].copy()
        choice=self.filter.value
        if choice=='Candidati convettivi':good=good[good.convective_pixels>0]
        if choice=='OT candidati':good=good[good.ot_pixels>0]
        if choice=='Solo nubi fredde':good=good[(good.cold_pixels>0)&(good.convective_pixels==0)]
        if choice=='Senza nubi fredde':good=good[good.cold_pixels==0]
        good=good.sort_values([self.order.value,'cold_pixels'],ascending=False,kind='stable')
        self.busy=True
        metric=self.order.value
        self.sample.options=[(f"{r.start_time[:16]} — {'media' if metric=='convective_pixels' else 'picco OT' if metric=='ot_peak_fraction' else 'picco'} {100*(r.convective_pixels/(16*224*224) if metric=='convective_pixels' else r[metric]):.2f}%",int(i)) for i,r in good.iterrows()]
        self.sample.value=self.sample.options[0][1] if len(good) else None
        self.busy=False
        if len(good):self.load_sample()
        else:
            self.example=None
            for output in [self.video,self.table,self.history,self.diagnostics]:output.clear_output()
            with self.message:
                clear_output();print('Nessuna clip in questo piccolo catalogo soddisfa il filtro. Non è una misura della frequenza atmosferica.')

    def load_sample(self, change=None):
        if self.busy or self.sample.value is None:return
        row=self.catalog.loc[self.sample.value]
        self.load(row.start_time,int(row.origin_y),int(row.origin_x),
                  label_root=Path(row.label_root) if self.domain_teacher else None)

    def load_custom(self, button=None):
        self.load(self.custom_time.value,self.custom_y.value,self.custom_x.value)

    def load(self, time, y, x, *, label_root=None):
        self.play.playing=False
        try:
            if self.domain_teacher:
                from .rdt.storage import load_domain_crop
                if label_root is None:
                    from .rdt.storage import frame_key
                    times=pd.date_range(pd.Timestamp(time),periods=16,freq="5min")
                    candidates=[directory for directory in self.label_roots
                                if all((directory/"frames"/frame_key(stamp)/"record.json").exists() for stamp in times)]
                    if not candidates:
                        raise FileNotFoundError("Full-domain labels not available for this clip")
                    label_root=self.label_root if self.label_root in candidates else candidates[0]
                if label_root not in self.label_roots:
                    raise ValueError("Unknown domain label archive")
                example=load_domain_crop(self.root,label_root,time,y,x)
                self.label_root=label_root
            else:
                example=load_example(self.root,time,y,x,warmup_frames=self.warmup,config=self.config,teacher_version=self.teacher_version,halo=self.halo)
        except (ValueError,FileNotFoundError) as error:
            self.example=None
            for output in [self.video,self.table,self.history,self.diagnostics]:output.clear_output()
            with self.message:
                clear_output()
                text=f'Clip non caricata ({type(error).__name__}):\n{error}'
                display(widgets.HTML(value='<pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+escape(text)+'</pre>'))
            return
        self.example=example
        self.busy=True
        self.frame.value=0;self.play.value=0
        ids=example['result']['objects'].track_id.unique()
        self.track.options=[(str(i),int(i)) for i in ids]
        self.track.value=int(ids[0]) if len(ids) else None
        self.busy=False
        with self.message:
            version='dominio completo · proxy UTLS v0 · ID globali' if self.domain_teacher else 'regole raffinate' if self.teacher_version=='refined_v2' else 'regole iniziali'
            clear_output();print(f"{example['times'][0]} UTC · {example['latitude']:.1f}°N, {example['longitude']:.1f}°E · {version}")
        self.show_frame()

    def show_frame(self, change=None):
        if self.busy or self.example is None:return
        if self.tabs.selected_index==2:
            self.show_history();return
        data=self.example;t=self.frame.value
        clip=data['clip'];r=data['result'];ir=clip[t,4];wv=clip[t,0]
        cmap=ListedColormap(CLASS_COLORS);norm=BoundaryNorm(np.arange(-.5,5.5),5)
        objects=r['objects'][r['objects'].frame==t]
        if self.tabs.selected_index==0:
            with self.video:
                clear_output(wait=True)
                fig,axes=plt.subplots(1,3,figsize=(14,5))
                for ax,arr,title in zip(axes[:2],[ir,wv],['IR 10.8','WV 6.2']):
                    im=ax.imshow(arr,vmin=180,vmax=300,cmap='RdBu_r');ax.set_title(title)
                    fig.colorbar(im,ax=ax,shrink=.7,label='K')
                axes[2].imshow(r['classes'][t],cmap=cmap,norm=norm)
                unknown=r['uncertain'][t] | r.get('phase_uncertain',np.zeros_like(r['uncertain']))[t]
                if unknown.any():axes[2].contourf(unknown,levels=[.5,1.5],colors='none',hatches=['////'])
                axes[2].set_title('Candidati delle regole')
                names=['Sfondo','Nube fredda','Sviluppo','Maturità','OT']
                legend=[Patch(facecolor=color,label=name) for color,name in zip(CLASS_COLORS,names)]
                legend.append(Patch(facecolor='white',hatch='////',label='Fase non valutabile' if self.teacher_version!='legacy' else 'Storia non valutabile'))
                fig.legend(handles=legend,loc='lower center',ncol=6,frameon=False)
                for ax in axes:ax.axis('off')
                fraction=100*np.isin(r['classes'][t],[2,3,4]).mean()
                fig.suptitle(f'Frame {t+1}/16 · +{t*5} min · area convettiva candidata {fraction:.2f}% dei pixel')
                fig.tight_layout(rect=[0,.08,1,.94]);display(fig);plt.close(fig)
        else:
            with self.diagnostics:
                clear_output(wait=True)
                fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
                im=axes[0].imshow(wv-ir,norm=btd_norm(),cmap='RdBu_r');axes[0].set_title('WV6.2 − IR10.8');fig.colorbar(im,ax=axes[0],label='Differenza BT (K)',extend='both')
                if self.domain_teacher:
                    field=self.threshold.value
                    if field in ['cell_id','plateau_mask']:
                        axes[1].imshow(ir,cmap='RdBu_r',vmin=180,vmax=300)
                        if field=='cell_id':
                            from skimage.segmentation import find_boundaries
                            mask=find_boundaries(r[field][t],mode='inner')
                            axes[1].imshow(np.ma.masked_where(~mask,mask),cmap=ListedColormap(['lime']),alpha=.9)
                        else:
                            mask=r[field][t]
                            axes[1].imshow(np.ma.masked_where(~mask,mask),cmap=ListedColormap(['lime']),alpha=.7)
                    else:
                        vmin,vmax,label=(0,2,'D') if field=='thermal_depth' else (0,1,'Indice di confidenza') if field=='proxy_confidence' else (180,300,'K')
                        im=axes[1].imshow(r[field][t],vmin=vmin,vmax=vmax,cmap='viridis')
                        fig.colorbar(im,ax=axes[1],label=label)
                    axes[1].set_title(dict(cell_id='Contorni adattivi',plateau_mask='Plateau selezionato',t_utls_proxy_k='Proxy UTLS/incudine',thermal_depth='Profondità termica D',proxy_confidence='Confidenza proxy',t_edge_k='Temperatura del bordo')[field])
                else:
                    axes[1].imshow(ir,cmap='RdBu_r',vmin=180,vmax=300)
                    level=[240,230,220,210].index(self.threshold.value);mask=r['levels'][t,level]>0
                    if mask.any() and not mask.all():axes[1].contour(mask,levels=[.5],colors=['lime'],linewidths=.9)
                    axes[1].set_title(f'Contorni ≤{self.threshold.value} K')
                if self.show_ids.value:
                    for row in objects.itertuples():
                        yy,xx=np.where(r['tracks'][t]==row.track_id)
                        axes[1].text(xx.mean(),yy.mean(),str(row.track_id),fontsize=8,
                                    bbox=dict(facecolor='white',alpha=.6,pad=1))
                for ax in axes:ax.axis('off')
                display(fig);plt.close(fig)
            with self.table:
                clear_output(wait=True)
                if objects.empty:print('Nessun oggetto rilevato.')
                else:
                    table=objects[['track_id','parents','event','area_km2','contrast_k','cooling_k_15min','growth_km2_15min','border','temporal_valid','class_id','ot_pixels']].copy()
                    if 'threshold_k' in objects:table.insert(4,'threshold_k',objects.threshold_k)
                    for col in ['t_edge_k','t_utls_instant_k','t_utls_proxy_k','proxy_confidence','proxy_reason']:
                        if col in objects:table[col]=objects[col]
                    if 'primary_parent' in objects:table.insert(2,'primary_parent',objects.primary_parent)
                    table['classe']=table.pop('class_id').map(dict(enumerate(CLASS_NAMES)))
                    for flag in ['growth_valid','phase_valid']:
                        if flag in objects:table[flag]=objects[flag]
                    display(table.round(2))

    def show_history(self, change=None):
        if self.busy or self.example is None:return
        with self.history:
            clear_output(wait=True)
            rows=self.example['result']['objects']
            rows=rows[rows.track_id==self.track.value]
            if rows.empty:print('Nessuna traccia disponibile.');return
            fig,axes=plt.subplots(1,3,figsize=(13,3))
            for ax,col,label in zip(axes,['area_km2','min_k','cooling_k_15min'],['Area (km²)','Minimo IR108 (K)','Raffreddamento (K/15min)']):
                ax.plot(rows.frame*5,rows[col],'-o');ax.set(xlabel='Minuti nella clip',ylabel=label);ax.grid(alpha=.2)
            if self.domain_teacher:
                axes[1].lines[0].set_label('Minimo')
                axes[1].plot(rows.frame*5,rows.t_utls_proxy_k,'--',label='Proxy UTLS')
                axes[1].plot(rows.frame*5,rows.t_edge_k,':',label='Bordo')
                axes[1].legend()
            cooling=self.config.cooling_15min_k if self.domain_teacher else self.config.cooling_k_15min if self.config else -2.5
            axes[2].axhline(cooling,color='red',ls='--')
            fig.tight_layout();display(fig);plt.close(fig)


    def close(self):
        self.play.playing=False;self.link.unlink()
