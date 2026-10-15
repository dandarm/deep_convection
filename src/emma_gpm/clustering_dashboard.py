"""Reactive notebook views with explicit global versus single-model controls."""
from pathlib import Path
import json
import base64
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from scipy.cluster.hierarchy import dendrogram

from .seviri_clips import SeviriZarrClipReader
from .embedding_clustering import day_mean_interval
from .physical_cluster_profiles import physical_separation


GROUPS = ['thermal', 'spatial', 'spectral', 'dynamic']
GROUP_LABELS = ['Termica', 'Spaziale', 'Spettrale', 'Temporale']
RUNS = ['mask50', 'mask75', 'mask90', 'random']
RUN_LABELS = ['50%', '75%', '90%', 'Casuale']


class ClusteringDashboard:
    def __init__(self, root):
        self.root = Path(root)
        self.loading = False
        self.reader = None
        self.clip = None
        self.clip_key = None
        self.split = widgets.Dropdown(options=[('Training', 'train'), ('Validation', 'validation'),
            ('Holdout nuovo', 'test_diagnostic')], value='test_diagnostic', description='Dati')
        self.pool = widgets.Dropdown(options=[('Media + tendenza temporale', 'mean_slope'),
            ('Solo media', 'mean')], value='mean_slope', description='Embedding', layout=widgets.Layout(width='350px'))
        self.mode = widgets.Dropdown(options=[('Albero, senza taglio', 'tree'),
            ('Albero, taglio in K gruppi', 'cut'), ('Densità: HDBSCAN', 'density')],
            value='tree', description='Analisi', layout=widgets.Layout(width='330px'))
        self.run = widgets.Dropdown(options=list(zip(RUN_LABELS, RUNS)), value='mask90', description='Encoder')
        self.method = widgets.Dropdown(options=[('Average / coseno', 'average'), ('Ward / euclidea', 'ward')],
            description='Linkage', layout=widgets.Layout(width='280px'))
        self.k = widgets.Dropdown(options=[4, 8, 16, 32], value=8, description='Taglio K')
        self.refresh = widgets.Button(description='Rileggi file', icon='refresh')
        self.group = widgets.Dropdown(options=[], description='Cluster')
        self.member = widgets.Dropdown(options=[], description='Clip', layout=widgets.Layout(width='650px'))
        self.descriptor = widgets.Dropdown(options=[], description='Misura', layout=widgets.Layout(width='650px'))
        self.play = widgets.Play(min=0, max=15, interval=350)
        self.frame = widgets.IntSlider(min=0, max=15, description='Frame')
        self.link = widgets.jslink((self.play, 'value'), (self.frame, 'value'))
        self.summary, self.structure, self.video, self.profiles, self.detail, self.paired = [widgets.Output() for _ in range(6)]
        self.neighbors, self.intercluster = widgets.Output(), widgets.Output()
        self.detail_section = widgets.Accordion(children=[widgets.VBox([self.descriptor, self.detail])], selected_index=None)
        self.detail_section.set_title(0, 'Facoltativo: una misura in unità fisiche')
        self.tabs = widgets.Tab(children=[self.summary, self.structure,
            widgets.VBox([self.group, self.member, widgets.HBox([self.play, self.frame]), self.video]),
            widgets.VBox([self.profiles, self.detail_section]), self.intercluster, self.neighbors, self.paired])
        for index, title in enumerate(['Sintesi gruppi', 'Albero / gruppi', 'Clip', 'Dentro i cluster',
                                        'Tra i cluster', 'Vicini: guadagno', '90% vs altri']):
            self.tabs.set_title(index, title)
        self.widget = widgets.VBox([
            widgets.HTML('<b>Confronto fra tutti gli encoder:</b> Dati, Embedding, Analisi, Linkage/Taglio quando visibili. '
                '<b>Dettaglio:</b> Encoder sceglie albero, clip e matrici fisiche. Tutti i menu aggiornano automaticamente; '
                'Rileggi file serve solo se i risultati sul disco sono cambiati.'),
            widgets.HBox([self.split, self.pool]), widgets.HBox([self.mode, self.method, self.k]),
            widgets.HBox([self.run, self.refresh]), self.tabs])
        for control in [self.split, self.pool, self.mode, self.method, self.k, self.run]:
            control.observe(self.reload, names='value')
        self.group.observe(self.update_members, names='value')
        self.member.observe(self.load_member, names='value')
        self.descriptor.observe(self.show_detail, names='value')
        self.frame.observe(self.show_frame, names='value')
        self.refresh.on_click(self.reload)
        self.reload()

    @property
    def directory(self):
        return self.root / self.split.value

    @property
    def prefix(self):
        method = 'hdbscan' if self.mode.value == 'density' else self.method.value
        return f'{self.run.value}_{self.pool.value}_{method}'

    def labels(self, run=None):
        if self.mode.value == 'tree':
            return None
        method = 'hdbscan' if self.mode.value == 'density' else self.method.value
        frame = pd.read_csv(self.directory / f'{run or self.run.value}_{self.pool.value}_{method}_assignments.csv')
        return frame['cluster' if self.mode.value == 'density' else f'k{self.k.value}'].to_numpy()

    def reload(self, change=None):
        if self.loading:
            return
        self.loading = True
        try:
            self.k.layout.display = '' if self.mode.value == 'cut' else 'none'
            self.method.layout.display = 'none' if self.mode.value == 'density' else ''
            self.group.layout.display = 'none' if self.mode.value == 'tree' else ''
            for output in [self.summary, self.structure, self.video, self.profiles, self.detail, self.paired,
                           self.neighbors, self.intercluster]:
                output.clear_output(wait=False)
            self.manifest = pd.read_csv(self.directory / 'manifest.csv')
            self.descriptors = pd.read_csv(self.directory / 'descriptors.csv')
            self.schema = json.loads((self.directory / 'descriptor_schema.json').read_text())
            previous = self.descriptor.value
            self.descriptor.options = list(self.schema)
            self.descriptor.value = previous if previous in self.schema else 'IR_108_tendency_k_15min'
            self.group.options = []
            if self.mode.value != 'tree':
                labels = self.labels()
                self.group.options = [('Non assegnate (non è un cluster)' if label < 0 else f'Cluster {label}', int(label))
                    for label in sorted(np.unique(labels), key=lambda value: (value < 0, value))]
                self.group.value = self.group.options[0][1] if self.group.options else None
            self.show_summary()
            self.show_structure()
            self.show_profiles()
            self.show_intercluster()
            self.show_neighbors()
            self.show_paired()
        except FileNotFoundError:
            with self.summary:
                print('Risultati non ancora disponibili per questa vista. Eseguire lo script corrispondente e poi Rileggi file.')
            self.member.options = []
            self.clip = None
            self.clip_key = None
        finally:
            self.loading = False
        if self.mode.value == 'tree' or self.group.value is not None:
            self.update_members()
        self.show_detail()

    def show_summary(self):
        with self.summary:
            physical_table = None
            if self.mode.value != 'tree':
                density = self.mode.value == 'density'
                metrics = pd.read_csv(self.root / ('density_metrics.csv' if density else 'cluster_metrics.csv'))
                selected = metrics[(metrics.split == self.split.value) & (metrics.pooling == self.pool.value)]
                if not density:
                    selected = selected[(selected.method == self.method.value) & (selected.clusters == self.k.value)].copy()
                    selected['coverage'] = 1.
                selected = selected.set_index('run').reindex(RUNS)
                table = selected[['clusters', 'coverage', 'silhouette', 'stability_ari']].copy()
                table['coverage'] *= 100
                table.columns = ['Gruppi', 'Clip assegnate %', 'Separazione', 'Stabilità']
                table.index = RUN_LABELS
                display(table.round(3))
                print('Separazione = silhouette. Stabilità = ARI ricampionando intere giornate; maggiore indica più riproducibilità.')
                if density:
                    print('HDBSCAN: K è un risultato. Accordo e silhouette usano solo clip assegnate: leggere sempre anche la copertura.')
                    print('La stabilità riguarda le clip assegnate in entrambi i ricampionamenti, non la riproducibilità delle esclusioni.')
                physical = pd.read_csv(self.root / ('density_descriptor_agreement.csv' if density else 'descriptor_agreement.csv'))
                physical = physical[(physical.split == self.split.value) & (physical.pooling == self.pool.value)
                                    & physical.group.isin(GROUPS)]
                if not density:
                    physical = physical[(physical.method == self.method.value) & (physical.clusters == self.k.value)]
                physical_table = physical.groupby(['run', 'group']).adjusted_eta2.mean().unstack().reindex(index=RUNS, columns=GROUPS)
            else:
                print('Albero completo: nessun numero di gruppi imposto, nessuna partizione o silhouette unica.')
                print('Per la coerenza locale indipendente dai gruppi, apri «Vicini: guadagno».')
            if physical_table is not None:
                fig, axis = plt.subplots(figsize=(8, 4))
                self.family_bars(axis, physical_table, 'Quanto i gruppi distinguono le osservabili fisiche', 'Varianza spiegata corretta')
                fig.tight_layout(); display(fig); plt.close(fig)
                print('La varianza spiegata misura la variabilità fra le medie dei gruppi rispetto alla variabilità totale; non è la media di un descrittore.')
                print('Dentro i cluster: omogeneità interna. Tra i cluster: distanze fra profili medi. Nessuna misura produce una graduatoria universale.')

    def show_neighbors(self):
        with self.neighbors:
            neighbors = pd.read_csv(self.root / 'neighbor_agreement.csv')
            selected = neighbors[(neighbors.split == self.split.value) & (neighbors.pooling == self.pool.value)
                                 & (neighbors.method == 'average')]
            columns = [name for family in GROUPS for name in self.descriptors
                       if self.schema[name]['group'] == family]
            self.neighbor_descriptor_scores = selected.pivot(index='run', columns='descriptor', values='gain').reindex(index=RUNS, columns=columns)
            # Wide figure inside a scrollable notebook panel: full descriptor
            # names and all four encoders stay visible without abbreviations.
            fig, axis = plt.subplots(figsize=(max(18, .55 * len(columns)), 7))
            x = np.arange(len(columns))
            for index, run in enumerate(RUNS):
                axis.bar(x + (index - 1.5) * .19,
                         self.neighbor_descriptor_scores.loc[run], width=.19, label=RUN_LABELS[index])
            axis.set_xticks(x, columns, rotation=90, fontsize=8)
            axis.set(title='Coerenza fisica dei vicini — ogni descrittore', ylabel='Guadagno')
            axis.axhline(0, color='gray', lw=.7)
            axis.legend(ncol=4); axis.grid(axis='y', alpha=.2)
            fig.subplots_adjust(bottom=.52, left=.04, right=.99, top=.92)
            fig.canvas.draw()
            lowest_label = min(tick.get_window_extent(fig.canvas.get_renderer()).y0
                               for tick in axis.get_xticklabels())
            brace_y = (lowest_label - 18 - axis.bbox.y0) / axis.bbox.height
            transform = axis.get_xaxis_transform()
            offset = 0
            for family, title in zip(GROUPS, GROUP_LABELS):
                count = sum(self.schema[name]['group'] == family for name in columns)
                left, right = offset - .45, offset + count - .55
                middle = (left + right) / 2
                # Horizontal underbrace in axes-y/data-x coordinates; the
                # centre notch and end hooks identify the whole tick span.
                axis.plot([left, left, middle-.25, middle, middle+.25, right, right],
                          [brace_y+.02, brace_y, brace_y, brace_y-.03,
                           brace_y, brace_y, brace_y+.02],
                          transform=transform, clip_on=False, color='black', lw=1)
                axis.text(middle, brace_y-.09, title, transform=transform,
                          ha='center', va='top', fontsize=11)
                if offset:
                    axis.axvline(offset-.5, color='gray', alpha=.3)
                offset += count
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
            display(HTML('<div style="overflow-x:auto;width:100%">'
                         '<img alt="Guadagno per descrittore, raggruppato per famiglia" '
                         f'style="max-width:none" src="data:image/png;base64,{encoded}"></div>'))
            plt.close(fig)
            print('«Guadagno» è il nome locale di una riduzione relativa di errore, non un indice universale di clustering.')
            print('0 = nessun vantaggio sulle coppie casuali; 0,5 = errore fisico dimezzato; 1 = descrittori identici; negativo = vicini meno simili.')
            print('Errore: differenza quadratica del descrittore. Dieci vicini di altre giornate; riferimento su tutte le coppie ammissibili, media con pari peso alle giornate.')
            print('Dipende solo da Dati ed Embedding; non da Linkage, Taglio K, HDBSCAN o Encoder selezionato. Confronta sempre tutti gli encoder.')

    def show_intercluster(self):
        with self.intercluster:
            self.profile_means = None
            self.intercluster_distances = {}
            self.separation_summary = None
            if self.labels() is None:
                print('Scegli HDBSCAN o un taglio per confrontare coppie di cluster.')
                return
            groups, counts, means, matrices, _, scores = physical_separation(*self.physical_data())
            self.profile_means = pd.DataFrame(means, index=groups, columns=self.descriptors.columns)
            self.intercluster_distances = matrices
            self.separation_summary = pd.Series(scores)
            if len(groups) < 2:
                print('Servono almeno due gruppi assegnati.')
                return
            self.matrix_plot(matrices, groups, f'Separazione fisica relativa — {self.run.value}',
                             'Distanza fra medie / dispersione dei due gruppi')
            fig, axis = plt.subplots(figsize=(8, 3))
            axis.bar(GROUP_LABELS + ['Complessiva'], [scores[g] for g in GROUPS + ['global']])
            axis.set(ylabel='Separazione media ↑', title='Sintesi delle coppie di cluster')
            axis.grid(axis='y', alpha=.2)
            fig.tight_layout(); display(fig); plt.close(fig)
            print('Valore alto = medie lontane rispetto alla dispersione interna; non è probabilità o correlazione.')
            print('S = sqrt(media delle differenze standardizzate al quadrato / media delle due varianze standardizzate sommate).')
            print('Diagonale zero per convenzione; grigio = gruppi <5 clip, famiglia costante o dispersione della coppia nulla.')
            print('Sintesi: media sulle coppie distinte valide; complessiva = media delle quattro famiglie, tutte richieste valide.')
            print('Scala colori comune alle quattro matrici della vista; confrontare la barra colori fra viste. Leggere anche K e copertura.')
            print('Differenze di forma a medie uguali non vengono rilevate. Non assegnati esclusi; input encoder invariati.')

    @staticmethod
    def family_bars(axis, table, title, ylabel):
        x = np.arange(4)
        for index, run in enumerate(RUNS):
            axis.bar(x + (index - 1.5) * .19, table.loc[run].to_numpy(), width=.19, label=RUN_LABELS[index])
        axis.set_xticks(x, GROUP_LABELS)
        axis.set(title=title, ylabel=ylabel)
        axis.axhline(0, color='gray', lw=.7); axis.legend(); axis.grid(axis='y', alpha=.2)

    def show_structure(self):
        with self.structure:
            fig, axis = plt.subplots(figsize=(11, 4))
            if self.mode.value == 'density':
                labels = self.labels()
                unique, counts = np.unique(labels, return_counts=True)
                axis.bar(range(len(unique)), counts, color=['gray' if label < 0 else 'tab:blue' for label in unique])
                axis.set_xticks(range(len(unique)), ['Non assegnate' if label < 0 else str(label) for label in unique])
                axis.set(title=f'HDBSCAN — encoder {self.run.value}: numero e dimensione dei gruppi trovati',
                         xlabel='Cluster (identificativi arbitrari)', ylabel='Clip')
                protocol = json.loads((self.root / 'density_protocol.json').read_text())
                print(f"Parametri comuni: gruppo minimo {100 * protocol['min_cluster_fraction']:.0f}% delle clip, almeno {protocol['min_cluster_floor']}; min_samples={protocol['min_samples']}.")
                print('Nessun K scelto. Questi parametri regolano la scala della densità: HDBSCAN non è privo di parametri.')
            else:
                tree = np.load(self.directory / f'{self.prefix}_linkage.npy', allow_pickle=False)
                n = len(tree) + 1
                def leaf_label(node):
                    return f'clip {node}' if node < n else f'{int(tree[node - n, 3])} clip'
                dendrogram(tree, truncate_mode='lastp', p=20, ax=axis, leaf_label_func=leaf_label)
                if self.mode.value == 'cut':
                    position = n - self.k.value
                    height = .5 * (tree[position - 1, 2] + tree[position, 2])
                    axis.axhline(height, color='black', linestyle='--', label=f'Taglio K={self.k.value}')
                    axis.legend()
                axis.set(title=f'Encoder {self.run.value} — {self.pool.label} — {self.method.label}',
                         xlabel='Rami aggregati per leggibilità; tutte le clip sono nell’albero', ylabel='Distanza di fusione')
                print('La gerarchia è costruita una volta senza K. Qui si mostrano gli ultimi 20 rami, non solo 20 clip.')
            fig.tight_layout(); display(fig); plt.close(fig)

    def update_members(self, change=None):
        if self.loading:
            return
        self.loading = True
        try:
            if self.mode.value == 'tree':
                options = [(f'{index}: {row.clip_id} — {row.start_time}', index)
                    for index, row in self.manifest.iterrows()]
            else:
                representatives = pd.read_csv(self.directory / f'{self.prefix}_representatives.csv')
                selected = representatives[representatives.cluster == self.group.value]
                if self.mode.value == 'cut':
                    selected = selected[selected.clusters == self.k.value]
                options = [(f'{row.role}: {row.clip_id} — {row.start_time}', int(row.index))
                    for row in selected.itertuples(index=False)]
            previous = self.member.value
            self.member.options = options
            self.member.value = previous if previous in [value for _, value in options] else (options[0][1] if options else None)
        except FileNotFoundError:
            self.member.options = []
        finally:
            self.loading = False
        self.load_member()

    def load_member(self, change=None):
        if self.loading or self.member.value is None:
            return
        key = (self.split.value, self.member.value)
        if key != self.clip_key:
            if self.reader is None:
                protocol = json.loads((self.root / 'protocol.json').read_text())
                self.reader = SeviriZarrClipReader(protocol['input_stats']['zarr_root'])
            row = self.manifest.iloc[self.member.value]
            self.clip = self.reader.load(row.start_time, origin_y=int(row.origin_y), origin_x=int(row.origin_x))
            self.clip_key = key
        self.play.value = 0; self.frame.value = 0
        self.show_frame()

    def show_frame(self, change=None):
        with self.video:
            clear_output(wait=True)
            if self.clip is None:
                return
            frame = self.frame.value
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for axis, data, title, limits in zip(axes,
                [self.clip[frame, 4], self.clip[frame, 0], self.clip[frame, 0] - self.clip[frame, 4]],
                ['IR 10.8', 'WV 6.2', 'WV 6.2 − IR 10.8'], [(180, 330), (180, 330), (-40, 10)]):
                shown = axis.imshow(data, vmin=limits[0], vmax=limits[1], cmap='RdBu_r')
                axis.set_title(f'{title} (K)'); axis.axis('off'); fig.colorbar(shown, ax=axis, shrink=.7)
            fig.suptitle(f'Frame {frame + 1}/16 — +{frame * 5} minuti, north-up/west-left')
            fig.tight_layout(); display(fig); plt.close(fig)

    def physical_data(self):
        moments = json.loads((self.root / 'descriptor_training_moments.json').read_text())
        columns = list(self.descriptors)
        return (self.descriptors.to_numpy(), self.labels(),
                [moments['mean'][name] for name in columns],
                [moments['std'][name] for name in columns],
                [self.schema[name]['group'] for name in columns])

    def matrix_plot(self, matrices, groups, title, colorbar):
        finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices.values()])
        vmax = max(float(finite.max()), 1e-12) if len(finite) else 1.
        palette = plt.get_cmap('viridis').copy(); palette.set_bad('lightgray')
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for axis, family, name in zip(axes.flat, GROUPS, GROUP_LABELS):
            shown = axis.imshow(matrices[family], cmap=palette, vmin=0, vmax=vmax)
            ticks = np.arange(0, len(groups), max(1, len(groups) // 12))
            axis.set_xticks(ticks, groups[ticks]); axis.set_yticks(ticks, groups[ticks])
            axis.set(title=name, xlabel='Cluster', ylabel='Cluster')
        fig.suptitle(title)
        fig.colorbar(shown, ax=axes.ravel().tolist(), label=colorbar)
        display(fig); plt.close(fig)

    def show_profiles(self):
        with self.profiles:
            self.internal_dispersion = None
            if self.labels() is None:
                print('Scegli HDBSCAN o un taglio per misurare la dispersione interna.')
                return
            groups, counts, _, _, dispersion, _ = physical_separation(*self.physical_data())
            self.internal_dispersion = pd.DataFrame(dispersion, index=groups)
            if not len(groups):
                print('Nessun gruppo trovato: tutte le clip sono non assegnate.')
                return
            x = np.arange(len(groups))
            fig, axis = plt.subplots(figsize=(max(8, .4*len(groups)), 4))
            for i, (family, title) in enumerate(zip(GROUPS, GROUP_LABELS)):
                axis.bar(x+(i-1.5)*.19, dispersion[family], width=.19, label=title)
            axis.set_xticks(x, [f'{g}\n({n})' for g,n in zip(groups,counts)])
            axis.set(xlabel='Cluster (numero di clip)', ylabel='Dispersione interna RMS ↓',
                     title=f'Compattezza fisica — {self.run.value}')
            axis.margins(y=.25)
            axis.legend(ncol=4); axis.grid(axis='y', alpha=.2)
            fig.tight_layout(); display(fig); plt.close(fig)
            print('Barra bassa = descrittori più omogenei nel gruppo. RMS delle deviazioni standard interne, in unità delle statistiche training.')
            print('Varianze empiriche ddof=0; gruppi <5 clip e famiglie costanti non valutati; non assegnati esclusi.')

    def show_detail(self, change=None):
        if self.loading or self.descriptor.value is None:
            return
        with self.detail:
            clear_output(wait=True)
            try:
                labels = self.labels()
            except FileNotFoundError:
                return
            if labels is None:
                print('Nessuna partizione attiva.')
                return
            name = self.descriptor.value
            spec = self.schema[name]
            print(f"{spec['definition']} | {spec['unit']}")
            groups = sorted(np.unique(labels[labels >= 0]))
            if not groups:
                return
            fig, axis = plt.subplots(figsize=(10, 3))
            axis.boxplot([self.descriptors[name][labels == group] for group in groups], showfliers=False,
                         showmeans=True, meanprops=dict(marker='^', markerfacecolor='tab:green', markeredgecolor='tab:green'))
            axis.set_xticks(range(1, len(groups) + 1), groups)
            axis.set(xlabel='Cluster', ylabel=spec['unit'], title=name)
            fig.tight_layout(); display(fig); plt.close(fig)
            print('Triangolo verde = media del descrittore nel cluster; linea nel box = mediana.')

    def show_paired(self):
        with self.paired:
            days = pd.to_datetime(self.manifest.start_time, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
            gains = {}
            for run in ['mask50', 'mask75', 'mask90']:
                with np.load(self.directory / f'{run}_{self.pool.value}_average_neighbors.npz', allow_pickle=False) as arrays:
                    gains[run] = arrays['gain_by_clip'].copy()
            fig, axes = plt.subplots(1, 2, figsize=(11, 3), sharey=True)
            for axis, reference in zip(axes, ['mask50', 'mask75']):
                for index, group in enumerate(GROUPS):
                    columns = [column for column, name in enumerate(self.descriptors) if self.schema[name]['group'] == group]
                    paired = np.nanmean((gains['mask90'] - gains[reference])[:, columns], axis=1)[:, None]
                    mean, low, high = day_mean_interval(paired, days)
                    axis.plot([low[0], high[0]], [index, index], color='gray')
                    axis.plot(mean[0], index, 'o', color='tab:blue')
                axis.axvline(0, color='black', linestyle='--')
                axis.set(title=f'90% − {reference[4:]}%', xlabel='Differenza di guadagno dei vicini')
            axes[0].set_yticks(range(4), GROUP_LABELS)
            fig.tight_layout(); display(fig); plt.close(fig)
            print('Destra favorisce 90%, sinistra l’altro modello. Intervalli descrittivi 95% appaiati per giornata.')
            print('Questo confronto dipende da Dati ed Embedding; non da cluster, linkage o encoder selezionato. Non è un test di significatività multiplo.')

    def close(self):
        if self.reader is not None:
            self.reader.close()
        self.link.unlink()
