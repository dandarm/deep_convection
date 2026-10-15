"""Read-only, full-domain day exploration of already committed teacher maps.

Kept outside rdt/ so adding display code does not change the running teacher's
scientific source fingerprint. No detection, tracking or label generation.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import zarr
from .rdt.storage import open_fields
from IPython.display import clear_output, display

from .rdt.storage import DomainFrameReader, frame_key, read_table
from .rdt_teacher import CLASS_COLORS
from .rdt_event_explorer import btd_norm, rank_frames, study_window, study_origin


FIELDS = [("Classi sul satellite", "semantic_label"), ("Contorni adattivi", "cell_id"),
          ("Plateau dell’incudine", "plateau_mask"), ("Proxy UTLS/incudine", "t_utls_proxy_k"),
          ("Profondità D", "thermal_depth"), ("Confidenza proxy", "proxy_confidence"),
          ("Candidati OT", "ot_mask")]


class DomainDayArchive:
    """Lazy full-frame reads, explicit missing slots, small bounded RAM cache."""

    def __init__(self, root, output, cache_frames=2):
        self.root, self.output = Path(root), Path(output)
        self.protocol = json.loads((self.output/"protocol.json").read_text())
        if self.protocol["root"] != str(self.root.resolve()):
            raise ValueError("The teacher maps belong to a different SEVIRI archive")
        if self.protocol["orientation"] != "north-up west-left":
            raise ValueError("Unsupported label orientation")
        self.cache_frames = int(cache_frames)
        if self.cache_frames < 1:
            raise ValueError("Cache size must be positive")
        self.reader = DomainFrameReader(self.root)
        self.cache = OrderedDict()
        self.extent, self.graticule = None, None
        self.refresh()

    def refresh(self):
        request = json.loads((self.output/"request.json").read_text())
        start, end = pd.Timestamp(request["start"]), pd.Timestamp(request["end_exclusive"])
        stamps = []
        for directory in sorted((self.output/"frames").glob("*")):
            if directory.name.endswith(".part") or not (directory/"record.json").exists():
                continue
            stamp = pd.to_datetime(directory.name, format="%Y%m%dT%H%M")
            if start <= stamp < end:
                stamps.append(stamp)
        self.committed = pd.DatetimeIndex(stamps)
        self.metrics = read_table(self.output/"frame_metrics.csv") if (self.output/"frame_metrics.csv").exists() else pd.DataFrame()
        if not self.metrics.empty:
            self.metrics["time"] = pd.to_datetime(self.metrics.time)

    def exploration(self):
        directory=self.output/'exploration'
        frames=read_table(directory/'frame_index.csv') if (directory/'frame_index.csv').exists() else pd.DataFrame()
        events=read_table(directory/'events.csv') if (directory/'events.csv').exists() else pd.DataFrame()
        if not frames.empty:
            frames['time']=pd.to_datetime(frames.time)
            frames=frames[frames.time.isin(self.committed)]
        if not events.empty:
            events['time']=pd.to_datetime(events.time)
            events=events[events.time.isin(self.committed)]
        return frames,events

    def day_options(self):
        days = self.committed.normalize().unique()
        return [(f"{day:%Y-%m-%d} · {int((self.committed.normalize()==day).sum())}/288 frame", day.strftime("%Y-%m-%d")) for day in days]

    def slots(self, day):
        """Keep the five-minute clock, including holes; never replace a slot."""
        start = pd.Timestamp(day).normalize()
        observed = self.committed[(self.committed >= start) & (self.committed < start+pd.Timedelta(days=1))]
        return pd.date_range(start, observed[-1], freq="5min") if len(observed) else pd.DatetimeIndex([])

    def _set_geometry(self):
        x, y, projection = self.reader.geometry
        fingerprint = hashlib.sha256(x.tobytes()+y.tobytes()+projection.encode()).hexdigest()
        if fingerprint != self.protocol["geometry_sha256"]:
            raise ValueError("Raw satellite geometry differs from the stored teacher maps")
        x = x if x[0] < x[-1] else x[::-1]
        y = y if y[0] > y[-1] else y[::-1]
        dx, dy = x[1]-x[0], y[1]-y[0]
        self.extent = ((x[0]-dx/2)/1000, (x[-1]+dx/2)/1000,
                       (y[-1]+dy/2)/1000, (y[0]-dy/2)/1000)
        xx, yy = np.meshgrid(np.linspace(x[0], x[-1], 128), np.linspace(y[0], y[-1], 64))
        lon, lat = Transformer.from_crs(CRS.from_user_input(projection), "EPSG:4326", always_xy=True).transform(xx, yy)
        self.graticule = (xx/1000, yy/1000, np.ma.masked_invalid(lon), np.ma.masked_invalid(lat))

    def load(self, stamp, field="semantic_label"):
        stamp = pd.Timestamp(stamp)
        if stamp not in self.committed:
            raise FileNotFoundError(f"Slot non disponibile: {stamp:%Y-%m-%d %H:%M} UTC")
        if stamp not in self.cache:
            raw = self.reader.load(stamp)
            if self.extent is None:
                self._set_geometry()
            with open_fields(self.output/"frames"/frame_key(stamp)) as group:
                if group.attrs.get("config_hash") != self.protocol["config_hash"] or pd.Timestamp(group.attrs["time"]) != stamp:
                    raise ValueError("Frame version/timestamp differs from its label archive")
                fields = {name: np.asarray(group[name][:]) for name in ("semantic_label", "valid_observation")}
            if fields["semantic_label"].shape != raw.shape[1:] or tuple(raw.shape[1:]) != tuple(self.protocol["shape"]):
                raise ValueError("Raw satellite and teacher maps have different shapes")
            # Copy the two display bands, releasing the other five. No day cube
            # is loaded or materialized, and invalid observations remain invalid.
            self.cache[stamp] = dict(ir=raw[4].copy(), wv=raw[0].copy(), fields=fields)
        entry = self.cache[stamp]
        if field not in entry["fields"]:
            with open_fields(self.output/"frames"/frame_key(stamp)) as group:
                entry["fields"][field] = np.asarray(group[field][:])
        self.cache.move_to_end(stamp)
        while len(self.cache) > self.cache_frames:
            self.cache.popitem(last=False)
        return entry

    def close(self):
        self.reader.close()
        self.cache.clear()


class RDTDomainDayViewer:
    def __init__(self, root, archives, *, preferred_day="2020-07-11"):
        self.root = Path(root)
        self.archives = {label: Path(path) for label, path in archives.items() if (Path(path)/"protocol.json").exists()}
        if not self.archives:
            raise FileNotFoundError("No committed full-domain label archives")
        self.preferred_day, self.archive, self.busy, self.figure = preferred_day, None, False, None
        self.source = widgets.Dropdown(options=list(self.archives), description="Scansione", layout=widgets.Layout(width="420px"))
        self.day = widgets.Dropdown(options=[], description="Giornata", layout=widgets.Layout(width="330px"))
        self.refresh_button = widgets.Button(description="Rileggi giornate", icon="refresh")
        self.field = widgets.Dropdown(options=FIELDS, description="Regole", layout=widgets.Layout(width="340px"))
        self.band = widgets.Dropdown(options=[("IR 10.8", "ir"), ("WV 6.2", "wv"), ("WV6.2 − IR10.8", "btd")], description="Satellite")
        self.grid = widgets.Checkbox(value=True, description="Riferimenti geografici")
        self.uncertain = widgets.Checkbox(value=False, description="Segnala fase non valutabile")
        self.step = widgets.Dropdown(options=[("5 min", 1), ("15 min", 3), ("30 min", 6)], value=1, description="Passo")
        self.play = widgets.Play(min=0, max=287, interval=1200)
        self.frame = widgets.IntSlider(min=0, max=287, description="Frame", continuous_update=False, layout=widgets.Layout(width="65%"))
        self.link = widgets.jslink((self.play, "value"), (self.frame, "value"))
        self.clock = widgets.HTML()
        self.conv_peak = widgets.Button(description="Picco convettivo")
        self.ot_peak = widgets.Button(description="Picco OT")
        self.scope=widgets.Dropdown(options=['Giornata visualizzata','Tutte le giornate calcolate'],value='Tutte le giornate calcolate',description='Ricerca',layout=widgets.Layout(width='400px'))
        self.rank_metric=widgets.Dropdown(options=[('Convezione · % pixel validi','convective_percent'),('OT · % pixel validi','ot_percent')],description='Graduatoria',layout=widgets.Layout(width='400px'))
        self.ranked=widgets.Dropdown(options=[],description='Picchi',layout=widgets.Layout(width='95%'))
        self.study_kind=widgets.Dropdown(options=[('Celle più estese','convection'),('Più pixel OT persistenti','ot')],description='Studio',layout=widgets.Layout(width='400px'))
        self.study=widgets.Dropdown(options=[],description='Traccia',layout=widgets.Layout(width='95%'))
        self.zoom_button=widgets.Button(description='Zoom · clip 16 frame',icon='search')
        self.index_status=widgets.HTML()
        self.zoom_output=widgets.Output()
        self.zoom_viewer=None
        self.output, self.activity_output = widgets.Output(), widgets.Output()
        self.widget = widgets.VBox([widgets.HBox([self.source, self.day, self.refresh_button]),
            widgets.HBox([self.play, self.frame, self.step]), self.clock,
            widgets.HBox([self.band, self.field]), widgets.HBox([self.grid, self.uncertain]),
            widgets.HBox([self.scope,self.conv_peak,self.ot_peak]),
            widgets.HBox([self.rank_metric]),self.ranked,self.index_status,self.output,
            widgets.Accordion(children=[widgets.VBox([self.study_kind,self.study,self.zoom_button,self.zoom_output])],
                              titles=('Zoom di studio · candidati per traccia',),selected_index=None)])
        self.source.observe(self.select_source, names="value")
        self.day.observe(self.select_day, names="value")
        self.frame.observe(self.show_frame, names="value")
        for control in (self.field, self.band, self.grid, self.uncertain):
            control.observe(self.show_frame, names="value")
        self.step.observe(self.change_step, names="value")
        self.refresh_button.on_click(self.refresh)
        self.conv_peak.on_click(lambda _: self.jump_to_peak("convective_objects"))
        self.ot_peak.on_click(lambda _: self.jump_to_peak("ot_pixels"))
        self.scope.observe(self.update_rankings,names='value')
        self.rank_metric.observe(self.update_rankings,names='value')
        self.ranked.observe(self.select_ranked,names='value')
        self.study_kind.observe(self.update_studies,names='value')
        self.zoom_button.on_click(self.show_study)
        self.select_source()

    def select_source(self, change=None):
        if self.busy:
            return
        self.play.playing = False
        if self.archive:
            self.archive.close()
        if self.zoom_viewer:
            self.zoom_viewer.close();self.zoom_viewer=None
            self.zoom_output.clear_output()
        self.archive = DomainDayArchive(self.root, self.archives[self.source.value])
        self.refresh()

    def refresh(self, button=None):
        self.play.playing = False
        self.archive.refresh()
        self.busy = True
        before = self.day.value
        options = self.archive.day_options()
        values = [value for _, value in options]
        self.day.options = options
        self.day.value = before if before in values else self.preferred_day if self.preferred_day in values else values[0] if values else None
        self.busy = False
        self.frame_index,self.event_index=self.archive.exploration()
        self.select_day()

    def select_day(self, change=None, *, target_time=None):
        if self.busy:
            return
        self.play.playing = False
        self.slots = self.archive.slots(self.day.value) if self.day.value else pd.DatetimeIndex([])
        self.busy = True
        self.frame.value = self.play.value = 0
        self.frame.max = self.play.max = max(0, len(self.slots)-1)
        if target_time is not None and target_time in self.slots:
            self.frame.value=self.play.value=int(self.slots.get_indexer([target_time])[0])
        self.busy = False
        self.activity = self.archive.metrics[self.archive.metrics.time.dt.normalize() == pd.Timestamp(self.day.value)].copy() if self.day.value and not self.archive.metrics.empty else pd.DataFrame()
        self.update_rankings()
        self.update_studies()
        self.show_frame()
        self.show_activity()

    def change_step(self, change=None):
        self.play.step = self.step.value

    def ranked_scope(self):
        frames=self.frame_index
        if self.scope.value=='Giornata visualizzata' and self.day.value and not frames.empty:
            frames=frames[frames.time.dt.normalize()==pd.Timestamp(self.day.value)]
        return frames

    def update_rankings(self, change=None):
        if self.busy:return
        rows=rank_frames(self.ranked_scope(),self.rank_metric.value)
        self.busy=True
        self.ranked.options=[(f"{row.time:%Y-%m-%d %H:%M} UTC · convezione {row.convective_percent:.3f}% · OT {row.ot_percent:.5f}%",row.time.isoformat()) for row in rows.itertuples()]
        self.ranked.value=None
        self.busy=False
        frames=self.ranked_scope()
        self.conv_peak.disabled=frames.empty or not (frames.convective_percent>0).any()
        self.ot_peak.disabled=frames.empty or not (frames.ot_percent>0).any()
        self.index_status.value=f"Indice: {len(self.frame_index):,}/{len(self.archive.committed):,} frame pubblicati. % sui pixel validi; picchi distanziati di almeno 90 minuti. Graduatoria provvisoria durante la scansione."

    def jump_to_time(self, stamp):
        stamp=pd.Timestamp(stamp)
        if stamp not in self.archive.committed:return
        self.play.playing=False
        self.busy=True
        self.day.value=stamp.strftime('%Y-%m-%d')
        self.busy=False
        self.select_day(target_time=stamp)

    def select_ranked(self, change=None):
        if not self.busy and self.ranked.value:
            self.jump_to_time(self.ranked.value)

    def jump_to_peak(self, column):
        metric='convective_percent' if column=='convective_objects' else 'ot_percent'
        frames=self.ranked_scope()
        if not frames.empty and (frames[metric]>0).any():
            self.jump_to_time(frames.sort_values([metric,'time'],ascending=[False,True]).iloc[0].time)

    def update_studies(self, change=None):
        if self.busy:return
        events=self.event_index
        if not events.empty:
            events=events[events.kind==self.study_kind.value].sort_values(['score','time','track_id'],ascending=[False,True,True]).head(100)
        self.busy=True
        self.study.options=[(f"{row.time:%Y-%m-%d %H:%M} UTC · {'area' if row.kind=='convection' else 'OT'} {row.score:,.1f} {'km²' if row.kind=='convection' else 'pixel'} · ID {row.track_id}",(int(row.track_id),row.time.isoformat())) for row in events.itertuples()]
        self.study.value=self.study.options[0][1] if len(self.study.options) else None
        self.zoom_button.disabled=self.study.value is None
        self.busy=False

    def show_study(self, button=None):
        if self.study.value is None:return
        from .rdt.storage import load_domain_crop
        from .rdt_dashboard import RDTClipViewer
        with self.zoom_output:
            clear_output(wait=True)
            track_id,time=self.study.value
            try:
                stamp=pd.Timestamp(time)
                entry=self.archive.load(stamp,'track_id')
                x,y,_=self.archive.reader.geometry
                origin_y,origin_x=study_origin(entry['fields']['track_id'],track_id,x,y)
                start=study_window(self.archive.committed,stamp)
                # The annual frames are symlinks into independent sequences;
                # use that sequence geometry/area cache for the study crop.
                label_root=(self.archive.output/"frames"/frame_key(stamp)).resolve().parent.parent
                example=load_domain_crop(self.root,label_root,start,origin_y,origin_x)
                if track_id not in example['result']['objects'].track_id.values:
                    raise ValueError('La traccia selezionata non è visibile nel crop centrato; serve un dominio di zoom più ampio.')
                labels=example['result']['classes']
                conv=np.isin(labels,[2,3,4]);ot=labels==4
                catalog=pd.DataFrame([dict(start_time=start.isoformat(),origin_y=origin_y,origin_x=origin_x,
                    teacher_version='multispectral_utls_v0',label_root=str(label_root),status='ok',
                    convective_pixels=int(conv.sum()),ot_pixels=int(ot.sum()),cold_pixels=int((labels>0).sum()),
                    convective_peak_fraction=float(conv.mean(axis=(1,2)).max()),ot_peak_fraction=float(ot.mean(axis=(1,2)).max()))])
                if self.zoom_viewer:self.zoom_viewer.close()
                self.zoom_viewer=RDTClipViewer(self.root,catalog)
                self.zoom_viewer.frame.value=int((stamp-start)/pd.Timedelta(minutes=5))
                self.zoom_viewer.play.value=self.zoom_viewer.frame.value
                if track_id in [value for _,value in self.zoom_viewer.track.options]:
                    self.zoom_viewer.track.value=track_id
                print('Crop centrato sulla traccia al picco; finestra completa, senza riempire gap. Candidati per traccia, non eventi meteorologici indipendenti validati.')
                display(self.zoom_viewer.widget)
            except (ValueError,FileNotFoundError,KeyError) as error:
                print(f'Zoom non disponibile: {error}')

    def _geographic_grid(self, ax):
        xx, yy, lon, lat = self.archive.graticule
        for values, candidates, suffix in ((lon, range(-20, 51, 10), "°E"), (lat, range(20, 81, 10), "°N")):
            if values.count() == 0:
                continue
            levels = [value for value in candidates if values.min() < value < values.max()]
            if levels:
                lines = ax.contour(xx, yy, values, levels=levels, colors="#444444", linewidths=.45, alpha=.5)
                ax.clabel(lines, fontsize=7, fmt=lambda value: f"{abs(value):g}{'°W' if suffix=='°E' and value<0 else suffix}")

    def show_frame(self, change=None):
        if self.busy:
            return
        with self.output:
            clear_output(wait=True)
            if not len(self.slots):
                self.clock.value = "Nessun frame pubblicato per questa giornata."
                return
            stamp = self.slots[self.frame.value]
            self.clock.value = f"<b>{stamp:%Y-%m-%d %H:%M} UTC</b> · {self.frame.value+1}/{len(self.slots)} · intero dominio"
            try:
                entry = self.archive.load(stamp, self.field.value)
                if self.uncertain.value:
                    for name in ("uncertainty", "phase_uncertainty"):
                        self.archive.load(stamp, name)
                if self.field.value == "ot_mask":
                    self.archive.load(stamp, "provisional_ot")
            except (ValueError, FileNotFoundError, KeyError) as error:
                self.play.playing = False
                print(f"{error}. Nessun frame sostituito.")
                return
            fields = entry["fields"]
            valid = fields["valid_observation"]
            count=int(valid.sum())
            if count:
                convection=int((np.isin(fields["semantic_label"],[2,3,4])&valid).sum())
                ot=int(((fields["semantic_label"]==4)&valid).sum())
                self.clock.value+=f" · convezione {100*convection/count:.3f}% · OT {100*ot/count:.5f}% dei pixel validi"
            if self.figure is not None:
                plt.close(self.figure)
            # Two native-aspect full-width maps; allow enough vertical space
            # so preserving geography does not shrink them into narrow panels.
            height = 2*13*entry["ir"].shape[0]/entry["ir"].shape[1]+1.6
            figure, panels = plt.subplots(2, 2, figsize=(15, height), constrained_layout=True,
                gridspec_kw={"width_ratios": [1, .025]})
            axes, color_axes = panels[:, 0], panels[:, 1]
            self.figure = figure
            extent = self.archive.extent
            display_band = entry["wv"]-entry["ir"] if self.band.value == "btd" else entry[self.band.value]
            scale=dict(norm=btd_norm()) if self.band.value=='btd' else dict(vmin=180,vmax=310)
            im = axes[0].imshow(np.ma.masked_where(~valid, display_band), extent=extent, origin="upper", cmap="RdBu_r", **scale)
            figure.colorbar(im, cax=color_axes[0], label="Differenza BT (K)" if self.band.value=='btd' else "K", extend='both' if self.band.value=='btd' else 'neither')
            if self.band.value=='btd':
                values=display_band[valid & np.isfinite(display_band)]
                if len(values):
                    self.clock.value+=f" · BTD nel frame: mediana {np.median(values):.1f} K, {100*(values>0).mean():.2f}% positiva · scala fissa −80/+10 K, bianco a 0"

            axes[0].set_title(dict(ir="IR 10.8", wv="WV 6.2", btd="WV6.2 − IR10.8")[self.band.value])
            field = self.field.value
            legend = []
            if field in ("semantic_label", "cell_id", "plateau_mask", "ot_mask"):
                color_axes[1].set_visible(False)
                axes[1].imshow(np.ma.masked_where(~valid, entry["ir"]), extent=extent, origin="upper", vmin=180, vmax=310, cmap="gray_r")
                if field == "semantic_label":
                    cmap = ListedColormap(CLASS_COLORS)
                    masked = np.ma.masked_where((fields[field] == 0) | ~valid, fields[field])
                    axes[1].imshow(masked, extent=extent, origin="upper", cmap=cmap, norm=BoundaryNorm(np.arange(-.5, 5.5), 5), alpha=.85)
                    legend = [Patch(color=color, label=label) for color, label in zip(CLASS_COLORS[1:], ["Nube / non confermata", "Sviluppo", "Maturità / decadimento", "OT candidato persistente"])]
                else:
                    if field == "cell_id":
                        from skimage.segmentation import find_boundaries
                        mask = find_boundaries(fields[field], mode="inner") & valid
                    else:
                        mask = fields[field] & valid
                    color = "#d400e8" if field == "ot_mask" else "lime"
                    axes[1].imshow(np.ma.masked_where(~mask, mask), extent=extent, origin="upper", cmap=ListedColormap([color]), alpha=.85)
                    if field == "ot_mask":
                        provisional = fields["provisional_ot"] & valid
                        axes[1].imshow(np.ma.masked_where(~provisional, provisional), extent=extent, origin="upper", cmap=ListedColormap(["#ffb000"]), alpha=.85)
                        legend = [Patch(color="#d400e8", label="Candidato persistente"), Patch(color="#ffb000", label="Provvisorio")]
            else:
                lo, hi, unit = (0, 2, "D") if field == "thermal_depth" else (0, 1, "Indice di confidenza") if field == "proxy_confidence" else (180, 310, "K")
                values = np.ma.masked_where(~valid | (fields["semantic_label"] == 0) | ~np.isfinite(fields[field]), fields[field])
                im = axes[1].imshow(values, extent=extent, origin="upper", vmin=lo, vmax=hi, cmap="viridis")
                figure.colorbar(im, cax=color_axes[1], label=unit)
            if self.uncertain.value:
                unknown = (fields["uncertainty"] | fields["phase_uncertainty"]) & valid & (fields["semantic_label"] > 0)
                yy, xx = np.indices(unknown.shape)
                stripes = unknown & ((xx+yy) % 12 < 2)
                axes[1].imshow(np.ma.masked_where(~stripes, stripes), extent=extent, origin="upper", cmap=ListedColormap(["#aaaaaa"]), alpha=.7)
                legend.append(Patch(facecolor="#aaaaaa", hatch="////", label="Fase non valutabile"))
            axes[1].set_title(dict((value, label) for label, value in FIELDS)[field])
            if legend:
                axes[1].legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, -.13), ncol=min(4, len(legend)), frameon=False)
            for ax in axes:
                ax.set_facecolor("#d9d9d9")
                ax.set(xlabel="x proiettata (km)", ylabel="y proiettata (km)")
                if self.grid.value:
                    self._geographic_grid(ax)
                ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
            display(figure)
            plt.close(figure)

    def show_activity(self):
        with self.activity_output:
            clear_output(wait=True)
            indexed=self.frame_index[self.frame_index.time.dt.normalize()==pd.Timestamp(self.day.value)] if self.day.value and not self.frame_index.empty else pd.DataFrame()
            if indexed.empty and self.activity.empty:
                print("Riepilogo giornaliero non ancora disponibile.")
                return
            from matplotlib.dates import DateFormatter, HourLocator
            figure, axes = plt.subplots(1, 2, figsize=(14, 3), constrained_layout=True)
            if not indexed.empty:
                frame=indexed.set_index('time').reindex(self.slots)
                columns=['convective_percent','ot_percent']
                titles=['Occupazione convettiva candidata','Occupazione OT persistente']
                units=['% pixel validi','% pixel validi']
            else:
                # Keep existing daily counts usable while the new index catches up.
                frame=self.activity.set_index('time').reindex(self.slots)
                columns=['convective_objects','ot_pixels']
                titles=['Celle candidate · percentuali in preparazione','Pixel OT · percentuali in preparazione']
                units=['Oggetti','Pixel']
            # Missing observations and unindexed frames remain gaps in the curves.
            for ax, column, title, unit in zip(axes,columns,titles,units):
                ax.plot(frame.index, frame[column], linewidth=1)
                ax.set(title=title, xlabel="Ora UTC", ylabel=unit)
                ax.xaxis.set_major_locator(HourLocator(interval=4))
                ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
                ax.grid(alpha=.2)
            display(figure)
            plt.close(figure)

    def close(self):
        self.play.playing = False
        self.link.unlink()
        if self.archive:
            self.archive.close()
        if self.zoom_viewer:
            self.zoom_viewer.close()
        if self.figure is not None:
            plt.close(self.figure)
