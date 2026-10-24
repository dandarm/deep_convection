#!/usr/bin/env python3
"""Build the HTML ATBD for RDT steps 2--4 from saved full-domain fields.

The renderer never detects or tracks cells inside the displayed crop.  It reads
the immutable full-domain teacher products, then crops them only for display.
The resulting HTML is self-contained and embeds the observational MP4.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.rdt.config import RDTTeacherConfig  # noqa: E402
from emma_gpm.rdt.storage import load_domain_crop  # noqa: E402


CLASS_COLORS = ["#00000000", "#94a3b8", "#f59e0b", "#ef4444", "#d946ef"]
CLASS_NAMES = {
    1: "nube/cella non convettiva",
    2: "sviluppo",
    3: "maturità/decadimento",
    4: "OT persistente",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path(
            os.environ.get(
                "SEVIRI_ZARR_ROOT",
                r"E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N",
            )
        ),
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=ROOT / "results" / "rdt_preview" / "atbd_local_20200917",
    )
    parser.add_argument("--start-time", default="2020-09-17T13:30:00")
    parser.add_argument("--origin-y", type=int, default=302)
    parser.add_argument("--origin-x", type=int, default=1223)
    parser.add_argument("--track-id", type=int, default=411)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures" / "videomae_pretraining",
    )
    return parser.parse_args()


def _track_rows(example: dict, track_id: int) -> pd.DataFrame:
    rows = example["result"]["objects"]
    rows = rows.loc[rows.track_id == track_id].copy().sort_values("frame")
    if rows.empty:
        raise ValueError(f"Track {track_id} is not visible in the selected crop")
    rows["time"] = pd.to_datetime(rows.time)
    return rows


def _metrics(example: dict, rows: pd.DataFrame, track_id: int) -> pd.DataFrame:
    result = example["result"]
    records = []
    by_frame = rows.set_index("frame")
    for frame, stamp in enumerate(example["times"]):
        mask = result["tracks"][frame] == track_id
        depth = result["thermal_depth"][frame]
        btd = result["BTD_062_108"][frame]
        row = by_frame.loc[frame] if frame in by_frame.index else None
        records.append(
            {
                "frame": frame,
                "time": pd.Timestamp(stamp),
                "pixels": int(mask.sum()),
                "max_depth": float(np.nanmax(depth[mask])) if mask.any() else np.nan,
                "max_btd_062_108": float(np.nanmax(btd[mask])) if mask.any() else np.nan,
                "class_id": int(row.class_id) if row is not None else 0,
                "provisional": int(row.provisional_ot_pixels) if row is not None else 0,
                "confirmed": int(row.ot_pixels) if row is not None else 0,
            }
        )
    return pd.DataFrame(records)


def _boundaries(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3)))


def render_video(
    example: dict,
    rows: pd.DataFrame,
    metrics: pd.DataFrame,
    track_id: int,
    output: Path,
    fps: float,
) -> None:
    result = example["result"]
    clip = example["clip"]
    frames = np.arange(len(example["times"]))
    series = rows.set_index("frame").reindex(frames)

    fig = plt.figure(figsize=(12.8, 8.0), facecolor="#07111f")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.08, 0.92), hspace=0.23, wspace=0.20)
    ax_ir = fig.add_subplot(grid[0, 0])
    ax_d = fig.add_subplot(grid[0, 1])
    ax_t = fig.add_subplot(grid[1, 0])
    ax_gate = fig.add_subplot(grid[1, 1])
    for axis in (ax_ir, ax_d, ax_t, ax_gate):
        axis.set_facecolor("#07111f")
        axis.tick_params(colors="#a9b7ca")
        for spine in axis.spines.values():
            spine.set_color("#334155")

    ir_artist = ax_ir.imshow(clip[0, 4], origin="upper", cmap="turbo_r", vmin=205, vmax=285)
    contour_artist = ax_ir.imshow(
        np.ma.masked_all(clip.shape[-2:]), origin="upper", cmap=ListedColormap(["#ffffff"]), alpha=0.95
    )
    plateau_artist = ax_ir.imshow(
        np.ma.masked_all(clip.shape[-2:]), origin="upper", cmap=ListedColormap(["#22d3ee"]), alpha=0.82
    )
    provisional_artist = ax_ir.imshow(
        np.ma.masked_all(clip.shape[-2:]), origin="upper", cmap=ListedColormap(["#f0abfc"]), alpha=1.0
    )
    ax_ir.set_title("IR 10.8 · cella, plateau e candidato OT", color="white")
    ax_ir.set(xlabel="colonna · ovest → est", ylabel="riga · nord → sud")
    cbar_ir = fig.colorbar(ir_artist, ax=ax_ir, fraction=0.046, pad=0.03)
    cbar_ir.set_label("K", color="white")
    cbar_ir.ax.tick_params(colors="white")

    d_artist = ax_d.imshow(
        np.ma.masked_all(clip.shape[-2:]), origin="upper", cmap="viridis", vmin=0, vmax=4
    )
    ax_d.set_title("Profondità termica relativa D", color="white")
    ax_d.set(xlabel="colonna · ovest → est", ylabel="riga · nord → sud")
    cbar_d = fig.colorbar(d_artist, ax=ax_d, fraction=0.046, pad=0.03, extend="max")
    cbar_d.set_label("D [1]", color="white")
    cbar_d.ax.tick_params(colors="white")

    ax_t.plot(frames, series.t_edge_k, color="#f8fafc", label="T_edge", linewidth=1.8)
    ax_t.plot(frames, series.t_utls_proxy_k, color="#22d3ee", label="T_proxy filtrata", linewidth=1.8)
    ax_t.plot(frames, series.min_k, color="#f97316", label="T_min", linewidth=1.8)
    cursor = ax_t.axvline(0, color="#d946ef", linewidth=1.5)
    ax_t.set(xlim=(0, 15), ylim=(235, 208), xlabel="frame a 5 minuti", ylabel="K")
    ax_t.set_title(f"Storia causale della traccia {track_id}", color="white")
    ax_t.grid(color="#334155", alpha=0.65, linewidth=0.6)
    legend = ax_t.legend(loc="lower left", facecolor="#111827", edgecolor="#334155")
    for label in legend.get_texts():
        label.set_color("white")

    ax_gate.axis("off")
    gate_text = ax_gate.text(
        0.02,
        0.98,
        "",
        transform=ax_gate.transAxes,
        va="top",
        ha="left",
        color="white",
        family="monospace",
        fontsize=10.0,
        linespacing=1.45,
    )
    ax_gate.set_title("Decisioni al timestamp corrente", color="white")
    title = fig.suptitle("", color="white", fontsize=16, y=0.985)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=2100,
        metadata={"title": "RDT-CW-inspired ATBD steps 2-4", "comment": "Full-domain labels cropped for display"},
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    # Integer figure dimensions at 100 dpi keep both H.264 axes even.
    with writer.saving(fig, str(output), dpi=100):
        for frame, stamp in enumerate(example["times"]):
            track_mask = result["tracks"][frame] == track_id
            plateau = result["plateau_mask"][frame] & track_mask
            provisional = result["provisional_ot"][frame] & track_mask
            contour = _boundaries(track_mask)
            ir_artist.set_data(clip[frame, 4])
            contour_artist.set_data(np.ma.masked_where(~contour, contour))
            plateau_artist.set_data(np.ma.masked_where(~plateau, plateau))
            provisional_artist.set_data(np.ma.masked_where(~provisional, provisional))
            depth = np.ma.masked_where(~track_mask | ~np.isfinite(result["thermal_depth"][frame]), result["thermal_depth"][frame])
            d_artist.set_data(depth)
            cursor.set_xdata([frame, frame])
            row = series.loc[frame]
            m = metrics.loc[metrics.frame == frame].iloc[0]

            temporal = bool(row.temporal_valid) if pd.notna(row.temporal_valid) else False
            growth_valid = bool(row.growth_valid) if pd.notna(row.growth_valid) else False
            cool = f"{row.cooling_k_15min:+.2f} K" if pd.notna(row.cooling_k_15min) else "N/D"
            growth = f"{row.growth_km2_15min:+.0f} km²" if pd.notna(row.growth_km2_15min) else "N/D"
            aor = f"{row.aor_from_parent:.3f}" if pd.notna(row.aor_from_parent) else "N/D"
            proxy = f"{row.t_utls_proxy_k:.2f} K" if pd.notna(row.t_utls_proxy_k) else "N/D"
            gate_text.set_text(
                f"Classe oggetto : {int(row.class_id)} · {CLASS_NAMES[int(row.class_id)]}\n"
                f"evento          : {row.event}\n"
                f"AOR primario    : {aor}\n"
                f"storia 15 min   : {'valida' if temporal else 'non disponibile'}\n"
                f"cooling 15 min  : {cool}\n"
                f"crescita 15 min : {growth} ({'valida' if growth_valid else 'non valida'})\n"
                f"contrasto       : {row.contrast_k:.2f} K\n"
                f"T_proxy         : {proxy}\n"
                f"confidenza      : {row.proxy_confidence:.3f}\n"
                f"max D nel crop  : {m.max_depth:.2f}\n"
                f"max BTD 6.2−10.8: {m.max_btd_062_108:+.2f} K\n"
                f"OT prov./conf.  : {int(m.provisional)} / {int(m.confirmed)} pixel"
            )
            title.set_text(
                f"{pd.Timestamp(stamp):%Y-%m-%d %H:%M} UTC · frame {frame + 1}/16 · output full-domain"
            )
            for _ in range(3):
                writer.grab_frame()
    plt.close(fig)


def render_preview(example: dict, track_id: int, frame: int, output: Path) -> None:
    result = example["result"]
    mask = result["tracks"][frame] == track_id
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    axes[0].imshow(example["clip"][frame, 4], cmap="turbo_r", vmin=205, vmax=285)
    axes[0].contour(mask, levels=[0.5], colors="white", linewidths=1.0)
    axes[0].contour(result["plateau_mask"][frame] & mask, levels=[0.5], colors="#00d5ff", linewidths=1.2)
    axes[0].set_title("IR10.8 · bianco: cella · ciano: plateau")
    depth = np.ma.masked_where(~mask, result["thermal_depth"][frame])
    image = axes[1].imshow(depth, cmap="viridis", vmin=0, vmax=4)
    fig.colorbar(image, ax=axes[1], label="D", extend="max")
    axes[1].set_title("D sulla stessa cella")
    labels = np.ma.masked_where(result["classes"][frame] == 0, result["classes"][frame])
    axes[2].imshow(example["clip"][frame, 4], cmap="gray_r", vmin=205, vmax=285)
    axes[2].imshow(labels, cmap=ListedColormap(CLASS_COLORS), norm=BoundaryNorm(np.arange(-0.5, 5.5), 5), alpha=0.72)
    provisional = result["provisional_ot"][frame] & mask
    yy, xx = np.where(provisional)
    axes[2].scatter(xx, yy, marker="x", color="#ff00ff", s=70, linewidth=2)
    axes[2].set_title("Classe 2 + candidato OT provvisorio")
    for axis in axes:
        axis.set(xlabel="ovest → est", ylabel="nord → sud")
    fig.suptitle(f"Caso osservativo · {example['times'][frame]:%Y-%m-%d %H:%M} UTC · track {track_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=145)
    plt.close(fig)


def _fmt(value: float, spec: str = ".2f") -> str:
    return "—" if pd.isna(value) else format(value, spec)


def _track_table(rows: pd.DataFrame) -> str:
    parts = []
    for row in rows.itertuples():
        parts.append(
            "<tr>"
            f"<td>{row.time:%H:%M}</td><td>{int(row.class_id)}</td><td>{html.escape(str(row.event))}</td>"
            f"<td>{row.area_km2:.0f}</td><td>{row.min_k:.2f}</td><td>{row.t_edge_k:.2f}</td>"
            f"<td>{_fmt(row.t_utls_instant_k)}</td><td>{_fmt(row.t_utls_proxy_k)}</td>"
            f"<td>{row.proxy_confidence:.3f}</td><td>{_fmt(row.cooling_k_15min)}</td>"
            f"<td>{_fmt(row.growth_km2_15min, '.0f')}</td><td>{int(row.provisional_ot_pixels)}/{int(row.ot_pixels)}</td>"
            "</tr>"
        )
    return "\n".join(parts)


def build_html(
    args: argparse.Namespace,
    example: dict,
    rows: pd.DataFrame,
    metrics: pd.DataFrame,
    video_path: Path,
    preview_path: Path,
    output_path: Path,
) -> None:
    cfg = RDTTeacherConfig()
    video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    preview_b64 = base64.b64encode(preview_path.read_bytes()).decode("ascii")
    row13 = rows.loc[rows.frame == 13].iloc[0]
    candidate13 = (
        example["result"]["provisional_ot"][13]
        & (example["result"]["tracks"][13] == args.track_id)
    )
    if candidate13.sum() != 1:
        raise ValueError("The documented case requires one provisional OT pixel at frame 13")
    candidate_depth = float(example["result"]["thermal_depth"][13][candidate13][0])
    candidate_btd = float(example["result"]["BTD_062_108"][13][candidate13][0])
    candidate_ir = float(example["clip"][13, 4][candidate13][0])
    page = f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ATBD RDT-CW-inspired · Step 2–4</title>
<style>
:root{{--bg:#f6f7f9;--fg:#172033;--muted:#526176;--panel:#fff;--line:#d8dee8;--code:#eef2f7;--accent:#155eef;--ok:#087f5b;--warn:#b45309}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0b111b;--fg:#e8edf5;--muted:#aab6c6;--panel:#121b29;--line:#2d3a4c;--code:#192536;--accent:#7aa7ff;--ok:#66d9b0;--warn:#fbbf24}}}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.55}}
main{{max-width:1120px;margin:auto;padding:28px 24px 70px}}h1{{font-size:1.85rem;margin:0 0 6px}}h2{{font-size:1.38rem;margin:42px 0 12px;border-top:1px solid var(--line);padding-top:28px}}h3{{font-size:1.08rem;margin:26px 0 8px}}p,li{{max-width:94ch}}a{{color:var(--accent)}}nav{{position:sticky;top:0;z-index:2;background:color-mix(in srgb,var(--bg) 92%,transparent);backdrop-filter:blur(8px);border:1px solid var(--line);padding:10px 12px;margin:22px 0;display:flex;gap:16px;flex-wrap:wrap}}
.meta,.small{{color:var(--muted)}}.small{{font-size:.91rem}}.box{{background:var(--panel);border:1px solid var(--line);padding:16px 18px;margin:14px 0}}.equation{{font-family:"Cambria Math","STIX Two Math",serif;font-size:1.08rem;background:var(--code);border-left:3px solid var(--accent);padding:12px 14px;margin:12px 0;overflow:auto}}code,pre{{background:var(--code)}}code{{padding:.1em .28em}}pre{{padding:13px;overflow:auto;border-left:3px solid var(--accent)}}table{{width:100%;border-collapse:collapse;margin:14px 0 22px;font-size:.93rem}}th,td{{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}}th{{font-weight:650}}.scroll{{overflow:auto}}video,img{{display:block;width:100%;height:auto}}.media{{background:#07111f;border:1px solid var(--line);padding:9px}}.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;align-items:stretch;margin:18px 0}}.flow div{{border:1px solid var(--line);background:var(--panel);padding:12px;text-align:center}}.flow div+div:before{{content:"→";position:absolute;transform:translate(-24px,18px);color:var(--accent);font-weight:bold}}.ok{{color:var(--ok)}}.warn{{color:var(--warn)}}@media(max-width:760px){{.flow{{grid-template-columns:1fr}}.flow div+div:before{{content:"↓";position:static;display:block;transform:none}}}}
</style></head><body><main>
<h1>ATBD · Step 2–4 del teacher RDT-CW-inspired</h1>
<p class="meta">Plateau e proxy UTLS/incudine → profondità D → tracking e fase → override Overshooting Top. Versione algoritmica <code>{cfg.version}</code>.</p>
<nav><a href="#scope">Campo di applicazione</a><a href="#step2">Step 2 · proxy e D</a><a href="#step3">Step 3 · tracking e ciclo</a><a href="#step4">Step 4 · OT e classi</a><a href="#case">Caso reale</a><a href="#qa">Qualità</a><a href="#refs">Riferimenti</a></nav>

<section id="scope"><h2>1. Campo di applicazione e tracciabilità</h2>
<p>Questo volume parte dagli output dello <a href="rdt_step1_random_020_frame01_atbd.html">Step 1</a>: maschera della cella <em>C<sub>i,t</sub></em>, temperatura di bordo <em>T<sub>edge,i,t</sub></em> e identificatore della componente. Il rilevamento, l'associazione temporale e tutte le decisioni sono stati calcolati sul dominio SEVIRI completo di 763 × 1762 pixel. Il riquadro 224 × 224 mostrato qui è estratto soltanto dopo la pubblicazione dei campi full-domain.</p>
<div class="flow"><div>7 BT SEVIRI<br>kelvin</div><div>celle e T<sub>edge</sub><br>full-domain</div><div>plateau, proxy<br>e D</div><div>tracking e<br>fase causale</div><div>OT persistente<br>e classi 0–4</div></div>
<table><tbody><tr><th>Sorgente radiometrica</th><td><code>{html.escape(str(args.zarr_root.resolve()))}</code></td></tr><tr><th>Archivio teacher</th><td><code>{html.escape(str(args.label_root.resolve()))}</code></td></tr><tr><th>Sequenza</th><td>{example['times'][0]:%Y-%m-%d %H:%M}–{example['times'][-1]:%H:%M} UTC, 16 osservazioni a 5 minuti</td></tr><tr><th>Riquadro diagnostico</th><td>origine nativa ({args.origin_y}, {args.origin_x}); centro {example['latitude']:.3f}°N, {example['longitude']:.3f}°E</td></tr><tr><th>Oggetto seguito</th><td>track ID {args.track_id}</td></tr><tr><th>Ordine spettrale</th><td>WV 6.2, WV 7.3, IR 8.7, IR 9.7, IR 10.8, IR 12.0, IR 13.4 µm</td></tr></tbody></table>
<p>Le sei differenze di brightness temperature sono definite in kelvin:</p>
<pre><code>b(p) = [WV062−IR108, WV073−IR108, IR087−IR108,
        IR097−IR108, IR108−IR120, IR134−IR108]</code></pre></section>

<section id="step2"><h2>2. Step 2 · plateau dell'incudine, proxy termica e profondità D</h2>
<h3>2.1 Insieme centrale candidato</h3>
<p>Per la cella <em>C<sub>i,t</sub></em> si calcola il quantile freddo <em>q<sub>0.10</sub></em> di IR 10.8. L'insieme centrale conserva le temperature uguali o superiori al quantile, così il nucleo più freddo viene escluso, e rimuove la fascia entro 1 K dal bordo:</p>
<div class="equation">A<sup>0</sup><sub>i,t</sub> = {{p ∈ C<sub>i,t</sub> : T(p) ≥ q<sub>0.10</sub>(T|C<sub>i,t</sub>) ∧ T(p) &lt; T<sub>edge,i,t</sub> − {cfg.plateau_edge_band_k:.1f} K}}</div>
<p>Su <em>A<sup>0</sup></em>, per ciascuna BTD <em>j</em>, si stimano centro e scala robusti:</p>
<div class="equation">m<sub>j</sub> = median(b<sub>j</sub>), &nbsp; s<sub>j</sub> = max(1.4826·median|b<sub>j</sub>−m<sub>j</sub>|, {cfg.spectral_mad_floor_k:.2f} K)</div>
<div class="equation">d<sub>spec</sub>(p) = max<sub>j=1…6</sub> |b<sub>j</sub>(p)−m<sub>j</sub>| / s<sub>j</sub></div>

<h3>2.2 Filtri di omogeneità e connettività</h3>
<ol><li>Calcolare <em>G(p)=||∇T(p)||</em> sul campo IR 10.8 grezzo.</li><li>Conservare <em>G(p)</em> entro il 60° percentile di <em>A<sup>0</sup></em>.</li><li>Conservare <em>d<sub>spec</sub>(p) ≤ {cfg.spectral_outlier_mad:.1f}</em>.</li><li>Se esiste il frame precedente, conservare la parte meno variabile del plateau: <em>|T<sub>t</sub>−T<sub>t−5min</sub>|</em> entro il 60° percentile dei candidati finiti.</li><li>Etichettare le componenti 8-connesse e scegliere quella di area fisica maggiore.</li><li>Accettare il plateau se contiene almeno {cfg.plateau_min_pixels} pixel e {cfg.plateau_min_area_km2:.0f} km²; altrimenti pubblicare proxy mancante e confidenza zero.</li></ol>

<h3>2.3 Mediana pesata e filtro causale</h3>
<p>Le quantità numeriche <em>G*</em> e <em>δ*</em> sono rispettivamente gradiente espresso in unità di 1 K pixel⁻¹ e variazione a 5 minuti espressa in unità di 1 K. Il peso di affidabilità è limitato fra 0,1 e 1 e poi moltiplicato per l'area geodetica del pixel:</p>
<div class="equation">r(p) = clip(1 / [(1+G*(p))(1+δ*(p))], 0.1, 1) · a(p)</div>
<div class="equation">T<sub>proxy,inst</sub>(i,t) = weighted-median<sub>p∈A</sub>(T(p); r(p))</div>
<div class="equation">T<sub>proxy</sub>(i,t) = median(T<sub>proxy,inst</sub>(i,t−10), T<sub>proxy,inst</sub>(i,t−5), T<sub>proxy,inst</sub>(i,t))</div>
<p>All'inizio della traccia la mediana usa soltanto i valori passati disponibili. La versione v0 applica correzione spettrale additiva pari a 0 K: i sei BTD determinano appartenenza e confidenza del plateau, mentre il valore radiometrico è la mediana pesata IR 10.8. Il risultato è un <strong>riferimento radiometrico SEVIRI dell'incudine/UTLS</strong>, definito dall'algoritmo qui riportato.</p>

<h3>2.4 Confidenza</h3>
<p>La confidenza <em>c∈[0,1]</em> è un indice euristico deterministico. I fattori sono moltiplicativi:</p>
<div class="equation">c = c<sub>area</sub> c<sub>N</sub> c<sub>disp</sub> c<sub>spec</sub> c<sub>jump</sub> c<sub>support</sub> c<sub>den</sub></div>
<table><thead><tr><th>Fattore</th><th>Definizione eseguibile</th></tr></thead><tbody>
<tr><td>c<sub>area</sub></td><td>min(1, area(A)/(2·{cfg.plateau_min_area_km2:.0f} km²))</td></tr><tr><td>c<sub>N</sub></td><td>min(1, N(A)/(2·{cfg.plateau_min_pixels}))</td></tr><tr><td>c<sub>disp</sub></td><td>1/(1 + MAD robusta IR10.8 / {cfg.plateau_dispersion_scale_k:.1f} K)</td></tr><tr><td>c<sub>spec</sub></td><td>1/(1 + dispersione spettrale robusta)</td></tr><tr><td>c<sub>jump</sub></td><td>1/(1 + |T<sub>inst,t</sub>−T<sub>inst,t−5</sub>|/{cfg.proxy_jump_scale_k:.1f} K)</td></tr><tr><td>c<sub>support</sub></td><td>confidenza del bordo restituita dallo Step 1</td></tr><tr><td>c<sub>den</sub></td><td>min(1, max(0,T<sub>edge</sub>−T<sub>proxy</sub>)/(2·{cfg.depth_min_denominator_k:.1f} K))</td></tr></tbody></table>

<h3>2.5 Profondità termica relativa</h3>
<div class="equation">D(p) = [T<sub>edge,i,t</sub> − T(p)] / [T<sub>edge,i,t</sub> − T<sub>proxy,i,t</sub>], &nbsp; p∈C<sub>i,t</sub></div>
<p>Il denominatore deve essere almeno {cfg.depth_min_denominator_k:.1f} K. In caso contrario <em>D</em> è mancante e il ramo OT è disabilitato. <em>D</em> è adimensionale e non possiede un limite superiore matematico:</p>
<table><thead><tr><th>Intervallo</th><th>Significato radiometrico</th></tr></thead><tbody><tr><td>D = 0</td><td>T coincide con T<sub>edge</sub>.</td></tr><tr><td>0 &lt; D &lt; 1</td><td>Pixel più freddo del bordo e più caldo del riferimento del plateau.</td></tr><tr><td>D = 1</td><td>T coincide con T<sub>proxy</sub>.</td></tr><tr><td>1 &lt; D ≤ 2</td><td>Escursione fredda oltre il plateau fino a una seconda ampiezza bordo–proxy.</td></tr><tr><td>D &gt; 2</td><td>Escursione molto fredda; il valore può crescere rapidamente quando il denominatore è vicino al minimo ammesso. Richiede i gate OT e l'ispezione della confidenza.</td></tr><tr><td>D &lt; 0</td><td>Possibile per pixel più caldi di T<sub>edge</sub> in presenza di differenze fra topologia smussata e BT grezza; è una diagnostica di coerenza.</td></tr></tbody></table></section>

<section id="step3"><h2>3. Step 3 · associazione, traiettoria e fase del ciclo di vita</h2>
<h3>3.1 Associazione fra osservazioni</h3><p>La maschera precedente è traslata del moto arrotondato stimato lungo la traiettoria primaria. Per ogni coppia precedente–corrente si usa area fisica:</p>
<div class="equation">AOR(P,C) = area(P<sub>adv</sub> ∩ C) / min[area(P<sub>adv</sub>), area(C)]</div>
<p>Si crea un arco se <em>AOR ≥ {cfg.min_aor:.2f}</em>. Tutti gli archi sopra soglia sono conservati. La traiettoria primaria è deterministica: relazione uno-a-uno per una continuazione; genitore di area maggiore in un merge semplice; figlio di area maggiore in uno split semplice. Le famiglie molti-a-molti ricevono evento <code>complex</code> e non ereditano una primaria.</p>
<h3>3.2 Diagnostiche a 15 minuti</h3><p>Per l'oggetto al tempo <em>t</em>, la storia è valutabile dopo tre intervalli RSS reali e in assenza di contatto col bordo o con osservazioni invalide:</p>
<div class="equation">cooling<sub>15</sub>(t) = T<sub>min</sub>(t) − T<sub>min</sub>(t−15 min)</div>
<div class="equation">growth<sub>15</sub>(t) = area(t) − area(t−15 min)</div>
<p><em>growth<sub>15</sub></em> è valido soltanto lungo tre associazioni ordinarie consecutive. Split e merge invalidano la crescita fino alla ricostruzione di tre intervalli ordinari. Nessuna differenza a 5 minuti viene moltiplicata per tre.</p>
<h3>3.3 Macchina a stati</h3>
<table><thead><tr><th>Uscita</th><th>Condizione</th></tr></thead><tbody><tr><td>Classe 1</td><td>Cella rilevata che non soddisfa l'ingresso convettivo e non eredita uno stato convettivo.</td></tr><tr><td>Classe 2 · sviluppo</td><td>storia e crescita valide; T<sub>mean</sub>−T<sub>min</sub> ≥ {cfg.contrast_k:.1f} K; cooling<sub>15</sub> ≤ {cfg.cooling_15min_k:.1f} K; growth<sub>15</sub> &gt; 0.</td></tr><tr><td>Classe 3 · maturità/decadimento</td><td>la primaria precedente era convettiva e la cella corrente e quella precedente non sono censurate dal bordo; si usa quando la regola di sviluppo corrente non è soddisfatta.</td></tr></tbody></table>
<p>I primi tre frame hanno diagnostiche temporali mancanti. Una lacuna diversa da 5 minuti azzera la storia causale. I pixel della cella mantengono un flag di incertezza quando la fase non è valutabile.</p></section>

<section id="step4"><h2>4. Step 4 · candidato OT, persistenza e mappa a cinque classi</h2>
<h3>4.1 Gate simultanei del candidato</h3><p>Un pixel entra nella maschera candidata soltanto dentro una cella di Classe 2 o 3 e quando tutti i criteri seguenti sono veri:</p>
<ol><li>proxy valido e confidenza ≥ {cfg.min_proxy_confidence:.2f};</li><li><em>D &gt; 1</em>;</li><li><em>T<sub>proxy</sub>−T(p) ≥ {cfg.ot_anvil_contrast_k:.1f} K</em>;</li><li>nel vicinato {(2*cfg.ot_local_radius_pixels+1)} × {(2*cfg.ot_local_radius_pixels+1)}, <em>T(p) ≤ local_min+{cfg.ot_local_contrast_k:.1f} K</em> e <em>local_mean−T(p) ≥ {cfg.ot_local_contrast_k:.1f} K</em>;</li><li><em>WV6.2−IR10.8 ≥ {cfg.ot_btd_062_108_k:.1f} K</em>;</li><li>distanza robusta a sei BTD ≤ {cfg.ot_spectral_outlier_mad:.1f};</li><li>componente 8-connessa con area fra {cfg.ot_min_area_km2:.0f} e {cfg.ot_max_area_km2:.0f} km² e rapporto fra assi ≤ {cfg.ot_max_axis_ratio:.1f}.</li></ol>
<h3>4.2 Persistenza</h3><p>La maschera candidata del frame precedente viene traslata con lo spostamento inter-frame della cella e dilatata mediante massimo 3 × 3. Il contatore cresce soltanto sui candidati correnti. Con <em>N<sub>persist</sub>={cfg.ot_persistence_frames}</em>, il primo riscontro è <code>provisional_ot</code>; il secondo riscontro consecutivo e associato è <code>ot_mask</code> e sovrascrive la classe oggetto con Classe 4 sui soli pixel confermati.</p>
<pre><code>0 sfondo
1 nube/cella rilevata non convettiva
2 cella convettiva in sviluppo
3 cella convettiva matura o in decadimento
4 pixel OT candidato persistente dentro una cella 2/3</code></pre></section>

<section id="case"><h2>5. Caso osservativo completo</h2>
<p>La traccia {args.track_id} passa dalla Classe 1 alla Classe 2 alle 14:30 UTC e alla Classe 3 alle 14:40 UTC. Alle 14:35 UTC un pixel supera simultaneamente i gate OT: <em>D={candidate_depth:.2f}</em>, IR10.8={candidate_ir:.2f} K, WV6.2−IR10.8={candidate_btd:.2f} K, confidenza proxy {row13.proxy_confidence:.3f}, T<sub>edge</sub>={row13.t_edge_k:.2f} K e T<sub>proxy</sub>={row13.t_utls_proxy_k:.2f} K. Il frame successivo non rinnova il candidato; l'algoritmo conserva quindi zero pixel OT confermati. Questo caso mostra sia la generazione del candidato sia l'effetto selettivo della persistenza.</p>
<div class="media"><video controls playsinline preload="metadata" aria-label="Animazione ATBD degli step 2, 3 e 4"><source src="data:video/mp4;base64,{video_b64}" type="video/mp4"></video></div>
<p class="small">Contorno bianco: track {args.track_id}; ciano: plateau selezionato; magenta: candidato OT provvisorio. La scala D è fissa 0–4; valori maggiori usano il colore terminale.</p>
<div class="box"><img src="data:image/png;base64,{preview_b64}" alt="Frame 14:35 con IR10.8, profondità D e candidato OT provvisorio"></div>
<h3>Tabella causale della traccia</h3><div class="scroll"><table><thead><tr><th>UTC</th><th>Classe</th><th>Evento</th><th>Area km²</th><th>Tmin K</th><th>Tedge K</th><th>Tinst K</th><th>Tproxy K</th><th>Conf.</th><th>Cooling15 K</th><th>Growth15 km²</th><th>OT prov./conf.</th></tr></thead><tbody>{_track_table(rows)}</tbody></table></div></section>

<section id="qa"><h2>6. Output, flag di qualità e criteri di accettazione</h2>
<pre><code>per pixel e timestamp:
  plateau_mask, t_utls_instant_k, t_utls_proxy_k, proxy_confidence
  thermal_depth, track_id, semantic_label
  provisional_ot, ot_mask, uncertainty, phase_uncertainty

per oggetto e timestamp:
  area_km2, t_min_k, t_mean_k, t_edge_k
  plateau_area_km2, plateau_dispersion_k, proxy_reason
  AOR, evento di lineage, cooling_15min, area_change_15min
  lifecycle_state, provisional_ot_pixels, ot_pixels</code></pre>
<ul><li>Il plateau deve essere riproducibile a parità di input, configurazione e storia.</li><li>Ogni valore temporale usa soltanto il timestamp corrente e quelli precedenti.</li><li>Proxy mancante, denominatore &lt; {cfg.depth_min_denominator_k:.1f} K, confidenza &lt; {cfg.min_proxy_confidence:.2f}, bordo, gap e split/merge devono restare distinguibili nei campi pubblicati.</li><li>La traslazione di tutte le BT di una costante lascia invariati topologia relativa e D entro la tolleranza numerica.</li><li>Una Classe 4 deve essere sottoinsieme di una cella convettiva e deve possedere due riscontri associati consecutivi.</li><li>La validazione scientifica deve confrontare bordi, traiettorie e OT con casi indipendenti; le soglie qui elencate sono configurazione v0.</li></ul>
<p class="warn"><strong>Interpretazione del probing:</strong> la decodifica delle pseudo-label mostra che il teacher deterministico è accessibile dalla rappresentazione congelata. Il protocollo completo richiede encoder casuale, checkpoint RGB adattato, backbone SEVIRI, ablazioni temporali/spettrali e teacher legacy a due canali per separare memoria spettrale, dinamica e termodinamica.</p></section>

<section id="refs"><h2>7. Fondamento scientifico e riferimenti</h2>
<p>La segmentazione adattiva e l'uso di evoluzione, raffreddamento ed espansione seguono il quadro concettuale del <a href="https://www.nwcsaf.org/AemetWebContents/ScientificDocumentation/Documentation/GEO/v2016/NWC-CDOP2-GEO-MFT-SCI-ATBD-Convection_v1.1.pdf">NWC SAF ATBD for Convection Products</a>. Le osservabili multispettrali SEVIRI e le rispettive bande sono documentate in <a href="https://www.esa.int/esapub/bulletin/bullet111/chapter3_bul111.pdf">ESA, The MSG System</a>. L'impiego di differenze di BT, firme WV–IR e anomalie fredde dell'incudine per caratterizzare convezione profonda e overshooting top è motivato da <a href="https://doi.org/10.1016/j.atmosres.2012.09.001">Mikuš e Strelec Mahović (2013)</a>, <a href="https://doi.org/10.1175/2009JAMC2286.1">Bedka et al. (2010)</a> e <a href="https://doi.org/10.1016/j.atmosres.2010.10.001">Bedka (2011)</a>.</p>
<p>L'elemento progettuale specifico di questa versione è la catena <em>plateau robusto multispettrale → mediana pesata IR10.8 → filtro causale → D relativo</em>. I riferimenti sostengono le osservabili e i vincoli fisici; coefficienti, quantili, confidenza e formula di D sono definiti qui come algoritmo v0 da validare.</p>
<p class="small">Metadati macchina: <code>{html.escape(json.dumps({'teacher_version': cfg.version, 'config_hash': cfg.config_hash, 'track_id': args.track_id, 'start_time': args.start_time, 'origin_y': args.origin_y, 'origin_x': args.origin_x, 'full_domain_before_crop': True}))}</code></p></section>
</main></body></html>"""
    output_path.write_text(page, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not (args.label_root / "protocol.json").exists():
        raise FileNotFoundError(
            "Missing full-domain teacher archive. Run build_rdt_pseudolabels.py "
            "for the selected 16-frame interval before rendering the ATBD."
        )
    example = load_domain_crop(
        args.zarr_root,
        args.label_root,
        args.start_time,
        args.origin_y,
        args.origin_x,
    )
    rows = _track_rows(example, args.track_id)
    metrics = _metrics(example, rows, args.track_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rdt_steps2_4_track{args.track_id}_atbd"
    video_path = args.output_dir / f"{stem}.mp4"
    preview_path = args.output_dir / f"{stem}_preview.png"
    html_path = args.output_dir / f"{stem}.html"
    summary_path = args.output_dir / f"{stem}_summary.json"
    render_video(example, rows, metrics, args.track_id, video_path, args.fps)
    render_preview(example, args.track_id, 13, preview_path)
    build_html(args, example, rows, metrics, video_path, preview_path, html_path)
    summary = {
        "html": str(html_path.resolve()),
        "video": str(video_path.resolve()),
        "preview": str(preview_path.resolve()),
        "label_root": str(args.label_root.resolve()),
        "source_root": str(args.zarr_root.resolve()),
        "full_domain_before_crop": True,
        "track_id": args.track_id,
        "times": [stamp.isoformat() for stamp in example["times"]],
        "crop_origin_native": [args.origin_y, args.origin_x],
        "crop_center_lon_lat": [example["longitude"], example["latitude"]],
        "classes": rows[["frame", "class_id"]].to_dict("records"),
        "provisional_ot_total": int(metrics.provisional.sum()),
        "confirmed_ot_total": int(metrics.confirmed.sum()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
