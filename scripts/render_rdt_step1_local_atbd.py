#!/usr/bin/env python3
"""Render a local SEVIRI case study and an HTML ATBD for RDT Step 1.

The output is a self-contained HTML document with an embedded MP4.  The case
study uses one real IR 10.8 um frame from the validated north-up SEVIRI Zarr
loader and visualizes regional minima, thermal flooding, first-merge saddles,
and the resulting cell-specific ``T_edge`` values.
"""

from __future__ import annotations

import argparse
import base64
from datetime import timedelta
import html
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from emma_gpm.seviri_clips import SeviriZarrClipReader  # noqa: E402
from render_rdt_step1_animation import (  # noqa: E402
    _cell_saddle_temperatures,
    _select_regional_minima,
    _watershed_labels,
)


IR108_INDEX = 4
COLORS = (
    "#ef4444",
    "#f59e0b",
    "#22c55e",
    "#06b6d4",
    "#3b82f6",
    "#a855f7",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path(
            os.environ.get(
                "SEVIRI_ZARR_ROOT",
                r"E:\Datasets\MSG-SEVIRI",
            )
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "catalogs" / "seviri_pretraining_random_100.csv",
    )
    parser.add_argument("--clip-id", default="random_020")
    parser.add_argument("--frame-index", type=int, default=1)
    parser.add_argument("--sigma-px", type=float, default=1.0)
    parser.add_argument("--max-seeds", type=int, default=6)
    parser.add_argument("--minimum-distance-px", type=int, default=30)
    parser.add_argument("--filter-size-px", type=int, default=11)
    parser.add_argument("--minimum-prominence-k", type=float, default=6.0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures" / "videomae_pretraining",
    )
    return parser.parse_args()


def cell_mask(
    temperature: np.ndarray,
    labels: np.ndarray,
    saddles: np.ndarray,
    level_k: float,
) -> np.ndarray:
    output = np.zeros(labels.shape, dtype=np.int16)
    for label in range(1, len(saddles)):
        upper = min(float(level_k), float(saddles[label]))
        output[(labels == label) & (temperature <= upper)] = label
    return output


def boundary(mask: np.ndarray) -> np.ndarray:
    inside = mask > 0
    return inside & ~ndimage.binary_erosion(inside, structure=np.ones((3, 3)))


def render_video(
    *,
    raw: np.ndarray,
    smoothed: np.ndarray,
    labels: np.ndarray,
    seeds: list[tuple[int, int]],
    saddles: np.ndarray,
    timestamp: pd.Timestamp,
    output: Path,
    fps: float,
) -> None:
    seed_temperatures = np.asarray([smoothed[position] for position in seeds])
    flood_start = float(np.min(seed_temperatures))
    flood_end = float(np.max(saddles[1:]))
    levels = np.linspace(flood_start, flood_end, 42)
    palette = ListedColormap(COLORS[: len(seeds)])

    fig, (map_axis, level_axis) = plt.subplots(
        1,
        2,
        figsize=(10.8, 6.2),
        gridspec_kw={"width_ratios": (1.45, 0.75)},
        facecolor="#07111f",
    )
    map_axis.set_facecolor("#07111f")
    base = map_axis.imshow(
        raw,
        cmap="turbo_r",
        vmin=205.0,
        vmax=285.0,
        origin="upper",
        interpolation="nearest",
    )
    overlay = map_axis.imshow(
        np.ma.masked_all(raw.shape),
        cmap=palette,
        vmin=1,
        vmax=max(len(seeds), 2),
        alpha=0.48,
        origin="upper",
        interpolation="nearest",
    )
    edge_artist = map_axis.imshow(
        np.ma.masked_all(raw.shape),
        cmap=ListedColormap(["white"]),
        vmin=0,
        vmax=1,
        alpha=0.95,
        origin="upper",
        interpolation="nearest",
    )
    seed_artist = map_axis.scatter(
        [],
        [],
        marker="x",
        s=65,
        linewidths=2.0,
        color="white",
        zorder=5,
    )
    map_axis.set_title("IR 10.8 µm osservato", color="white", pad=8)
    map_axis.set_xlabel("colonna · ovest → est", color="#cbd5e1")
    map_axis.set_ylabel("riga · nord → sud", color="#cbd5e1")
    map_axis.tick_params(colors="#94a3b8")
    colorbar = fig.colorbar(base, ax=map_axis, orientation="horizontal", pad=0.11)
    colorbar.set_label("Brightness temperature [K]", color="white")
    colorbar.ax.tick_params(colors="white")

    level_axis.set_facecolor("#07111f")
    level_axis.set_title("Intervallo minimo–saddle", color="white", pad=8)
    level_axis.set_xlim(0.5, len(seeds) + 0.5)
    lower = float(np.floor(seed_temperatures.min() - 3.0))
    upper = float(np.ceil(saddles[1:].max() + 3.0))
    level_axis.set_ylim(upper, lower)
    level_axis.set_xticks(np.arange(1, len(seeds) + 1))
    level_axis.set_xticklabels([f"C{i}" for i in range(1, len(seeds) + 1)])
    level_axis.set_ylabel("Brightness temperature [K]", color="#cbd5e1")
    level_axis.tick_params(colors="#94a3b8")
    level_axis.grid(axis="y", color="#334155", linewidth=0.6, alpha=0.7)
    for spine in level_axis.spines.values():
        spine.set_color("#475569")
    for index, (minimum, saddle) in enumerate(
        zip(seed_temperatures, saddles[1:], strict=True), start=1
    ):
        color = COLORS[index - 1]
        level_axis.plot([index, index], [minimum, saddle], color=color, linewidth=4)
        level_axis.scatter(index, minimum, s=42, color=color, marker="v", zorder=3)
        level_axis.scatter(index, saddle, s=42, color=color, marker="s", zorder=3)
        level_axis.text(
            index,
            minimum - 0.8,
            f"{minimum:.1f}",
            color="white",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        level_axis.text(
            index,
            saddle + 0.8,
            f"{saddle:.1f}",
            color="white",
            ha="center",
            va="top",
            fontsize=8,
        )
    current_level = level_axis.axhline(
        flood_start, color="white", linewidth=1.5, linestyle="--"
    )
    legend = level_axis.text(
        0.5,
        0.02,
        "▽ T_min    ■ T_edge",
        transform=level_axis.transAxes,
        color="#cbd5e1",
        ha="center",
        va="bottom",
    )
    legend.set_in_layout(False)

    title = fig.suptitle("", color="white", fontsize=16, y=0.985)
    subtitle = fig.text(
        0.5,
        0.935,
        "",
        color="#cbd5e1",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.86, wspace=0.24)
    seed_offsets = np.asarray([(column, row) for row, column in seeds])
    empty = np.ma.masked_all(raw.shape)
    final_cells = cell_mask(smoothed, labels, saddles, flood_end)
    final_boundary = boundary(final_cells)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=1800,
        metadata={
            "title": "RDT-CW-inspired Step 1 local SEVIRI case",
            "comment": "Regional minima, thermal flooding, saddle and T_edge",
        },
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(output), dpi=90):
        for _ in range(10):
            title.set_text(f"SEVIRI · {timestamp:%Y-%m-%d %H:%M} UTC")
            subtitle.set_text("Campo IR 10.8 µm north-up, west-left")
            overlay.set_data(empty)
            edge_artist.set_data(empty)
            seed_artist.set_offsets(np.empty((0, 2)))
            current_level.set_ydata([flood_start, flood_start])
            writer.grab_frame()
        for _ in range(10):
            seed_artist.set_offsets(seed_offsets)
            subtitle.set_text("Marker: minimi regionali termicamente persistenti")
            writer.grab_frame()
        for level in levels:
            cells = cell_mask(smoothed, labels, saddles, float(level))
            overlay.set_data(np.ma.masked_where(cells == 0, cells))
            current_level.set_ydata([level, level])
            subtitle.set_text(
                f"Sublevel set Tₛ ≤ τ · τ = {level:.1f} K · arresto al primo saddle"
            )
            writer.grab_frame()
        edge_artist.set_data(
            np.ma.masked_where(~final_boundary, final_boundary.astype(float))
        )
        current_level.set_ydata([flood_end, flood_end])
        for _ in range(18):
            subtitle.set_text(
                "T_edge: livello del primo merge, assegnato separatamente a ogni cella"
            )
            writer.grab_frame()
    plt.close(fig)


def build_html(
    *,
    args: argparse.Namespace,
    video_path: Path,
    output_path: Path,
    timestamp: pd.Timestamp,
    seed_temperatures: np.ndarray,
    saddles: np.ndarray,
    prominences: np.ndarray,
    shape: tuple[int, int],
) -> None:
    video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    rows = "\n".join(
        "<tr>"
        f"<td>C{index}</td>"
        f"<td>{minimum:.2f}</td>"
        f"<td>{edge:.2f}</td>"
        f"<td>{prominence:.2f}</td>"
        "</tr>"
        for index, (minimum, edge, prominence) in enumerate(
            zip(seed_temperatures, saddles[1:], prominences, strict=True), start=1
        )
    )
    source_manifest = html.escape(str(args.manifest.resolve()))
    page = f"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ATBD · Step 1 RDT-CW-inspired fino a T_edge</title>
<style>
:root {{ color-scheme: light dark; --bg:#f7f8fa; --fg:#172033; --muted:#526176; --panel:#ffffff; --line:#d7dde6; --code:#eef2f7; --accent:#155eef; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0b111b; --fg:#e8edf5; --muted:#aab6c6; --panel:#121b29; --line:#2c3a4d; --code:#192536; --accent:#77a7ff; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.55; }}
main {{ max-width:1040px; margin:0 auto; padding:28px 22px 60px; }}
h1 {{ font-size:1.75rem; font-weight:600; margin:0 0 6px; }}
h2 {{ font-size:1.25rem; font-weight:600; margin:32px 0 10px; }}
h3 {{ font-size:1.05rem; font-weight:600; margin:22px 0 8px; }}
p,li {{ max-width:88ch; }}
.meta {{ color:var(--muted); margin:0 0 24px; }}
.video {{ background:#07111f; border:1px solid var(--line); padding:10px; }}
video {{ display:block; width:100%; height:auto; }}
table {{ width:100%; border-collapse:collapse; margin:14px 0 20px; }}
th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ font-weight:600; }}
code,pre {{ background:var(--code); color:var(--fg); }}
code {{ padding:.1em .3em; }}
pre {{ padding:13px; overflow:auto; border-left:3px solid var(--accent); }}
.equation {{ font-family:"Cambria Math","STIX Two Math",serif; font-size:1.08rem; padding:10px 12px; background:var(--code); border-left:3px solid var(--accent); overflow:auto; }}
.definition {{ border-left:3px solid var(--line); padding-left:14px; }}
.small {{ color:var(--muted); font-size:.92rem; }}
a {{ color:var(--accent); }}
</style>
</head>
<body><main>
<h1>ATBD · Step 1: rilevamento termico adattivo fino a T<sub>edge</sub></h1>
<p class="meta">Caso SEVIRI reale · IR 10.8 µm · {timestamp:%Y-%m-%d %H:%M} UTC · dominio locale {shape[1]} × {shape[0]} pixel</p>

<h2>1. Scopo e output</h2>
<p>Lo step trasforma un campo bidimensionale di brightness temperature IR 10.8 µm in un insieme di celle termiche individuali. A ogni cella <em>i</em> associa un marker <em>m<sub>i</sub></em>, un identificatore, una maschera spaziale e una temperatura di bordo <em>T<sub>edge,i</sub></em>. La temperatura di bordo è il livello termico del primo evento di fusione della cella con il bacino di un altro minimo regionale.</p>

<h2>2. Quantità e definizioni</h2>
<table>
<thead><tr><th>Simbolo</th><th>Definizione operativa</th><th>Unità/dominio</th></tr></thead>
<tbody>
<tr><td>Ω</td><td>Griglia finita dei pixel del sotto-dominio, orientata con nord in alto e ovest a sinistra.</td><td>insieme di pixel</td></tr>
<tr><td>T(p)</td><td>Brightness temperature IR 10.8 µm osservata nel pixel p.</td><td>kelvin</td></tr>
<tr><td>T<sub>s</sub>(p)</td><td>Campo impiegato per la topologia: convoluzione di T con un kernel gaussiano di deviazione standard σ.</td><td>kelvin</td></tr>
<tr><td>N<sub>8</sub>(p)</td><td>Insieme dei pixel che condividono con p un lato oppure un vertice.</td><td>massimo 8 pixel</td></tr>
<tr><td>M<sub>i</sub></td><td>Componente connessa di minimo regionale: plateau per il quale ogni pixel adiacente esterno possiede T<sub>s</sub> maggiore del valore del plateau.</td><td>insieme connesso</td></tr>
<tr><td>ℳ</td><td>Insieme finito dei marker conservati dopo il filtro locale, il filtro del 2,5° percentile freddo, la distanza minima e il limite numerico sul numero di marker.</td><td>insieme di marker</td></tr>
<tr><td>L(τ)</td><td>Sublevel set del campo termico smussato.</td><td>{{p ∈ Ω : T<sub>s</sub>(p) ≤ τ}}</td></tr>
<tr><td>C<sub>i</sub>(τ)</td><td>Componente 8-connessa di L(τ) che contiene il marker m<sub>i</sub>.</td><td>insieme di pixel</td></tr>
<tr><td>s<sub>ij</sub></td><td>Minimo valore di τ per il quale C<sub>i</sub>(τ) e C<sub>j</sub>(τ) appartengono alla stessa componente connessa.</td><td>kelvin</td></tr>
<tr><td>T<sub>edge,i</sub>(ℳ)</td><td>Primo livello di fusione della cella i rispetto ai marker conservati: min<sub>j≠i</sub> s<sub>ij</sub>. La watershed conserva l'identità dei bacini al livello di contatto.</td><td>kelvin</td></tr>
<tr><td>P<sub>i</sub></td><td>Prominenza termica del minimo: T<sub>edge,i</sub> − T<sub>min,i</sub>.</td><td>kelvin</td></tr>
</tbody></table>

<div class="equation">L(τ) = {{ p ∈ Ω : T<sub>s</sub>(p) ≤ τ }}</div>
<div class="equation">s<sub>ij</sub> = inf {{ τ : C<sub>i</sub>(τ) e C<sub>j</sub>(τ) risultano connesse }}</div>
<div class="equation">T<sub>edge,i</sub>(ℳ) = min<sub>j≠i, m<sub>j</sub>∈ℳ</sub> s<sub>ij</sub></div>

<h2>3. Sequenza algoritmica</h2>
<ol>
<li>Acquisire T in kelvin dal loader SEVIRI, conservando ordine geografico e valori radiometrici originali.</li>
<li>Calcolare T<sub>s</sub> esclusivamente per le operazioni topologiche. Le statistiche fisiche successive continuano a usare T.</li>
<li>Individuare i minimi regionali di T<sub>s</sub> con un filtro minimo locale. Nell'esempio si conservano candidati entro il 2,5° percentile freddo del dominio, ordinati per temperatura, separati dalla distanza minima e limitati al numero massimo configurato. Il risultato è ℳ.</li>
<li>Costruire la watershed marker-controlled di T<sub>s</sub>. Ogni pixel riceve l'identità del marker raggiungibile attraverso il cammino con minore livello massimo.</li>
<li>Aumentare τ. La componente C<sub>i</sub>(τ) cresce quando nuovi pixel entrano in L(τ).</li>
<li>Per ogni coppia di bacini adiacenti, misurare il più basso livello di contatto s<sub>ij</sub>.</li>
<li>Assegnare T<sub>edge,i</sub> al primo contatto della cella i e congelarne la crescita a quel livello.</li>
<li>Calcolare P<sub>i</sub>. Il valore candidato iniziale P<sub>i</sub> ≥ {args.minimum_prominence_k:.1f} K identifica i minimi termicamente persistenti da sottoporre agli step successivi.</li>
</ol>

<h2>4. Esempio osservativo</h2>
<div class="video"><video controls playsinline preload="metadata" aria-label="Animazione del flooding termico e della determinazione di T edge"><source src="data:video/mp4;base64,{video_b64}" type="video/mp4"></video></div>
<p class="small">La mappa usa T osservata come sfondo. Marker, watershed e livelli di fusione sono calcolati su T<sub>s</sub>. Nel grafico laterale il triangolo indica T<sub>min</sub>, il quadrato indica T<sub>edge</sub> e la linea tratteggiata indica il livello τ corrente.</p>

<h3>Valori del caso</h3>
<table>
<thead><tr><th>Cella</th><th>T<sub>min</sub> [K]</th><th>T<sub>edge</sub> [K]</th><th>P [K]</th></tr></thead>
<tbody>{rows}</tbody>
</table>

<h2>5. Parametri dell'esempio</h2>
<table>
<tbody>
<tr><th>Clip</th><td>{html.escape(args.clip_id)}</td></tr>
<tr><th>Frame</th><td>{args.frame_index + 1}/16</td></tr>
<tr><th>Cadenza</th><td>5 minuti</td></tr>
<tr><th>σ gaussiana</th><td>{args.sigma_px:.2f} pixel</td></tr>
<tr><th>Connettività</th><td>8</td></tr>
<tr><th>Filtro dei minimi</th><td>finestra {args.filter_size_px} × {args.filter_size_px} pixel</td></tr>
<tr><th>Coda fredda dei candidati</th><td>T<sub>s</sub> ≤ quantile 0,025 del sotto-dominio</td></tr>
<tr><th>Distanza fra marker</th><td>{args.minimum_distance_px} pixel</td></tr>
<tr><th>Numero massimo di marker</th><td>{args.max_seeds}</td></tr>
<tr><th>Manifest sorgente</th><td><code>{source_manifest}</code></td></tr>
</tbody></table>

<h2>6. Output numerici richiesti</h2>
<pre><code>cell_id[y,x]       : int32, 0 fuori dalle celle
marker_id          : int32
marker_row/column  : int32
t_min_k            : float32
t_edge_k           : float32
prominence_k       : float32
boundary_mask[y,x] : bool
quality_flags      : bit mask</code></pre>
<p>I flag di qualità devono registrare almeno: marker sul bordo del dominio, assenza di un saddle interno, contatto con dati mancanti, area insufficiente e sensibilità del risultato a σ.</p>

<h2>7. Parametri da sottoporre a validazione</h2>
<p>La struttura topologica — sublevel set, primo merge e T<sub>edge</sub> specifica della cella — costituisce la definizione dell'algoritmo. Il valore T<sub>edge,i</sub>(ℳ) dipende dall'insieme dei marker ℳ: un marker vicino rimosso può spostare il primo merge verso un livello più caldo. σ, percentile dei candidati, finestra dei minimi, distanza minima, limite sul numero di marker, prominenza e area minima costituiscono quindi la configurazione numerica. La validazione deve misurarne la stabilità rispetto alla risoluzione SEVIRI, alla posizione nel disco, alla stagione e alla morfologia multicella.</p>

<h2>8. Riferimento metodologico</h2>
<p>Il principio della soglia adattiva specifica dell'oggetto e del contorno più caldo che mantiene separate torri vicine è documentato nell'<a href="https://www.nwcsaf.org/AemetWebContents/ScientificDocumentation/Documentation/GEO/v2016/NWC-CDOP2-GEO-MFT-SCI-ATBD-Convection_v1.1.pdf">ATBD NWC SAF dei prodotti di convezione</a> e nella <a href="https://www.nwcsaf.org/rdt_description">descrizione ufficiale RDT-CW</a>. La formulazione mediante sublevel set, saddle e watershed rende tale principio esplicito e riproducibile sulla griglia SEVIRI.</p>

<p class="small">Metadati macchina: <code>{html.escape(json.dumps({'clip_id': args.clip_id, 'frame_index': args.frame_index, 'timestamp': timestamp.isoformat(), 'sigma_px': args.sigma_px, 'connectivity': 8}))}</code></p>
</main></body></html>"""
    output_path.write_text(page, encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    selected = manifest.loc[manifest.clip_id == args.clip_id]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one manifest row for {args.clip_id!r}; found {len(selected)}."
        )
    row = selected.iloc[0]
    start_time = pd.Timestamp(row.start_time)
    if start_time.tzinfo is not None:
        start_time = start_time.tz_convert("UTC").tz_localize(None)
    timestamp = start_time + timedelta(minutes=5 * args.frame_index)

    with SeviriZarrClipReader(args.zarr_root) as reader:
        clip = reader.load(
            start_time,
            origin_y=int(row.origin_y),
            origin_x=int(row.origin_x),
        )
    raw = np.asarray(clip[args.frame_index, IR108_INDEX], dtype=np.float32)
    smoothed = ndimage.gaussian_filter(raw, sigma=args.sigma_px)
    valid = np.isfinite(smoothed)
    seeds = _select_regional_minima(
        smoothed,
        valid,
        max_seeds=args.max_seeds,
        filter_size=args.filter_size_px,
        minimum_distance=args.minimum_distance_px,
    )
    labels = _watershed_labels(smoothed, valid, seeds)
    saddles = _cell_saddle_temperatures(smoothed, labels, seeds)
    seed_temperatures = np.asarray([smoothed[position] for position in seeds])
    prominences = saddles[1:] - seed_temperatures
    if np.count_nonzero(prominences >= args.minimum_prominence_k) < 2:
        raise RuntimeError("The selected case has fewer than two persistent minima.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rdt_step1_{args.clip_id}_frame{args.frame_index:02d}"
    video_path = args.output_dir / f"{stem}.mp4"
    html_path = args.output_dir / f"{stem}_atbd.html"
    summary_path = args.output_dir / f"{stem}_summary.json"

    render_video(
        raw=raw,
        smoothed=smoothed,
        labels=labels,
        seeds=seeds,
        saddles=saddles,
        timestamp=timestamp,
        output=video_path,
        fps=args.fps,
    )
    build_html(
        args=args,
        video_path=video_path,
        output_path=html_path,
        timestamp=timestamp,
        seed_temperatures=seed_temperatures,
        saddles=saddles,
        prominences=prominences,
        shape=raw.shape,
    )
    summary = {
        "clip_id": args.clip_id,
        "frame_index": args.frame_index,
        "timestamp": timestamp.isoformat(),
        "shape": list(raw.shape),
        "seeds_row_column": [list(position) for position in seeds],
        "t_min_k": seed_temperatures.tolist(),
        "t_edge_k": saddles[1:].tolist(),
        "prominence_k": prominences.tolist(),
        "sigma_px": args.sigma_px,
        "candidate_cold_quantile": 0.025,
        "connectivity": 8,
        "video": str(video_path.resolve()),
        "html": str(html_path.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
