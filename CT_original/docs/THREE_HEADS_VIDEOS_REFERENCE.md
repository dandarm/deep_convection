# Catalogo dei Video Multi-Periodo a 3 Teste (CMA + CT + RDT_original) e Confronti Visibile Diurno

Documento di riferimento per i video lunghi multi-periodo (giornata intera di **Medicane Ianos `2020-09-17`** da 288 frame e **7 sequenze lunghe da 13–17 ore del `2022`**) generati con **`RDT_original`** ([`NwcSafRdtTracker`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/pipeline.py), con proxy scalare di tropopausa di scena e `separate_observation_support=True`) e con la calibrazione costiera geostazionaria di [`NwcSafCmaCtTracker`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/ct_nwcsaf/pipeline.py).

---

## 0. Player HTML Interattivo Frame-by-Frame

Per ispezionare agilmente tutti i 10 video lunghi a 3 teste (incluse le **3 giornate intere 24h del 2020** e i **7 eventi convettivi estremi del 2022**) e i 4 confronti diurni nel visibile spostandosi istantaneamente di singolo frame tramite cursore o frecce da tastiera (`←` / `→`):
- Player nella documentazione: [`CT_original/docs/video_player.html`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/docs/video_player.html)
- Player nella cartella locale: [`locale_loki/video_player.html`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/locale_loki/video_player.html)
- Player nella cartella video su `E:\`: [`player.html`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/player.html)

---

## 1. Specifiche di Rendering e Tassonomia Visualizzata

- **Script di generazione:**
  - Dashboard a 7 canali SEVIRI IR + 3 Teste: [`CT_original/scripts/render_probing_spec_video.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/scripts/render_probing_spec_video.py)
  - Confronto diurno Visibile (`VIS006`, `VIS008`, `Day Natural Color RGB`) + `IR_108` + 3 Teste: [`CT_original/scripts/render_visible_vs_cma_comparison.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/scripts/render_visible_vs_cma_comparison.py)
- **Spazio profondo oltre il lembo terrestre (`~is_earth_disk`):** Nero pieno `#000000` `(0, 0, 0)`, nettamente distinto dal cielo sereno `0: Clear Sky` (`#1E293B`).
- **Preservazione degli Overshooting Top (OT) nel ridimensionamento dei pannelli:** Prima del sottocampionamento `NEAREST` da `(763, 1762)` a `(311, 596)`, viene applicato un filtro morfologico di massimo a priorità di classe (`ndimage.maximum_filter`) sui pixel convettivi e sugli Overshooting Top (`4: Severe / OT`, `#EB0055`), garantendo che nessun nucleo di OT (anche di 1–4 pixel nativi) venga perso nel pannello Full HD.
- **Tassonomia delle 3 Teste:**
  - **Testa 1 — Cloud Mask:** `0: Clear Sky` (`#1E293B`), `1: Cloud Filled` (`#F1F5F9`), `2: Cloud Contaminated` (`#94A3B8`).
  - **Testa 2 — Cloud Type:** `0: Clear Sky` (`#1E293B`), `1: Low Cloud` (`#F59E0B`), `2: Mid Cloud` (`#10B981`), `3: High Opaque` (`#EF4444`), `4: Cirrus & Multilayer` (`#06B6D4`).
  - **Testa 3 — RDT Convezione (`RDT_original`):** `0: Clear / Non-Cold` (`#1E293B`), `1: Cold Anvil` (Blu acceso `#3B82F6`), `2: Triggering / Growing` (Giallo `#FAE014`), `3: Mature / Decaying` (Arancione `#FF8800`), `4: Severe / OT` (Rosso-Magenta `#EB0055`).

---

## 2. Periodi di Riferimento — Dashboard Completa a 3 Teste (3 Giornate Intere 2020 + 7 Outbreak Lunghi 2022)

Cartella di output: `E:\Datasets\Deep_convection\staging\nwcsaf_cma_ct_pilot\three_heads_convective_videos\`
File di riepilogo JSON: [`three_heads_videos_summary.json`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_videos_summary.json)

| # | Tag Periodo | Descrizione Sinottica | Inizio Spin-Up UTC | Finestra Video Renderizzata UTC | Frame (5 min) | Durata Video (3 fps) | Pixel Convettivi (`RDT_original`) | Picco Frame (px) | Pixel OT (`4: Severe / OT`) | Recall in `CMA` |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `20200917_medicane_ianos` | Medicane Ianos (Mar Ionio — 24h Intere) | `2020-09-17 00:00` | `2020-09-17 00:00–23:55` (24h) | **288** | **96.0s (1m 36s)** | **`14,631,039`** | `110,741` | **`13,936`** | **`99.88%`** |
| 1 | `20200711_summer_convective_outbreak` | Outbreak Temporalesco Estivo (24h Intere) | `2020-07-11 00:00` | `2020-07-11 00:00–23:55` (24h) | **288** | **96.0s (1m 36s)** | **`8,164,376`** | `48,671` | **`27,901`** | **`99.76%`** |
| 2 | `20200715_alpine_med_storms` | Sistemi Convettivi Alpini & Mediterranei (24h Intere) | `2020-07-15 00:00` | `2020-07-15 00:00–23:55` (24h) | **288** | **96.0s (1m 36s)** | **`10,054,052`** | `51,531` | **`13,648`** | **`99.76%`** |
| 1 | `20220628_alpine_west_med_severe` | Supercelle Alpine & Med Occidentale | `2022-06-28 09:00` | `2022-06-28 10:00–22:55` (13h) | **156** | **52.0s** | **`11,762,114`** | `109,729` | **`40,246`** | **`99.97%`** |
| 2 | `20220912_deep_tropopause_storms` | Convezione Profonda & Top 197.5 K | `2022-09-12 10:00` | `2022-09-12 11:00–23:55` (13h) | **156** | **52.0s** | **`4,588,034`** | `30,341` | **`23,028`** | **`99.72%`** |
| 3 | `20220522_euro_supercells` | Outbreak Europeo di Supercelle & MCS | `2022-05-22 10:00` | `2022-05-22 11:00–23:55` (13h) | **156** | **52.0s** | **`9,174,521`** | `89,347` | **`19,627`** | **`99.87%`** |
| 4 | `20220528_med_continental_mcs` | MCS & Temporali Diurni Continentali | `2022-05-28 08:00` | `2022-05-28 09:00–21:55` (13h) | **156** | **52.0s** | **`8,051,489`** | `64,576` | **`17,974`** | **`99.80%`** |
| 5 | `20220925_extreme_ot_mcs` | MCS Estremo Tirreno/Adriatico — Top 199 K | `2022-09-25 00:00` | `2022-09-25 00:30–13:55` (13.5h) | **162** | **54.0s** | **`7,245,539`** | `66,819` | **`13,923`** | **`99.55%`** |
| 6 | `20220924_med_mcs_outbreak` | Outbreak MCS V-Shaped Mediterraneo & Italia | `2022-09-24 06:00` | `2022-09-24 07:00–23:55` (17h) | **204** | **68.0s (1m 08s)** | **`9,533,191`** | `89,302` | **`3,855`** | **`99.61%`** |
| 7 | `20220506_widespread_med_convection` | Ciclone Convettivo Mediterraneo Esteso | `2022-05-06 00:00` | `2022-05-06 00:30–13:55` (13.5h) | **162** | **54.0s** | **`2,875,317`** | `31,294` | **`1,912`** | **`98.93%`** |

### Percorsi dei File MP4, GIF e Anteprime PNG (10 Periodi Lunghi)

0. **`20200917_medicane_ianos` (Medicane Ianos — Mar Ionio, 24h Intere, 288 frame):**
   - MP4: [`three_heads_probing_20200917_medicane_ianos.mp4`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20200917_medicane_ianos.mp4)
   - GIF: [`three_heads_probing_20200917_medicane_ianos.gif`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20200917_medicane_ianos.gif)
   - PNG: [`three_heads_preview_20200917_medicane_ianos_20200917T1130Z.png`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_preview_20200917_medicane_ianos_20200917T1130Z.png)

1. **`20200711_summer_convective_outbreak` (Outbreak Temporalesco Estivo, 24h Intere, 288 frame):**
   - MP4: [`three_heads_probing_20200711_summer_convective_outbreak.mp4`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20200711_summer_convective_outbreak.mp4)
   - GIF: [`three_heads_probing_20200711_summer_convective_outbreak.gif`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20200711_summer_convective_outbreak.gif)
   - PNG: [`three_heads_preview_20200711_summer_convective_outbreak_20200711T1720Z.png`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_preview_20200711_summer_convective_outbreak_20200711T1720Z.png)

2. **`20200715_alpine_med_storms` (Sistemi Convettivi Alpini & Mediterranei, 24h Intere, 288 frame):**
   - MP4: [`three_heads_probing_20200715_alpine_med_storms.mp4`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20200715_alpine_med_storms.mp4)
   - GIF: [`three_heads_probing_20200715_alpine_med_storms.gif`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20200715_alpine_med_storms.gif)
   - PNG: [`three_heads_preview_20200715_alpine_med_storms_20200715T1505Z.png`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_preview_20200715_alpine_med_storms_20200715T1505Z.png)

1. **`20220628_alpine_west_med_severe` (Supercelle Alpine & Med Occidentale, 13h, 156 frame):**
   - MP4: [`three_heads_probing_20220628_alpine_west_med_severe.mp4`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20220628_alpine_west_med_severe.mp4)
   - GIF: [`three_heads_probing_20220628_alpine_west_med_severe.gif`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20220628_alpine_west_med_severe.gif)
   - PNG: [`three_heads_preview_20220628_alpine_west_med_severe_20220628T1850Z.png`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_preview_20220628_alpine_west_med_severe_20220628T1850Z.png)

2. **`20220912_deep_tropopause_storms` (Convezione Profonda & Top 197.5 K, 13h, 156 frame):**
   - MP4: [`three_heads_probing_20220912_deep_tropopause_storms.mp4`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20220912_deep_tropopause_storms.mp4)
   - GIF: [`three_heads_probing_20220912_deep_tropopause_storms.gif`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20220912_deep_tropopause_storms.gif)
   - PNG: [`three_heads_preview_20220912_deep_tropopause_storms_20220912T1830Z.png`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_preview_20220912_deep_tropopause_storms_20220912T1830Z.png)

3. **`20220522_euro_supercells` (Outbreak Europeo di Supercelle & MCS, 13h, 156 frame):**
   - MP4: [`three_heads_probing_20220522_euro_supercells.mp4`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20220522_euro_supercells.mp4)
   - GIF: [`three_heads_probing_20220522_euro_supercells.gif`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20220522_euro_supercells.gif)
   - PNG: [`three_heads_preview_20220522_euro_supercells_20220522T2040Z.png`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_preview_20220522_euro_supercells_20220522T2040Z.png)

4. **`20220528_med_continental_mcs` (MCS & Temporali Diurni Continentali, 13h, 156 frame):**
   - MP4: [`three_heads_probing_20220528_med_continental_mcs.mp4`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20220528_med_continental_mcs.mp4)
   - GIF: [`three_heads_probing_20220528_med_continental_mcs.gif`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20220528_med_continental_mcs.gif)
   - PNG: [`three_heads_preview_20220528_med_continental_mcs_20220528T1720Z.png`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_preview_20220528_med_continental_mcs_20220528T1720Z.png)

5. **`20220925_extreme_ot_mcs` (MCS Estremo Tirreno/Adriatico — Top 199 K, 13.5h, 162 frame):**
   - MP4: [`three_heads_probing_20220925_extreme_ot_mcs.mp4`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20220925_extreme_ot_mcs.mp4)
   - GIF: [`three_heads_probing_20220925_extreme_ot_mcs.gif`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20220925_extreme_ot_mcs.gif)
   - PNG: [`three_heads_preview_20220925_extreme_ot_mcs_20220925T1000Z.png`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_preview_20220925_extreme_ot_mcs_20220925T1000Z.png)

6. **`20220924_med_mcs_outbreak` (Outbreak MCS V-Shaped Mediterraneo & Italia, 17h, 204 frame):**
   - MP4: [`three_heads_probing_20220924_med_mcs_outbreak.mp4`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20220924_med_mcs_outbreak.mp4)
   - GIF: [`three_heads_probing_20220924_med_mcs_outbreak.gif`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20220924_med_mcs_outbreak.gif)
   - PNG: [`three_heads_preview_20220924_med_mcs_outbreak_20220924T1220Z.png`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_preview_20220924_med_mcs_outbreak_20220924T1220Z.png)

7. **`20220506_widespread_med_convection` (Ciclone Convettivo Mediterraneo Esteso, 13.5h, 162 frame):**
   - MP4: [`three_heads_probing_20220506_widespread_med_convection.mp4`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20220506_widespread_med_convection.mp4)
   - GIF: [`three_heads_probing_20220506_widespread_med_convection.gif`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_probing_20220506_widespread_med_convection.gif)
   - PNG: [`three_heads_preview_20220506_widespread_med_convection_20220506T1145Z.png`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/three_heads_convective_videos/three_heads_preview_20220506_widespread_med_convection_20220506T1145Z.png)

---

## 3. Periodi di Riferimento — Confronto Diurno Visibile (`VIS006`, `VIS008`, `Day Natural Color RGB`) vs 3 Teste Pure-IR (4 Periodi)

Cartella di output: `E:\Datasets\Deep_convection\staging\nwcsaf_cma_ct_pilot\visible_vs_cma_daytime_comparisons\`
File di riepilogo JSON: [`visible_comparison_summary.json`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/visible_vs_cma_daytime_comparisons/visible_comparison_summary.json)

| # | Tag Periodo | Finestra Diurna UTC | Frame (cadenza 15m) | Video MP4 | Anteprima PNG Picco |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `20261027_med_atlantic` | `2026-10-27 09:15–12:45 UTC` | 15 | [`visible_vs_cma_probing_20261027_med_atlantic.mp4`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/visible_vs_cma_daytime_comparisons/visible_vs_cma_probing_20261027_med_atlantic.mp4) | [`visible_vs_cma_preview_20261027_med_atlantic_20261027T1245Z.png`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/visible_vs_cma_daytime_comparisons/visible_vs_cma_preview_20261027_med_atlantic_20261027T1245Z.png) |
| 2 | `20250720_severe_summer` | `2025-07-20 12:15–14:55 UTC` | 11 | [`visible_vs_cma_probing_20250720_severe_summer.mp4`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/visible_vs_cma_daytime_comparisons/visible_vs_cma_probing_20250720_severe_summer.mp4) | [`visible_vs_cma_preview_20250720_severe_summer_20250720T1345Z.png`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/visible_vs_cma_daytime_comparisons/visible_vs_cma_preview_20250720_severe_summer_20250720T1345Z.png) |
| 3 | `20250605_continental_outbreak` | `2025-06-05 12:15–14:55 UTC` | 11 | [`visible_vs_cma_probing_20250605_continental_outbreak.mp4`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/visible_vs_cma_daytime_comparisons/visible_vs_cma_probing_20250605_continental_outbreak.mp4) | [`visible_vs_cma_preview_20250605_continental_outbreak_20250605T1445Z.png`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/visible_vs_cma_daytime_comparisons/visible_vs_cma_preview_20250605_continental_outbreak_20250605T1445Z.png) |
| 4 | `20250801_alpine_po_valley` | `2025-08-01 11:15–13:55 UTC` | 11 | [`visible_vs_cma_probing_20250801_alpine_po_valley.mp4`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/visible_vs_cma_daytime_comparisons/visible_vs_cma_probing_20250801_alpine_po_valley.mp4) | [`visible_vs_cma_preview_20250801_alpine_po_valley_20250801T1345Z.png`](file:///E:/Datasets/Deep_convection/staging/nwcsaf_cma_ct_pilot/visible_vs_cma_daytime_comparisons/visible_vs_cma_preview_20250801_alpine_po_valley_20250801T1345Z.png) |
