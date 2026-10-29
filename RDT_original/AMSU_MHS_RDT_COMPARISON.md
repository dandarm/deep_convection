# Confronto Georeferenziato tra RDT Puro (SEVIRI) e Istantanee PMW AMSU/MHS sui Medicane

Questo documento descrive la metodologia, l'architettura del codice e la procedura operativa per replicare il confronto spaziale e temporale tra le maschere convettive del nostro **RDT Puro su MSG-SEVIRI** (`RDT_original`) e immagini statiche georeferenziate provenienti da sensori esterni (in particolare radiometri a microonde passive **AMSU-B / MHS** a $183 \pm 7\text{ GHz}$), anche quando sono disponibili **esclusivamente i file immagine (`.jpg` / `.png`)** senza gli array numerici o gli script che li hanno generati.

---

## 1. Obiettivo Scientifico

I radiometri a microonde passive in orbita polare (**AMSU/MHS**, canale nella banda del vapore acqueo a $183.31 \pm 7\text{ GHz}$) penetrano i cirri stratiformi superiori e misurano direttamente lo **scattering da idrometeore ghiacciate (graupel, grandine, aggregati densi)** nelle torri convettive profonde:
- **Forti depressioni di temperatura di brillanza ($T_B \lesssim 220\text{–}150\text{ K}$, colori giallo/verde/ciano/blu)** e i **marker neri `x`** identificano le celle convettive attive attorno all'occhio del Medicane.
- Il nostro **RDT Puro (`NWC SAF GEO-RDT-CW`, schema `GEN` + ancillari IR + correzione di parallasse)** opera invece con cadenza di **5 minuti** sui 7 canali termici di **MSG-SEVIRI** senza ausilio di dati NWP o radar in ingresso.

L'obiettivo dello script [`RDT_original/scripts/render_amsu_mhs_rdt_comparison.py`](scripts/render_amsu_mhs_rdt_comparison.py) è:
1. Calibrare la trasformazione cartografica direttamente da ciascuna figura AMSU/MHS;
2. Estrarre lo slot SEVIRI RSS a 5 minuti più vicino all'istante effettivo del sorvolo;
3. Riproiettare la temperatura di brillanza `SEVIRI IR 10.8` (corretta per parallasse) e la maschera semantica a 6 classi di `RDT Puro` esattamente sulla griglia pixel dell'immagine AMSU/MHS;
4. Produrre una tavola comparativa a **3 pannelli sincronizzati** e calcolare le metriche di sovrapposizione spaziale.

---

## 2. Calibrazione Cartografica Diretta da Immagini Statiche (`calibrate_amsu_image_geometry`)

Quando si dispone soltanto di figure raster (`.jpg` o `.png`) dotate di assi graduati e reticolo geografico interno (graticola tratteggiata a passo di $0.5^\circ$), la funzione `calibrate_amsu_image_geometry()` ricostruisce la mappa continua $(x_{\text{px}}, y_{\text{px}}) \mapsto (\text{lat}, \text{lon})$ in quattro passaggi automatici:

### 2.1 Normalizzazione della risoluzione e rilevamento del riquadro mappa (`box`)
- Le immagini vengono portate a una griglia canonica `1200 x 1200` pixel (per immagini ad altissima risoluzione, es. `6000 x 6000` come nel caso di *Rolf*, viene applicato anche un *min-pooling* $5 \times 5$ sul canale di luminanza per preservare il contrasto dei puntini di 1 pixel della graticola).
- I quattro bordi neri del riquadro geografico `(x0, y0, x1, y1)` vengono individuati cercando le righe e colonne con densità di pixel scuri non colorati superiore al $50\%$.

### 2.2 Rilevamento automatico delle tacche sugli assi e *snapping* alla graticola
- **Asse sinistro (Latitudine, passo $0.5^\circ$)**: nella fascia marginale a sinistra di `x0` vengono segmentate le componenti connesse (`scipy.ndimage.label`) dei caratteri tipografici (`39°N`, `38.5°N`, ...). I baricentri verticali grezzi vengono ordinati dall'alto (`lat_max`) al basso (`lat_min`) e poi **agganciati con precisione al pixel** ai minimi locali di luminanza del reticolo tratteggiato subito all'interno del bordo sinistro (`x0 + 6 .. x0 + 95`).
- **Asse inferiore (Longitudine, passo $0.5^\circ$)**: nella fascia sotto `y1` vengono segmentate le etichette verticali (`13.25°E`, `13.75°E`, ...). I baricentri orizzontali vengono ordinati da sinistra (`lon_min`) a destra (`lon_max`) e agganciati ai minimi locali dei meridiani tratteggiati subito sopra il bordo inferiore (`y1 - 95 .. y1 - 6`).
- Su entrambi gli assi viene eseguito un fit polinomiale robusto per regolarizzare eventuali tacche parzialmente sovrapposte ai bordi.

### 2.3 Modellazione della proiezione conica/cilindrica (convergenza dei meridiani e curvatura dei paralleli)
Nelle mappe regionali mediterranee i meridiani non sono perfettamente verticali ma convergono verso il polo (con meridiano centrale $\text{lon}_0 \approx 20.0^\circ\text{E}$):
- Lo script esegue una scansione sub-pixel dell'inclinazione media dei meridiani $\bar{s}_v = \Delta x / \Delta y$ e dell'inclinazione dei paralleli $\bar{s}_h = \Delta y / \Delta x$ massimizzando il contrasto lungo le linee tratteggiate interne al riquadro.
- Include quindi il termine di **convergenza meridiana locale**:
  $$s_v(\text{lon}) = \bar{s}_v + 0.0160 \cdot (\text{lon} - \text{lon}_{\text{center}})$$
  da cui, per ogni pixel $(x_{\text{px}}, y_{\text{px}})$ del riquadro mappa, si ottengono le coordinate geografiche esatte $(\text{lat}(y_{\text{px}}, x_{\text{px}}),\, \text{lon}(y_{\text{px}}, x_{\text{px}}))$.

### 2.4 Riproiezione da GEOS-SEVIRI con Correzione di Parallasse
1. A partire dal frame SEVIRI a 7 canali orientato `north-up, west-left`, il campo `IR 10.8` viene corretto per lo spostamento apparente di parallasse geostazionaria (`apply_parallax_correction_2d`, satellite MSG a $9.5^\circ\text{E}$) esattamente come già fatto per la maschera semantica `RDT Puro`.
2. Le coordinate $(\text{lon}, \text{lat})$ di ciascun pixel del riquadro AMSU/MHS vengono proiettate nello spazio metrico geostazionario `(x_geos, y_geos)` tramite `pyproj.Transformer` (dalla `projection_wkt` dello Zarr SEVIRI) e convertite in indici continui `(row_sev, col_sev)`.
3. Il fondo `SEVIRI IR 10.8` viene campionato con interpolazione bilineare (`order=1`), mentre la maschera semantica a classi discrete (`0..6`) viene campionata a vicino più prossimo (`nearest-neighbor`).

---

## 3. Allineamento Temporale e Utilizzo delle Cache RDT Puro

### 3.1 Differenza tra orario del nome file e orario di sorvolo nel titolo
Nei dataset di passaggi polari (AMSU/MHS, SSMIS, GMI, ATMS), il timestamp riportato nel **nome del file** (es. `AMSU_20211029_0742_apollo.jpg`) corrisponde spesso all'inizio del granulo orbitale, mentre l'istante effettivo in cui il satellite sorvola l'occhio del ciclone è stampato nel **titolo della figura** (`AMSU 29/10/2021 08:19`, ossia **37 minuti dopo**).
- Nella struttura `AmsuSnapshotSpec` sono registrati entrambi gli orari (`file_time_utc` e `overpass_time_utc`).
- L'accoppiamento con gli slot SEVIRI RSS (cadenza 5 minuti) viene sempre effettuato rispetto a `overpass_time_utc`, garantendo $|\Delta t| \le 2.5\text{ min}$.

### 3.2 Lettura delle cache `_cache_YYYYMMDD_pure.npz` e continuità temporale
Per evitare di ricalcolare l'intero ciclo giornaliero RDT:
1. **Giorni e orari già presenti in `_cache_YYYYMMDD_pure.npz`**: lo script individua lo slot tramite il vettore `timestamps_unix` salvato nella cache. Per evitare che un'oscillazione di split/merge su un singolo frame da 5 minuti (`Classe 2..6 -> Classe 1 -> Classe 2..6`, invisibile nei video a 24 fps perché dura $0.04\text{ s}$) degradi l'istantanea statica, viene applicato un filtro di persistenza temporale su 1 frame: se un pixel è classificato come nube fredda (`1`) allo slot $t$ ma è convettivo ($\ge 2$) sia a $t - 5\text{ min}$ sia a $t + 5\text{ min}$, conserva la classe convettiva di $t - 5\text{ min}$.
2. **Passaggi esterni alla finestra `--tight-window` (`_cache_YYYYMMDD_extra_slots.npz`)**: se un passaggio AMSU cade in un'ora del giorno non inclusa nel ritaglio `--tight-window` usato durante il rendering del video (6 casi su 32), lo script esegue automaticamente un **warmup causale di 6 frame (30 minuti)** direttamente dallo Zarr giornaliero (`E:\Datasets\MSG-SEVIRI\YYYY\YYYYMMDD.zarr`) e salva il risultato in `_cache_YYYYMMDD_extra_slots.npz`, così le esecuzioni successive sono istantanee.

---

## 4. Struttura della Tavola a 3 Pannelli e Legenda Classi

Ogni file `compare_<nome_file_amsu>_rdt_pure.png` contiene:
1. **Pannello 1 (Sinistra)**: Immagine originale AMSU/MHS ($T_B$ a $183 \pm 7\text{ GHz}$) con barra colori originale, ellisse del ciclone e marker `x` convettivi.
2. **Pannello 2 (Centro)**: `RDT Puro` sovrapposto a `SEVIRI IR 10.8` riproiettato nella stessa identica cornice geografica; la graticola tratteggiata e le annotazioni nere dell'immagine AMSU (coste, ellisse, marker `x`) sono riportate in **ciano** per verificare a colpo d'occhio l'esattezza dell'allineamento geografico.
3. **Pannello 3 (Destra)**: **Sovrapposizione diretta** delle celle `RDT Puro` (riempimento semitrasparente + contorno nitido + alone bianco esterno per le celle convettive) sopra la mappa `AMSU/MHS`, preservando in primo piano i marker `x` neri, l'ellisse e le coste.

**Codifica colori RDT Puro**:
- **Blu acciaio (`Classe 1`)**: Nube fredda non convettiva
- **Giallo (`Classe 2`)**: *Triggering* (Innesco convettivo)
- **Arancione (`Classe 3`)**: *Growing* (Sviluppo / espansione e raffreddamento)
- **Rosso cremisi (`Classe 4`)**: *Mature* (Maturità)
- **Rosso mattone (`Classe 5`)**: *Decaying* (Decadimento)
- **Magenta acceso (`Classe 6`)**: *Overshooting Top (OT)* confermato

---

## 5. Come Eseguire lo Script e Come Aggiungere Nuove Immagini in Futuro

### 5.1 Esecuzione da riga di comando (PowerShell)

Dalla radice del repository:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path

# Rigenera tutte le 32 tavole di confronto e il CSV riepilogativo (~15 secondi con cache calda)
.\.venv\Scripts\python.exe RDT_original\scripts\render_amsu_mhs_rdt_comparison.py

# Filtra per un singolo Medicane (es. Ianos, Apollo, Daniel) o per un singolo file
.\.venv\Scripts\python.exe RDT_original\scripts\render_amsu_mhs_rdt_comparison.py --filter ianos
.\.venv\Scripts\python.exe RDT_original\scripts\render_amsu_mhs_rdt_comparison.py --filter AMSU_20200917_1903_ianos
```

### 5.2 Procedura per aggiungere nuovi passaggi satellitari o nuovi Medicane

Per replicare il confronto su nuove immagini geolocalizzate (AMSU/MHS, SSMIS, GMI, ATMS, ecc.):
1. Verificare che esista lo Zarr giornaliero SEVIRI in `E:\Datasets\MSG-SEVIRI\YYYY\YYYYMMDD.zarr` e la cache `_cache_YYYYMMDD_pure.npz` nella cartella dell'evento in `RDT_original/results/medicanes/<Cartella_Evento>/` (generabile con `scripts/process_medicane_windows.py`).
2. Leggere sull'immagine:
   - L'orario esatto stampato nel **titolo della figura** (`YYYY-MM-DD HH:MM` UTC);
   - Il valore della **prima e ultima tacca di latitudine** sull'asse sinistro (`lat_max`, `lat_min`);
   - Il valore della **prima e ultima tacca di longitudine** sull'asse inferiore (`lon_min`, `lon_max`).
3. Aggiungere una riga a `AMSU_SNAPSHOTS` in [`RDT_original/scripts/render_amsu_mhs_rdt_comparison.py`](scripts/render_amsu_mhs_rdt_comparison.py):
   ```python
   AmsuSnapshotSpec(
       filename="AMSU_YYYYMMDD_HHMM_nome.jpg",
       medicane="NomeMedicane",
       event_folder="NomeCartellaEvento_YYYYMMDD_YYYYMMDD",
       day_str="YYYYMMDD",
       file_time_utc="HH:MM",
       overpass_time_utc="YYYY-MM-DD HH:MM",
       lat_max=39.00,
       lat_min=34.00,
       lon_min=13.25,
       lon_max=18.75,
   ),
   ```
4. Lanciare `render_amsu_mhs_rdt_comparison.py --filter AMSU_YYYYMMDD_HHMM_nome`.

---

## 6. Riepilogo dei 32 Passaggi Analizzati (8 Medicane, 2011–2023)

I risultati completi sono salvati in `RDT_original/results/medicanes_amsu_mhs_comparison/amsu_mhs_rdt_pure_summary.csv`:

| # | File AMSU/MHS | Medicane | Sorvolo (Titolo UTC) | Slot SEVIRI UTC | $\Delta t$ | Area Conv. RDT | Pixel OT | $\text{IR}_{10.8,\min}$ | Overlap PMW $\le 25\text{ km}$ |
| :-: | :--- | :--- | :--- | :--- | :-: | --: | --: | --: | --: |
| 01 | `AMSU_20111106_084201_rolf.jpg` | Rolf | `2011-11-06 09:19` | `2011-11-06 09:20` | $+1\text{m}$ | `6.80%` | `0` | `223.9 K` | `2.7%` |
| 02 | `AMSU_20111108_094200_rolf.jpg` | Rolf | `2011-11-08 10:18` | `2011-11-08 10:20` | $+2\text{m}$ | `12.58%` | `0` | `222.2 K` | **`87.7%`** |
| 03 | `AMSU_20111108_194800_rolf.jpg` | Rolf | `2011-11-08 19:58` | `2011-11-08 20:00` | $+2\text{m}$ | `27.51%` | `0` | `222.7 K` | **`86.5%`** |
| 04 | `AMSU_20141107_0757_qendresa.jpg` | Qendresa | `2014-11-07 08:36` | `2014-11-07 08:35` | $-1\text{m}$ | `0.00%` | `0` | `222.2 K` | `0.0%` |
| 05 | `AMSU_20141108_2015_qendresa.jpg` | Qendresa | `2014-11-08 20:24` | `2014-11-08 20:25` | $+1\text{m}$ | `0.45%` | `0` | `230.1 K` | `0.0%` |
| 06 | `AMSU_20161028_0742_trixie.jpg` | Trixie | `2016-10-28 08:20` | `2016-10-28 08:20` | $0\text{m}$ | `4.79%` | `282` | `215.2 K` | `37.7%` |
| 07 | `AMSU_20161028_1930_trixie.jpg` | Trixie | `2016-10-28 19:40` | `2016-10-28 19:40` | $0\text{m}$ | `24.46%` | `716` | `211.7 K` | **`100.0%`** |
| 08 | `AMSU_20161028_2018_trixie.jpg` | Trixie | `2016-10-28 20:26` | `2016-10-28 20:25` | $-1\text{m}$ | `25.20%` | `348` | `211.9 K` | **`100.0%`** |
| 09 | `AMSU_20161030_0745_trixie.jpg` | Trixie | `2016-10-30 08:25` | `2016-10-30 08:25` | $0\text{m}$ | `12.83%` | `51` | `216.6 K` | **`100.0%`** |
| 10 | `AMSU_20161030_0842_trixie.jpg` | Trixie | `2016-10-30 09:19` | `2016-10-30 09:20` | $+1\text{m}$ | `11.48%` | `46` | `216.9 K` | **`100.0%`** |
| 11 | `AMSU_20161031_0821_trixie.jpg` | Trixie | `2016-10-31 08:58` | `2016-10-31 09:00` | $+2\text{m}$ | `8.17%` | `0` | `223.9 K` | **`73.2%`** |
| 12 | `AMSU_20171116_0739_numa.jpg` | Numa | `2017-11-16 08:18` | `2017-11-16 08:20` | $+2\text{m}$ | `4.81%` | `1098` | `215.5 K` | `26.9%` |
| 13 | `AMSU_20171116_0836_numa.jpg` | Numa | `2017-11-16 09:14` | `2017-11-16 09:15` | $+1\text{m}$ | `1.12%` | `32` | `216.5 K` | `1.9%` |
| 14 | `AMSU_20171117_1909_numa.jpg` | Numa | `2017-11-17 19:18` | `2017-11-17 19:20` | $+2\text{m}$ | `1.92%` | `16` | `220.6 K` | `0.3%` |
| 15 | `AMSU_20171117_2006_numa.jpg` | Numa | `2017-11-17 20:13` | `2017-11-17 20:15` | $+2\text{m}$ | `0.92%` | `0` | `222.5 K` | `0.2%` |
| 16 | `AMSU_20171118_0839_numa.jpg` | Numa | `2017-11-18 09:17` | `2017-11-18 09:15` | $-2\text{m}$ | `18.79%` | `0` | `214.9 K` | `7.6%` |
| 17 | `AMSU_20200916_0733_ianos.jpg` | Ianos | `2020-09-16 08:13` | `2020-09-16 08:15` | $+2\text{m}$ | `18.89%` | `215` | `211.9 K` | **`99.7%`** |
| 18 | `AMSU_20200916_2012_ianos.jpg` | Ianos | `2020-09-16 20:20` | `2020-09-16 20:20` | $0\text{m}$ | `10.40%` | `0` | `212.5 K` | **`99.6%`** |
| 19 | `AMSU_20200917_1903_ianos.jpg` | Ianos | `2020-09-17 19:12` | `2020-09-17 19:10` | $-2\text{m}$ | `10.02%` | `0` | `215.5 K` | **`99.9%`** |
| 20 | `AMSU_20211028_1833_apollo.jpg` | Apollo | `2021-10-28 18:43` | `2021-10-28 18:45` | $+2\text{m}$ | `44.67%` | `66` | `212.9 K` | **`100.0%`** |
| 21 | `AMSU_20211029_0742_apollo.jpg` | Apollo | `2021-10-29 08:19` | `2021-10-29 08:20` | $+1\text{m}$ | `56.57%` | `0` | `213.3 K` | **`99.9%`** |
| 22 | `AMSU_20211029_1930_apollo.jpg` | Apollo | `2021-10-29 19:39` | `2021-10-29 19:40` | $+1\text{m}$ | `37.41%` | `0` | `217.6 K` | **`99.9%`** |
| 23 | `AMSU_20211029_2024_apollo.jpg` | Apollo | `2021-10-29 20:31` | `2021-10-29 20:30` | $-1\text{m}$ | `31.95%` | `0` | `219.9 K` | **`99.4%`** |
| 24 | `AMSU_20211030_1909_apollo.jpg` | Apollo | `2021-10-30 19:19` | `2021-10-30 19:20` | $+1\text{m}$ | `21.11%` | `0` | `223.8 K` | **`99.2%`** |
| 25 | `AMSU_20230228_1954_juliette.jpg` | Juliette | `2023-02-28 20:03` | `2023-02-28 20:05` | $+2\text{m}$ | `17.36%` | `0` | `226.4 K` | **`99.2%`** |
| 26 | `AMSU_20230228_2048_juliette.jpg` | Juliette | `2023-02-28 20:56` | `2023-02-28 20:55` | $-1\text{m}$ | `0.79%` | `0` | `227.1 K` | `0.0%` |
| 27 | `AMSU_20230301_1933_juliette.jpg` | Juliette | `2023-03-01 19:42` | `2023-03-01 19:40` | $-2\text{m}$ | `0.83%` | `0` | `229.6 K` | `0.0%` |
| 28 | `AMSU_20230302_0815_juliette.jpg` | Juliette | `2023-03-02 08:53` | `2023-03-02 08:55` | $+2\text{m}$ | `32.27%` | `0` | `220.2 K` | **`100.0%`** |
| 29 | `AMSU_20230302_0903_juliette.jpg` | Juliette | `2023-03-02 09:42` | `2023-03-02 09:40` | $-2\text{m}$ | `37.48%` | `0` | `219.5 K` | **`100.0%`** |
| 30 | `AMSU_20230906_192100_daniel.jpg` | Daniel | `2023-09-06 19:30` | `2023-09-06 19:30` | $0\text{m}$ | `32.65%` | `253` | `212.5 K` | **`100.0%`** |
| 31 | `AMSU_20230909_1912_daniel.jpg` | Daniel | `2023-09-09 19:21` | `2023-09-09 19:20` | $-1\text{m}$ | `22.78%` | `263` | `209.6 K` | **`91.5%`** |
| 32 | `AMSU_20230909_200000_daniel.jpg` | Daniel | `2023-09-09 20:08` | `2023-09-09 20:10` | $+2\text{m}$ | `23.36%` | `390` | `209.5 K` | **`87.1%`** |

---

## 7. Nota Tecnica per Sviluppi Futuri: Profondità di Ricorsione Multi-Soglia (`depth <= 12`)

Dall'analisi incrociata con i passaggi AMSU/MHS di **Qendresa** (`2014-11-07 08:35 UTC`) e **Numa** (`2017-11-16 08:20 UTC`) è emerso un dettaglio algoritmico importante in [`RDT_original/rdt_nwcsaf/detection.py`](rdt_nwcsaf/detection.py) (riga 346):
- La funzione `_descend()` esplora l'albero gerarchico delle isoterme da `twarm_k = 278.15 K` ($+5^\circ\text{C}$) a `tcold_k = 198.15 K` ($-75^\circ\text{C}$) con passo `deltat_k = 1.0 K`, incrementando `depth` di `1` a ogni biforcazione valida (`len(valid_children) >= 2`).
- Quando un vasto sistema frontale/sinottico mediterraneo si biforca 13 volte già nelle soglie calde tra $278.15\text{ K}$ e $265.15\text{ K}$, la condizione `if depth <= 12:` interrompe ulteriori suddivisioni sotto $265\text{ K}$:
  1. Se la macro-componente a $265\text{ K}$ ha area $\le S_{\max} = 400\,000\text{ km}^2$ (es. *Apollo* e *Juliette*), viene conservata come un'unica grande cella con base calda ($T_{\text{BT}} \approx 265\text{ K}$);
  2. Se invece supera $400\,000\text{ km}^2$ (es. *Qendresa* `2014-11-07 08:35 UTC`), viene scartata dal controllo `s_bt <= cfg.smax_km2` (riga 461) senza che le singole torri fredde interne ($T_{\min} \approx 214\text{ K}$) vengano estratte.
- Qualora in futuro si ricalcolino le cache `_cache_YYYYMMDD_pure.npz`, portare il limite di profondità da `depth <= 12` a `depth <= 80` (pari all'escursione completa di 80 livelli da $1\text{ K}$) permetterà di isolare sempre le singole torri convettive anche all'interno di scudi frontali sinottici $> 400\,000\text{ km}^2$.
