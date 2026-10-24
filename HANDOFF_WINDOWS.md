# Handoff verso Windows

Questo file permette di riprendere il feasibility study su un altro computer
anche se la cronologia locale della task Codex non è disponibile.

## Stato verificato

- Repository: `https://github.com/dandarm/deep_convection`
- Branch principale: `main`
- Pilot: Mediterraneo ed Europa meridionale, maggio-settembre 2020
- Dominio: 10°W-45°E, 25°N-55°N
- Tolleranza temporale EMMA-GPM: massimo 30 minuti, con offset effettivo
  conservato nei cataloghi
- Stato dei test al primo rilascio: 5 test superati
- I file grezzi e intermedi non sono versionati su GitHub

Il pilot pubblicato usa il GPM Precipitation Feature Database. I risultati PF
sono un primo catalogo riproducibile e non equivalgono alla copertura completa
degli swath GMI/DPR. Consultare `REPORT.md` e `PLAN_COMPLETAMENTO.md` prima di
estendere o interpretare le metriche.

## 1. Recuperare gli artefatti versionati

In PowerShell:

```powershell
git clone https://github.com/dandarm/deep_convection.git D:\Progetti\deep_convection
cd D:\Progetti\deep_convection
```

Cambiare `D:\Progetti` con la cartella desiderata. Aprire poi questa directory
come progetto nella app Codex desktop.

## 2. Impostare la directory dati sul nuovo computer

Usare, per esempio:

```text
E:\Datasets\Deep_convection
```

Impostare la variabile per la sessione PowerShell corrente:

```powershell
$env:EMMA_GPM_DATA_ROOT = "E:\Datasets\Deep_convection"
```

Per renderla persistente per l'utente:

```powershell
[Environment]::SetEnvironmentVariable(
  "EMMA_GPM_DATA_ROOT",
  "E:\Datasets\Deep_convection",
  "User"
)
```

Riavviare Codex o il terminale dopo l'impostazione persistente.

## 3. Ricreare l'ambiente

```powershell
cd D:\Progetti\deep_convection
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest -q
```

Alcune dipendenze scientifiche possono essere più semplici in WSL2. Se si usa
WSL2, conservare progetto e dataset nello stesso ambiente e usare percorsi
Linux coerenti.

## 4. Ricostruire da zero i dati del pilot

Questa è la procedura consigliata sul computer con la connessione più veloce.
Ricostruisce ciò che era presente nella directory dati Linux senza copiare i
file fra computer.

Dimensioni osservate nella prima esecuzione:

- archivio EMMA 2020: circa 23 MiB;
- cinque file mensili GMI PF: circa 551 MiB;
- cinque file mensili DPR PF: circa 515 MiB;
- inventario CMR JSON e Parquet: circa 28 MiB;
- directory dati completa dopo preparazione e QA: circa 1,4 GiB.

Eseguire dalla radice del repository, mantenendo
`EMMA_GPM_DATA_ROOT` impostata:

```powershell
$python = ".\.venv\Scripts\python.exe"

# Download pubblici: EMMA e PF GMI/DPR, maggio-settembre 2020
& $python scripts\download_emma.py
& $python scripts\download_gpm_pf.py

# Preparazione e subset sul dominio mediterraneo
& $python scripts\prepare_emma.py
& $python scripts\prepare_gpm_pf.py

# Inventario dei granuli nativi GPM V07; non scarica i granuli
& $python scripts\query_cmr.py

# Cataloghi EMMA -> GPM e GPM -> EMMA
& $python scripts\build_catalogs.py

# Metriche e figure
$env:MPLCONFIGDIR = "$env:TEMP\mplconfig-emma-gpm"
& $python scripts\compute_metrics.py

# Dieci casi di QA con centri reali dei pixel DPR. Lo script usa richieste HTTP
# parziali ai contenitori orbitali pubblici e salva checkpoint per orbita.
& $python scripts\extract_raw_dpr_cases.py
& $python scripts\plot_qc_cases.py

# Verifica finale
$env:PYTHONPATH = "src"
& $python -m compileall -q src scripts tests
& $python -m pytest -q
git status --short
```

I download pubblici attesi sono esattamente questi:

```text
raw/emma/EMMA_MCS_Climatology_Europe_2020_v1.tar.gz
raw/gpm_pf_1c/GPM.202005.HDF5 ... GPM.202009.HDF5
raw/gpm_pf_dpr/pf_202005_level2.HDF ... pf_202009_level2.HDF
```

`download_earthdata.py` non fa parte della ricostruzione del pilot da 1,4 GiB.
Non lanciarlo sull'intero inventario: il download nativo completo DPR del periodo
è stato stimato in circa 756,8 GiB. Usarlo solo in una fase successiva, con un
token Earthdata e un limite esplicito, per esempio `--limit 1`.

Il downloader scrive inizialmente file con suffisso `.part`; un file incompleto
non deve essere rinominato o usato come dato valido. Se una fase fallisce,
conservare il log, verificare i file già completi e chiedere a Codex di ripartire
dall'ultimo passaggio riuscito senza cancellare gli output validi.

## 5. Prompt da dare alla nuova task

Se l'handoff nativo della chat non è disponibile, iniziare una nuova task nel
progetto clonato con questo prompt:

> Leggi integralmente README.md, HANDOFF_WINDOWS.md, REPORT.md e
> PLAN_COMPLETAMENTO.md prima di agire. Questo computer non contiene la directory
> dati del pilot Linux: ricostruiscila da zero seguendo la sezione 4 di
> HANDOFF_WINDOWS.md, usando `E:\Datasets\Deep_convection` come
> `EMMA_GPM_DATA_ROOT`. Scarica soltanto EMMA 2020, i cinque mesi GMI PF, i cinque
> mesi DPR PF e l'inventario CMR necessari al pilot maggio-settembre 2020. Non
> scaricare in massa i granuli GPM nativi da Earthdata e non avviare alcun
> addestramento. Esegui preparazione, matching, metriche, dieci casi QA e test;
> conserva i valori effettivi dei time offset e mantieni nel catalogo GPM anche
> le osservazioni fuori EMMA. Tratta EMMA esclusivamente come weak supervision,
> non come ground truth indipendente. La rete è più veloce su questo computer:
> completa i download pubblici, ma rendi ogni fase verificabile e riprendibile,
> senza cancellare file validi. Alla fine confronta conteggi e schemi con gli
> output versionati, riporta eventuali differenze e lascia `git status` pulito o
> spiega precisamente ogni modifica.

## 6. Handoff nativo della chat

Questa task è stata creata nella preview Linux, dove `Connections` non era
disponibile. Se una futura versione rende visibile Remote su entrambi i sistemi
e il progetto Git è salvato su entrambi, si potrà usare `Hand off`; fino ad
allora usare il prompt della sezione 5.

Non copiare alla cieca l'intera directory `.codex` tra sistemi operativi:
potrebbe includere credenziali, configurazioni locali e percorsi non portabili.

## 7. Aggiornamento 2026-09-11: chiusura del pilot pixel-level

Il pilot è stato ricostruito su Windows usando
`E:\Datasets\Deep_convection` come `EMMA_GPM_DATA_ROOT`. Non sono stati
scaricati in massa granuli GPM nativi da Earthdata e non è stato avviato alcun
training. I campi pixel-level aggiuntivi provengono dai contenitori orbitali PF
pubblici e sono stati letti con richieste HTTP parziali e checkpoint per orbita.

### Interpretazione consolidata dei dati

EMMA fornisce maschere orarie di MCS. Una `track_uid` identifica lo stesso MCS
seguito nelle ore successive e va interpretata come un evento distinto, non
come una singola cella della griglia. Le celle EMMA sono gli elementi spaziali
della maschera; il numero di celle misura l'area discretizzata dell'oggetto, non
il numero di eventi. EMMA è sempre weak supervision e non è una ground truth
indipendente per GPM.

Le PF sono oggetti di precipitazione derivati dai prodotti GPM. `PRECIP` per
GMI e `NSPRECIP` per DPR sono invece retrieval continui del rate di
precipitazione, espressi in mm/h. Nei grafici sono indicati rispettivamente come
`Precipitation-rate retrieval GMI` e
`Near-surface precipitation-rate retrieval DPR`. Non sono PF: applicando una
soglia al retrieval e raggruppando i pixel connessi si possono costruire gli
oggetti PF. I cataloghi conservano anche tutte le osservazioni GPM fuori EMMA.

### Figure pixel-level

Sono state selezionate 30 scene, sei per ciascun mese da maggio a settembre, e
sono state prodotte figure individuali a due pannelli GMI/DPR. Non si usano
ellissi: vengono mostrati i pixel dei retrieval ricampionati sulla griglia EMMA.
La maschera EMMA è rossa, semitrasparente, con bordo rosso scuro disegnato in
primo piano; le coste sono visibili; la colorbar riporta soltanto
`Rate di precipitazione [mm/h]`. I trenta file correnti sono identificati in
`results/presentation_precipitation_figure_manifest.csv`. Nella directory delle
figure è stato conservato anche un PNG storico valido, ma non appartiene al
manifest corrente.

### Overlap ottenuto dai pixel reali

La misura usa i pixel con rate maggiore di 0,1 mm/h, ricampionati per centro
più vicino sulla griglia EMMA. Nei 30 casi selezionati:

- GMI presenta almeno un'intersezione in 26 scene su 30; il 15,2% delle celle
  precipitanti ricampionate ricade dentro EMMA e le celle precipitanti coprono
  il 60,9% dell'area EMMA aggregata;
- DPR presenta almeno un'intersezione in 26 scene su 30; il 23,5% delle celle
  precipitanti ricampionate ricade dentro EMMA e le celle precipitanti coprono
  il 27,8% dell'area EMMA aggregata;
- 41 componenti GMI su 4.732 e 66 componenti DPR su 1.819 toccano EMMA.

Le percentuali nelle due direzioni non sono equivalenti. La frazione di
precipitazione dentro EMMA usa come denominatore l'area precipitante GPM; la
frazione di EMMA coperta usa come denominatore l'area EMMA. La maggiore
copertura EMMA di GMI è coerente con una swath più ampia, mentre DPR osserva una
fascia più stretta. La selezione delle 30 scene è volutamente ricca di overlap e
non stima la probabilità naturale di coincidenza.

Il manifest era stato costruito con 27 scene positive secondo le ellissi PF e
tre scene contestuali. Il controllo con i pixel reali ne conferma 26: almeno un
caso positivo secondo l'ellisse non presenta overlap nel retrieval rasterizzato.
Questa differenza documenta il limite dell'approssimazione ellittica.

### Conteggi e ordine di grandezza

Nel periodo maggio-settembre 2020 il dominio contiene 343 track MCS distinte,
2.652 maschere-oggetto orarie e 1.672 ore con almeno un oggetto EMMA. Il catalogo
completo contiene 92.004 PF GMI in 1.169 orbite e 34.680 PF DPR in 938 orbite.
Le PF con co-occorrenza EMMA secondo la geometria PF sono 161 per GMI e 155 per
DPR.

La sola proiezione lineare del tasso MJJAS sull'anno bisestile usa il fattore
`366/153 = 2,392`. Produce circa 821 eventi EMMA, 6.344 maschere-oggetto EMMA,
4.000 scene orarie positive, 220.088 PF GMI e 82.960 PF DPR per anno. Le sole PF
in overlap diventerebbero circa 385 GMI e 371 DPR. È un ordine di grandezza, non
una stima climatologica: estendere a tutto l'anno il tasso della stagione calda
può sovrastimare la frequenza invernale.

### Output e riproducibilità aggiunti

- `catalogs/presentation_scene_manifest_2020.csv`;
- `results/presentation_precipitation_figure_manifest.csv`;
- `results/presentation_precipitation_map_settings.csv`;
- `results/pixel_retrieval_overlap_selected_scenes.csv`;
- `results/pixel_retrieval_overlap_selected_summary.csv`;
- `results/training_sample_order_of_magnitude.csv`;
- `$EMMA_GPM_DATA_ROOT/interim/orbital_pixel_feature_masks_2020.parquet`;
- `scripts/select_presentation_scenes.py`;
- `scripts/extract_orbital_pixel_feature_masks.py`;
- `scripts/plot_presentation_precipitation_maps.py`;
- `scripts/measure_pixel_retrieval_overlap.py`.

Il file pixel-level unificato contiene 162.797 righe relative alle 30 orbite.
L'ultima verifica ha dato 30 figure catalogate, nessun file mancante, nessun
`.part` residuo e 7 test superati su 7.
