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
D:\Datasets\Deep_convection
```

Impostare la variabile per la sessione PowerShell corrente:

```powershell
$env:EMMA_GPM_DATA_ROOT = "D:\Datasets\Deep_convection"
```

Per renderla persistente per l'utente:

```powershell
[Environment]::SetEnvironmentVariable(
  "EMMA_GPM_DATA_ROOT",
  "D:\Datasets\Deep_convection",
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
> HANDOFF_WINDOWS.md, usando `D:\Datasets\Deep_convection` come
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
