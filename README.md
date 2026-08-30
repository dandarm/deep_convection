# EMMA–GPM Mediterranean feasibility pilot

Pipeline riproducibile per costruire cataloghi di co-occorrenza fra EMMA e GPM
nel dominio 10°W–45°E, 25–55°N, maggio–settembre 2020. Non addestra modelli.

Il risultato corrente è un pilot basato su GPM Precipitation Features. Le
frazioni PF sono lower bound di co-occorrenza, non copertura completa degli
swath. Leggere `REPORT.md` prima di usare i cataloghi.

I raw e gli intermedi non sono versionati. Prima di eseguire la pipeline:

```bash
export EMMA_GPM_DATA_ROOT=/home/daniele/Datasets/Deep_convection
```

Senza questa variabile, il fallback è la directory locale `data/`, mantenuta
fuori da Git tramite `.gitignore`.

## Ambiente

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`pyhdf` richiede il runtime HDF4; l'ambiente locale verificato è già presente
solo se è stato creato seguendo i comandi sopra.

## Riproduzione del pilot PF

Con i download già presenti:

```bash
.venv/bin/python scripts/prepare_emma.py
.venv/bin/python scripts/prepare_gpm_pf.py
.venv/bin/python scripts/build_catalogs.py
MPLCONFIGDIR=/tmp/mplconfig-emma-gpm .venv/bin/python scripts/compute_metrics.py
```

Download da zero:

```bash
.venv/bin/python scripts/download_emma.py
.venv/bin/python scripts/download_gpm_pf.py
.venv/bin/python scripts/query_cmr.py
```

Il download PF usa file mensili pubblici. L'inventario CMR non richiede login;
i file nativi GES DISC sì.

## QA DPR pixel-level

```bash
.venv/bin/python scripts/extract_raw_dpr_cases.py
MPLCONFIGDIR=/tmp/mplconfig-emma-gpm .venv/bin/python scripts/plot_qc_cases.py
```

L'estrazione salva un Parquet per orbita in
`data/interim/raw_dpr_case_orbits/`. Se viene interrotta, al riavvio salta le
orbite già completate.

## Download nativo autenticato

Creare un account Earthdata, autorizzare GES DISC e impostare il token senza
salvarlo nel repository:

```bash
export EARTHDATA_TOKEN='...'
.venv/bin/python scripts/download_earthdata.py \
  "$EMMA_GPM_DATA_ROOT/interim/cmr_gpm_v07_mjjas2020.parquet" \
  "$EMMA_GPM_DATA_ROOT/raw/gpm_native/dpr" \
  --collection DPR_2A --limit 1
```

Rimuovere il token dall'ambiente dopo l'uso. Preferire sempre il subset
server-side: il set completo dei granuli DPR inventariati è circa 756,8 GiB.

## Output principali

- `catalogs/emma_to_gpm_pf_2020.parquet`: una riga per MCS EMMA.
- `catalogs/gpm_to_emma_pf_2020.parquet`: tutte le feature GPM, incluse quelle
  fuori EMMA.
- `catalogs/gpm_granule_inventory_cmr_v07_2020.csv`: inventario nativo NASA.
- `results/metrics_summary.csv`: metriche sintetiche.
- `results/counts_by_month.csv` e `counts_by_geo_5deg.csv`.
- `results/qc_case_metrics.csv`: esito dei dieci controlli.
- `figures/qc_cases_overview.png` e `figures/qc_cases/`.

## Test

```bash
.venv/bin/python -m compileall -q src scripts tests
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Configurazione del pilot: `config/pilot_2020.json`. Stato e percorso di
estensione: `PLAN_COMPLETAMENTO.md`.
