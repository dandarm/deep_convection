# `CMA_original` — EUMETSAT NWC SAF Cloud Mask (`GEO-CMA v6.0` / `v2025`, NWC-004)

Implementazione autonoma e fedele all'ATBD ufficiale del detector **Cloud Mask (`GEO-CMA v6.0`, `v2025`)** di EUMETSAT NWC SAF (<https://www.nwcsaf.org/cma_v2025>), progettata per operare sia con campi ancillari NWP sia in **modalità pura 7 canali IR SEVIRI H24** (`WV_062`, `WV_073`, `IR_087`, `IR_097`, `IR_108`, `IR_120`, `IR_134`).

## Le 3 Classi Positive (+ 1 Classe Negativa `0 = Cloud-free`)

Il pacchetto espone esplicitamente entrambe le classificazioni a 4 categorie definite nell'ATBD NWC SAF:

1. **`cma_cover` ([`CmaCoverClass`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py#L36-L49) — ATBD Sezione 2.2.1.2.2, p. 13):**
   - `0 = CLOUD_FREE` *(Classe negativa)*: Cielo sereno
   - `1 = CLOUD_CONTAMINATED` *(Classe positiva 1)*: Parzialmente nuvoloso / sub-pixel / bordo nube / cirro sottile semitrasparente
   - `2 = CLOUD_FILLED` *(Classe positiva 2)*: Pienamente coperto da nube opaca ad alta confidenza
   - `3 = SNOW_OR_ICE` *(Classe positiva 3)*: Neve continentale o ghiaccio marino in cielo sereno

2. **`cma_cloudsnow` ([`CmaCloudSnowClass`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py#L52-L63) — ATBD Sezione 2.2.2.4, p. 59):**
   - `0 = CLOUD_FREE` *(Classe negativa)*: Cielo sereno
   - `1 = CLOUD` *(Classe positiva 1)*: Nube (eccetto cirro sottile sopra neve/ghiaccio)
   - `2 = THIN_ICE_CLOUD_OVER_SNOW` *(Classe positiva 2)*: Cirro sottile di ghiaccio sopra neve/ghiaccio
   - `3 = SNOW_OR_ICE` *(Classe positiva 3)*: Neve o ghiaccio marino

Oltre a:
- **`cma`** ([`CmaBinaryClass`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py#L28-L33)): `0 = CLOUD_FREE`, `1 = CLOUDY`
- **`cma_dust`**, **`cma_volcanic`**, **`cma_smoke`**: Rilevatori di plume di polvere desertica, cenere vulcanica e fumo (`0..2`)
- **`cma_testlist`**: Bitmask a 28 bit dei test attivati ([`CmaTestBit`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py#L119-L149))
- **`cma_status_flag`** (`11 bit`) e **`cma_quality`** (`0..4`).

## Struttura della Directory

- [`cma_nwcsaf/`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/__init__.py): Pacchetto Python del detector `GEO-CMA v6.0`
- [`tests/test_cma_nwcsaf.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/tests/test_cma_nwcsaf.py): Suite di test fisici e sintetici per tutte le classi e i bit del detector
- [`scripts/benchmark_cma_on_seviri.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/scripts/benchmark_cma_on_seviri.py): Benchmark empirico su clip SEVIRI reali a 7 canali
- [`docs/ATBD_PURE_SEVIRI_CMA.md`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/docs/ATBD_PURE_SEVIRI_CMA.md): Descrizione teorica e matematica dell'algoritmo scritto
- [`docs/CMA_ATBD_SPECIFICATION.md`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/docs/CMA_ATBD_SPECIFICATION.md): Specifica ufficiale delle classi e dei 28 bit `GEO-CMA v2025`

## Verifica e Benchmark da PowerShell

```powershell
$env:PYTHONPATH = "$(Resolve-Path 'src').Path;$(Resolve-Path 'CMA_original').Path"
.\.venv\Scripts\python.exe -m pytest -q CMA_original/tests/test_cma_nwcsaf.py
.\.venv\Scripts\python.exe CMA_original/scripts/benchmark_cma_on_seviri.py
```
