# NWC SAF Cloud Mask (`GEO-CMA v6.0`) & Cloud Type (`GEO-CT v5.0`) — Implementazione Python (`CT_original`)

Implementazione fedele, modulare e autonoma in Python dei prodotti operativi **Cloud Mask (`GEO-CMA v6.0`, NWC-004)** e **Cloud Type (`GEO-CT v5.0`, NWC-008)** di **EUMETSAT NWC SAF (`v2025`)**, derivata direttamente dai documenti ufficiali ATBD (*Algorithm Theoretical Basis Document*, Issue 1.1.1, 30 Maggio 2025) e UM (*User Manual*, Issue 1.1.1, 30 Maggio 2025) pubblicati su <https://www.nwcsaf.org/cma_v2025>.

---

## Struttura della Directory `CT_original/`

- [`ct_nwcsaf/`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/ct_nwcsaf/__init__.py): Pacchetto Python principale
  - [`config.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/ct_nwcsaf/config.py): Enumerazioni ufficiali NWC SAF (`CmaClass`, `CmaCloudSnowClass`, `CmaDustClass`, `CmaVolcanicClass`, `CmaSmokeClass`, `CmaTestBit` a 28 bit, `CmaStatusBit` a 11 bit, `CmaQuality`, `CtClass` a 15 classi, `CtCumuliformClass`, `CtMultilayerClass`, `CtStatusBit` a 6 bit, `RdtBridgeCtClass` a 6 classi) e configurazione [`NwcSafCmaCtConfig`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/ct_nwcsaf/config.py#L148-L296).
  - [`ancillary.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/ct_nwcsaf/ancillary.py): Calcolo della secante dell'angolo zenitale satellitare $\sec\theta_{\text{sat}}$, maschera costiera morfologica, ingestione opzionale di campi NWP ($T_{\text{surf}}, T_{950}, T_{850}, T_{700}, T_{500}, T_{\text{tropo}}, \text{TCWV}$) e recupero autonomo Pure-SEVIRI a 7 canali IR con tracker causale EMA della temperatura superficiale.
  - [`cma.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/ct_nwcsaf/cma.py): Processore completo **NWC SAF `GEO-CMA v6.0`** (test SST split-window, contrasto termico $T_{10.8}$ con correzione *extreme cooling*, split-window $T_{10.8}-T_{12.0}$, fase ghiaccio $T_{8.7}-T_{10.8}$, nubi basse ad alto angolo $T_{10.8}-T_{8.7}$, test spettrale IR a 7 canali per nubi basse/medie, tessitura spaziale locale $3\times 3$ omogenea per tipo di superficie, differenza temporale a 15 minuti $\Delta_{15\text{min}}T_{10.8}$, coerenza a 1 ora con *region growing* 8-connesso, *spatial filtering* finale per inversioni notturne fredde $T_{7.3}-T_{10.8} > 0.5\text{ K}$ / coste / bordi neve / pixel isolati, rilevamento polvere sahariana `cma_dust` e cenere vulcanica `cma_volcanic`, bitmask a 28 bit `cma_testlist` e qualità `cma_quality`).
  - [`ct.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/ct_nwcsaf/ct.py): Processore completo **NWC SAF `GEO-CT v5.0` a 15 classi** (soglie verticali dinamiche `maxT108vh`, `maxT108hi`, `maxT108med`, `maxT108low`, `T108interthr`, `T108T73thrlow = 4.0·sec + 8.5 K`, separazione nubi semitrasparenti sottili/medie/spesse, nubi frazionarie e nubi **multilayer `Classe 14`**, guardia del vapore acqueo $T_{10.8}-T_{7.3}$ e riclassificazione per inversione termica nei bassi strati tramite $T_{8.7}-T_{10.8}$, container `ct_cumuliform`, `ct_multilayer`, `ct_quality` con margine di sicurezza di $0.2\text{ K}$ e ponte a 6 classi `rdt_bridge_class`).
  - [`pipeline.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/ct_nwcsaf/pipeline.py): Tracker stateful [`NwcSafCmaCtTracker`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/ct_nwcsaf/pipeline.py#L70-L162) e runner di sequenza [`run_nwcsaf_cma_ct_sequence`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/ct_nwcsaf/pipeline.py#L165-L190).
- [`docs/`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/docs):
  - [`ATBD_PURE_SEVIRI_CMA_CT.md`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/docs/ATBD_PURE_SEVIRI_CMA_CT.md): Descrizione teorica e algoritmica completa del codice sviluppato (in italiano, con diagrammi Mermaid, equazioni e tabelle).
  - [`CMA_CT_ATBD_SPECIFICATION.md`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/docs/CMA_CT_ATBD_SPECIFICATION.md): Specifica matematica autorevole estratta dall'ATBD e dall'User Manual ufficiali NWC SAF v2025.
- [`scripts/benchmark_ct_on_seviri.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/scripts/benchmark_ct_on_seviri.py): Script di benchmark empirico su sequenze SEVIRI reali a 7 canali del 2020 e confronto con il pre-filtro leggero a 6 classi in `src/emma_gpm/rdt/cloud_type_ir.py` e `RDT_original`.
- [`tests/test_ct_nwcsaf.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CT_original/tests/test_ct_nwcsaf.py): Suite di test unitari e sintetici che verifica tutte le 15 classi, le equazioni dell'ATBD, i filtri spaziali/temporali, i flag di qualità e il rilevamento di polvere e cenere vulcanica.

---

## Esecuzione Rapida di Test e Benchmark

Dalla radice del repository in PowerShell:

```powershell
$env:PYTHONPATH = "$((Resolve-Path 'src').Path);$((Resolve-Path 'CT_original').Path)"
.\.venv\Scripts\python.exe -m pytest CT_original/tests/test_ct_nwcsaf.py -v
.\.venv\Scripts\python.exe CT_original/scripts/benchmark_ct_on_seviri.py
```
