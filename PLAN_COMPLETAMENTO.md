# Piano di completamento - feasibility study EMMA-GPM

Ultimo aggiornamento: 2026-08-30, Europe/Rome. Il pilot PF è completato; la
fase nativa full-season resta una estensione separata bloccata da Earthdata.

## Stato persistente raggiunto

- EMMA 2020 ufficiale scaricato da Zenodo, checksum MD5 verificato ed estratto:
  `$EMMA_GPM_DATA_ROOT/raw/emma/` e `$EMMA_GPM_DATA_ROOT/interim/emma/2020/`.
- 3.672 maschere orarie maggio-settembre lette e validate.
- Tabella degli oggetti orari e lifecycle EMMA prodotta:
  `$EMMA_GPM_DATA_ROOT/interim/emma_hourly_objects_2020.parquet` e
  `$EMMA_GPM_DATA_ROOT/interim/emma_lifecycle_2020.parquet`.
- Cinque mesi GPM GMI 1C-PF e DPRrpf scaricati (circa 1,1 GB complessivi):
  `$EMMA_GPM_DATA_ROOT/raw/gpm_pf_1c/` e `$EMMA_GPM_DATA_ROOT/raw/gpm_pf_dpr/`.
- Feature nel dominio preparate: 92.004 GMI-PF e 34.680 DPR-PF.
- Inventario NASA CMR V07 prodotto: 1.054 granuli DPR, 1.258 GMI e 1.081
  Combined che intersecano il dominio/periodo secondo i poligoni CMR.
- Cataloghi PF preliminari prodotti:
  `catalogs/gpm_to_emma_pf_2020.*` e `catalogs/emma_to_gpm_pf_2020.*`.
- Metriche e diagnostiche aggregate prodotte in `results/` e `figures/`:
  `metrics_summary.csv`, conteggi mensili/geografici, offset temporali e
  duplicazioni per sistema-orbita.
- Test locali presenti in `tests/test_matching.py`: 3 test superati.
- Accesso NASA GES DISC/OPeNDAP testato: risposta HTTP 401, perché non sono
  presenti credenziali Earthdata nell'ambiente.

## QA pixel-level completata

- Estrazione via HTTP Range completata per 7 orbite uniche usate nei dieci casi.
- I checkpoint per orbita sono in
  `$EMMA_GPM_DATA_ROOT/interim/raw_dpr_case_orbits/`; il file unificato è
  `$EMMA_GPM_DATA_ROOT/interim/raw_dpr_visual_case_pixels.parquet`.
- Il manifest dei casi è `catalogs/visual_case_manifest_2020.csv`.
- Le dieci mappe definitive distinguono rigorosamente: hit = centri DPR dentro
  la maschera target; miss = centri DPR locali entro l'ellisse della singola
  precipitation feature, senza sovrapposizione EMMA.
- Tutti i cinque hit e cinque miss hanno `case_validation_passed=True` e sono
  stati verificati visivamente nell'overview.

## Chiusura del pilot PF

- Cataloghi A e B: completati in CSV e Parquet.
- Metriche mensili, geografiche, offset e duplicazioni: completate.
- Dieci mappe QC: completate e validate.
- `REPORT.md` e `README.md`: completati.
- Smoke test: compilazione completata e 3/3 test superati.

La riproduzione da zero e i comandi di download sono documentati in `README.md`.

## Estensione nativa DPR/GMI (blocco esterno)

Per il catalogo definitivo footprint/pixel di tutta la warm season servono un
account Earthdata gratuito e un token esportato come `EARTHDATA_TOKEN`. Non
salvare il token nel repository.

1. Scaricare o subsettare i granuli dell'inventario CMR con
   `scripts/download_earthdata.py` oppure OPeNDAP autenticato.
2. Preferire subset per bounding box/variabili e non granuli interi: i 1.054
   granuli DPR completi riportano circa 0,7 GB ciascuno, quindi il trasferimento
   integrale sarebbe dell'ordine di 0,8 TB.
3. Eseguire il matching sui centri pixel reali e produrre cataloghi nativi
   separati dai PF.
4. Per GMI, conservare separatamente le geolocazioni dei diversi canali/IFOV e
   non trattare le TB come retrieval ERA5-independent quando si usano campi
   GPROF ausiliari.

## Criterio di completamento

Il pilot PF soddisfa il criterio: cataloghi, metriche, 10 mappe e report sono
riproducibili e verificati. Il pilot nativo sarà completo solo dopo avere ottenuto
i subset autenticati e ricalcolato le metriche di intercettazione usando pixel o
footprint reali; fino ad allora le frazioni PF sono lower bound di co-occorrenza,
non coverage satellitare.
