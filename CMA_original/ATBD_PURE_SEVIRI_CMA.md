# Algoritmo NWC SAF Cloud Mask (`GEO-CMA v6.0` / `v2025`, NWC-004) in Modalità Pura 7 Canali IR SEVIRI

> Documento principale di specifica e descrizione algoritmica del pacchetto [`CMA_original/cma_nwcsaf/`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/__init__.py). La copia gemella nella cartella documentazione è disponibile in [`CMA_original/docs/ATBD_PURE_SEVIRI_CMA.md`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/docs/ATBD_PURE_SEVIRI_CMA.md).

---

## 1. Perché il Cloud Mask NWC SAF è un Detector a 3 Classi Positive (+ 1 Classe Negativa)

Nell'ecosistema EUMETSAT NWC SAF (`GEO-CMA`), oltre alla maschera binaria semplificata (`cma`: `0 = Cloud-free`, `1 = Cloudy`), il prodotto Cloud Mask definisce **due rappresentazioni a 4 livelli** costituite da **1 classe negativa (`0 = Cloud-free`)** e **3 classi positive**:

### A. Categorie di Copertura Nuvolosa (`cma_cover` — ATBD Sezione 2.2.1.2.2, p. 13 e MPEF/NWC SAF classico)
È la classificazione fisica della frazione/opacità di copertura del pixel:
- **`0 = CLOUD_FREE` (Classe Negativa):** Pixel libero da nubi e libero da neve/ghiaccio.
- **`1 = CLOUD_CONTAMINATED` (Classe Positiva 1 — Parzialmente nuvoloso / Sub-pixel / Semitrasparente):** Pixel contaminato da nubi sub-pixel, bordi nuvolosi eterogenei, cirri sottili semitrasparenti in cristalli di ghiaccio, nubi sottili sopra neve/ghiaccio, o nubi incipienti intercettate dai test di tessitura spaziale ($3\times 3$) o di raffreddamento temporale rapido ($15\text{ min}$) o prossime alla soglia radiometrica (entro il margine di sicurezza).
- **`2 = CLOUD_FILLED` (Classe Positiva 2 — Pienamente coperto da nube opaca):** Pixel completamente riempito da un corpo nuvoloso opaco che supera nettamente il margine di sicurezza radiometrico nei test termici/assorbimento ($\text{IR}_{10.8}$, $\text{SST}$, $\text{CO}_2\ 13.4\,\mu\text{m}$, $\text{WV}\ 7.3\,\mu\text{m}$, o strato basso opaco $\text{IR}_{8.7}-\text{IR}_{10.8}$) ed è circondato da un intorno $3\times 3$ prevalentemente nuvoloso ($\ge 5/9$ pixel).
- **`3 = SNOW_OR_ICE` (Classe Positiva 3 — Contaminato da Neve o Ghiaccio marino):** Pixel privo di nubi ma coperto da neve continentale su terra o ghiaccio marino su mare.

### B. Contenitore Cloud & Snow v2025 (`cma_cloudsnow` — ATBD Sezione 2.2.2.4, Tabella 4, p. 59)
È il contenitore ufficiale introdotto per gli utenti neve/criosfera in `GEO-CMA v2025`:
- **`0 = CLOUD_FREE` (Classe Negativa):** Cielo sereno su superficie libera da neve/ghiaccio.
- **`1 = CLOUD` (Classe Positiva 1):** Nube (tutte le nubi opache, semitrasparenti o sub-pixel, ad eccezione dei cirri sottili sopra neve/ghiaccio).
- **`2 = THIN_ICE_CLOUD_OVER_SNOW` (Classe Positiva 2):** Nube sottile di ghiaccio sovrapposta a superficie innevata o ghiacciata ($\text{BTD}_{10.8-12.0} > 1.35\text{ K}$ e $\text{BTD}_{7.3-10.8} > -12.0\text{ K}$ su mappa neve/ghiaccio attiva).
- **`3 = SNOW_OR_ICE` (Classe Positiva 3):** Neve continentale o ghiaccio marino in cielo sereno.

---

## 2. Riepilogo dei Moduli in `CMA_original`

- [`cma_nwcsaf/config.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py): Enumerazioni [`CmaBinaryClass`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py#L28-L33), [`CmaCoverClass`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py#L36-L49), [`CmaCloudSnowClass`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py#L52-L63), [`CmaDustClass`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py#L66-L72), [`CmaVolcanicClass`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py#L75-L81), [`CmaSmokeClass`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py#L84-L90), [`CmaQuality`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py#L93-L100), [`CmaStatusBit`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py#L103-L116), [`CmaTestBit`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py#L119-L149) e dataclass [`NwcSafCmaConfig`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/config.py#L152-L269).
- [`cma_nwcsaf/ancillary.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/ancillary.py): Calcolo di $\sec\theta_{\text{sat}}$ e stima autonoma causale 7 canali IR di $T_{\text{surf}}$ e del profilo troposferico.
- [`cma_nwcsaf/detector.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/detector.py): Esecuzione dei 12 passi di `GEO-CMA v6.0` su singolo frame (`run_cma_frame`).
- [`cma_nwcsaf/pipeline.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/pipeline.py): Tracker causale multi-temporale [`NwcSafCmaTracker`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/pipeline.py#L54-L148) e runner su sequenze [`run_nwcsaf_cma_sequence`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/CMA_original/cma_nwcsaf/pipeline.py#L151-L216).
