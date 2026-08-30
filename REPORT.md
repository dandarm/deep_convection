# Feasibility study SEVIRI–GPM indipendente da EMMA

## Esito in breve

Il pilot **EMMA–GPM Precipitation Feature (PF)** per maggio–settembre 2020 è
completo e riproducibile. Sono stati prodotti entrambi i cataloghi richiesti,
metriche, inventario dei granuli nativi NASA e dieci controlli pixel-level.

Il risultato più importante è anche il limite principale: le frazioni 33,82%
(GMI) e 19,83% (DPR) misurano la co-occorrenza con *precipitation features*
selezionate, non la copertura completa degli swath GPM. Sono quindi **lower
bound di feature co-occurrence**; l'assenza di una feature non dimostra che il
satellite non sia passato sopra l'MCS. Il catalogo nativo full-season, necessario
per una vera metrica di overpass, resta bloccato dall'assenza di credenziali
Earthdata nell'ambiente.

## Materiale locale esaminato

All'avvio il workspace non conteneva proposal, file dedicati a Laviola o altro
materiale progettuale. Durante lo studio sono stati acquisiti e controllati:

- il codice sorgente EMMA-Tracker rilevante per i significati di `mcs_id`,
  `robust_mcs_id` e `mcs_id_merge_split`, ora archiviato fuori dal repository;
- il manuale del GPM Precipitation Feature Database, in
  `/home/daniele/Datasets/Deep_convection/references/`.

Se i riferimenti Laviola o il proposal esistono altrove, non sono stati usati e
andranno aggiunti in una revisione successiva.

## Fonti e versioni effettivamente usate

| Risorsa | Versione / formato | Uso nel pilot | Accesso verificato |
|---|---|---|---|
| EMMA European MCS climatology | v1.0; NetCDF4/HDF5 orario; 2020 | maschere `mcs_id` e `robust_mcs_id` | download pubblico Zenodo, checksum MD5 verificato |
| GPM PF Ggpf | produzione V07A archiviata nel 2022; HDF5 mensile | feature GMI nell'intero swath | archivio TAMUCC pubblico |
| GPM PF DPRrpf | produzione V07A archiviata nel 2022; HDF4 mensile | feature radar DPR | archivio TAMUCC pubblico |
| GPM PF Level-1 orbitale | V07A; HDF5 | centri pixel DPR reali nei dieci casi QC | lettura pubblica via HTTP Range |
| NASA CMR | collezioni GES DISC V07 | inventario DPR, GMI 1C e Combined | metadata pubblici, senza login |
| GES DISC / OPeNDAP | GPM V07 HDF5 | previsto per il catalogo nativo | test HTTP 401: Earthdata richiesto |

Fonti ufficiali: [dataset EMMA su Zenodo](https://zenodo.org/records/18234276),
[articolo EMMA-Tracker](https://gmd.copernicus.org/articles/19/5119/2026/),
[codice EMMA-Tracker](https://github.com/DavidKneidinger/emma-tracker/),
[directory prodotti GPM](https://gpm.nasa.gov/data/directory),
[ATBD DPR Level 2 V07A](https://gpm.nasa.gov/sites/default/files/2022-06/ATBD_DPR_V07A.pdf),
[documentazione Earthdata per download autenticato](https://urs.earthdata.nasa.gov/documentation/for_users/data_access/curl_and_wget)
e [documentazione PF](https://atmos.tamucc.edu/trmm/document.html).

### EMMA

Il file 2020 verificato dichiara:

- `source = GPM IMERG v7 Final Run; ECMWF ERA5 Reanalysis`;
- dominio ufficiale 30–70°N, 20°W–40°E, griglia regolare 0,1°;
- warm season maggio–settembre;
- soglia di fase MCS: almeno 3500 km² per almeno quattro ore continue;
- `mcs_id`: ID della traccia MCS per l'intero lifecycle;
- `robust_mcs_id`: ID presente soltanto nella fase matura/in-phase;
- `mcs_id_merge_split`: lifecycle esteso alla famiglia di merging/splitting;
- valore 0: background; ID unici solo entro il singolo anno.

Il sorgente del tracker chiarisce che la fase robusta richiede simultaneamente
soglia di area e criterio convettivo basato sul Lifted Index. L'articolo conferma
che la climatologia è costruita da IMERG Final e instabilità ERA5 e ne raccomanda
un uso climatologico/process-based, non come record meteorologico definitivo.
EMMA è quindi **weak supervision**, non ground truth fisica indipendente.

### Prodotti GPM raccomandati

1. **2A-DPR Level 2** — target radar primario. Contiene geolocazione,
   `typePrecip`, precipitation rate, echo-top/storm-top e profili di riflettività.
   L'ATBD codifica le classi maggiori come
   `typePrecip // 10_000_000`: 1 stratiform, 2 convective, 3 other; valori
   negativi indicano no-rain/missing. La riflettività deriva direttamente
   dall'eco radar ma subisce calibrazioni/correzioni; precipitation rate e
   classificazione sono retrieval algoritmici, non misure in-situ.
2. **GMI Level 1C** — target microwave preferito. Le brightness temperatures
   sono calibrate/intercalibrate e geolocalizzate; sono osservazioni indipendenti
   da SEVIRI ed EMMA. La [documentazione Level 1C NASA](https://gpm.nasa.gov/taxonomy/term/1410)
   indica GMI come riferimento di calibrazione.
3. **2B Combined DPR–GMI** — utile per precipitation rate e struttura
   combinata, ma è un retrieval vincolato congiuntamente da radar e radiometro,
   non un target osservativo puro. Il prodotto V07 è
   [GPM_2BCMB_07](https://disc.gsfc.nasa.gov/datasets/GPM_2BCMB_07/summary).
4. **GPM PF Database** — ottimo indice rapido e fonte di proprietà aggregate,
   ma non sostituisce il dato nativo. DPRrpf e Ggpf raggruppano pixel contigui con
   precipitation rate >0,1 mm h⁻¹ e salvano un'ellisse fittata. Eventuali campi
   ERA5/ambientali del database non devono essere usati come target indipendenti.

Per indipendenza si intende qui “target costruito senza SEVIRI e senza label
EMMA”. Non è indipendenza statistica assoluta: EMMA usa IMERG, che a sua volta
fonde osservazioni microwave dell'ecosistema GPM. Per il training futuro è
preferibile usare direttamente DPR e TB GMI, mantenendo EMMA soltanto come
annotazione ausiliaria.

## Pilot riproducibile

- Periodo: `2020-05-01 00:00:00Z`–`2020-09-30 23:59:59Z`.
- Dominio richiesto: 10°W–45°E, 25–55°N.
- Copertura EMMA effettiva nel dominio: centri griglia 9,95°W–40,05°E e
  29,95–54,95°N. Le feature nelle fasce 25–29,95°N e 40,05–45°E sono marcate
  `outside_emma_coverage`, non erroneamente `outside_emma`.
- Anno 2020 scelto perché è completo in EMMA e GPM e coincide con la warm
  season preferita nel protocollo.

### Definizione rigorosa del matching

1. Per ogni feature GPM si calcola l'ora EMMA più vicina.
2. `time_offset_minutes = observation_time - emma_time`; segno e valore reale
   sono sempre conservati.
3. Tolleranza inclusiva: `abs(time_offset_minutes) <= 30`. In caso esatto a
   +30 minuti viene scelta l'ora precedente, in modo deterministico.
4. Il prototipo PF confronta i centri della griglia EMMA con l'ellisse fittata
   (`R_major`, `R_minor`, orientamento) della feature. Gli assi documentati sono
   trattati come lunghezze complete e divisi per due. Se l'ellisse manca, viene
   usato soltanto il centro della feature e l'approssimazione è registrata.
5. La classe EMMA è una tra `robust_mcs`, `mcs_nonrobust_phase`,
   `outside_emma`, `outside_emma_coverage` e `outside_time_tolerance`.
6. Per la QA, invece, sono state lette le coordinate reali dei centri pixel DPR.
   Il centro pixel è dichiarato esplicitamente come approssimazione del
   footprint: non è stato usato il solo centroide dell'MCS per dichiarare hit.

## Cataloghi

### A. EMMA → GPM

`catalogs/emma_to_gpm_pf_2020.parquet` e `.csv`, una riga per `track_uid`
(`year:mcs_id`). Campi principali:

- lifecycle, ore osservate, estensione e area massima EMMA;
- presenza della fase robusta ufficiale;
- numero di feature e orbite GMI/DPR associate;
- flag di osservazione totale, in fase robusta e in sviluppo/decadimento;
- campo testuale che impedisce di interpretare l'assenza PF come no-overpass.

### B. GPM → EMMA

`catalogs/gpm_to_emma_pf_2020.parquet` e `.csv`, 126.684 righe, incluse tutte
le feature fuori EMMA. Campi principali:

- sensore, orbita, `feature_uid`, tempo, latitudine e longitudine;
- geometria PF e proprietà fisiche disponibili;
- `emma_time`, offset, copertura, `mcs_id`, `robust_mcs_id` e classe EMMA;
- GMI: area/pioggia, rain rate massimo, PCT/TB minima, IWP e geometria;
- DPR: pixel stratiformi/convettivi, pioggia per classe, near-surface
  reflectivity/precipitation, echo-top 20/30/40 dBZ;
- `dpr_dominant_class_pf` e frazione convettiva, derivate dai conteggi
  `NCONV_DPR`/`NSTRAT_DPR`. Il campo sorgente avverte che non si tratta del
  `typePrecip` nativo per pixel e che la classe `other` non è disponibile nel
  riepilogo PF.

I dieci casi DPR pixel-level estratti dal contenitore PF orbitale sono inoltre in
`$EMMA_GPM_DATA_ROOT/interim/raw_dpr_visual_case_pixels.parquet`; qui sono disponibili le tre
classi maggiori di `typePrecip` per ogni centro pixel selezionato.

## Risultati quantitativi verificati

### MCS e intercettazioni PF

| Metrica | Risultato |
|---|---:|
| MCS EMMA distinti che intersecano il pilot | 343 |
| MCS con almeno una fase robusta ufficiale nel dominio | 343 |
| Oggetti MCS orari nel dominio | 2.652 |
| Oggetti orari in fase robusta | 2.633 |
| Oggetti orari sviluppo/decadimento | 19 |
| MCS con almeno una GMI-PF associata | 116 / 343 = 33,82% |
| MCS con almeno una DPR-PF associata | 68 / 343 = 19,83% |

Queste due percentuali non sono overpass fractions complete.

### Feature dentro e fuori EMMA

| Sensore | Totale nel dominio | robust MCS | fase non robusta | fuori EMMA | fuori copertura EMMA |
|---|---:|---:|---:|---:|---:|
| GMI-PF | 92.004 | 160 | 1 | 76.068 | 15.775 |
| DPR-PF | 34.680 | 155 | 0 | 29.727 | 4.798 |

Fra le DPR-PF, la classe dominante derivata dai soli pixel classificati è
stratiform per 17.192 feature e convective per 15.222; 601 sono pari merito e
1.665 non hanno pixel convective/stratiform classificati. Questi non sono
conteggi nativi della classe `other`.

### Distribuzione mensile

| Mese 2020 | GMI-PF | DPR-PF |
|---|---:|---:|
| maggio | 22.639 | 8.177 |
| giugno | 22.975 | 8.317 |
| luglio | 17.181 | 6.931 |
| agosto | 11.700 | 4.827 |
| settembre | 17.509 | 6.428 |

I conteggi per classi EMMA e per celle geografiche 5°×5° sono in
`results/counts_by_month.csv` e `results/counts_by_geo_5deg.csv`. I bin più
popolati non vanno interpretati come climatologia: riflettono sia la frequenza
dei sistemi sia il campionamento orbitale e la selezione PF.

### Offset temporale

- minimo −29,996 min; massimo +30,000 min;
- mediana +2,169 min; media +0,928 min;
- percentile 5: −27,300 min; percentile 95: +27,017 min.

Nessun valore è stato nascosto o sostituito con il solo flag di tolleranza.

### Duplicazioni

Tra le associazioni positive:

- DPR: 74 gruppi sistema-orbita, 39 con più feature; 81 feature eccedenti la
  prima; massimo 8 feature per sistema-orbita;
- GMI: 132 gruppi sistema-orbita, 25 con più feature; 29 feature eccedenti la
  prima; massimo 3 feature per sistema-orbita.

I conteggi di “MCS intercettati” usano quindi `nunique(track_uid)` e non il
numero grezzo di feature.

## Controllo visivo

Sono stati verificati cinque hit e cinque miss, uno per mese per ciascun tipo.
Tutti i casi hanno `case_validation_passed=True`:

- nei cinque hit, da 892 a 2.489 centri DPR cadono dentro la maschera target;
- nei cinque miss, i centri DPR interni all'ellisse PF locale associati a EMMA
  sono esattamente zero;
- una stessa strisciata può colpire un MCS altrove pur contenendo una feature
  locale classificata miss. Le mappe e il catalogo mantengono distinta questa
  situazione da un vero “nessun overpass”.

Overview: `figures/qc_cases_overview.png`; metriche dei casi:
`results/qc_case_metrics.csv`; pannelli singoli: `figures/qc_cases/`.

## Accesso, volumi e tempi di download

CMR ha restituito, per il periodo/dominio richiesto, 1.054 granuli 2A-DPR,
1.258 GMI-1C e 1.081 Combined. Il poligono CMR che interseca il dominio non
implica che ogni pixel del granulo sia interno al dominio.

| Collezione nativa | Volume CMR riportato | Tempo continuo a 1 MB/s |
|---|---:|---:|
| GMI Level 1C | 38,61 GiB | circa 11,0 h |
| DPR Level 2 | 756,81 GiB | circa 9,0 giorni |
| Combined 2B | 200,25 GiB | circa 2,37 giorni |
| Totale dei tre | 995,67 GiB | circa 11,8 giorni |

Se “1 Mb/s” indica **megabit/s** e non megabyte/s, i tempi vanno moltiplicati
per otto. I circa 1,12 GB di PF effettivamente scaricati corrispondono invece a
circa 19 minuti teorici a 1 MB/s, esclusi latenza e overhead.

Il download integrale non è consigliato. Earthdata/GES DISC richiede account
gratuito, autorizzazione dell'applicazione GES DISC e cookie/token; l'OPeNDAP
anonimo testato ha risposto 401. `scripts/download_earthdata.py` legge un token
solo da `EARTHDATA_TOKEN`, scrive prima un `.part` e rinomina il file alla fine.
La strategia raccomandata è subset per dominio e variabili.

## Limiti e bias

- Selezione PF condizionata alla presenza di precipitation rate >0,1 mm h⁻¹.
- Ellisse PF fittata, non footprint/poligono nativo; può sovra- o sottostimare
  l'intersezione con una maschera irregolare.
- Il centro pixel usato nella QA non rappresenta l'intera IFOV.
- EMMA ha copertura più stretta del dominio richiesto; la distinzione
  `outside_emma_coverage` è indispensabile.
- Un offset fino a 30 minuti può essere rilevante per sistemi in rapido moto o
  forte evoluzione.
- EMMA usa precipitazione, LI ERA5, tracking e filtri lifecycle; non è una
  misura fisica indipendente.
- DPR rate e `typePrecip` sono retrieval radar; Combined e GPROF sono retrieval
  ancora più dipendenti dall'algoritmo. Le TB GMI L1C sono il target MW più
  vicino all'osservazione strumentale.
- Effetti di superficie, orografia, neve e mare modificano le firme microwave e
  devono essere stratificati nel futuro dataset.
- Pilot e PF sono V07A/V07. Non vanno mischiati silenziosamente con V08.

## Estensione al 2014–2024

Per il catalogo definitivo servono:

1. Credenziali Earthdata e autorizzazione GES DISC/PPS.
2. Una versione congelata per tutta la serie, preferibilmente V07 per coerenza
   con il pilot, oppure una re-elaborazione completa V08 con test di rottura.
3. Query CMR per anno e subset server-side di soli scan che intersecano il
   dominio e delle sole variabili richieste.
4. Checkpoint per granulo/orbita, manifest con checksum e retry, già adottati
   nel prototipo dei casi DPR.
5. Matching sui centri pixel nativi; quando disponibili, poligoni IFOV distinti
   per frequenza GMI. Non usare il centroide dell'MCS.
6. Identificatore `year:mcs_id`, perché gli ID EMMA si ripetono tra anni.
7. Cataloghi nativi separati dai PF e mantenimento di tutte le osservazioni
   GPM fuori EMMA.
8. Validazione stratificata per mese, regione, superficie e distanza dal bordo
   swath, con deduplicazione sistema-orbita.

Non viene fornita una stima del numero 2014–2024: richiede inventari CMR annuali
effettivi. Anche il tempo di processamento pixel-level non è stato estrapolato,
perché dipende dalla strategia di subset e dall'I/O disponibile.

## Implicazione per la fase modellistica

Il dataset progettato è scientificamente adatto al test successivo: congelare
l'encoder SEVIRI pre-addestrato con weak supervision EMMA e addestrare una
linear probe sulle classi DPR convective/stratiform/other. Una buona probe
indicherebbe che l'embedding contiene già informazione organizzata sulla
struttura convettiva radar. Questa fase non è stata avviata nel presente studio.
