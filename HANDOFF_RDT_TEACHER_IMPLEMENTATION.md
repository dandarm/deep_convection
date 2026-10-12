# Handoff tecnico: teacher RDT-CW-inspired multispettrale

Questo documento definisce il lavoro per implementare il teacher deterministico
usato nel probing del VideoMAE. La specifica incorpora la decisione scientifica
più recente: il teacher principale usa tutti i sette canali SEVIRI per stimare
un riferimento radiometrico UTLS, mentre IR 10.8 µm continua a governare la
geometria delle celle.

La prima implementazione deve rimanere esplicitamente **RDT-CW-inspired**. Non
deve essere presentata come replica esatta del prodotto operativo.

## 1. Decisione che sostituisce il piano precedente

Le versioni correnti di `README.md`, `RDT_plan.md` e `AGENTS.md` descrivono un
teacher limitato a WV 6.2 e IR 10.8 e includono soglie assolute a 240, 230, 220,
210 e 215 K. La decisione aggiornata è:

- rimuovere la scala fissa 240/230/220/210 K dalla segmentazione principale;
- sostituire il limite `IR 10.8 <= 215 K` dell'OT con un criterio relativo
  all'incudine e al proxy UTLS;
- usare tutti i sette canali nel teacher principale;
- conservare il teacher fisso a due canali come baseline e ablation separata;
- mantenere invariati il significato delle cinque classi, il tracking per
  sovrapposizione e la distinzione temporale sviluppo versus maturità/decadimento.

Prima del merge dell'implementazione, aggiornare i tre documenti sopra per
eliminare la contraddizione. Il pretraining VideoMAE rimane completamente
separato e non deve essere modificato.

## 2. Input e invarianti

Il teacher riceve brightness temperature non normalizzate in kelvin:

```text
clip_tchw: float32, shape (16, 7, H, W)
time: 16 timestamp crescenti, cadenza 5 minuti
channel order:
  WV_062, WV_073, IR_087, IR_097, IR_108, IR_120, IR_134
orientation: north-up, west-left
```

Usare `SeviriZarrClipReader` come punto di ingresso, ma chiamarlo senza
`global_mean` e `global_std`. La normalizzazione globale serve al VideoMAE, non
al teacher fisico. Rifiutare clip con frame o canali mancanti, tempo non
crescente, valori non finiti o ordine spettrale diverso.

Quando disponibile, leggere dalle coordinate proiettate l'area reale di ogni
pixel. Non convertire il limite di 100 km² in un numero fisso di pixel senza
tenere conto della geometria geostazionaria.

## 3. Output richiesti

Output minimo per ogni clip:

```text
semantic_label: uint8  (T,H,W), valori 0..4
cell_id:        int32  (T,H,W), 0 fuori dalle celle
thermal_depth:  float32(T,H,W), D o NaN fuori dalle celle
t_edge_k:       float32(T,H,W), valore della cella o NaN
t_utls_proxy_k: float32(T,H,W), valore della cella o NaN
proxy_confidence: float32(T,H,W), intervallo [0,1]
ot_mask:        bool   (T,H,W)
```

Produrre inoltre una tabella per oggetto e timestamp con almeno:

```text
time, cell_id, track_id, parent_track_ids, child_track_ids,
area_km2, t_min_k, t_mean_k, t_edge_k, t_utls_proxy_k,
proxy_confidence, contrast_k, cooling_15min_k, area_change_15min_km2,
aor_from_parent, lifecycle_state, config_hash
```

Gli array possono essere salvati in Zarr e la tabella in Parquet. Questi
artefatti sono dataset intermedi e non vanno aggiunti a Git.

## 4. Pipeline deterministica

### 4.1 Feature spettrali

Calcolare in kelvin, senza standardizzazione per canale:

```text
BTD_062_108 = WV_062 - IR_108
BTD_073_108 = WV_073 - IR_108
BTD_087_108 = IR_087 - IR_108
BTD_097_108 = IR_097 - IR_108
BTD_108_120 = IR_108 - IR_120
BTD_134_108 = IR_134 - IR_108
```

Conservare sia i campi sia statistiche robuste per cella. Non assumere che
tutte le differenze debbano avere lo stesso segno. Le firme dipendono da
opacità, angolo di vista e stato atmosferico.

### 4.2 Rilevamento adattivo delle celle

Usare IR 10.8 per la topologia:

1. Applicare un leggero smoothing configurabile soltanto alla copia usata per
   trovare la topologia. Temperature e diagnostiche devono usare i dati grezzi.
2. Individuare i minimi regionali significativi. Il valore iniziale candidato
   per la prominence è 6 K, coerente con la separazione termica descritta
   nell'ATBD RDT-CW, ma deve rimanere configurabile.
3. Costruire un min-tree oppure una watershed marker-controlled della
   superficie `IR_108`.
4. Far crescere ogni minimo verso temperature più calde. `T_edge` è il livello
   più caldo che mantiene la torre distinta dai minimi vicini, cioè il livello
   immediatamente precedente al saddle di fusione.
5. Per una torre isolata, non lasciare che il bacino occupi l'intero dominio.
   Fermarlo al bordo di supporto nuvoloso/ghiacciato ricavato dalle feature
   multispettrali e dal massimo gradiente termico chiuso.
6. Applicare 8-connettività e scartare le componenti sotto `min_area_km2`, con
   valore iniziale 100 km².

Il punto 5 è il principale elemento da validare. Non introdurre di nascosto
una nuova soglia termica universale per risolverlo. Se il bordo esterno non è
determinabile, assegnare bassa confidenza o rifiutare l'oggetto.

### 4.3 Selezione del plateau dell'incudine

Per ogni cella e timestamp costruire un insieme di pixel candidati `A`:

- escludere il nucleo più freddo, inizialmente il decile inferiore di IR 10.8;
- escludere una fascia configurabile a contatto con `T_edge`;
- preferire pixel con basso gradiente spaziale di IR 10.8;
- quando esiste una storia, preferire pixel con piccola variazione temporale;
- richiedere coerenza spettrale nelle sei BTD.

Per la prima versione, la coerenza spettrale può essere definita in modo
robusto e non supervisionato: calcolare mediana e MAD del vettore BTD nella
parte centrale della cella e rimuovere gli outlier multivariati. In questo modo
tutti i canali contribuiscono a identificare l'incudine senza inventare pesi
fisici non ancora calibrati.

L'insieme `A` deve essere esteso, connesso e non dominato da pochi pixel. Se il
numero o l'area dei pixel validi è insufficiente, il proxy è mancante o a bassa
confidenza. Non sostituirlo con il minimo IR 10.8.

### 4.4 Proxy UTLS versione 0

Definire:

```text
T_anvil = weighted_median(IR_108[p] for p in A)
```

I pesi devono diminuire con il gradiente spaziale e con il valore assoluto
della tendenza temporale. Limitare i pesi per evitare che un singolo pixel
domini la mediana.

La versione iniziale usa:

```text
T_UTLS_proxy_v0 = T_anvil
```

Le feature WV/O₃/CO₂ partecipano alla selezione e alla confidenza del plateau,
ma non si devono inventare coefficienti additivi. Preparare un'interfaccia
`spectral_correction(features) -> delta_k` che restituisca zero nella versione
0. Una correzione diversa da zero richiederà calibrazione e validazione
scientifica separata.

Regolarizzare il proxy lungo ogni track con una mediana temporale corta e
causale. Registrare sempre sia il valore istantaneo sia quello filtrato.

La confidenza deve diminuire quando:

- l'area del plateau è piccola;
- la dispersione IR 10.8 è elevata;
- le feature spettrali sono eterogenee;
- il proxy cambia troppo rapidamente;
- `T_edge - T_UTLS_proxy` è troppo piccolo per una normalizzazione stabile.

Non chiamare questo valore temperatura di tropopausa WMO. Il nome corretto è
**SEVIRI-derived UTLS/anvil temperature proxy**.

### 4.5 Profondità termica normalizzata

Per ogni pixel appartenente alla cella:

```text
D = (T_edge - IR_108) / (T_edge - T_UTLS_proxy)
```

Verificare che il denominatore sia positivo e sufficientemente grande. In caso
contrario impostare `D = NaN`, abbassare la confidenza e impedire l'override OT.

Interpretazione:

- `D` vicino a 0: bordo della torre;
- `0 < D < 1`: livelli progressivamente più profondi;
- `D` vicino a 1: incudine prossima al riferimento UTLS;
- `D > 1`: candidato più freddo dell'incudine, non ancora OT confermato.

### 4.6 Tracking

Mantenere la metrica prevista:

```text
AOR(A,B) = area(A intersect B) / min(area(A), area(B))
link se AOR >= 0.15
```

Usare l'area fisica quando varia nel dominio. Conservare tutti i candidati che
superano la soglia per descrivere split e merge. Assegnare un `track_id`
stabile e relazioni padre-figlio esplicite.

Cooling ed espansione a 15 minuti devono confrontare oggetti collegati a tre
frame di distanza, non confondere un incremento a 5 minuti con una misura a
15 minuti. Nei primi tre frame produrre valori mancanti, senza replicare dati.

### 4.7 Classi semantiche

La semantica resta:

```text
0 background
1 nube fredda/non convettiva
2 cella convettiva in sviluppo
3 cella convettiva matura o in decadimento
4 Overshooting Top
```

Routing:

1. Fuori dalle celle adattive: Classe 0.
2. Una nuova cella entra in Classe 2 se supera i criteri configurati di
   contrasto, cooling e crescita. I valori iniziali restano 10 K,
   -2.5 K/15 min e area change positiva.
3. Una cella già convettiva passa o rimane in Classe 3 quando il cooling
   rallenta, si stabilizza o diventa warming. Non richiedere nuovamente il
   criterio di cooling rapido, altrimenti la Classe 3 sarebbe irraggiungibile.
4. Le celle che non entrano nello stato convettivo sono Classe 1.
5. La Classe 4 sovrascrive soltanto pixel di celle in Classe 2 o 3.

L'override OT richiede congiuntamente:

- `D > 1` con proxy valido e sufficientemente confidente;
- minimo locale e contrasto rispetto al plateau dell'incudine;
- firma multispettrale coerente, includendo BTD 6.2–10.8, 9.7–10.8 e
  13.4–10.8;
- limiti configurabili di area e morfologia;
- persistenza temporale configurabile oppure un flag `provisional_ot` al primo
  frame e `confirmed_ot` dopo la persistenza richiesta.

`BTD_062_108 >= 1 K` può rimanere come valore iniziale candidato. Non usare
più `IR_108 <= 215 K` nel teacher adattivo.

## 5. Architettura software proposta

Preferire un package separato:

```text
src/emma_gpm/rdt/
  __init__.py
  config.py
  features.py
  detection.py
  utls_proxy.py
  tracking.py
  lifecycle.py
  pipeline.py

scripts/build_rdt_pseudolabels.py

tests/test_rdt_features.py
tests/test_rdt_detection.py
tests/test_rdt_utls_proxy.py
tests/test_rdt_tracking.py
tests/test_rdt_lifecycle.py
tests/test_rdt_pipeline.py
```

API pubblica minima:

```python
result = build_rdt_teacher_labels(
    clip_tchw,
    times=times,
    x=x,
    y=y,
    channels=THERMAL_CHANNELS,
    config=config,
)
```

Usare dataclass congelate per configurazione e risultati. Salvare nei metadati
la configurazione serializzata, l'hash, il commit Git e la versione
dell'algoritmo. Non inserire stato globale o soglie nascoste nel codice.

Prevedere almeno due modalità:

```text
multispectral_utls_v0   teacher principale aggiornato
legacy_fixed_2ch        baseline del piano iniziale
```

## 6. Test obbligatori

### Test sintetici

- Un minimo isolato con anelli concentrici produce una sola cella e il
  `T_edge` atteso.
- Due minimi separati da un saddle producono due celle fino al livello di
  fusione corretto.
- Traslazione di tutte le temperature di una costante lascia invariati
  topologia e `D`, entro la tolleranza numerica.
- L'aggiunta di un piccolo nucleo più freddo non trascina il proxy verso il
  minimo OT.
- Canali riordinati, NaN, tempi mancanti e denominatore non valido vengono
  rifiutati o marcati esplicitamente.
- Split, merge e AOR hanno casi unitari con lineage noto.
- La Classe 3 è raggiungibile da una track precedentemente in sviluppo.
- L'OT non può apparire fuori da Classi 2 e 3.

### Test su casi reali

- Salvare diagnostiche per alcuni episodi con celle isolate, sistemi
  multicella, cirri sottili e bordi del disco.
- Visualizzare IR 10.8, cell ID, `T_edge`, plateau, proxy, `D` e label finale.
- Controllare stabilità del proxy frame per frame e distribuzione della
  confidenza.
- Confrontare sempre il teacher adattivo con `legacy_fixed_2ch`.

Le figure e gli output diagnostici restano fuori da Git.

## 7. Criteri di accettazione della prima implementazione

- Pipeline deterministica a parità di input e configurazione.
- Nessuna dipendenza da EMMA, GPM, OPERA, radar, NWP o pseudo-label esterne.
- Input esclusivamente SEVIRI in kelvin e nell'ordine canonico a sette canali.
- Nessuna soglia assoluta 240/230/220/210/215 K nel teacher principale.
- Output completi, con confidenza e diagnostiche intermedie ispezionabili.
- Test sintetici e suite esistente tutti verdi.
- README, piano RDT e AGENTS coerenti con il nuovo teacher multispettrale.
- Il pretraining VideoMAE e i suoi manifest restano invariati.

Comandi minimi di verifica:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe -m pytest -q
```

## 8. Rischi scientifici e controlli

Il teacher multispettrale usa gli stessi sette canali disponibili al probe.
Questo migliora il fondamento fisico delle pseudo-label, ma rende più facile
imparare scorciatoie spettrali. Il protocollo deve quindi conservare:

- teacher legacy a due canali come baseline;
- probe con soli WV 6.2 e IR 10.8;
- probe con tutti i sette canali;
- ablazione senza IR 13.4;
- frame order shuffled e repeated single-frame;
- encoder casuale, RGB adattato e SEVIRI preaddestrato con decoder identico.

Una buona segmentazione dimostra decodificabilità del teacher. Non basta per
dimostrare apprendimento della termodinamica reale.

## 9. Riferimenti

1. NWC SAF, *Algorithm Theoretical Basis Document for Convection Products*:
   <https://www.nwcsaf.org/AemetWebContents/ScientificDocumentation/Documentation/GEO/v2016/NWC-CDOP2-GEO-MFT-SCI-ATBD-Convection_v1.1.pdf>
2. ESA, *The MSG System*:
   <https://www.esa.int/esapub/bulletin/bullet111/chapter3_bul111.pdf>
3. Mikuš and Strelec Mahović, 2013:
   <https://doi.org/10.1016/j.atmosres.2012.09.001>
4. Bedka et al., 2010:
   <https://doi.org/10.1175/2009JAMC2286.1>
5. Bedka, 2011:
   <https://doi.org/10.1016/j.atmosres.2010.10.001>

Le formule `T_UTLS_proxy_v0 = T_anvil` e `D` sono scelte progettuali da
validare. I riferimenti giustificano le osservabili e i vincoli, non una
retrieval esatta della temperatura di tropopausa dai soli canali SEVIRI.
