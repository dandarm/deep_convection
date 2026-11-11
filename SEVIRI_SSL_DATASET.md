# Dataset SEVIRI SSL: procedura della selezione da 17.568 clip

Questo documento specifica la costruzione del dataset
`ssl_continuous_iou020_20261111`. La selezione contiene **17.568 clip di training**
su **93 date dal 1° gennaio al 29 giugno 2022**, con **2.928 clip per mese**.
Comprende **134 clip che attraversano mezzanotte** e utilizza **26.735 istanti
SEVIRI distinti**. La validation principale comprende 512 clip; la validation
estesa ne comprende 10.240.

La configurazione eseguibile è
[`configs/ssl_continuous_iou020_20261111.json`](configs/ssl_continuous_iou020_20261111.json).
Le figure e la descrizione sintetica sono in
[`notebooks/study_seviri_ssl_dataset.ipynb`](notebooks/study_seviri_ssl_dataset.ipynb).

## 1. Dati scaricati e inventario

Le osservazioni scaricate sono conservate in archivi Zarr giornalieri sotto
`/data1/deep_convection_data/MSG-SEVIRI`, nella struttura
`anno/AAAAMMGG.zarr`. Ogni archivio contiene i tempi di acquisizione, le
coordinate della griglia geostazionaria, la proiezione e le temperature di
brillanza in Kelvin. L'ordine dei sette canali è:

`WV_062, WV_073, IR_087, IR_097, IR_108, IR_120, IR_134`.

L'inventario registra le dimensioni degli array, i timestamp, le coordinate e
i blocchi di dati fisicamente presenti. I file di coordinate devono essere
completi e stabili. Un blocco di dati termici mancante rende indisponibili le
osservazioni necessarie ai crop che lo attraversano. Il controllo pixel per
pixel dei valori avviene durante la preparazione delle clip.

L'inventario congelato usato per questa selezione è indicato dalla chiave
`inventory` della configurazione. Il file `preparation_plan.json` del dataset
registra la configurazione effettiva, i record delle sorgenti e l'identità del
codice di preparazione. L'inventario di copertura mostrato nel notebook può
essere aggiornato dopo nuovi trasferimenti e descrive l'archivio disponibile
alla data riportata nella figura.

## 2. Dominio, clip e date ammesse

Il dominio geografico è **20°W–40°E, 30–70°N**. La validità geografica richiede
che il centro di ogni pixel dell'intero crop ricada entro questi limiti.

Ogni clip è definita da un timestamp iniziale e da un'origine sulla griglia
nativa. Contiene **16 frame consecutivi a cadenza di 5 minuti**, sette canali e
un crop di **224 × 224 pixel**, con forma `(16, 7, 224, 224)`. Il primo e
l'ultimo frame distano 75 minuti. Tutti i pixel richiesti devono essere finiti
e compresi fra 100 e 400 K. Le coordinate determinano l'orientamento finale
con nord in alto e ovest a sinistra. Il contenuto conserva l'ordine temporale
e spettrale delle osservazioni.

La chiave `training_days` elenca le 93 date ammesse per questa raccolta.
I giorni riservati sono definiti dal protocollo di riferimento. L'intero
intervallo di ogni clip di training deve rispettare una separazione di almeno
24 ore dalle date riservate alla validation e alle altre valutazioni.

La validation estesa contiene **512 clip per ciascuna di 20 date** tra gennaio
e giugno 2022. La validation principale da 512 clip è un sottoinsieme della
validation estesa. I relativi manifest e i file materializzati sono mantenuti
identici per il confronto fra modelli.

## 3. Input fissati per riprodurre questa selezione

La raccolta da 17.568 clip è costruita mediante estensione di un insieme
fissato. Il parametro `parent_dataset` identifica gli input:

- `candidates.csv`: **21.358 candidati già accettati**, con coordinate,
  timestamp, file e statistiche radiometriche;
- `training_crop_moments.csv` e `training/dataset/train.csv`: **16.536 clip
  che devono rimanere nella selezione finale**;
- `candidate_clips/` e `daily/`: array compressi e relativi record di preparazione.

Questi input, insieme all'inventario e al seed, definiscono la riproduzione
esatta della raccolta. I file già presenti sono riutilizzati mediante hard
link. Tutti i candidati fissati partecipano ai controlli di sovrapposizione,
compresi quelli disponibili per il successivo bilanciamento mensile.

## 4. Proposte continue nello spazio e nel tempo

Il builder è
[`scripts/build_seviri_ssl_continuous_dataset.py`](scripts/build_seviri_ssl_continuous_dataset.py).
Per ogni data determina tutti gli inizi di clip completi a risoluzione di un
frame, quindi a passi di cinque minuti. Quando la data successiva è ammessa,
legge anche i frame necessari del suo archivio. Le finestre attraversano
mezzanotte e i cambi di mese o anno quando i timestamp sono consecutivi, le
osservazioni sono disponibili e la griglia nativa è compatibile.

Per ciascun inizio determina le origini intere di pixel che soddisfano i
vincoli geografici e radiometrici per tutti i 16 frame. Una proposta estrae
uniformemente un inizio ammissibile, poi un'origine ammissibile per quella
finestra. Ogni proposta è valutata una volta. L'estrazione spaziale usa la
risoluzione di un pixel e consente origini in posizioni adiacenti.

Il seed principale è **20261111**, come specificato nella configurazione e nel
piano congelato. Il generatore di ogni data deriva dal seed principale, dal
numero ordinale della data e dall'identificatore del campionatore continuo.
L'elaborazione usa due fasi deterministiche di giornate non adiacenti. La
seconda fase include tutte le aggiunte della prima nel contesto delle
sovrapposizioni. Il numero di worker determina la concorrenza della
preparazione; il risultato resta determinato dagli input e dai seed.

## 5. Limiti alle osservazioni condivise

Una proposta viene confrontata con tutte le clip che condividono almeno un
istante osservato, comprese quelle che iniziano nella giornata precedente o
successiva. I criteri sono applicati a ogni coppia pertinente:

- **IoU spaziale ≤ 0,20.** L'IoU è l'area di intersezione dei due crop divisa
  per l'area coperta da almeno uno dei due. Tutte le coppie selezionate che
  condividono frame rispettano questo limite.
- **Al massimo il 20% delle posizioni pixel–istante di una clip condiviso
  con ciascuna altra clip.** Il conteggio combina l'area comune e il numero
  di frame comuni. Questo limite è applicato a ogni coppia che include una
  nuova proposta; le coppie formate soltanto da input fissati mantengono i
  vincoli della loro selezione.

Il secondo limite è definito per coppia. Clip su aree disgiunte possono
condividere tutti gli istanti di osservazione; l'area comune determina quanto
contribuisce la coincidenza temporale al riuso delle osservazioni.

## 6. Arresto della ricerca

Ogni blocco contiene **1.024 proposte** con finestre e origini già ammissibili.
Dopo il blocco si calcola la frazione accettata dai controlli di
sovrapposizione. La ricerca sulla data termina quando si verifica una delle
seguenti condizioni:

1. **Dieci blocchi consecutivi** hanno una frazione accettata **inferiore al 10%**.
   Un blocco con accettazione almeno pari al 10% azzera il contatore consecutivo.
2. Sono state valutate **32.768 proposte** sulla data.

Nell'esecuzione che ha prodotto questa raccolta, tutte le 93 date si sono
fermate dopo dieci blocchi: **952.320 proposte complessive** e **1.395 nuovi
candidati accettati**. Il pool risultante contiene **22.753 candidati**.
La bassa accettazione descrive il rendimento della ricerca randomizzata con
questi vincoli e input fissati.

## 7. Bilanciamento e manifest finale

La selezione finale conserva le 16.536 clip vincolate. Il mese con meno
candidati disponibili stabilisce la quota mensile comune, rispettando il
tetto `target_samples` e la divisibilità richiesta per la distribuzione su
quattro GPU. In questa esecuzione la quota è **2.928 clip per mese**, per un
totale di **17.568**. Il tetto di configurazione è 131.072 clip; la capacità
del pool determina la quota effettiva.

All'interno di ogni mese, l'allocazione considera le clip già fissate e
assegna le aggiunte alle date meno rappresentate con capacità disponibile.
Lo stesso criterio ripartisce le aggiunte nei quattro intervalli giornalieri
di sei ore. La scelta fra candidati di uno stesso gruppo è casuale e
riproducibile, con seed **20261114**. Il manifest finale è ordinato per
timestamp e origine spaziale. La raccolta comprende **1.032 clip selezionate
in aggiunta agli input vincolati**; queste possono provenire dal pool fissato
o dalle nuove proposte accettate.

## 8. Materializzazione, statistiche e utilizzo

Ogni clip selezionata è disponibile come array float16 Kelvin completo,
compresso senza perdita mediante Blosc/LZ4. Una clip che attraversa
mezzanotte contiene già tutti i suoi frame nello stesso file. La lettura degli
Zarr e l'estrazione dei crop avvengono nella preparazione; il dataloader usa i
file di clip.

Una sola media globale e una sola deviazione standard sono calcolate su
tutte le clip di training selezionate, tutti i frame, i canali e i pixel.
Le osservazioni condivise contribuiscono per ogni loro occorrenza nel dataset.
I valori sono **256,6 K** e **19,5 K**, con precisione completa conservata in
`channel_stats.json`. Le stesse costanti di training sono usate per la
validation del relativo modello.

La directory finale è:

`/data1/deep_convection_data/ssl_continuous_iou020_20261111`

I riferimenti principali sono `training/dataset/train.csv`, le due
`validation.csv`, `training/dataset/channel_stats.json`, `clips/metadata.json`,
`sampling_ready.json` e `dataset_ready.json`. Coordinate e timestamp in
`candidates.csv` e nel manifest consentono di estendere la selezione
conservando gli input già materializzati.

## 9. Significato dei conteggi nell'overview dell'archivio

**Geometric clip candidates** enumera ogni finestra di 16 frame consecutivi,
con partenza spostata di un frame alla volta, e ogni origine ammissibile di
un crop 224 × 224, spostata di un pixel alla volta lungo entrambi gli assi.
I crop vicini condividono gran parte delle loro osservazioni. Le finestre
attraversano archivi giornalieri consecutivi compatibili. Il totale somma
queste combinazioni su tutto l'archivio inventariato nel dominio geografico
concordato, verificando la presenza fisica dei blocchi necessari. Lo screening
radiometrico e i vincoli di selezione del training vengono applicati nelle
fasi successive.

Nell'inventario rilevato l'11 novembre 2026, **66.326 frame** sostengono
**65.864 finestre temporali complete**, di cui **3.210 attraverso mezzanotte**.
L'enumerazione delle origini spaziali produce **32.997.527.176 candidati
geometrici** nel dominio concordato. Questi valori sono calcolati
dall'inventario nel notebook e si aggiornano dopo nuovi trasferimenti.

**Selected training clips** riporta le **17.568 righe del manifest finale**
sulle 93 date scelte. Il conteggio ottenibile applicando il campionatore ad
altre date viene determinato da una nuova esecuzione con il relativo
inventario, le riserve temporali e gli stessi parametri di accettazione.

**All-archive frames** conta gli istanti osservati distinti. **Coverage at
5-min cadence** assegna cinque minuti a ogni istante e somma anche i periodi
separati. La mappa di copertura mensile mostra la percentuale degli istanti
previsti dal calendario per cui sono disponibili osservazioni utilizzabili
nel dominio.

## 10. Esecuzione e applicazione a nuovi dati

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/build_seviri_ssl_continuous_dataset.py \
  --config configs/ssl_continuous_iou020_20261111.json --workers 4

.venv/bin/python scripts/prepare_seviri_spaced_validation.py \
  --config configs/ssl_continuous_iou020_20261111.json --wait-for-training
```

Per nuovi anni si prepara un inventario aggiornato e una configurazione con
output distinto. `training_days` indica le date considerate; omettendo questa
chiave, il builder considera le date dell'inventario ammesse dalle riserve
temporali. `parent_dataset` permette di conservare un insieme già selezionato;
la sua omissione inizializza il campionatore con un insieme vuoto. Il
bilanciamento richiede capacità sufficiente nei mesi aggiunti per mantenere
le selezioni vincolate. Durata, cadenza, dominio e parametri di accettazione
sono definiti esplicitamente nella configurazione.

Le verifiche comprendono la continuità dei 16 timestamp, la validità dei
pixel, il dominio, le sovrapposizioni attraverso mezzanotte, la separazione
tra split e la conservazione degli input fissati. La preparazione e i dati
finalizzati conservano i propri manifest e il codice necessario a
identificare l'esecuzione.
