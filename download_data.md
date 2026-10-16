# Download dei dati SEVIRI RSS

Guida operativa per il PC GPU, aggiornata il 15 ottobre 2026. Lo script è
[`scripts/acquire_seviri_year.py`](scripts/acquire_seviri_year.py); la descrizione
storica dell’acquisizione e dei recovery è in
[`VIDEOMAE_PRETRAINING.md`](VIDEOMAE_PRETRAINING.md).

## Dati e percorsi

Lo script interroga la collezione EUMETSAT Rapid Scan High Rate SEVIRI L1.5
per l’anno richiesto, scarica i prodotti nativi e pubblica Zarr giornalieri
verificati nella stessa radice dell’archivio esistente:

```text
/media/fenrir/disk1/Datasets/MSG-SEVIRI/
├── 2020/
├── 2021/
│   └── 20210101.zarr
└── _acquisition/
    └── 2021/
        ├── acquisition.log
        ├── download.pid
        ├── progress.json
        ├── last_error.json
        ├── completed_days/
        ├── native_buffer/
        ├── processing/
        └── recovery/
```

I file di stato e le sottocartelle compaiono quando necessari. `acquisition.log`
e `download.pid` sono creati dai comandi sotto; non sono generati automaticamente
dallo script.

Ogni anno usa lo stesso dominio geografico, 20°W–40°E e 30–70°N, con forma
regionale `(7,763,1762)`, brightness temperature in Kelvin e ordine:
`WV_062`, `WV_073`, `IR_087`, `IR_097`, `IR_108`, `IR_120`, `IR_134`.
Gli Zarr usano float16 e conservano le coordinate native; i loader deducono
l’orientamento north-up/west-left dalle coordinate. La cadenza nominale è di
cinque minuti: slot mancanti o prodotti rifiutati restano gap documentati,
senza interpolazione o ripetizione dei frame.

L’ambiente `.venv` del PC è già predisposto. Le dipendenze della sola
acquisizione, tra cui EUMDAC e Satpy, sono in `requirements-download.txt`;
non richiedono GPU o PyTorch. Sono incluse anche in `requirements.txt`.
Le credenziali vengono lette dal file locale:

```text
/media/fenrir/disk1/danieleda/deep_convection/env
```

Il file `env` è escluso da Git. Non copiarne il contenuto nei log o nei documenti.

## Preparare il terminale

Aprire un terminale Bash. Questo blocco va eseguito anche in un nuovo terminale
usato soltanto per monitorare il download. Cambiare `2021` per scegliere l’anno.

```bash
cd /media/fenrir/disk1/danieleda/deep_convection

SEVIRI_YEAR=2021
SEVIRI_ROOT=/media/fenrir/disk1/Datasets/MSG-SEVIRI
SEVIRI_ENV_FILE="$PWD/env"
SEVIRI_WORKERS=8
SEVIRI_CONVERSION_WORKERS=16
SEVIRI_CONVERSION_THREADS=2
SEVIRI_CONVERSION_PREFETCH=4
SEVIRI_CROP_BATCH=64
SEVIRI_STATUS_DIR="$SEVIRI_ROOT/_acquisition/$SEVIRI_YEAR"
mkdir -p "$SEVIRI_STATUS_DIR"
```

## Avviare in background

```bash
nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python scripts/acquire_seviri_year.py \
  --year "$SEVIRI_YEAR" \
  --zarr-root "$SEVIRI_ROOT" \
  --env-file "$SEVIRI_ENV_FILE" \
  --workers "$SEVIRI_WORKERS" \
  --preferred-satellite MSG3 \
  --conversion-workers "$SEVIRI_CONVERSION_WORKERS" \
  --conversion-threads "$SEVIRI_CONVERSION_THREADS" \
  --conversion-prefetch "$SEVIRI_CONVERSION_PREFETCH" \
  --crop-batch-size "$SEVIRI_CROP_BATCH" \
  --recover-native-read-errors \
  >> "$SEVIRI_STATUS_DIR/acquisition.log" 2>&1 &

echo $! > "$SEVIRI_STATUS_DIR/download.pid"
```

Il processo continua anche chiudendo il terminale. Il log viene accodato,
conservando la cronologia dei tentativi precedenti.

- `--workers 8`: otto download nativi in parallelo; non è il numero di worker RDT.
  Sono **thread nello stesso processo Python**: vedere un solo PID non significa
  che il download sia sequenziale. Questi worker gestiscono i trasferimenti,
  mentre la conversione giornaliera è una fase distinta.
  Per passare da quattro a otto, arrestare il processo esistente e riprendere
  con `SEVIRI_WORKERS=8`; non avviare due acquisizioni dello stesso anno e archivio.
  Sul PC GPU è in corso la prova con otto worker. Le misure brevi con quattro
  e otto trasferimenti hanno mostrato forte variabilità, senza dimostrare un
  vantaggio stabile: valutare la velocità su intervalli lunghi durante i
  download, escludendo conversione, verifica e ripartenza.
- `--preferred-satellite MSG3`: risolve soltanto uno slot offerto contemporaneamente
  da satelliti diversi, scegliendo MSG3. Se c'è un unico prodotto RSS di MSG2,
  lo conserva: il servizio ha già alternato MSG3 e MSG2 nell'archivio. Non è un
  filtro che elimina tutti i frame MSG2. Senza il flag i duplicati restano
  bloccanti; duplicati dello stesso satellite, nomi non riconosciuti o assenza
  del satellite preferito in uno slot ambiguo restano sempre un errore.
- `--conversion-workers`: processi separati di lettura, calibrazione e crop dei
  nativi. Sono distinti dagli otto thread di download. Zero mantiene il percorso
  seriale originale. Un solo processo valida e scrive gli Zarr in ordine temporale;
  recovery e quarantena restano nel processo principale.
- `--conversion-threads`: thread Dask per processo di conversione. Il prefetch
  limita i task in attesa a `conversion-workers × conversion-prefetch`.
- `--crop-batch-size`: frame accumulati in RAM per ogni scrittura Zarr; non crea
  le clip spaziali 224×224 del probing o del pretraining. Batch grandi e prefetch
  sovrappongono preparazione e scrittura, usando più RAM. Un frame float32 a sette
  canali occupa circa 36 MiB; stack temporanei e processi aumentano il picco reale.
  I valori sopra sono per il PC GPU da 256 GiB, non per un portatile.
- I limiti dei thread numerici e il batch di conversione contengono la competizione
  con RDT per CPU e memoria. Otto worker possono aumentare il throughput di rete,
  ma non accelerano la conversione e non garantiscono il doppio della velocità.
- Il lock per anno impedisce due acquisizioni simultanee dello stesso anno.
  Controllare il processo prima di rilanciare: un nuovo tentativo bloccato dal
  lock potrebbe sovrascrivere `download.pid` con il PID del tentativo terminato.

Per una sola esecuzione in primo piano, usare lo stesso comando Python senza
`nohup`, redirezioni e `&`; in quel caso la chiusura del terminale può interromperlo.

## Monitorare

Log continuo:

```bash
tail -f "$SEVIRI_STATUS_DIR/acquisition.log"
```

`Ctrl+C` interrompe soltanto `tail`, lasciando attivo il download in background.
Il log mostra file scaricati/riusati, retry, frame convertiti e giornate pubblicate.
La rete può essere poco attiva mentre lo script converte o verifica una giornata.

Stato aggiornato ogni cinque secondi:

```bash
watch -n 5 "cat '$SEVIRI_STATUS_DIR/progress.json'"
```

Il JSON viene aggiornato principalmente per giornata. `current_day` indica il
giorno in elaborazione; `completed_days` conta le giornate passate in rassegna,
comprese eventuali giornate senza prodotti. Non misura direttamente i frame
scaricati o la completezza della copertura. I report `completed_days/*.json`
contengono `stored_frames`, `stored_times`, `missing_nominal_slots` e prodotti
rifiutati. Anche il contatore può ripartire da zero durante una ripresa,
mentre lo script verifica le giornate già completate.

Con il nuovo convertitore il JSON mostra anche `phase`: verifica delle giornate
già complete, download, conversione, verifica/pubblicazione. Durante la conversione
aggiorna dopo ogni batch `converted_frames`, `frames_to_convert`,
`conversion_frames_per_second`, worker, thread, prefetch e batch. Il throughput
include avvio dei processi e scritture; dopo pochi frame è ancora transitorio.
I report giornalieri conservano tempi e configurazione della conversione.

Processo e spazio libero:

```bash
ps -p "$(cat "$SEVIRI_STATUS_DIR/download.pid")" -o pid,etime,%cpu,%mem,args
df -h "$SEVIRI_ROOT"
```

Se il PID non corrisponde più al processo atteso, cercare il comando effettivo:

```bash
pgrep -af '[p]ython.*scripts/acquire_seviri_year.py'
```

Per vedere anche i thread del processo:

```bash
ps -L -p "$(cat "$SEVIRI_STATUS_DIR/download.pid")" -o pid,tid,stat,%cpu,comm
```

Durante i trasferimenti, i file incompleti terminano con `.nat.part`.
Questo comando mostra le dimensioni in byte: più file che crescono fra due
aggiornamenti confermano trasferimenti paralleli. Un file può sparire dalla
lista quando il download termina; in conversione la lista può essere vuota.

```bash
watch -n 5 "find '$SEVIRI_STATUS_DIR/native_buffer' -name '*.nat.part' -printf '%s %f\n'"
```

## Arrestare e riprendere

Per arrestare, verificare prima con `ps` che il PID appartenga al download
atteso, poi inviare SIGTERM:

```bash
kill -TERM "$(cat "$SEVIRI_STATUS_DIR/download.pid")"
```

Per riprendere, rieseguire il blocco **Avviare in background**, con lo stesso anno,
radice e opzioni di recovery/esclusione eventualmente aggiunte. Le giornate
completate vengono verificate e saltate; i nativi già validi vengono riusati.
La giornata interrotta può dover essere riconvertita. Lo script gestisce staging
e backup prima di pubblicare il nuovo Zarr e mantiene i frame preesistenti.

I nuovi nativi del buffer giornaliero vengono rimossi soltanto dopo la verifica
e pubblicazione degli Zarr. I prodotti originali esterni al buffer non vengono
rimossi; recovery e quarantena conservano i file previsti dai relativi report.

### Conversione parallela e RAM sul PC GPU

La configurazione del 16 ottobre 2026 usa 16 processi, due thread Dask per
processo, prefetch quattro e batch 64. I convertitori preparano frame diversi;
il processo principale li riceve in ordine, valida metadati e timestamp,
gestisce gli errori nativi e rimane l'unico writer dello Zarr. La coda ha al
massimo 64 task; non si accumula un intero anno in RAM. Lo stop via SIGTERM
chiude il pool; durante la conversione la giornata non ancora pubblicata può
dover essere ricalcolata alla ripresa. I giorni completi e i nativi validi
rimangono riutilizzabili.

Benchmark locale, stesso batch 2, 12 nativi reali e scrittura inclusa:
seriale 30,71 s, quattro processi 10,67 s, otto processi 10,04 s. Kelvin grezzi,
coordinate, metadati e chunk Zarr compressi risultano identici.
Su ulteriori 72 frame, otto processi/batch2/prefetch2 richiedono 42,26 s con
picco RSS cumulata 2,94 GiB; dodici/batch32/prefetch4 39,63 s e 7,16 GiB;
sedici/batch64/prefetch4 36,40 s e 9,87 GiB. Tutti i valori e le coordinate
salvati coincidono; batch diversi possono avere chunk diversi per le coordinate
temporali, senza cambiare i dati. Le misure includono l'avvio dei processi;
la cache disco non è stata svuotata e il carico del PC può influire. Report e
script sono in `_acquisition/2021/conversion_benchmark/`, fuori da Git.

La RAM vista come cache del filesystem contiene già molti dati riutilizzabili.
Aumentare batch/prefetch usa ulteriore memoria dei processi; scegliere i valori
con throughput e picco misurati, non in base alla sola percentuale di RAM libera.

## Diagnosticare un errore

Gli stati principali sono `running`, `failed` e `completed`. Con `failed`:

```bash
cat "$SEVIRI_STATUS_DIR/last_error.json"
tail -n 60 "$SEVIRI_STATUS_DIR/acquisition.log"
```

Dopo cinque tentativi falliti sulla stessa giornata lo script si arresta.
`last_error.json` può riferirsi a un tentativo precedente già superato: leggere
anche `progress.json` e la coda del log per sapere se l’errore è ancora attuale.

`--recover-native-read-errors` gestisce uno specifico errore di lettura nativa
(`buffer is smaller than requested size`): riscarica il prodotto per confronto.
Un payload identico che resta illeggibile viene documentato e messo in
quarantena, lasciando un gap; altri errori restano bloccanti.
Duplicati di slot nel catalogo del provider richiedono diagnosi e non vengono
risolti da questa opzione.

Nel caso verificato del **7 gennaio 2021 alle 09:00 UTC**, il servizio passa
da MSG2 a MSG3 offrendo entrambi nello slot del passaggio. I giorni 1–4 usavano
MSG3, il 5–6 comprendevano la sostituzione con MSG2; l'8–10 tornano interamente
MSG3. Il campionamento dei mesi successivi trova anche giornate solo MSG2:
la selezione preserva questa continuità RSS senza richiedere un unico satellite
per tutto l'anno. I numeri 2/3 identificano satelliti diversi della famiglia MSG.

`selection_policy.json` conserva la preferenza dell'anno e impedisce di
cambiarla accidentalmente durante la ripresa. Mantenere quindi lo stesso flag.
Le giornate completate prima dell'introduzione della regola vengono verificate
e conservate. `product_selection/AAAAMMGG.json` registra tutti i prodotti scelti,
satellite e slot, le alternative escluse e i frame preesistenti non sostituiti;
il report giornaliero distingue prodotti offerti e prodotti selezionati.
Eventuali nativi alternativi già nel buffer vengono conservati con SHA-256 in
`unselected_native/AAAAMMGG/`, senza convertirli per errore al posto del prodotto
scelto. I nuovi frame sono scritti in ordine temporale anche quando cambia
il prefisso del satellite. Nessuno slot è riempito, duplicato o interpolato.

`--exclude-native NOME_ESATTO.nat` è disponibile per escludere esplicitamente
un prodotto già diagnosticato, conservandolo in quarantena. Non riutilizzare
alla cieca le esclusioni specifiche del 2020 per altri anni. Conservare le
stesse esclusioni nei successivi comandi di ripresa.

## Altri anni e convivenza con RDT

Per il 2022, cambiare `SEVIRI_YEAR=2022` nel blocco iniziale e rilanciare. Dati,
lock e stato saranno separati per anno. Preferire un anno alla volta mentre
RDT è attivo; aumentare i download paralleli solo se rete, disco e memoria lo
consentono. Lo spazio libero va controllato con `df`: lo script controlla una
soglia di **50 GiB** prima di avviare una giornata e si ferma se è insufficiente.

L’acquisizione scarica e converte dati; non lancia automaticamente training,
calcolo RDT, probing o campionamento di clip. Il job RDT corrente usa un
**inventario congelato**: i nuovi anni scaricati non entrano automaticamente
nel suo totale o nelle graduatorie. Per analizzarli servirà un successivo job
RDT o un’estensione esplicita con provenienza e checkpoint compatibili.

Dataset, nativi, Zarr, buffer, log e credenziali restano fuori da Git.

## Avviare da un altro PC

Usare **Linux oppure WSL su Windows**, con Python 3.11 o 3.12: lo script usa
`fcntl` per il lock e non funziona direttamente in Python Windows. Non serve
una GPU. Su una nuova macchina:

```bash
git clone https://github.com/dandarm/deep_convection.git
cd deep_convection
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-download.txt
```

Se il repository è già presente, eseguire `git pull --ff-only` prima
dell'installazione. Creare localmente il file `env` con le proprie credenziali
EUMETSAT, usando le variabili `EUMETSAT_CONSUMER_KEY` e
`EUMETSAT_CONSUMER_SECRET`. Il file non arriva con il clone e non va committato.

Per esempio, mentre il PC GPU scarica **2021**, il secondo può scaricare **2022**:

```bash
SEVIRI_YEAR=2022
SEVIRI_ROOT="$HOME/Datasets/MSG-SEVIRI"
SEVIRI_ENV_FILE="$PWD/env"
SEVIRI_WORKERS=4
SEVIRI_CONVERSION_WORKERS=2
SEVIRI_CONVERSION_THREADS=2
SEVIRI_CONVERSION_PREFETCH=2
SEVIRI_CROP_BATCH=8
SEVIRI_STATUS_DIR="$SEVIRI_ROOT/_acquisition/$SEVIRI_YEAR"
mkdir -p "$SEVIRI_STATUS_DIR"
```

Eseguire poi il blocco **Avviare in background** sopra, che usa queste variabili.
Partire con quattro worker sul secondo PC e aumentare a otto solo dopo aver
misurato la velocità. Se i due PC condividono connessione o account, i limiti
disponibili possono essere condivisi: più worker non implicano più banda.
Usare un disco locale con spazio sufficiente, anche per il buffer dei nativi.

Ogni PC conserva i propri Zarr, report di acquisizione e stato. La stessa
procedura produce dominio, canali, unità e formato uguali; non avviare due
writer sullo stesso anno di un archivio condiviso. L'eventuale trasferimento
dei dati fra PC è separato dal push del codice: conservare anche
`_acquisition/ANNO/completed_days/` e i report di recovery/quarantena per la
provenienza. Questi dati non vengono trasferiti tramite Git.
