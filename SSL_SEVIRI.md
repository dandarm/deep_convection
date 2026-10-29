# Confronto degli obiettivi SSL SEVIRI

Implementazione iniziale dell'handoff
[`handoff_codex_ssl_seviri.md`](handoff_codex_ssl_seviri.md), avviata il
24 ottobre 2026. Il primo esperimento prioritario è la previsione causale.
Il trainer VideoMAE storico, il teacher RDT e i relativi risultati restano
separati da questa nuova pipeline.

## Stato del workspace e trasferimento

All'avvio: commit `0c447e7`, nessun checkpoint VideoMAE e nessun archivio
SEVIRI locale. I risultati V1/V2 descritti nei documenti precedenti sono
risultati del computer di origine, non misure riprodotte qui. I manifest
da 2/100 clip sono solo materiale di integrazione; quelli dei training V2 e
le riserve diagnostiche vanno trasferiti insieme ai relativi checkpoint.

Dal 27 ottobre 2026, la destinazione dell'upload è
`/data1/deep_convection_data/MSG-SEVIRI/`, sul disco `/dev/sdb`. Questa raccolta
contiene anche una copia del 2020 già ricevuto. Gli originali del 2020 e la
cache usata dal training Base restano sul disco di sistema, nei percorsi
registrati dai piani di esecuzione. I dettagli e le verifiche della migrazione
sono riportati nell'ultima sezione di questo documento.

Cartelle predisposte **fuori dal repository** all'avvio del progetto:

```text
/home/isac/deep_convection_data/
  MSG-SEVIRI/
    2020/
      20200912.zarr/
      ...
  previous_runs/
```

Per PowerShell nativo, dal PC Windows con OpenSSH Client installato:

```powershell
Get-Command scp
scp -r -J UTENTE_BRIDGE@IP_BRIDGE "E:\Datasets\MSG-SEVIRI" "isac@IP_WORKSPACE:/data1/deep_convection_data/"
```

La directory `MSG-SEVIRI` viene copiata dentro la radice di destinazione.
Sostituire utente del bridge, IP del bridge e IP del workspace raggiungibile
dal bridge. Le porte SSH predefinite sono 22. `scp` non riprende automaticamente
un trasferimento interrotto.

Per una sincronizzazione riprendibile usare **WSL/Ubuntu**, dove il drive E
è montato sotto `/mnt/e`. `rsync` deve essere disponibile su entrambi i lati:

```bash
rsync -a --partial --partial-dir=.rsync-partial --info=progress2 \
  -e "ssh -J UTENTE_BRIDGE@IP_BRIDGE -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  /mnt/e/Datasets/MSG-SEVIRI/ \
  isac@IP_WORKSPACE:/data1/deep_convection_data/MSG-SEVIRI/
```

Lo slash finale sulla sorgente copia il contenuto. Il bridge inoltra la
connessione SSH tra PC e workspace. Rilanciare lo stesso comando dopo
un'interruzione, usando la destinazione `/data1/` indicata sopra. Per porte
diverse e una prova `--dry-run`, usare
[`scripts/upload_seviri.example.sh`](scripts/upload_seviri.example.sh).
La verifica delle chiavi host SSH rimane abilitata.

Nel terminale **MSYS2** di Windows, la sorgente del drive E è invece `/e`:

```bash
rsync -rt --partial --partial-dir=.rsync-partial --info=progress2 \
  -e "ssh -J UTENTE_BRIDGE@IP_BRIDGE -o Compression=no -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  /e/Datasets/MSG-SEVIRI/ \
  isac@IP_WORKSPACE:/data1/deep_convection_data/MSG-SEVIRI/
```

Al rilancio, `progress2` può ripartire da 0%: il contatore riguarda la nuova
esecuzione e l'inventario viene ricostruito progressivamente. Con gli stessi
percorsi e queste opzioni, i file completi con dimensione e data invariati
vengono saltati; i parziali conservati possono essere riutilizzati. La
percentuale descrive i dati considerati nell'esecuzione corrente. Per fissare
il totale fin dall'inizio si può aggiungere `--no-inc-recursive`: rsync deve
prima completare la scansione di tutti i file locali. Con l'inventario
incrementale predefinito, `ir-chk` indica che la scoperta dei file è ancora
in corso e il totale può continuare a crescere.

I piccoli manifest, `channel_stats.json`/`train*_stats.json`, `run_config.json`,
`plan.json`, `clustering_reserved_days.json` e i checkpoint dei run precedenti
possono essere copiati sotto `previous_runs/`. Le statistiche storiche sono
legate al percorso originale degli Zarr: il nuovo preparatore le ricalcola
sullo stesso manifest dopo la migrazione, senza falsificarne la provenienza.

L'upload iniziato dall'utente viene seguito da
[`scripts/watch_seviri_upload.py`](scripts/watch_seviri_upload.py). Il watcher
richiede metadata, coordinate e **tutti i chunk fisicamente presenti** della
clip candidata; controlla stabilità dei file, hash, 16 timestamp consecutivi,
sette canali, orientamento e radiometria finita in Kelvin. Una cartella Zarr
parziale non è considerata una clip valida. Il watcher non scrive nell'archivio.
Alla prima clip valida avvia un solo smoke CUDA e termina. Stato, PID/comando,
log e report sono in `results/ssl_upload_smoke/`; il comando riproducibile è:

```bash
.venv/bin/python -u scripts/watch_seviri_upload.py \
  --zarr-root /home/isac/deep_convection_data/MSG-SEVIRI \
  --output-dir results/ssl_upload_smoke
```

Un lock impedisce due watcher concorrenti. Il monitor supporta il formato
Zarr v2 corrente, chunk temporali di un frame e sette bande per chunk.
Lo smoke su una clip controlla esecuzione e numerica; le sue statistiche
locali non diventano statistiche di pretraining scientifico.

## Protocollo implementato

Il loader continua a leggere campioni completi `(16,7,224,224)` e conserva
ordine dei canali, coordinate, Kelvin e trasformazione affine globale. La
separazione causale avviene **prima** della chiamata al modello:

| Parte | Indici da zero | Tempo rispetto all'ultimo input |
|---|---|---|
| Storia | 0–9 | −45, −40, …, 0 minuti |
| Target +15 | 12 | +15 minuti |
| Target +30 | 15 | +30 minuti |

I frame 10/11/13/14 vengono comunque validati nel cubo originale; non entrano
nell'encoder né sono target degli orizzonti iniziali. Non vengono inventati
frame. I dieci frame storici producono cinque tubelet; le posizioni sinusoidali
vengono costruite per questo prefisso, senza includere un tubelet che attraversi
il confine input/target. L'attenzione è bidirezionale **entro il passato**:
non serve una maschera triangolare per la previsione dal tempo finale.

[`scripts/pretrain_seviri_ssl.py`](scripts/pretrain_seviri_ssl.py) espone:

| `--objective` | Input all'encoder e loss |
|---|---|
| `future` | Storia completa; MSE o Huber sui due frame futuri |
| `mvm` | Storia mascherata; ricostruzione V2 dei soli target storici nascosti |
| `joint` | Encoder condiviso, due passaggi sulla storia: MVM mascherato e future completo; `lambda_mvm * MVM + lambda_future * future` |
| `residual` | Storia completa; baseline advettiva fissa più residuo previsto, una sola loss sul futuro ricostruito |

Backbone Small 384×12, patch 16×16, tubelet 2, sette canali. La head comune
concatena temporalmente i token di ciascuna posizione, applica convoluzioni
spaziali e ricostruisce i pixel con una convoluzione trasposta. Mantiene
l'ordine temporale, senza pooling temporale globale. L'inizializzazione della
head è uguale fra obiettivi a seed uguale, indipendentemente dal decoder MAE.

Il nuovo controllo MVM usa **10 frame storici** per rendere confrontabile
l'informazione disponibile. Il MVM storico a 16 frame resta un controllo
separato. Il joint esegue due forward e costa più compute; stessi batch,
epoche e optimizer step non significano stesso costo computazionale. Registrare
entrambi nel confronto finale. Nessun risultato dei quattro obiettivi può
essere dedotto confrontando loss di task differenti.

Masking storico: `tube` (default 90%), `random`, `block`. I blocchi sono cuboidi
2×2×2 in unità tubelet/patch, in ordine casuale; l'ultimo può essere parziale
per ottenere uguale numero di token mascherati per batch. Il decoder V2 usa
running-cell 50%; la loss resta sui token nascosti all'encoder. La validation
ricrea maschere fisse per indice della clip.

La baseline advettiva iniziale è **traslazione globale mediante correlazione
di fase**, stimata esclusivamente dagli ultimi due IR108 storici. Limite
iniziale ±12 pixel per cinque minuti, velocità costante, stesso movimento per
tutti i canali. Non è ancora optical flow denso né una misura del vento.
Il warping campiona all'indietro `x-dx, y-dy`; quando la sorgente cade fuori
crop, la previsione usa esplicitamente la persistenza. Le osservazioni e i
target non sono modificati. La maschera di supporto distingue i pixel con
sorgente disponibile: tutti i metodi sono valutati sia sull'intero crop sia
sullo stesso supporto advettivo. Il residuo contiene anche errori del moto.

## Preparazione dei manifest

L'input è una coppia di manifest esistenti train/validation con colonne
`start_time,origin_y,origin_x`. Il preparatore verifica i pixel e salva un
protocollo comune. La validation deve cominciare **oltre 24 ore dopo l'ultimo
frame** del training; nessuna divisione casuale delle finestre sovrapposte.
Le intere giornate diagnostiche 3–9 marzo 2020 vengono escluse automaticamente
dal training, anche se il JSON storico non è ancora stato trasferito.
`--reserved-days FILE.json` aggiunge altre giornate dalla chiave `days`.

Esempio con i manifest V2 trasferiti, da eseguire sul workspace **dopo**
il completamento dell'upload delle giornate interessate:

```bash
export SEVIRI_ZARR_ROOT=/home/isac/deep_convection_data/MSG-SEVIRI
.venv/bin/python scripts/prepare_seviri_ssl.py \
  --zarr-root "$SEVIRI_ZARR_ROOT" \
  --manifest /home/isac/deep_convection_data/previous_runs/videomae_v2_small_scaling/train2000.csv \
  --validation-manifest /home/isac/deep_convection_data/previous_runs/videomae_v2_small_scaling/validation.csv \
  --output-dir results/ssl_dataset
```

Senza `--count`, ordine e candidati restano invariati. Per espandere il pool,
generare prima candidati SEVIRI con gli strumenti esistenti, rispettando gli
split, poi sottocampionare con `--count N --uniform-fraction 0.5`. Metà della
selezione è uniforme; il resto privilegia gli strati di variabilità bassa,
media e alta con masse 1:2:4, senza duplicati. Si salvano media, quantile 90%
e frazione oltre 2 K delle differenze IR108 fra frame **storici** consecutivi.
`--compensated` usa la variazione dopo compensazione della traslazione.
La validation non viene sovracampionata. Questi score non definiscono classi
convettive; con `N` uguale al pool non si aumenta il peso degli esempi variabili.

Output: `train.csv`, `validation.csv`, `candidate_variability.csv`,
`channel_stats.json`, `protocol.json`. La coppia scalare media/std usa tutti
i pixel, canali e 16 frame del **solo training selezionato**. La normalizzazione
rimane identica fra i quattro esperimenti. L'utilizzo dei futuri di training
per stimare due costanti globali non introduce target per-clip nell'encoder.

## Training, ripresa e valutazione

Ambiente del workspace: `.venv`, installato da `requirements-training.txt`
e dalle dipendenze di analisi/RDT per la suite completa. `requirements.txt`
limita xarray a `<2026.9`: la versione 2026.9 non accetta più il vecchio
encoding `compressor` dell'acquisizione. Il formato dei dati e il codice
di acquisizione non sono stati cambiati.

Primo pilot, una epoca:

```bash
.venv/bin/python scripts/pretrain_seviri_ssl.py \
  --config config/seviri_ssl_pilot.json \
  --manifest results/ssl_dataset/train.csv \
  --validation-manifest results/ssl_dataset/validation.csv \
  --global-stats results/ssl_dataset/channel_stats.json \
  --output-dir results/ssl_future_pilot
```

Le opzioni CLI prevalgono sul JSON. Sostituire `--objective` e directory di
output per gli altri tre obiettivi, mantenendo i medesimi manifest, statistiche,
seed e budget. Default CUDA bf16, microbatch 2, accumulo 2, LR costante 1.5e-4.
La configurazione è da misurare sui dati reali prima di aumentare batch/durata.

Per riprendere, usare gli stessi argomenti, aumentare `--epochs` se necessario
e aggiungere `--resume results/ssl_future_pilot/checkpoint_latest.pt`. Il
checkpoint atomico per epoca include modello, optimizer, scheduler, RNG
Python/NumPy/Torch/CUDA, generatori di sampler/loader/masking/validation,
statistiche e identità del protocollo/sorgenti. Un'epoca interrotta viene
ripetuta. Cambiamenti del protocollo vengono rifiutati. I CSV vengono riallineati
all'epoca salvata. Le directory non vuote richiedono una ripresa esplicita.

Output di ciascun run:

- `run_config.json`, `channel_stats.json`: configurazione, seed, hash, commit,
  hardware, precisione, eventuale inizializzazione e provenienza;
- `checkpoint_latest.pt`, `checkpoint_best.pt`: stato riprendibile;
- `loss_steps.csv`, `loss_epochs.csv`: loss distinte, throughput e memoria;
- `validation_latest.json`, `validation_best.json`: baseline e metriche;
- `forecast_latest.png`: stessa clip di validation, IR108 con scala 180–330 K
  condivisa fra osservazioni, persistenza, advezione e previsione;
- `encoder/`: encoder finale riutilizzabile in formato Hugging Face.

Metriche per sette canali e ogni orizzonte: MAE/RMSE in K, errore sui gradienti
in K/pixel, tendenza dal tempo iniziale e fra orizzonti in K/min, errore sui
minimi locali osservati 3×3 e sul minimo assoluto, errore sul contrasto P95−P05.
L'errore di tendenza da `t=0` è algebricamente l'errore radiometrico diviso per
l'orizzonte; non è una prova indipendente di informazione dinamica. Le metriche
accumulano somme/conteggi prima di calcolare RMSE. Si riportano anche risultati
per tre livelli di variabilità storica, con soglie stimate sul training.
Best usa MSE futura, oppure MSE MVM per il controllo MVM.

Per un'ablazione aggiungere agli stessi argomenti del run:

```bash
--resume results/ssl_future_pilot/checkpoint_best.pt \
--evaluate-only --temporal-ablation shuffle
```

`shuffle` permuta i primi nove input e mantiene l'ultimo frame di riferimento;
`last_frame` ripete solo l'ultimo. Sono interventi diagnostici espliciti,
esterni al loader e mai augmentation implicite. Baseline e decomposizione
residuale restano calcolate sulla storia originale, così cambia solo l'input
all'encoder. I seed della permutazione dipendono dall'indice della clip.

Il probing con head comune è disponibile con `--objective future
--freeze-encoder --encoder-from PERCORSO`. Accetta `encoder/`, `model/` o un
checkpoint atomico locale. Conserva architettura, attivazioni e LayerNorm
del sorgente, trasferisce strettamente i pesi e ricostruisce le posizioni
temporali per la storia. L'encoder resta in eval e non riceve gradienti.
Usare identiche head, normalizzazione, manifest e budget fra encoder; senza
`--encoder-from`, `--freeze-encoder` costituisce il controllo casuale.
Un encoder storico che abbia visto le date del nuovo holdout non è un controllo
valido, anche se il codice ne consente tecnicamente il caricamento.

## Verifiche e prossimi esperimenti

Smoke senza osservazioni, con encoder ridotto e clip generate:

```bash
.venv/bin/python scripts/pretrain_seviri_ssl.py \
  --synthetic-smoke --objective future --device cpu --epochs 2 \
  --output-dir results/ssl_synthetic_future

.venv/bin/python scripts/smoke_seviri_ssl.py \
  --output results/ssl_development/full_size_cuda_preflight.json

PYTHONPATH=src .venv/bin/python -m compileall -q src scripts tests
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Il secondo comando usa Small completo e 224×224, confronta FP32/bf16,
backward e aggiornamento dell'encoder per tutti e quattro gli obiettivi.
Il report non viene sovrascritto: scegliere un nuovo percorso per ripeterlo.
I test verificano causalità, indici temporali, segno del warping, bordi,
equivalenza loss residuo/futuro, masking, caricamento checkpoint, encoder
congelato, normalizzazione training-only, sampling e ripresa bit-per-bit su CPU.

Il pilot con holdout separato, quattro obiettivi, sonde radiometriche e
confronto degli embedding osservati/sola advezione è documentato più avanti.
Restano per un esperimento scientifico esteso: validazione della qualità del
moto, eventuale flow denso, sweep del peso delle loss e protocollo multi-seed
con test indipendente. I test d'integrazione non quantificano il valore
scientifico delle rappresentazioni.

### Verifica nel workspace del 24 ottobre 2026

- Suite completa: **237 passed**; compilazione Python e `pip check` riusciti.
  Restano warning di deprecazione delle dipendenze, senza test falliti.
- Ambiente verificato: Python 3.12.3, Torch 2.14.1+cu130, Transformers 5.20.2,
  xarray 2026.7.0; inventario completo in
  `results/ssl_development/environment.txt`.
- Smoke sintetico con Small completo riuscito per i quattro obiettivi:
  `results/ssl_development/full_size_cuda_preflight.json`.
- Smoke **su osservazioni reali arrivate durante l'upload** riuscito su GPU
  NVIDIA H200 MIG per tutti e quattro gli obiettivi: clip del 12 settembre
  2020, 08:20–09:35 UTC, origine nativa `(224,672)`, storia fino alle 09:05,
  target alle 09:20 e 09:35. Canali e orientamento verificati, valori
  212.625–306.25 K. Loss e backward finiti, encoder aggiornato; massimo scarto
  relativo bf16/FP32 `1.2124e-5`, picco di memoria riservata PyTorch 0.672 GiB
  con batch 1. Queste misure non stimano il throughput di un training lungo.
  Report: `results/ssl_upload_smoke/real_cuda_preflight.json`; file sorgenti
  verificati prima/dopo e relativi hash in `sample.json`.
- Gli store reali contengono osservazioni aggiunte dopo un precedente download
  parziale: l'ordine fisico dei chunk può differire dal tempo. Il watcher
  risolve i timestamp negli indici nativi, come richiede il loader esistente,
  senza riscrivere l'archivio. La regressione è coperta da un test.

Il watcher ha completato il suo singolo smoke ed è terminato; l'upload può
proseguire. Non sono stati avviati training scientifici o selezioni di modello
sulla singola clip usata per l'integrazione.

## Pilot su 4.096 clip durante l'upload — 24 ottobre 2026

Il dataset congelato è `results/ssl_pilot_20261024/dataset/`. Non materializza
immagini né clip: i manifest referenziano gli Zarr originali. La preparazione
usa soltanto coordinate e chunk fisicamente presenti e stabili, con mapping
fra timestamp cronologici e indici nativi. Gli hash prima/dopo la validazione
escludono sostituzioni dei file mentre l'upload prosegue; i chunk aggiunti
successivamente non ampliano automaticamente il dataset.

- Training: **4.096 clip**, 2.752 del 12 settembre e 1.344 del 13 settembre;
  402 partenze temporali e 74 origini spaziali distinte.
- Validation: **512 clip** del 14 settembre, 129 partenze e 70 origini.
- Ultima osservazione training: 13 settembre 11:55 UTC. Prima osservazione
  validation: 14 settembre 12:00 UTC; separazione temporale effettiva **24 h 5 min**.
- Pool training validato: 8.192 clip. Metà della selezione finale è uniforme;
  metà privilegia terzili di variabilità storica IR108 con pesi 1:2:4.
  I candidati sono prima bilanciati per giorno, origine e fascia di sei ore.
  La validation mantiene il bilanciamento spazio-temporale senza pesi radiometrici.
- Reticolo di origini ogni 112 pixel, inclusi gli ultimi indici ammissibili
  sui bordi. Sono ammessi overlap spaziali e temporali **dentro** lo split.
  I crop non finiti/fuori disco sono rifiutati, senza riempimenti.
- Media globale: **263.96726726439204 K**; deviazione standard globale:
  **20.818200820284428 K**, su tutti i 23.018.340.352 valori del training.
  Nessun dato di validation contribuisce a queste costanti.

Il footprint spaziale copre il 93,26% dei pixel finiti nell'immagine IR108 del
12 settembre 2020 alle 00:00 UTC per il training e il 90,27% per la validation.
È una diagnostica geometrica riferita a quella specifica osservazione;
la verifica è salvata in `results/ssl_notebook_review_20261025/spatial_reference.json`. Il dominio
rettangolare include anche pixel fuori disco, e le finestre 224×224 interamente
valide non possono sempre raggiungere ogni bordo curvo osservabile.
L'inventario conserva anche store ancora incompleti e motivi di esclusione.

Per creare un altro snapshot, scegliere una **nuova** directory:

```bash
.venv/bin/python scripts/build_seviri_ssl_pilot_dataset.py \
  --zarr-root /home/isac/deep_convection_data/MSG-SEVIRI \
  --output-dir results/ssl_pilot_20261024/dataset \
  --count 4096 --candidates 8192 --validation-count 512 --workers 8
```

### Coda dei training e confronto delle rappresentazioni

La coda esegue da zero future, MVM, joint e residuo, **tre epoche** ciascuno.
Tutti partono dagli stessi pesi dell'encoder; le tre head predittive hanno
anch'esse inizializzazione identica. Small completo 384×12, 10 input storici,
bf16, microbatch16, accumulo2, batch effettivo32, AdamW/LR1.5e−4 costante,
8 worker, task I/O di due clip, tube90% e decoder running-cell50%.
Gli aggiornamenti sono 128 per epoca; joint richiede due forward e non ha
lo stesso costo computazionale degli altri obiettivi.

Seguono **cinque sonde future comuni**, due epoche ciascuna: encoder casuale
e i quattro encoder finali congelati. Identiche head, seed, dati e optimizer.
MVM entra così in un confronto predittivo che non confonde task diversi.
Gli hash dei pesi dell'encoder prima/dopo il probing vengono verificati
nell'analisi. Ogni run salva anche le metriche di validation per ogni epoca,
oltre a best/latest; i confronti finali usano i checkpoint **finali**, con
budget uguale entro ciascuna fase, senza selezione retrospettiva del best.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -u scripts/run_seviri_ssl_comparison.py \
  --zarr-root /home/isac/deep_convection_data/MSG-SEVIRI \
  --dataset results/ssl_pilot_20261024/dataset \
  --output-dir results/ssl_pilot_20261024 \
  --epochs 3 --probe-epochs 2 --batch-size 16 --workers 8
```

Ripetere lo stesso comando riprende gli incompleti dall'ultima epoca e salta
i run conclusi; un lock impedisce due supervisori. Piano e sorgenti sono
congelati e verificati, così come gli hash dei chunk prima/dopo la coda.
PID/comando in `launcher.json`, stato in `queue_status.json`, log separati.

L'analisi, da eseguire dopo il completamento della coda:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -u scripts/analyze_seviri_ssl_comparison.py \
  --queue-dir results/ssl_pilot_20261024 --batch-size 16 --workers 4
```

Produce metriche radiometriche, shuffle dei primi nove frame mantenendo fisso
l'ultimo, ripetizione dell'ultimo frame, embedding mean+slope (768 elementi),
sonde ridge alpha10 su 14 osservabili e confronto con storie di sola advezione.
Lo scaler degli embedding e le sonde ridge sono fittati solo sul training;
non modificano la normalizzazione dei pixel. La storia sintetica advettiva
parte dal primo frame e usa il moto stimato dagli ultimi due input storici;
è un controllo diagnostico, non un input al training né una ricostruzione vera.
Fuori dal supporto del warping resta il primo frame; il CSV riporta la frazione
di supporto disponibile, perché anche i bordi possono cambiare gli embedding.
La sensibilità dell'embedding non dimostra da sola una rappresentazione utile.

### Notebook dedicato

Aprire `notebooks/study_seviri_ssl_training.ipynb` con il kernel
`Python (deep_convection)`. La revisione del 25 ottobre presenta in inglese
le ipotesi del documento di progetto, le definizioni dei dati e delle quattro
loss, le metriche, le sonde e l'interpretazione dei confronti. Ogni nome delle
legende è definito nelle celle Markdown; i grafici contengono titoli, assi,
unità e legende. La modalità predefinita usa figure dirette; `USE_WIDGETS = True`
abilita i controlli interattivi. Dopo l'aggiornamento, riavviare il kernel ed
eseguire le celle dalla configurazione iniziale.

La cella delle metriche legge i risultati di validation salvati in JSON.
Nella verifica della revisione, la lettura e il rendering hanno richiesto
circa 0,26 s. La previsione Future di una clip, con entrambi gli orizzonti,
ha richiesto in media 5,94 ms con input sulla GPU; il percorso completo da
Zarr alla previsione su CPU ha richiesto 41,9 ms con i file nella cache del
sistema. Metodo di misura, risultati e copia eseguita del notebook sono in
`results/ssl_notebook_review_20261025/`.

Per ricostruire l'ambiente di training, analisi
e notebook su Linux:

```bash
.venv/bin/python -m pip install -r requirements-analysis.txt -r requirements-notebook.txt
.venv/bin/python -m ipykernel install --user --name deep-convection --display-name "Python (deep_convection)"
```

Le celle autonome includono:

1. copertura spaziale, date e quantità del dataset;
2. **Monitoraggio e loss per epoca**, Aggiorna/Auto30s, pannelli separati
   MVM, previsione end-to-end e sonde congelate, in K²;
3. metriche selezionabili per canale e +15/+30 minuti;
4. osservazioni, persistenza, advezione e previsioni con scale comuni;
5. player della ricostruzione MVM sui dieci frame storici, errori solo nascosti;
6. ablazioni temporali e diagnostiche degli embedding/sonde radiometriche.

Le visualizzazioni usano quattro clip fissate per rango di variabilità nella
validation. La cache float16 è solo grafica: le metriche sono calcolate prima
della quantizzazione. Il notebook versionabile resta senza output; le copie
eseguite e le figure sono sotto `results/ssl_pilot_20261024/`, fuori da Git.
Variabile opzionale `SEVIRI_SSL_QUEUE` per esplorare una coda successiva.

Si tratta di un pilot con tre giornate, clip correlate e un solo seed: serve
a debugging, profiling e primi confronti. Non sostituisce stagioni/anni
separati, più seed e un test indipendente per una conclusione scientifica.

### Primi risultati del pilot

I quattro pretraining hanno completato tre epoche ciascuno, con 384
aggiornamenti dell'optimizer. Le cinque sonde comuni hanno completato due
epoche ciascuna, con 256 aggiornamenti e encoder congelato. Tutti i numeri
seguenti riguardano i checkpoint finali e le stesse **512 clip di validation**.
La RMSE globale aggrega gli errori quadratici dei sette canali e dei due
orizzonti prima della radice: non è la media delle RMSE.

| Encoder/obiettivo | Previsione end-to-end, RMSE K | Sonda comune congelata, RMSE K |
|---|---:|---:|
| Casuale | — | 4.287 |
| MVM | — | 4.347 |
| Future | 3.886 | 3.868 |
| Joint | 4.113 | 3.964 |
| Residuo advettivo | 3.869 | 3.830 |

Baseline sullo stesso holdout: **persistenza 4.467 K**, **advezione 4.109 K**.
MVM ha una loss finale di ricostruzione mascherata pari a 0.276088 in unità
normalizzate; questa loss non è una previsione futura e non va inserita nella
stessa classifica. In questo breve pilot, la sonda MVM non migliora il
controllo casuale; Future e Residual mostrano un vantaggio che richiede
conferma su più periodi e seed. I soli 0.016 K fra Future e Residual end-to-end
non giustificano una conclusione sulla superiorità di un obiettivo.

L'aggregato nasconde differenze utili: su IR108 a +15 minuti Future ha RMSE
4.899 K contro 4.606 K dell'advezione; a +30 minuti scende a 5.383 K contro
6.516 K. Le sue immagini sono però più lisce e l'errore sul contrasto P95−P05
è 2.769/2.972 K contro 0.529/1.005 K dell'advezione. Residual conserva meglio
il contrasto (0.516/1.079 K) grazie alla baseline incorporata, senza migliorare
tutte le metriche dei gradienti. RMSE, minimi, contrasto e gradienti vanno
quindi letti insieme, per canale e orizzonte. Nelle previsioni delle sonde
sono visibili anche artefatti a patch: la riduzione della RMSE non equivale
alla conservazione del dettaglio spaziale.

La validation comprende 246 clip a bassa variabilità, 125 media e 141 alta,
usando i terzili del **training selezionato** (0.745107 e 1.439074 K), distinti
dai terzili del pool usati per il campionamento. Future peggiora l'advezione
nelle scene meno variabili (2.296 contro 2.017 K) e la migliora nelle più
variabili (5.578 contro 6.253 K). Questi gruppi non sono classi di convezione.

Le ablazioni end-to-end mantengono target e baseline cronologiche, cambiando
soltanto gli input dell'encoder:

| Modello | Cronologico, RMSE K | Primi nove frame permutati | Ultimo frame ripetuto |
|---|---:|---:|---:|
| Future | 3.886 | 4.642 | 4.125 |
| Joint | 4.113 | 4.222 | 4.295 |
| Residual | 3.869 | 5.543 | 4.080 |

Future e Residual peggiorano quando si altera la storia. Questo mostra una
dipendenza dalla sequenza, ma non isola termodinamica, convezione o moto reale:
gli interventi modificano anche la distribuzione degli input. Il controllo
statico misura il beneficio della storia fornita all'encoder oltre all'ultimo
frame. Nel modello Residual la baseline mantiene comunque il moto stimato
dalla storia originale: non è un'ablazione dell'intera pipeline a singolo frame.

Le 70 sonde ridge (14 osservabili × 5 encoder) mostrano che il controllo
casuale è forte per molte statistiche semplici. Sul cambiamento medio IR108
a +30 minuti, R² è 0.851 per il casuale, 0.847 per MVM, 0.846 per Future,
0.845 per Joint e 0.838 per Residual. Sul MAE del residuo advettivo futuro
a +30 minuti, invece, Residual raggiunge R² 0.916 contro 0.897 del casuale,
0.849 Future, 0.811 Joint e 0.772 MVM. È un risultato specifico del descrittore
e del pilot; la rappresentazione non migliora uniformemente tutti i compiti.

Gli embedding salvati sono 4.096 training e 512 validation per ciascun
encoder, con 768 componenti. Ogni controllo embedding contiene 512 clip ×
tre interventi (shuffle, ultimo frame, sola advezione). L'analisi finale
include 140 righe di metriche canale/orizzonte, 420 stratificate per
variabilità e 24 valutazioni temporali. I cinque hash encoder sono invariati
durante il probing; i nove export Hugging Face coincidono con i checkpoint.
Gli hash dei 14.032 file sorgenti selezionati sono stati ricontrollati dopo
training e analisi, mentre l'upload proseguiva. Il controllo indipendente è
salvato in `results/ssl_pilot_20261024/completion_audit.json`.

La suite completa aggiornata passa **246 test**; compilazione Python riuscita.
I warning delle dipendenze sono di deprecazione. Log della verifica:
`results/ssl_pilot_20261024/final_pytest.log`.

Le sette celle di codice del notebook sono state eseguite senza errori;
verificati anche i callback dei menu, l'avvio/arresto del monitor automatico,
i quattro esempi e più frame della ricostruzione. La copia con output è
`results/ssl_pilot_20261024/study_seviri_ssl_training.executed.ipynb`;
il report dei controlli è `notebook_validation.json` nella stessa cartella.
Le figure di loss, previsioni, ricostruzione e ablazioni sono state ispezionate.
Il Play usa il collegamento JavaScript standard di ipywidgets: il test
automatico esercita il kernel e i callback, non un browser.

## Confronto 4.096 / 8.192 clip, 150 epoche — 25 ottobre 2026

La serie di esperimenti `results/ssl_scaling_20261025/` usa due training set annidati.
`n4096/dataset/train.csv` conserva esattamente il manifest del pilot:
2.752 clip del 12 settembre e 1.344 del 13 settembre. `n8192/` aggiunge
4.096 clip del 15 settembre, selezionate da 8.192 candidati validati:
2.048 mediante campionamento uniforme e 2.048 mediante campionamento
ponderato per variabilità storica IR108. Gli Zarr caricati dal 12 al 17
settembre forniscono sequenze utilizzabili; ogni crop selezionato ha tutti
i chunk fisici necessari per i suoi 16 frame. Le giornate successive sono
escluse dallo snapshot congelato della serie di esperimenti.

Entrambe le dimensioni usano le stesse 512 clip di validation del
17 settembre. L'ultima osservazione del training ampliato precede la prima
osservazione di validation di 24 ore e 5 minuti. La validation storica del
14 settembre rimane associata al pilot. I risultati della nuova serie di esperimenti
si confrontano sulla nuova validation comune.

Le normalizzazioni globali sono calcolate sul rispettivo training set:
media/deviazione standard circa 264,0/20,8 K per 4.096 clip e 264,2/20,8 K
per 8.192 clip. I file JSON conservano la precisione del calcolo. Le soglie
delle classi Low/Medium/High sono invece comuni e provengono dalle 4.096
clip originarie, circa 0,75 e 1,44 K. Il flag
`--variability-reference-manifest` verifica che il riferimento sia un
sottoinsieme del training e lega il relativo hash al checkpoint.

Il programma dei training comprende quattro obiettivi e cinque probe per
dimensione, quindi **18 run da 150 epoche**. Per ciascun obiettivo si esegue
prima il training da 4.096 clip e poi quello da 8.192. Tutti partono dal seed
20261024. La configurazione iniziale usava microbatch 16 e accumulo 2,
con AdamW, learning rate costante 0,00015 e bfloat16. La revisione esecutiva
descritta sotto passa a batch 32 e accumulo 1. Le probe hanno la stessa architettura e inizializzazione della
head e ricevono il training set della rispettiva dimensione. Questa scelta
misura l'effetto dei dati sull'intera procedura di pretraining e probing;
la probe con encoder casuale fornisce il riferimento per l'ampliamento del
solo campione usato dalla head.

A 150 epoche i due dataset comportano rispettivamente 19.200 e 38.400
aggiornamenti. Le curve e le tabelle salvano anche confronti a pari numero
di aggiornamenti: per esempio 150 epoche su 4.096 clip e 75 su 8.192.
Il criterio finale usa i pesi dell'ultima epoca prevista. Il numero di epoche
misura l'avanzamento di ciascun run; le repliche statistiche richiederanno
ulteriori seed e periodi di osservazione.

Preparazione, con ripresa dei blocchi di candidati già validati:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python scripts/build_seviri_ssl_scaling_dataset.py \
  --base-dataset results/ssl_pilot_20261024/dataset \
  --zarr-root /home/isac/deep_convection_data/MSG-SEVIRI \
  --output-dir results/ssl_scaling_20261025 \
  --count 8192 --candidates 8192 --validation-count 512 --workers 8
```

Comando della configurazione iniziale, conservato come riferimento storico. La ripresa attuale usa il supervisore della revisione esecutiva descritto nella sezione seguente:

```bash
.venv/bin/python -u scripts/run_seviri_ssl_scaling.py \
  --output-dir results/ssl_scaling_20261025 \
  --zarr-root /home/isac/deep_convection_data/MSG-SEVIRI \
  --epochs 150 --probe-epochs 150 --batch-size 16 --workers 8 --seed 20261024
```

Il supervisore usa lock esclusivi e verifica i file sorgente del dataset e
gli hash del codice. Salta i run conclusi e riprende gli altri dall'ultimo
checkpoint atomico completo, inclusi optimizer, scheduler e generatori
casuali. `launcher.json` registra il PID del supervisore; `queue_status.json`
registra il processo attivo, e `queue.log` registra i passaggi tra i run.
Prima di rilanciare un comando, verificare il processo indicato dal launcher.

Il notebook unificato `notebooks/study_seviri_ssl_training.ipynb` legge la serie di esperimenti e
mostra conteggi, avanzamento, curve per obiettivo e metriche per canale e
orizzonte con scala verticale comune a +15/+30 minuti. Il supervisore
aggiorna ogni 30 secondi `training_status.csv`, `learning_curves.csv`,
`paired_comparisons.csv` e `latest_channel_metrics.csv` dai risultati
committati. A training concluso esegue per entrambe le dimensioni l'analisi
temporale, le ridge probe, i controlli dei pesi congelati e gli esempi.

La verifica iniziale controlla l'inclusione esatta delle 4.096 clip originali,
l'identità delle validation, la separazione temporale, le statistiche e
le soglie comuni; è salvata in `dataset_audit.json`. La suite aggiornata
passa 261 test, inclusi campionamento, ripresa, gruppi comuni e confronti
a pari budget; log in `results/ssl_scaling_20261025/pytest.log`.

L'avvio GPU è verificato fino al checkpoint della seconda epoca di Future
su 4.096 clip, con modello, optimizer, scheduler e RNG salvati. La seconda
epoca richiede circa 51 secondi di training e 10 secondi di validation.
Usando l'ultima epoca del pilot per stimare ciascun obiettivo e scalando il
tempo di training con il numero di clip, la proiezione iniziale per i 18 run
è circa 65 ore. Il costo della nuova dimensione verrà misurato dal relativo
run. `launch_audit.json` registra verifica dei processi, checkpoint e stima;
`notebook_snapshots/startup.executed.ipynb` conserva il notebook eseguito
durante l'avvio. Le metriche finali saranno disponibili al termine della coda.

## Clip su file e revisione dell'esecuzione — 25 ottobre 2026

La revisione `execution_revisions/clip_files_v1/revision.json` adotta clip
complete compresse con LZ4 senza perdita. La cartella
`/home/isac/deep_convection_data/ssl_clip_files_20261025` contiene 8.192 clip
uniche di training e 512 di validation, per 39,6 GiB. Il manifest da 4.096
riutilizza i file del sottoinsieme comune. Ogni file contiene temperature
Kelvin float16 nella forma 16 × 7 × 224 × 224, con orientamento e ordine
temporale verificati dal reader originale. Il salvataggio ha richiesto circa
2,5 minuti; ogni clip è stata decompressa e confrontata bit per bit con i
valori float16 sorgente. I byte scritti sono stati riletti prima della
pubblicazione. Il loader verifica il checksum a ogni lettura.

Quattro thread preparano batch completi in memoria host pinned. Il trasferimento
usa float16; conversione a float32, normalizzazione globale e controllo dei
valori finiti vengono eseguiti sulla GPU. Le statistiche globali rimangono
quelle del rispettivo training set. Training e validation usano entrambi
questo percorso. Il batch contiene 32 clip e produce un aggiornamento dei
pesi. La configurazione precedente accumulava due batch da 16 per ottenere
lo stesso batch effettivo di 32.

| Dataset | Batch di training precedenti | Batch di training attuali | Aggiornamenti per epoca |
|---|---:|---:|---:|
| 4.096 clip | 256 | 128 | 128 |
| 8.192 clip | 512 | 256 | 256 |
| Validation, 512 clip | 32 | 16 | — |

MVM su 8.192 clip ha adottato la revisione dopo il checkpoint dell'epoca 8.
I tre run già conclusi conservano i risultati a 150 epoche. La ripresa
trasferisce modello, optimizer, scheduler e generatori casuali dal checkpoint
verificato. Gli hash del protocollo originale, dei checkpoint di origine,
della nuova implementazione e della collezione di clip sono registrati nella
revisione. Il checkpoint successivo contiene anche la cronologia della modifica.
Il confronto delle identità preserva tutti i parametri scientifici e il
batch effettivo. Il nuovo raggruppamento può modificare l'ordine delle
riduzioni numeriche e l'assegnazione delle maschere casuali ai campioni.

Per riprendere la serie con la revisione già preparata:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -u scripts/run_seviri_ssl_series.py \
  --output-dir results/ssl_scaling_20261025 \
  --revision results/ssl_scaling_20261025/execution_revisions/clip_files_v1/revision.json
```

`scripts/materialize_ssl_clip_files.py` prepara una nuova collezione in una
cartella vuota; accetta più opzioni `--manifest`, una per ciascun manifest
da includere, e deduplica le clip condivise. Il supervisore conserva i lock
esclusivi e verifica sorgenti, manifest e revisione prima dei run.

Le misure su epoche complete sono salvate in
`execution_revisions/clip_files_v1/measurements/` e confrontate in
`comparison.json`; il notebook delle prestazioni presenta tempi, batch e
attività CUDA. I tempi a regime escludono la prima epoca dopo l'avvio e le
epoche registrate con Nsight. I file `timing_epochs/epoch_*.json` distinguono
training, validation, salvataggio e durata complessiva. La percentuale CUDA
misura la frazione del tempo con kernel attivi sulla partizione assegnata;
la seconda misura include anche copie e inizializzazioni della memoria.

La suite completa passa 273 test, inclusi conservazione dei valori,
normalizzazione, ordine e ripresa del campionamento, rifiuto di file alterati
e vincoli della revisione. Il log è in
`execution_revisions/clip_files_v1/tests.log`.

## Monitoraggio della serie tramite SSH

`scripts/status_seviri_ssl.py` usa la libreria standard di Python e legge
manifest della serie, avanzamento e tempi già salvati. Verifica PID, comando
e stato dei processi del supervisore e del training. Un PID assente, riciclato
per un altro comando o sospeso viene identificato nel riepilogo. Un processo
presente con avanzamento fermo da oltre tre minuti viene segnalato per una
verifica; `--stale-after` permette di impostare questa soglia in secondi.

Da PowerShell su Windows, sostituire utente del bridge e indirizzi:

```powershell
ssh -J UTENTE_BRIDGE@IP_BRIDGE isac@IP_WORKSPACE "python3 /home/isac/deep_convection/scripts/status_seviri_ssl.py"
```

Per mantenere aperto il riepilogo e aggiornarlo ogni 15 secondi:

```powershell
ssh -t -J UTENTE_BRIDGE@IP_BRIDGE isac@IP_WORKSPACE "python3 /home/isac/deep_convection/scripts/status_seviri_ssl.py --watch 15"
```

Ctrl+C chiude il monitor; il supervisore e i training proseguono nei loro
processi indipendenti. Il report comprende gli esperimenti conclusi sui 18
previsti, le epoche salvate sul totale pianificato, la fase corrente
(training, validation o analisi finale), l'epoca e il batch correnti,
l'età dell'ultimo aggiornamento e la tabella completa degli esperimenti.
Le date sono mostrate nel fuso Europe/Paris.

La percentuale complessiva indica la quota delle epoche pianificate già
salvate. La stima temporale riguarda il singolo esperimento attivo e usa
la mediana delle ultime cinque epoche disponibili nella revisione esecutiva
corrente, includendo il salvataggio quando il relativo tempo è disponibile.
I costi delle epoche dipendono dall'obiettivo e dal numero di clip.

`--json` restituisce un oggetto JSON per interrogazione, utilizzabile in
script esterni; `--series /percorso/della/serie` seleziona un'altra serie
compatibile. Il comando compie letture dei metadati e può essere usato
durante il training senza caricare modelli o clip.

## Studio scientifico unificato e confronto dei risultati disponibili

`notebooks/study_seviri_ssl_training.ipynb` riunisce il confronto fra i quattro
obiettivi SSL e quello fra i dataset da 4.096 e 8.192 clip. La sorgente
predefinita è `results/ssl_scaling_20261025`, con validation del 17 settembre.
`study_seviri_ssl_scaling.ipynb` contiene il rimando al notebook principale.
Il notebook delle prestazioni rimane dedicato a caricamento, velocità e GPU.

La configurazione acquisisce uno snapshot dei progressi salvati. Il modulo
`src/emma_gpm/ssl_study.py` controlla gli hash dei manifest e delle statistiche,
l'identità della validation fra dimensioni e la completezza dei report.
Ogni confronto legge `validation_epochs/epoch_*.json` per un'epoca completata
comune ai modelli disponibili. Le tabelle esplicitano modelli disponibili,
modelli pianificati, epoca scelta e stato finale o provvisorio del confronto.
L'RMSE globale viene ricostruito dalle medie degli errori quadratici per
canale e orizzonte, coerentemente con il pooling su tutti i pixel.

I risultati comprendono:

- previsioni Future, Joint e Residual rispetto a Persistence e Advection;
- confronto fra dimensioni a pari epoche e pari aggiornamenti dei pesi;
- curve separate di training e validation e minimo registrato nella traiettoria;
- errori per canale e orizzonte, con scala comune a +15/+30 minuti;
- prestazioni nei gruppi comuni di variabilità Low, Medium e High;
- gradienti, minimi termici, contrasti e variazioni tra i due istanti previsti;
- ricostruzione storica MVM/Joint e confronto comune mediante probe congelate;
- riduzione dell'RMSE dei probe rispetto all'encoder casuale a pari dimensione ed epoca;
- incremento dell'errore dopo permutazione della storia o ripetizione dell'ultimo frame;
- figure di validation già salvate e disponibilità delle analisi temporali e degli embedding.

Le comparazioni finali usano l'epoca 150. I minimi osservati durante il training
descrivono la traiettoria di ottimizzazione. Le probe e le analisi finali
vengono incluse progressivamente quando i relativi prodotti sono disponibili.
Il codice legge metriche e immagini salvate; l'esecuzione del notebook lascia
proseguire i processi di training e il loro protocollo congelato.

Figure, CSV, copie precedenti e notebook eseguito sono in
`results/ssl_scaling_20261025/notebook_review_20261025/`. Il notebook sorgente
rimane privo di output salvati. Per aggiornare l'intero confronto, rieseguire
le celle dalla configurazione iniziale con il kernel `Python (deep_convection)`.

La verifica finale del 26 ottobre 2026 include 18 training completati a 150
epoche e le analisi finali di entrambi i dataset. I controlli salvati confermano
che i pesi dei dieci encoder congelati sono rimasti invariati durante il fitting
dei probe. Il notebook esplicita anche la conservazione degli estremi termici
e i risultati dei ridge probe, per leggere il ranking radiometrico insieme
alle proprietà delle rappresentazioni.

## Estensione a 16.384 clip: Future e Joint

La serie `results/ssl_extension_20261026` esegue i due obiettivi approvati il
26 ottobre 2026. Ogni modello parte dagli stessi pesi casuali del protocollo
precedente, con seed 20261024, VideoMAE-Small, AdamW a 0.00015 e bf16.
Il budget è di 75 epoche, 512 batch e aggiornamenti per epoca, batch effettivo
32 e 38.400 aggiornamenti totali. Future precede Joint nell'esecuzione sequenziale.

Il dataset conserva tutte le 8.192 clip del training precedente. Le 8.192 aggiunte
sono selezionate da un pool validato di 16.384 candidati disgiunti mediante la
stessa miscela di campionamento uniforme e pesato sulla variabilità. I candidati
provengono dalle giornate di training del 12, 13 e 15 settembre 2020. La validation
rimane identica, con 512 clip del 17 settembre; le soglie di variabilità continuano
a derivare dal training originale da 4.096 clip. Media e deviazione standard
globali sono ricalcolate su tutti i pixel delle 16.384 clip finali.

La preparazione riproducibile usa:

```bash
.venv/bin/python scripts/build_seviri_ssl_extension.py \
  --base-series results/ssl_scaling_20261025 \
  --output-dir results/ssl_extension_20261026 \
  --zarr-root /home/isac/deep_convection_data/MSG-SEVIRI \
  --count 16384 --candidates 16384 --workers 8

.venv/bin/python scripts/extend_ssl_clip_files.py \
  --manifest results/ssl_extension_20261026/n16384/dataset/train.csv \
  --manifest results/ssl_extension_20261026/n16384/dataset/validation.csv \
  --reuse-root /home/isac/deep_convection_data/ssl_clip_files_20261025 \
  --output-dir /home/isac/deep_convection_data/ssl_clip_files_20261026_n16384 \
  --zarr-root /home/isac/deep_convection_data/MSG-SEVIRI --workers 8

.venv/bin/python -u scripts/run_seviri_ssl_selected.py \
  --output-dir results/ssl_extension_20261026 \
  --reference-series results/ssl_scaling_20261025 \
  --clip-root /home/isac/deep_convection_data/ssl_clip_files_20261026_n16384 \
  --zarr-root /home/isac/deep_convection_data/MSG-SEVIRI \
  --epochs 75 --batch-size 32 --workers 4
```

I piani e i sorgenti vengono congelati prima del training. Il supervisore controlla
gli hash e riprende ciascun modello dall'ultimo checkpoint completo quando viene
rilanciato. I report per canale, orizzonte, variabilità e struttura sono prodotti
a ogni epoca. La selezione comprende soltanto i due training approvati.

Il comando di stato predefinito segue la serie indicata in
`results/ssl_active_series.json`. Per selezionare questa estensione esplicitamente:

```bash
python3 /home/isac/deep_convection/scripts/status_seviri_ssl.py \
  --series /home/isac/deep_convection/results/ssl_extension_20261026 --watch 15
```

### Risultati completati dell'estensione

Future e Joint hanno completato 75 epoche ciascuno. Il confronto principale
usa 38.400 aggiornamenti per modello, equivalenti a 150 epoche su 8.192 clip
e 75 epoche su 16.384 clip. La validation è identica, con 512 clip.

| Modello | RMSE con 8.192 clip | RMSE con 16.384 clip | Riduzione |
|---|---:|---:|---:|
| Future | 3,77 K | 3,67 K | 2,8% |
| Joint | 3,99 K | 3,92 K | 1,9% |

Future migliora in tutte le 14 combinazioni canale–orizzonte; Joint migliora
nei cinque canali IR a entrambi gli orizzonti, mentre WV_062 e WV_073 mostrano
errori maggiori. Le medie degli ultimi dieci checkpoint condivisi a pari
aggiornamenti mantengono il vantaggio del dataset più grande.

Nel canale IR108 a +30 minuti, l'errore sulla temperatura minima della clip
passa da 6,03 a 5,98 K per Future e da 6,67 a 6,83 K per Joint. L'advezione
rimane a 2,26 K su questa metrica. Il confronto distingue accuratezza
radiometrica media e conservazione degli estremi freddi.

Le curve Future mostrano picchi transitori di errore: con 16.384 clip, alle
epoche 36 e 46 l'RMSE sale a circa 8,8 K e recupera nelle epoche successive.
Anche la loss di training aumenta in prossimità dei picchi. L'origine di queste
oscillazioni richiede un controllo dedicato dell'ottimizzazione; la graduatoria
riportata usa il budget finale registrato e la tabella dei checkpoint finali.

Il notebook unificato mostra il confronto aggiornato. Figure, tabelle, audit
dei parametri e copia eseguita sono in
`results/ssl_extension_20261026/comparison_review/`. Il confronto riguarda un
seed e una giornata di validation, con clip sovrapposte. Le nuove clip ampliano
il campionamento delle tre giornate già presenti nel training.


## VideoMAE-Base: 32.768 clip, Future e Joint, 150 epoche

La configurazione autorizzata il 26 ottobre 2026 aumenta insieme la capacità
del modello, il numero di clip campionate e il numero di aggiornamenti.
`results/ssl_base_32768_20261026` raccoglie preparazione, misure e training.
L'encoder Base usa 768 componenti, 12 blocchi, 12 teste di attenzione e circa
88 milioni di parametri; il decoder MVM ha quattro blocchi da 384 componenti.
La testa di previsione mantiene larghezza 192. Future ha circa 90 milioni
di parametri complessivi, Joint circa 98 milioni. Tutti i pesi iniziali sono
casuali, con seed 20261024.

Future e Joint vengono eseguiti in sequenza, ciascuno per 150 epoche, con
batch fisico ed effettivo 32 e bf16. Ogni epoca comprende 1.024 aggiornamenti;
il totale è 153.600 per obiettivo. Learning rate 0,00015, AdamW, weight decay
0,05, mascheramento storico al 90% e pesi delle due loss Joint pari a uno
restano quelli del protocollo precedente. Le statistiche globali sono
ricalcolate su tutte le clip del nuovo training set.

Il manifest conserva le 16.384 clip precedenti e aggiunge 16.384 finestre
uniche su 12, 13 e 15 settembre 2020. Il campionamento aggiuntivo usa per metà
estrazione uniforme e per metà pesi di variabilità storica IR108. Il pool
aggiuntivo validato comprende 32.768 candidati esclusi dal manifest precedente.
La validation mantiene le stesse 512 clip del 17 settembre, con separazione
temporale minima di 24 ore. L'aumento delle finestre sfrutta ulteriormente
le tre giornate osservate; la loro sovrapposizione introduce osservazioni condivise.

Per rispettare lo spazio disponibile, i nuovi training leggono quattro file
NPY giornalieri, con i valori float16 Kelvin originali, in
`/home/isac/deep_convection_data/cache_native_seviri_20261026_n32768`.
La raccolta occupa circa 20 GiB logici, di cui circa 10 GiB riutilizzati dalla
cache precedente mediante hard link. Il percorso conserva ordine temporale,
canali e orientamento geografico. Ogni file viene verificato tramite SHA256
all'apertura; durante l'uso vengono controllati dimensione, data di modifica
e inode. Ogni batch viene controllato per valori finiti prima del modello.
Lo shuffle riguarda l'ordine delle clip e viene rinnovato a ogni epoca.

Il benchmark con il loader di produzione confronta esattamente i bit float16
di 64 clip casuali con i file per clip precedenti. Su H200 MIG con circa
32 GiB disponibili, batch 32 usa circa 14 GiB per Future e 17 GiB per Joint.
I 32 aggiornamenti misurati dopo il riscaldamento richiedono rispettivamente
circa 0,32 e 0,46 secondi; l'attesa media del loader è circa 2 millisecondi.
Queste misure brevi precedono la verifica delle epoche complete. I risultati
sono in `frame_loader_benchmark/result.json`, i test di integrazione con dati
reali e ripresa in `runtime_smoke.json`, la suite software in `software_tests.log`.

Lancio e ripresa della serie, con rilevamento automatico dei checkpoint:

```bash
.venv/bin/python -u scripts/run_seviri_ssl_selected.py \
  --output-dir results/ssl_base_32768_20261026 \
  --reference-series results/ssl_extension_20261026 \
  --reference-case n16384 \
  --variability-reference-manifest results/ssl_scaling_20261025/n4096/dataset/train.csv \
  --frame-cache /home/isac/deep_convection_data/cache_native_seviri_20261026_n32768 \
  --zarr-root /home/isac/deep_convection_data/MSG-SEVIRI \
  --model-size base --epochs 150 --batch-size 32 --gradient-accumulation 1 --workers 4
```

Il supervisore conserva piani immutabili e copie dei sorgenti, esegue la
validation a ogni epoca e salva checkpoint atomici con optimizer e generatori
casuali. Lo stato remoto usa `scripts/status_seviri_ssl.py`; il notebook
`study_seviri_ssl_training.ipynb` include progressi, risultati Base disponibili
e confronti Small/Base al numero di aggiornamenti condiviso.

La prima epoca completa di Future è stata verificata nella notte tra il
26 e il 27 ottobre: 329 secondi di training, 8 secondi di validation su tutte
le 512 clip e circa 3 secondi per checkpoint e log, per un totale di
340 secondi. Il picco di memoria allocata è circa 14 GiB. Il checkpoint
contiene 1.024 aggiornamenti; Future procede dalla seconda epoca e Joint
è in coda nel supervisore. La durata iniziale stimata per l'intera sequenza
è circa 34 ore, combinando l'epoca completa Future e il benchmark Joint.
I dati di verifica sono in `implementation_audit.json`; il notebook eseguito
con i primi risultati Base è in `comparison_review/` della nuova serie.


## Archiviazione dei dataset conclusi su /data1 — 27 ottobre 2026

Le raccolte `ssl_clip_files_20261025` e `ssl_clip_files_20261026_n16384`
risiedono ora in `/data1/deep_convection_data/`, sul filesystem ext4 di
`/dev/sdb`. I rispettivi percorsi in `/home/isac/deep_convection_data/`
sono collegamenti simbolici alle raccolte trasferite. La migrazione ha
preservato i metadati e tutti gli hard link tra file condivisi, verificando
SHA256 per 25.606 percorsi corrispondenti a 16.902 file fisici. Sono stati
liberati circa 77 GiB; lo spazio disponibile sul disco di sistema è passato
da circa 9,6 a 86 GiB.

Il training Base continua a usare
`/home/isac/deep_convection_data/cache_native_seviri_20261026_n32768`.
Durante questa prima operazione, l'rsync in ingresso ha continuato a scrivere
in `/home/isac/deep_convection_data/MSG-SEVIRI`. Le verifiche successive confermano
la lettura dei dataset Small attraverso i percorsi originali, l'integrità
degli input Base e la prosecuzione dei due processi.

Le revisioni di esecuzione di eventuali future riprese dei checkpoint Small
dovranno registrare il nuovo percorso canonico `/data1/`, poiché il trainer
confronta anche il percorso risolto del backend di input. I piani e i checkpoint
storici conservano i loro metadati originali. I report e la corrispondenza tra
percorsi sono in `results/storage_migration_20261027/`.


## Destinazione dell'upload Zarr su /data1 — 27 ottobre 2026

Dopo l'arresto del trasferimento confermato dall'utente, l'intero archivio già
ricevuto è stato copiato in `/data1/deep_convection_data/MSG-SEVIRI/`: circa
140 GB in 662.993 file. Ogni file è stato confrontato con l'originale tramite
SHA256; un confronto finale ha verificato dimensioni, date di modifica,
permessi e struttura delle directory.

L'archivio del 2022 risiede ora su `/data1`; il vecchio percorso
`/home/isac/deep_convection_data/MSG-SEVIRI/2022` è un collegamento simbolico
alla nuova posizione. La directory principale `MSG-SEVIRI` sotto `/home/isac/`
rimane una directory reale, con gli originali del 2020 al loro posto.
Anche i quattro file NPY di `cache_native_seviri_20261026_n32768` rimangono
fisicamente sul disco di sistema. I percorsi canonici registrati dal
supervisore Base conservano quindi la loro identità.

Il controllo successivo conferma che tutti i 78.652 file originali del 2020,
i quattro file della cache e i 24 sorgenti registrati dal piano Base sono
invariati. Il loader ha riaperto una clip per ciascuna delle quattro giornate,
verificando forma, valori finiti e provenienza della cache; anche le
statistiche globali restano compatibili. Joint ha continuato nell'epoca 13/150
con lo stesso processo di training.

Per riprendere l'upload dal PC Windows, utilizzare il comando rsync precedente
con la destinazione remota `isac@IP_WORKSPACE:/data1/deep_convection_data/MSG-SEVIRI/`.
La copia verificata contiene anche il 2020, permettendo a rsync di saltare
i file già completi con dimensione e data di modifica corrispondenti.
I nuovi dati caricati vengono così collocati sul disco capiente. La nuova
radice `/data1/` va usata anche per preparare dataset dalle osservazioni che
arriveranno; i training già avviati conservano la loro radice originale.

Lo spazio disponibile sul disco di sistema è ora circa 190 GiB. Su `/data1`
restano circa 3,8 TB disponibili, sufficienti per completare l'archivio locale
da circa 600 GB. Il report di integrità e il riepilogo della migrazione sono
in `results/archive_migration_20261027/`.


## Confronti unificati e protocollo dei probe — 28 ottobre 2026

Il notebook `study_seviri_ssl_training.ipynb` integra le tre serie registrate
in un unico insieme di risultati: Small a 4.096, 8.192 e 16.384 clip, Base a
32.768 clip. Tutti i 22 training registrati sono conclusi. Il lettore verifica
manifest, validation comune e annidamento dei campioni prima del confronto.
Le sezioni aggiuntive dedicate alle sole 16.384 clip sono state assorbite nei
confronti comuni.

Le curve dei quattro obiettivi usano colori per MVM, Future, Joint e Residual,
e stili di linea per architettura e numero di clip. Mostrano training e
validation rispetto a epoche e optimizer updates. La loss è espressa in K²:
per Joint è la somma pesata dei due errori quadratici medi, con i pesi
registrati; per gli altri obiettivi è il rispettivo errore di ricostruzione
o previsione. Le scale logaritmiche mostrano l'intera traiettoria. I livelli
di loss si interpretano entro ciascun obiettivo; gli errori di previsione
successivi valutano lo stesso target futuro.

Il confronto Small da 8.192 a 16.384 clip conserva architettura e 38.400
aggiornamenti: Future migliora da 3,77 a 3,67 K, Joint da 3,99 a 3,92 K.
L'ampliamento campiona più finestre nelle stesse tre giornate. Base a 32.768
clip termina 153.600 aggiornamenti con Future a 3,78 K e Joint a 3,70 K.
Il confronto fra Small e Base a 37.888 aggiornamenti, ultimo valore condiviso
dai report di epoca, dà rispettivamente 3,68 e 3,70 K per Future, 3,92 e
3,71 K per Joint. Capacità e campionamento cambiano insieme; nuovi periodi
osservativi e seed permetterebbero di valutare la riproducibilità del risultato.

I probe di previsione disponibili sono quelli delle serie da 4.096 e 8.192
clip. Ogni encoder resta congelato; una nuova testa con convoluzioni 1×1,
3×3 e trasposta viene addestrata per 150 epoche sugli stessi manifest di
training del relativo encoder. La testa Small ha circa 1,4 milioni di
parametri. La validation usa le 512 clip del 17 settembre, separate nel tempo
dai dati di fitting. I controlli degli hash confermano i pesi congelati.
Le sonde radiometriche separate usano una regressione ridge lineare su
embedding aggregati, con lo stesso split, per recuperare descrittori
osservativi. Un confronto di trasferimento richiederebbe un dataset comune
per il fitting delle sonde, con giornate escluse da tutti i pretraining,
seguito da un ulteriore periodo di valutazione.

La ricostruzione storica è mostrata con barre ed esempi spaziali. La figura
chiamata `Reconstruction` conserva le osservazioni nei pixel visibili e
riempie quelli mascherati con le predizioni; questo descrive la precedente
dicitura `Composite reconstruction`. Il testo definisce anche il condizionamento
per variabilità storica e le due coordinate dei diagrammi RMSE/Gradient MAE.
Output eseguiti e verifiche sono in `results/notebook_revision_20261028/`:
notebook eseguito in circa 11 secondi e suite completa con 302 test superati.

La successiva revisione dei grafici, sempre del 28 ottobre, è in
`results/notebook_revision_20261028_plots/`. Le traiettorie usano scala
logaritmica anche per epoche e aggiornamenti. I confronti per canale
affiancano tutti i training nello stesso pannello, con Persistence e Advection
come linee di riferimento; +15 e +30 minuti mantengono gli stessi limiti
verticali per ciascun canale. La riga `All channels` aggrega gli errori
quadratici prima della radice per RMSE. Per le altre metriche usa la media
dei valori per canale, esplicitata nel testo.

Il confronto a budget uguale mostra la riduzione di RMSE fra dataset adiacenti,
con il budget esatto nel titolo. La variabilità storica mostra il miglioramento
percentuale rispetto ad Advection entro Low, Medium e High: verifica se il
vantaggio del modello riguarda anche le sequenze che cambiano rapidamente.
Le tabelle sono disattivate nella presentazione predefinita e disponibili con
`SHOW_TABLES = True`; i dati numerici restano nei CSV. Anche i risultati ridge
e il confronto dei probe con Random hanno una rappresentazione grafica.

Il testo specifica gli output dei probe: due immagini future a sette canali
di 224×224 pixel per la testa convoluzionale, 14 descrittori scalari per le
ridge. Le sintesi delle ultime due finestre da 30 epoche Base sono salvate in
`late_training_windows.csv`. La mediana dell'RMSE di validation di Joint è
circa 3,72 K in entrambe le finestre, mentre l'RMSE di training scende da
1,94 a 1,87 K. Per Future le mediane di validation passano da 3,77 a 3,78 K.
Questa evidenza orienta il prossimo confronto verso una maggiore diversità
di giornate, mantenendo inizialmente fissa l'architettura; un controllo mirato
del budget Small Joint a 16.384 clip resta utile. La revisione si esegue in
circa 13 secondi, mostra 22 figure e supera la suite completa di 304 test.

## Confronto Small/Base con 16.384, 32.768 e 65.536 clip — 28 ottobre 2026

La serie `results/ssl_factorial_20261028` contiene otto esperimenti: Future e
Joint per Small con 32.768 clip, Base con 16.384 clip, Small con 65.536 clip e
Base con 65.536 clip, in questo ordine. Ogni training parte da pesi casuali,
usa 150 epoche, batch effettivo 32, AdamW con learning rate 0,00015 e weight
decay 0,05, precisione bf16, quattro thread di lettura e shuffle a ogni epoca.
Le configurazioni restano identiche entro ciascuna architettura. Il seed è
20261024; la validation conserva le 512 clip del 17 settembre 2020.

Il budget per epoca è di 512 aggiornamenti con 16.384 clip, 1.024 con 32.768 e
2.048 con 65.536. A 150 epoche i budget sono rispettivamente 76.800, 153.600 e
307.200 aggiornamenti. Il precedente Small con 16.384 clip aveva 75 epoche:
il suo confronto a pari aggiornamenti corrisponde all'epoca 75 del nuovo Base.
I checkpoint Small/Base con 32.768 o 65.536 clip condividono sia la numerosità
sia i budget di epoche e aggiornamenti. I confronti fra numerosità diverse
devono continuare a distinguere questi due budget.

L'estensione conserva integralmente il manifest da 32.768 clip e aggiunge
32.768 finestre estratte dalle osservazioni del 12 ottobre 2010, 13 aprile
2012, 30 settembre 2015 e 28 febbraio 2016. Il pool aggiuntivo contiene 49.152
clip con valori finiti in tutti i sette canali e nei sedici frame, intervalli
esatti di cinque minuti e temperature comprese fra 100 e 400 K. Le finestre
candidate richiedono la presenza fisica di tutti i blocchi Zarr necessari.
Il campionamento conserva una componente uniforme del 50% e una componente
stratificata del 50% sulla variazione storica di IR 10.8, con pesi relativi
1:2:4 per i tre livelli di variabilità. La selezione procede senza ripetizioni;
il protocollo registra il numero finale di clip per data.

Il numero di clip aumenta insieme alla varietà delle giornate osservate.
Questa selezione di date documenta una copertura meteorologica circoscritta;
la validation comune misura la previsione sul 17 settembre 2020. La media e
la deviazione standard globali del nuovo dataset sono ricalcolate su tutti i
pixel, canali e frame delle 65.536 clip di training. Small e Base condividono
esattamente manifest, statistiche e sorgenti per questa numerosità.

I nuovi Zarr selezionati vengono copiati e verificati tramite SHA256 in
`/data1/deep_convection_data/ssl_sources_20261028_n65536/`. La sottodirectory
`2020` di questo nuovo archivio punta alle sorgenti storiche sotto
`/home/isac/deep_convection_data/MSG-SEVIRI/2020/`. I training a 16.384 e
32.768 continuano a utilizzare i loro percorsi e manifest originali.
La cache a frame nativi dell'estensione è
`/home/isac/deep_convection_data/cache_native_seviri_20261028_n65536/`:
circa 40 GiB complessivi, di cui circa 20 GiB condivisi con gli array del
2020 mediante hard link. Gli array delle nuove date aggiungono circa 20 GiB.
La destinazione dell'upload rimane `/data1/deep_convection_data/MSG-SEVIRI/`.

Preparazione, inventario, verifiche e selezione sono riproducibili mediante
`scripts/build_seviri_ssl_new_dates.py`; gli argomenti esatti sono conservati
in `results/ssl_factorial_20261028/data_preparation/launcher.json`. Il file
`dataset_ready.json` viene pubblicato al completamento dei controlli sulle
clip, sulle statistiche e sui checksum. I confronti diretti fra clip lette
da Zarr e da cache sono registrati nel protocollo del dataset.

Il supervisore conserva piani immutabili, copia delle sorgenti eseguibili e
checksum dei dataset. Esegue un training alla volta e riprende da
`checkpoint_latest.pt` quando una precedente esecuzione è interrotta. Il
checkpoint conserva anche optimizer, scheduler e generatori casuali. Per
avviare o riprendere la matrice, dopo aver verificato che il precedente
supervisore sia terminato:

```bash
cd /home/isac/deep_convection
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  scripts/run_seviri_ssl_matrix.py \
  --output-dir results/ssl_factorial_20261028 \
  --config results/ssl_factorial_20261028/requested_matrix.json
```

Per controllare la serie dalla sessione SSH:

```bash
python3 /home/isac/deep_convection/scripts/status_seviri_ssl.py --watch 15
```

Il monitor mostra encoder, obiettivo, clip, epoche salvate, batch corrente,
età dell'ultimo aggiornamento e stato dei processi. I CSV della serie
conservano separatamente `encoder_size` e l'obiettivo di apprendimento.
Le verifiche software e i brevi training di integrazione sono in
`results/ssl_factorial_20261028/`; i run sotto `smoke/` usano sottoinsiemi
isolati e sono registrati separatamente dagli otto esperimenti scientifici.
La verifica del 28 ottobre supera 310 test. I quattro brevi training su dati
reali completano due epoche ciascuno su 128 clip, con 32 clip di validation.
Base Joint riprende poi il checkpoint e completa una terza epoca, passando da
otto a dodici aggiornamenti. Con batch 32 i picchi di memoria allocata sono
circa 8 GiB per Small Future, 9 GiB per Small Joint, 14 GiB per Base Future e
17 GiB per Base Joint.
