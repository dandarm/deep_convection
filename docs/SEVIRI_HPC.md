# Ripresa dello studio SSL su Leonardo

## Correzione delle verifiche ripetute — 4 novembre 2026

Il job 63393282 ha salvato cinque epoche per Base/Future con batch globale 608
e cinque con batch globale 128. A regime il training impiega rispettivamente
159 e 175 secondi per epoca. Queste sono prove di prestazioni e batch size.

Il primo run ha completato anche la validation estesa. La sua preparazione ha
impiegato circa 51 minuti, con attività GPU concentrata negli ultimi 22 secondi.
Il costruttore del dataset ripeteva in ciascuno dei quattro processi i checksum
dei file NPY e degli archivi Zarr originali. Questo era un errore di caricamento.
Il job è stato fermato dopo il secondo checkpoint alla quinta epoca, durante
la preparazione della seconda validation estesa; tutti i pesi sono conservati.

I dataset vengono trattati come immutabili dopo la preparazione e il trasferimento
verificato. Il loader ordinario apre i file NPY senza calcolarne i checksum e
senza riaprire gli archivi Zarr. Anche la copia da disco a `/dev/shm` usa una
normale copia locale, senza verifiche aggiuntive dei byte. Restano i controlli
di forma, canali, orientamento, tempi e valori finiti dei campioni, insieme alle
identità dei manifest e della normalizzazione necessarie alla riproducibilità.
Un controllo completo dei file richiede una richiesta esplicita attraverso
`FrameFileDataset(..., verify_integrity=True)` o `stage_frame_cache(..., verify=True)`.

`evaluate_seviri_ssl_checkpoint_ddp.py` permette di completare la valutazione
di un checkpoint già salvato senza aggiornamenti dei pesi. Registra separatamente
il tempo di preparazione e quello di calcolo per ciascun validation set.

## Proseguimento Base/Future a 150 epoche

La fase `full` del controller usa quattro A100, 32 CPU e 480 GiB di RAM.
Prima completa la validation estesa del checkpoint Base/Future con batch globale
128 e verifica che l'RMSE sul validation set da 512 clip coincida con quello
già salvato, entro 0,0001 K. Poi riprende il checkpoint alla quinta epoca del run
con batch globale 608, ossia 152 clip per GPU, fino a 150 epoche totali.
I 216 aggiornamenti per epoca portano il totale finale a 32.400 aggiornamenti.
Il learning rate rimane 0,00015 e il weight decay 0,05.

La ripresa conserva pesi, momenti e contatori AdamW, stati casuali di ogni rank
e indice dell'epoca. Una revisione esplicita associa il checkpoint precedente
alle sole modifiche del loader e del codice di ripresa, e alla nuova posizione
della copia RAM. Modifiche a modello, dati, loss, batch size o iperparametri
vengono rifiutate. Le riprese successive usano direttamente il checkpoint della
nuova esecuzione, con la stessa versione del codice.

`full_training.json` contiene il traguardo totale delle epoche. Aumentarlo e
rilanciare la stessa esecuzione prosegue dal `training/checkpoint_latest.pt`.
La validation ordinaria viene calcolata a ogni epoca; quella estesa ogni dieci
epoche e all'ultima. La ripresa completa anche una validation finale interrotta
dopo il salvataggio dell'ultimo checkpoint.

## Training con frame condivisi nella RAM del nodo

La revisione del 3 novembre 2026 aggiunge `--stage ram-pilots` al controller.
La prima prova con batch globale 32 ha misurato circa 2,3 secondi per
aggiornamento sui frame mappati da Lustre, con numerose riletture e molta RAM
disponibile. La revisione copia una volta i circa 172 GiB di frame di training
e validation in `/dev/shm`, condivisi dai quattro processi. Dal 4 novembre
questa copia locale usa direttamente i dataset già verificati; i metadati, i valori
float16, i manifest e i controlli sulle sorgenti conservano la stessa identità.
Il percorso RAM è stabile per esperimento e viene ricreato e verificato alla
ripresa su un nodo diverso. L'uscita del job rimuove la propria copia temporanea.

Prima del training vengono misurati batch di 8, 32 e 144 clip per GPU,
con quattro aggiornamenti di riscaldamento e 32 aggiornamenti cronometrati
per batch. Il benchmark legge clip reali e separa preparazione/trasferimento,
forward, backward con DDP e optimizer. Registra anche throughput, letture dal
filesystem e picchi di memoria su ciascuna GPU. I tempi includono sincronizzazioni
di misura; quelli delle epoche successive descrivono la pipeline ordinaria.
La scelta del batch richiede che la memoria misurata rimanga entro il 92%
della VRAM disponibile. La ricerca parte da 144 clip per GPU e verifica valori
adiacenti in passi di otto clip, usando la pipeline reale per affinare il limite.

La prova con il batch massimo viene eseguita per prima, seguita dal batch
di 32 clip per GPU. Entrambe usano Base/Future e cinque epoche. Il riferimento
precedente con batch globale 32 conserva il checkpoint alla quarta epoca;
i risultati riportano il numero effettivo di epoche e aggiornamenti di ogni run.
La validation da 512 clip resta a ogni epoca; quella da 10.240 clip viene
calcolata al termine delle cinque epoche.

Il diagnostico Leonardo `63394589` ha verificato la pipeline su quattro
A100 da 64 GiB, con la revisione congelata `0d28a74`. Il riferimento precedente
su Lustre misurava circa 2,3 s per aggiornamento con batch globale 32, circa
14 clip/s. I benchmark con frame in RAM hanno misurato:

| Clip per GPU | Batch globale | Secondi per aggiornamento | Clip/s | VRAM massima osservata per GPU |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 32 | 0,059 | 540 | 6 GiB |
| 32 | 128 | 0,17 | 740 | 14 GiB |
| 144 | 576 | 0,70 | 830 | 55 GiB |

Sono misure su 32 aggiornamenti dopo quattro di riscaldamento. Tutti i processi
hanno registrato zero byte di letture a blocchi durante la finestra misurata.
Con 144 clip per GPU, la telemetria ha rilevato un utilizzo medio tra il 98%
e il 100% sulle quattro GPU e una potenza media tra 370 e 400 W. L'avvio,
la copia verificata in RAM e le validation sono misurati separatamente.
I tempi dell'intera epoca serviranno a confermare il throughput sul training
completo. I report locali sono in `results/ssl_hpc_ram_20261103/`; quelli HPC
in `$WORK/deep_convection/ddp_ram_20261103/`.

## Prove distribuite su un nodo con quattro GPU

`pretrain_seviri_ssl_ddp.py` distribuisce un singolo modello sulle quattro
GPU mediante PyTorch DDP. Il campionatore costruisce una permutazione globale
diversa a ogni epoca e assegna ciascuna clip a un solo processo. Ogni processo
contribuisce alla media dei gradienti. Il batch globale è il prodotto del
batch per GPU e del numero di GPU; l'accumulo è uno in queste prove.

Le prime tre prove usano Base/Future, 131.072 clip, cinque epoche,
seed 20261024, learning rate 0,00015 e weight decay 0,05. Il riferimento usa
32 clip globali, otto per GPU. La seconda prova usa 32 clip per GPU,
128 globali. La terza usa la massima batch per GPU verificata
con granularità di otto clip e un limite del 92% della memoria disponibile.
Se la capacità misurata coincide con uno dei batch precedenti, una sola
esecuzione fornisce entrambi i confronti; `pilots_plan.json` registra questa scelta.
Una ricerca preliminare include optimizer, backward e comunicazione DDP;
la rifinitura usa la stessa compilazione del training. Numero di aggiornamenti,
clip al secondo, secondi per aggiornamento e memoria di ciascuna GPU vengono
registrati separatamente. A parità di epoche, aumentare il batch riduce il
numero di aggiornamenti dell'optimizer: le curve vanno interpretate anche
rispetto a questo conteggio.

La validation da 512 clip viene eseguita a ogni epoca; quella da 10.240 clip
ogni dieci epoche e all'ultima epoca, quindi alla quinta in queste prove.
Entrambe sono distribuite con indici globali delle maschere invariati.
Le metriche combinano somme degli errori e conteggi dei pixel prima di
calcolare gli RMSE. La selezione del checkpoint migliore usa la validation
da 512 clip. I checkpoint conservano lo stato casuale di ciascun processo
e verificano la stessa topologia alla ripresa.

Lo script `scripts/hpc/seviri_ssl_ddp.sbatch` richiede un nodo, quattro A100,
32 CPU e 480 GiB di RAM. I frame di training e validation occupano insieme
circa 172 GiB; la memoria restante ospita processi, cache e buffer pinned.
Un solo processo Slurm avvia `torchrun` con quattro processi, uno per GPU.
Le impostazioni DDP riprendono `Demetra/moduli/videomae/utils.py`: grafo
statico, gradienti nelle viste dei buffer di comunicazione e bucket da 64 MiB.
La telemetria `gpu_JOBID.csv` registra utilizzo, memoria e potenza delle quattro
GPU ogni due secondi.

La procedura `run_seviri_ssl_ddp_trials.py` esegue prima la fase `capacity`,
poi `pilots` sullo stesso tipo di nodo. I file `runtime_files.json` e
`source_archive/` congelano il codice. La fase dei training controlla che
entrambi i trasferimenti siano verificati prima di accedere ai dati.
La validation estesa risiede in
`$WORK/deep_convection_datasets/seviri_validation_10240_20261102`, mentre
il training conserva la posizione `$FAST` descritta sotto. La mappatura delle
sorgenti include entrambi gli archivi e le rispettive impronte temporali.

Riferimenti: [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
e [configurazione Booster](https://docs.hpc.cineca.it/_downloads/d2425378307005ba649bfd226ce266b1/Leonardo_Booster_EF.pdf).

Lo studio da 131.072 clip comprende Small e Base per MVM, Future e Joint,
150 epoche per esperimento. Small–MVM è sospeso dopo tre epoche complete e
12.288 aggiornamenti. Il checkpoint conserva pesi, optimizer, scheduler e
generatori casuali. Gli altri cinque esperimenti partono da zero.

## Posizioni e integrità dei dati

Il clone Git è in `$HOME/deep_convection`. I dati risiedono in
`$FAST/deep_convection_datasets/seviri_131072_20261102/`:

- `frames/`: 14 file NumPy giornalieri, circa 71 GiB, Kelvin float16;
- `dataset/`: manifest delle 131.072 clip di training e 512 di validation,
  statistiche globali e inventario congelato;
- `sources/`: circa 24 GiB di Zarr originali congelati, utilizzati dai
  controlli di provenienza del lettore;
- `transfer_plan.json` e `verification.json`: identità e verifica SHA256
  dell'intero trasferimento; i timestamp originali sono registrati a parte.

Le clip sono definite da timestamp e origine spaziale nei manifest. Le
osservazioni comuni a più clip occupano un solo file giornaliero. Il
trasferimento preserva valori, ordine dei sette canali, orientamento,
campionamento, dominio e statistiche globali del protocollo locale.

### Validation storica e validation estesa

La validation da 512 clip usata fino ai dataset da 65k rimane il riferimento
per confrontare quei training. Il dataset da 131.072 clip usa altri 512 crop
della stessa giornata, 17 settembre 2020: il ricampionamento ha modificato
posizioni e finestre temporali. La preparazione della validation estesa conserva
copie identiche di entrambi i manifest, chiamate `legacy_512.csv` e
`current_128k_512.csv`. Per confrontare direttamente tutti i checkpoint sul
riferimento storico serve valutarli sulle medesime clip `legacy_512`.

`configs/ssl_validation_extended.json` definisce un secondo insieme di
10.240 clip: 20 giornate fra il 5 gennaio e il 15 giugno 2022, con 512 clip
per giornata e 128 per fascia UTC di sei ore. Le giornate sono distribuite
lungo il periodo coperto dalle osservazioni già caricate. Le origini spaziali
sono campionate alla risoluzione di un pixel; ogni pixel del crop appartiene
al dominio 20°W–40°E, 30–70°N. La selezione richiede tutti i sette canali,
16 frame a cadenza regolare, valori finiti e temperature fra 100 e 400 K.
Il campionamento uniforme delle proposte valide in ciascuna fascia mantiene
lo stesso peso per le giornate e per gli intervalli della giornata.

La valutazione estesa misura il trasferimento alle condizioni osservate nel
primo semestre 2022. Le finestre spaziali e temporali possono sovrapporsi
entro una giornata; i report per giornata permettono di osservare quanto
l'errore dipenda dalle diverse situazioni meteorologiche. Il numero di clip
descrive il numero di finestre valutate. La copertura di giornate, stagioni
ed eventi determina quali condizioni atmosferiche entrano nel confronto.

Il controllo dei manifest riserva queste giornate e una separazione minima
di 24 ore dalle osservazioni di training, considerando anche l'ultimo frame
di ciascuna clip. La riserva è inclusa fra i sorgenti congelati dei nuovi
esperimenti. Ogni checkpoint viene valutato usando la propria media e
deviazione standard globali, calcolate sul rispettivo training.

La cadenza proposta è: validation piccola a ogni epoca, validation estesa
ogni dieci epoche e sul checkpoint migliore selezionato dalla validation
piccola. Gli esperimenti HPC già preparati conservano il protocollo congelato;
il comando seguente esegue la valutazione aggiuntiva separatamente e salva
metriche globali e per giornata. La cadenza periodica resta da collegare alla
prossima configurazione di training, dopo il trasferimento del nuovo dataset.

```bash
python scripts/evaluate_seviri_ssl_validation.py \
  --dataset results/ssl_validation_10k_20261102 \
  --run-dir results/ssl_factorial_20261028/small_n65536/future \
  --checkpoint best \
  --output-dir results/validation_extended/small_65536_future \
  --batch-size 32 --precision bf16
```

Il parametro `--smoke-samples 32` esegue una verifica tecnica parziale,
identificata da `scope: smoke` nel risultato. La valutazione completa produce
`scope: full_validation`. Per MVM e Joint, i batch rispettano i confini delle
giornate per attribuire correttamente anche la loss di ricostruzione.

La preparazione locale si trova in `results/ssl_validation_10k_20261102/`;
le copie congelate delle sorgenti e i frame sono sotto
`/data1/deep_convection_data/ssl_validation_10240_20261102/`.
`progress.json` descrive lo stato di preparazione, mentre
`dataset_ready.json` viene pubblicato dopo le verifiche finali. Il dataset
esteso richiede un trasferimento separato sul cluster; `$WORK` è la
destinazione prevista, dato lo spazio residuo nella quota `$FAST`.

`SEVIRI_STORAGE_ROOTS` indica un JSON con le nuove posizioni fisiche delle
due radici. I percorsi canonici conservati nei manifest, nei metadati e nel
checkpoint mantengono il loro significato di provenienza. Il lettore continua
a verificare checksum, dimensioni, timestamp, sequenze temporali e canali;
l'hash della mappatura entra nella revisione esplicita di esecuzione.

Il filesystem di destinazione osservato sul cluster arrotonda i timestamp ai
secondi, anche quando si richiede esplicitamente precisione maggiore.
`transport_times.json` conserva dimensioni e tempi originali ed è vincolato
tramite SHA256 alla mappatura. Per le sorgenti trasferite, il lettore verifica
il contenuto SHA256 di ogni file prima di usare il tempo canonico per
ricostruire l'impronta originale. Un cambiamento dei byte viene rifiutato
anche a dimensioni e timestamp fisici invariati. Questo controllo aggiunge
una lettura iniziale delle sorgenti, mentre i batch usano i frame NumPy.

Il trasferimento diretto avviato il 2 novembre 2026 è controllabile sulla
macchina di origine tramite
`~/.local/state/seviri-leonardo-transfer-20261102/status.json` e
`transfer.log`. Rieseguire `python3` sul file `transfer.py` nella stessa
directory riprende le copie parziali e ripete la verifica finale. Il campo
`status: verified` viene scritto solo dopo la verifica sul cluster.

## Ambiente Python

Il modulo `cineca-ai/4.3.0` usato da Demetra contiene PyTorch 2.2 e
Transformers 4.38. Questo studio richiede le versioni del checkpoint:
PyTorch 2.14.1 e Transformers 5.20.2. Usare un ambiente separato in `$WORK`:

```bash
module load python/3.11.7
unset PYTHONPATH
python3 -m venv "$WORK/deep_convection/venv"
source "$WORK/deep_convection/venv/bin/activate"
python -m pip install --upgrade pip setuptools
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$HOME/deep_convection/requirements-hpc.txt"
python -m pip check
```

La build CUDA 12.6 va verificata sul nodo GPU con il driver installato.
Una ripresa su A100 conserva lo stato dell'ottimizzazione; operazioni CUDA
e compilazione possono introdurre differenze di arrotondamento rispetto a H200.
Registrare le versioni e misurare tempi, loss finite e memoria prima del run lungo.

`scripts/hpc/seviri_ssl_smoke.sbatch` esegue un test CUDA sintetico breve per
MVM, Future e Joint con bf16, AdamW fused e compilazione. Verifica lo stack
software su una GPU allocata tramite Slurm; usa clip e modelli piccoli.
Il successivo benchmark sui dati reali deve misurare Small e Base con batch 32
prima di avviare l'intero array scientifico.

## Preparazione dei sei esperimenti

La directory `$WORK/deep_convection/migration_20261102` contiene il checkpoint
locale, i report delle tre epoche, la configurazione originale, i sorgenti
precedenti e un inventario SHA256. Dopo il completamento della verifica dati:

```bash
cd "$HOME/deep_convection"
python scripts/prepare_seviri_ssl_hpc.py \
  --dataset-root "$FAST/deep_convection_datasets/seviri_131072_20261102" \
  --seed-root "$WORK/deep_convection/migration_20261102" \
  --output-root "$WORK/deep_convection/ssl_131072_20261102"
```

La preparazione richiede un output vuoto, verifica gli inventari e congela
i sorgenti in `source_archive/`. Aggiornare successivamente il clone Git
lascia disponibile la versione congelata per la ripresa dei job. La revisione
di esecuzione lega il checkpoint iniziale ai nuovi sorgenti e alle radici
fisiche, conservando batch effettivo e configurazione scientifica.

Gli indici dell'array Slurm sono, nell'ordine: Small MVM/Future/Joint,
Base MVM/Future/Joint. Per ispezionare il comando di un indice:

```bash
python scripts/run_seviri_ssl_hpc.py \
  --plan "$WORK/deep_convection/ssl_131072_20261102/hpc_plan.json" \
  --index 0 --dry-run
```

## Slurm e risorse

`scripts/hpc/seviri_ssl_array.sbatch` riprende la partizione Booster,
l'account `iscrc_same-d2` e i limiti ai thread numerici dallo script
`Demetra/moduli/videomae/sbatch_specialization.sh`. L'account è stato
verificato nelle associazioni Slurm dell'utente. Ogni task alloca una GPU,
otto CPU e 120 GiB RAM, con al massimo quattro task contemporanei.

Ogni GPU esegue un esperimento completo. Batch 32, accumulo 1, bf16,
quattro thread di lettura, prefetch 4, AdamW fused, TF32 e compilazione
conservano le impostazioni dello studio locale. La RAM richiesta permette
di ospitare i circa 71 GiB dei frame nella cache del filesystem, oltre ai
buffer di lettura e allo stato del processo; il guadagno va misurato.

I nodi Booster hanno `/tmp` su RAM con capacità limitata: i dati restano
nel filesystem condiviso `$FAST`. Checkpoint, log e cache del compilatore
sono in `$WORK`. Riferimenti: [Leonardo](https://docs.hpc.cineca.it/hpc/leonardo.html)
e [filesystem CINECA](https://docs.hpc.cineca.it/hpc/hpc_data_storage.html).

Dopo un test GPU breve riuscito, il comando di invio è:

```bash
cd "$WORK/deep_convection/ssl_131072_20261102/slurm_logs"
sbatch "$HOME/deep_convection/scripts/hpc/seviri_ssl_array.sbatch"
```

La durata massima richiesta è 23 ore e 50 minuti. Un'ora prima della
scadenza Slurm richiede l'arresto: il trainer termina l'epoca, valida e salva
il checkpoint. Il wrapper restituisce 75 solo per questa pausa controllata;
lo script rimette in coda quel task. Gli errori mantengono un codice diverso
e richiedono diagnosi. Se il tempo termina prima del salvataggio, resta
disponibile il checkpoint dell'ultima epoca completa.

`job_status_<indice>.json` riporta job Slurm, nodo, comando, stato ed epoche
completate; ogni directory esperimento conserva `progress.json`, loss e
checkpoint. Per richiedere una pausa manuale al termine dell'epoca, creare
`stop_requests/<indice>.stop`; con questo script la pausa viene seguita da
requeue. Per sospendere l'intera serie usare il controllo dei job Slurm.
