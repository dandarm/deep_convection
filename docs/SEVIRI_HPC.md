# Ripresa dello studio SSL su Leonardo

## Prove distribuite su un nodo con quattro GPU

`pretrain_seviri_ssl_ddp.py` distribuisce un singolo modello sulle quattro
GPU mediante PyTorch DDP. Il campionatore costruisce una permutazione globale
diversa a ogni epoca e assegna ciascuna clip a un solo processo. Ogni processo
contribuisce alla media dei gradienti. Il batch globale è il prodotto del
batch per GPU e del numero di GPU; l'accumulo è uno in queste prove.

La prima coppia di esperimenti usa Base/Future, 131.072 clip, cinque epoche,
seed 20261024, learning rate 0,00015 e weight decay 0,05. Il riferimento usa
32 clip globali, otto per GPU. Il secondo usa la massima batch per GPU verificata
con granularità di otto clip e un limite del 92% della memoria disponibile.
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
