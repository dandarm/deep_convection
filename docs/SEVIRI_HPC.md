# Ripresa dello studio SSL su Leonardo

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
