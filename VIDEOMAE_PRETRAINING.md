# Pretraining VideoMAE su SEVIRI

Questa fase usa esclusivamente sequenze SEVIRI non etichettate. EMMA, GPM,
OPERA e le pseudo-label RDT non entrano nel dataset, nel campionamento o nella
loss. Il nome storico della directory Zarr contiene ancora `emma`, ma lo store
contiene soltanto brightness temperature SEVIRI calibrate.

## Modello

Il protocollo corrente usa **VideoMAE V2 Small a sette canali**, con dual
masking; i run precedenti documentati sotto usavano V1. L'adattamento
`src/emma_gpm/videomae_v2.py` riusa i componenti Transformer HF e implementa
il forward dual-mask e il running cell masking del
[codice ufficiale V2](https://github.com/OpenGVLab/VideoMAEv2), dal
[paper CVPR 2023](https://arxiv.org/abs/2303.16727). Non usa pesi RGB, etichette
o la fase supervisionata post-pretraining del paper. La normalizzazione
globale atmosferica sostituisce intenzionalmente i target RGB normalizzati
per patch. Le dimensioni restano ViT-S (encoder 384×12, decoder 192×4).

Il modello principale è VideoMAE-Small con input `(B,16,7,224,224)`, tubelet
temporale 2 e patch spaziale 16. La patch embedding ha pesi
`(384,7,2,16,16)` e il decoder ricostruisce 3584 valori per tubelet. Il modello
completo dello smoke test contiene 25.210.304 parametri.

Gli input usano una sola trasformazione affine globale: una media e una
deviazione standard calcolate congiuntamente su tutti i pixel e tutti i sette
canali del solo training split. Non esistono standardizzazioni per canale, clip,
frame o patch. La loss ricostruisce questi valori senza normalizzazione locale,
preservando temperatura assoluta e differenze interbanda.

Il loader ispeziona le coordinate proiettate e restituisce sempre le clip con
nord in alto e ovest a sinistra. Nei file Zarr locali `x` è decrescente e `y`
è crescente, quindi entrambi gli assi vengono invertiti dopo il ritaglio. Non
si usano rotazioni, riflessioni o inversioni temporali come augmentation.

## Smoke test CPU

Installazione:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

Esecuzione completa VideoMAE-Small, due clip e una epoca:

```powershell
& .\.venv\Scripts\python.exe scripts\pretrain_videomae.py `
  --device cpu `
  --epochs 1 `
  --batch-size 1 `
  --initialization scratch `
  --no-cpu-smoke-decoder `
  --output-dir results\videomae_pretraining_smoke_full_decoder
```

Il manifest `catalogs/seviri_pretraining_smoke.csv` contiene due clip
cronologiche complete da 16 frame a cinque minuti. Sono state selezionate dai
campi SEVIRI sulla base della presenza di nubi fredde, senza consultare alcuna
etichetta esterna. Le statistiche prodotte da sole due clip servono unicamente
allo smoke test e non devono essere riutilizzate nel training reale.

Su un altro computer si può cambiare la radice dati senza modificare il codice:

```powershell
$env:SEVIRI_ZARR_ROOT = "D:\path\to\rss_7ch_bt"
```

## DataLoader CUDA e integrazione su Linux

Il DataLoader supporta `--num-workers`, `--prefetch-factor` e
`--pin-memory` / `--no-pin-memory`, oltre a `--loader-batch-size` per separare
le clip per task di lettura dal batch dell'optimizer. Il valore deve dividere
il batch di training; zero conserva un task per batch completo. I task piccoli
vengono ricomposti nell'ordine originale, incluso l'ultimo batch incompleto,
senza cambiare shuffle del manifest, ordine dei frame o mascheramento.
Con worker attivi usa processi `spawn`,
store Zarr aperti separatamente in ogni processo e worker persistenti fra le
epoche. Il prefetch predefinito è un batch per worker. La memoria pinned è
attiva automaticamente su CUDA e consente trasferimenti non bloccanti.
Il default resta zero worker per compatibilità con gli smoke test CPU.
Il reader legge i chunk selezionati direttamente tramite Zarr, evitando la
costruzione di grafi Dask per l'intero store giornaliero. Nei worker la
concorrenza Zarr è limitata a otto richieste e quattro thread, con un thread
Blosc, per contenere la moltiplicazione di thread fra processi.

Esempio di integrazione su 100 clip, dieci epoche e decoder completo:

```bash
.venv/bin/python scripts/pretrain_videomae.py \
  --zarr-root /media/fenrir/disk1/Datasets/MSG-SEVIRI \
  --manifest catalogs/seviri_pretraining_random_100.csv \
  --output-dir results/videomae_pretraining_smoke_gpu_100_b32_w16_io2_e10 \
  --device cuda --epochs 10 --batch-size 32 \
  --num-workers 16 --loader-batch-size 2 --prefetch-factor 2 \
  --pin-memory --cpu-threads 4 \
  --initialization scratch --no-cpu-smoke-decoder
```

Lo script registra media della loss pesata per campione e throughput di ogni
epoca, configurazione del DataLoader e picchi di memoria CUDA nel riepilogo.
Salva inoltre `loss_steps.csv` dopo ogni step e `loss_epochs.csv` dopo ogni
epoca, con tempi di attesa dati e durata degli aggiornamenti sincronizzati.
Alla conclusione genera `loss_per_epoch.png`, con la sola loss di training
media pesata per campione; non è una curva di validation.
La media globale e la deviazione standard vengono ancora calcolate prima del
training sul manifest corrente. Questi run sono test d'integrazione in FP32:
le 100 clip e le loro statistiche non bastano per il pretraining scientifico.
Di default modello e optimizer vengono esportati alla conclusione.
`--checkpoint-every 1` salva inoltre a ogni fine epoca un checkpoint atomico
con modello, optimizer, scheduler a learning rate costante, storico delle
loss e RNG Python/NumPy/PyTorch/CUDA, del sampler e del DataLoader.
`--resume checkpoint_latest.pt` riprende dall'ultima epoca completa,
verificando identità del dataset, normalizzazione e configurazione.
Un'interruzione durante un'epoca richiede di ripetere solo quell'epoca.
I sample letti dai worker non usano trasformazioni stocastiche; lo stato del
sampler è separato dal seed dei worker per preservare l'ordine alla ripresa.

Misure RTX 3090, 100 clip, batch 32, FP32 e dieci epoche (1 ottobre 2026):

| Lettura / worker / clip per task | Clip/s, incluso avvio worker | Clip/s, epoche 2–10 | Utilizzo GPU medio |
|---|---:|---:|---:|
| Dask / 4 / 32 | 5,65 | 5,84 | 6,1% |
| Zarr diretto / 8 / 4 | 26,76 | 31,86 | 29,8% |
| Zarr diretto / 16 / 2 | 31,06 | 40,01 | 33,8% |

I throughput escludono il calcolo iniziale delle statistiche e il salvataggio
finale. La telemetria GPU è campionata circa ogni 100 ms durante il training.
Il picco di VRAM riservata è 11,44 GiB in tutti i run. Statistiche identiche
e differenze massime fra loss entro 1e-7 confermano il comportamento numerico
equivalente. Con 16 worker il tempo di attesa dati scende dal 92,5% al 57,7%:
resta un limite da misurare su un archivio e manifest più grandi. I risultati
sono misure d'integrazione su questi dati locali e non garantiscono la stessa
scala di miglioramento con storage, CPU o dataset differenti.

### Manifest più grande per benchmark I/O

La generazione supporta validazione parallela e calcolo delle statistiche
globali nello stesso passaggio. I risultati sono raccolti nell'ordine
deterministico dei candidati, indipendentemente dal numero di worker.

```bash
.venv/bin/python scripts/build_random_seviri_clip_manifest.py \
  --zarr-root /media/fenrir/disk1/Datasets/MSG-SEVIRI \
  --output results/videomae_pretraining_smoke_large4096/manifest.csv \
  --count 4096 --max-attempts 40000 --seed 20261001 --num-workers 16 \
  --stats-output results/videomae_pretraining_smoke_large4096/channel_stats.json

.venv/bin/python scripts/pretrain_videomae.py \
  --zarr-root /media/fenrir/disk1/Datasets/MSG-SEVIRI \
  --manifest results/videomae_pretraining_smoke_large4096/manifest.csv \
  --global-stats results/videomae_pretraining_smoke_large4096/channel_stats.json \
  --output-dir results/videomae_pretraining_smoke_large4096/train_onfly \
  --device cuda --epochs 2 --batch-size 32 --loader-batch-size 2 \
  --num-workers 16 --prefetch-factor 2 --pin-memory --cpu-threads 4 \
  --initialization scratch --no-cpu-smoke-decoder
```

`--global-stats` verifica hash SHA-256 del manifest, radice Zarr, ordine dei
canali, statistiche scalari e numero totale di pixel. Le statistiche delle
100 clip non possono essere riutilizzate per il manifest ampliato.
Le 4.096 clip ampliano solo i crop: l'archivio corrente conserva 166 partenze
temporali complete su 14 giorni, quindi questo rimane un benchmark tecnico.
Un futuro confronto con un file per sample deve usare lo stesso manifest,
normalizzazione, seed, masking e batch; deve riportare spazio, tempo di
materializzazione e throughput separatamente. I sample grezzi north-up in
Kelvin float16 occuperebbero circa 42,9 GiB, rispetto ai circa 9 GiB degli
Zarr locali; i sample float32 circa 85,8 GiB, esclusi piccoli header.
La cache del filesystem può contenere entrambi i dataset su questa macchina:
i tempi a cache calda non dimostrano il throughput di archivi più grandi
della RAM. Non viene materializzato alcun dataset in questo benchmark Zarr.

Baseline misurata su 4.096 clip uniche, stesso archivio e configurazione sopra:

| Epoca | Secondi | Clip/s | GPU media | Loss media |
|---|---:|---:|---:|---:|
| 1 | 111,43 | 36,76 | 40,7% | 0,44134 |
| 2 | 67,37 | 60,80 | 67,7% | 0,25491 |

Il throughput complessivo è 45,81 clip/s (178,8 s di training), con picco di
VRAM riservata 11,40 GiB. L'attesa dati nella seconda epoca è il 25,5% del
tempo. La prima passata include avvio dei worker e apertura degli store nei
processi; non è stata forzata una cache disco fredda. Il manifest e le nuove
statistiche si trovano in `results/videomae_pretraining_smoke_large4096/`,
i checkpoint, CSV e grafici in `train_onfly/`. I 256 aggiornamenti sono
completati e tutte le loss sono finite.

### Confronto con un file per sample

`scripts/materialize_seviri_samples.py` salva un `.npy` contiguo per sample,
con Kelvin float16 e forma `(16,7,224,224)`, già north-up e west-left. Rifiuta
qualsiasi conversione non esattamente reversibile a float32 rispetto alla
sorgente, rilegge ogni file e ne registra SHA-256 e dimensione. Ogni sample e
il metadata finale sono pubblicati con rename da un file temporaneo; senza
`metadata.json` la cache non è considerata completa. I file rimangono fuori Git.

```bash
.venv/bin/python scripts/materialize_seviri_samples.py \
  --zarr-root /media/fenrir/disk1/Datasets/MSG-SEVIRI \
  --manifest results/videomae_pretraining_smoke_large4096/manifest.csv \
  --output-dir /media/fenrir/disk1/Datasets/MSG-SEVIRI-samples-4096 \
  --num-workers 16
```

Ripetere il comando di training Zarr sopra cambiando soltanto la directory di
output in `results/videomae_pretraining_smoke_large4096/train_materialized`
e aggiungendo `--sample-root /media/fenrir/disk1/Datasets/MSG-SEVIRI-samples-4096`.
Il loader verifica manifest, provenance, metadati, numero e dimensione dei
file; al caricamento rifiuta forma, dtype o valori non finiti inattesi.
La stessa trasformazione globale viene applicata ai Kelvin letti dai `.npy`.
I checksum attestano i file riletti durante la materializzazione, ma non
vengono ricalcolati a ogni step di training.

Materializzazione effettiva delle 4.096 clip: 42,88 GiB in 87,16 s.
La directory di output deve essere vuota; non viene sovrascritta una cache
esistente. Confronto a manifest, statistiche, seed, masking, modello e batch
di training identici, due epoche FP32:

| Storage / worker / clip per task | Training totale (s) | Clip/s totali | Clip/s epoca 2 | GPU media |
|---|---:|---:|---:|---:|
| Zarr / 16 / 2 | 178,81 | 45,81 | 60,80 | 50,9% |
| NPY / 16 / 2 | 172,04 | 47,62 | 63,44 | 53,2% |
| NPY / 4 / 32 | 164,82 | 49,70 | 53,97 | 54,8% |

Il confronto a configurazione identica mostra un miglioramento di throughput
di circa il 4%, non un salto significativo. Con quattro worker il totale di
due epoche migliora ancora, ma la seconda epoca è più lenta. Ogni
configurazione è stata misurata una volta; piccoli scarti possono dipendere
dalla variabilità del sistema. Le misure escludono i 87 s di preparazione e
non forzano una cache filesystem fredda. Sample sorgente e salvati sono
confrontati esattamente durante la materializzazione; i tensori normalizzati
sono identici nei controlli successivi. Le loss restano praticamente
sovrapponibili, con piccoli scarti FP32 in CUDA senza determinismo forzato.
Riepilogo e grafico: `results/videomae_pretraining_smoke_large4096/storage_comparison.*`.

### Continuazione del run Zarr fino a 150 epoche

Il run `results/videomae_pretraining_smoke_large4096/train_onfly_150`
continua le due epoche Zarr già completate fino a 150 epoche totali
(148 nuove epoche), con dataset non materializzato, batch 32, 16 worker,
task da due clip, FP32, LR costante 1,5e-4 e checkpoint a ogni epoca.
È una prova prolungata sul manifest di integrazione, non il pretraining
scientifico su anni disgiunti. Pesi e optimizer precedenti vengono caricati
con `--warm-start .../train_onfly`; lo stato RNG del vecchio run non era
salvato e viene reinizializzato una volta. Da questo segmento in avanti la
ripresa completa usa i nuovi checkpoint.

`launcher.json` conserva comando e PID del supervisore separato dalla chat.
`run_config.json` conserva seed, manifest, commit, hash dello script e
configurazione; `progress.json` riporta l'ultima epoca completata. CSV e
`loss_per_epoch.png` vengono aggiornati durante il run. Il monitor locale
salva `nvidia_smi.csv` con campionamento richiesto ogni 100 ms.

Per riprendere, usare gli stessi argomenti del comando in `launcher.json`,
rimuovere `--warm-start` e il suo percorso e aggiungere
`--resume results/videomae_pretraining_smoke_large4096/train_onfly_150/checkpoint_latest.pt`.
La directory di output deve restare la stessa. I CSV vengono riallineati
all'ultima epoca effettivamente salvata, scartando eventuali righe di un'epoca
interrotta. Checkpoint, dati e log restano locali.

## Visualizzazione

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
& .\.venv\Scripts\python.exe -m jupyter lab `
  notebooks\visualize_seviri_pretraining_clips.ipynb
```

Il notebook mostra sette colonne sincronizzate, quattro differenze spettrali e
controlli Play/pausa/slider sui 16 frame. Tutti i canali grezzi condividono la
scala assoluta fissa 180–330 K. Una sezione finale permette inoltre di scegliere
fra undici video separati sul dominio 20°W–40°E, 30–60°N.

## Passaggio al training reale

In fondo al notebook, la sezione **Ricostruzione VideoMAE sulle clip di
training** contiene tre celle autonome: spiegazione, caricamento del modello
e player. Usare il kernel `.venv` con `requirements-notebook.txt` e
`requirements-training.txt`. Il player carica il modello finale a 150 epoche
oppure un checkpoint atomico dell'ultima epoca completa, applica le statistiche
globali del run e mostra sette bande per originale, input mascherato,
ricostruzione composita ed errore. Play e slider scorrono i 16 frame senza
cambiare la tube mask, fissata per clip con seed riproducibile.
I pixel visibili sono copiati, quindi MAE/RMSE in Kelvin e la baseline della
media globale si misurano esclusivamente sui pixel mascherati. Le celle
precedenti del notebook restano invariate e non servono per questa sezione.
Le ricostruzioni su sample del training sono diagnostiche: una validation
indipendente richiede episodi/anni non utilizzati dal modello, senza finestre
sovrapposte e usando le sole statistiche del training.

Prima del training GPU occorre costruire un manifest SEVIRI-only esteso,
calcolare le statistiche su tutto il training split e definire anni disgiunti
per training, validation e test. Il primo esperimento principale partirà da
pesi casuali; il checkpoint RGB adattato a sette canali resterà un confronto
controllato separato.

## Coda masking e aumento dei crop (ottobre 2026)

La coda `scripts/run_seviri_masking_experiments.py` costruisce un nuovo split
cronologico dagli Zarr disponibili: gli ultimi tre giorni con finestre complete
(23 agosto, 2 e 17 settembre 2020) sono riservati alla validation. Ogni clip di
training deve terminare almeno 24 ore prima del primo giorno di validation;
il 22 agosto viene escluso. Restano 106 partenze temporali di training e 48
di validation. I manifest contengono 4.096 crop di training e 512 di validation,
tutti letti e verificati, senza materializzazione. I timestamp dei frame non
si sovrappongono fra split. La separazione di 24 ore non garantisce che tutti
siano episodi meteorologici indipendenti: l'archivio frammentario rimane piccolo.

Il precedente modello da 150 epoche ha visto anche le date della nuova
validation, quindi non viene riutilizzato. La coda allena da zero tre
VideoMAE-Small, con lo stesso seed, manifest, normalizzazione globale e budget
di 150 epoche, cambiando soltanto il tube masking: 50%, 75%, 90%. Il 90%
resta il riferimento canonico; gli altri due sono ablazioni. Le percentuali
effettive dipendono dall'arrotondamento sui 196 patch spaziali.

Ogni epoca valuta gli stessi 512 crop, in ordine fisso, con masking al 90% e
seed 314159 più indice del sample. La maschera è costante attraverso il tempo
e fra epoche/modelli, indipendente dai batch e dai generatori del training.
La validation usa esclusivamente media/deviazione standard del training,
misura soltanto pixel mascherati e registra MSE normalizzata, RMSE Kelvin
aggregata e per canale, più una baseline che predice la media globale.
Il miglior checkpoint di ogni run viene scelto per la MSE di validation.
Questa selezione richiede un ulteriore test indipendente per una valutazione
scientifica finale. Le loss di training con masking diversi hanno difficoltà
diverse e non sono il criterio di selezione.

Configurazione comune: bf16, microbatch 16, due microbatch per aggiornamento
(batch effettivo 32), LR costante 1,5e-4, 16 worker, task I/O da due clip,
prefetch 2, quattro worker separati per la validation, checkpoint atomici
per ogni epoca. Prima del training, un preflight su dati reali confronta loss
bf16/FP32 (scarto relativo massimo ammesso 2%), verifica backward, gradienti
finiti e step optimizer al 50% di masking, richiedendo picco VRAM riservata
non superiore a 21 GiB. Non si usa un GradScaler per bf16. L'accumulo pesa
correttamente anche un ultimo gruppo incompleto.

```bash
.venv/bin/python scripts/run_seviri_masking_experiments.py \
  --zarr-root /media/fenrir/disk1/Datasets/MSG-SEVIRI \
  --output-dir results/videomae_pretraining_smoke_masking_sweep \
  --epochs 150 --large-epochs 150 --minimum-improvement 0.05
```

Il run da 40.000 crop parte automaticamente **solo se** il miglior masking
riduce l'RMSE di validation di meno del 5% rispetto al nuovo run al 90%.
La soglia operativa rende eseguibile il criterio «se non va meglio» e viene
salvata nel piano. In caso di parità prevale l'ordine 50%, 75%, 90%.
I 40.000 crop includono i 4.096 iniziali come prefisso e provengono dallo
stesso training split, con nuove statistiche globali calcolate su tutti i loro
pixel. Il modello riparte da zero con il masking selezionato e la validation
rimane identica. Il confronto si fa in Kelvin perché le due coppie di
statistiche possono differire. A parità di 150 epoche, il run grande esegue
circa dieci volte gli aggiornamenti: non è un confronto a compute uguale.
Quarantamila crop distinti non equivalgono a quarantamila episodi indipendenti.

Output sotto `results/videomae_pretraining_smoke_masking_sweep/`:

- `plan.json`, `temporal_split.json`, manifest e statistiche del solo training;
- `queue_status.json` e `queue.log` per la coda;
- `mask50/`, `mask75/`, `mask90/`: log, CSV training/validation, plot per epoca,
  `checkpoint_latest.pt`, `checkpoint_best.pt`, `best_validation.json`;
- `masking_comparison.json` e `selection.json` al termine delle tre prove;
- `large40000/` e `dataset_comparison.json` soltanto se scatta l'espansione.

Rilanciare **lo stesso comando** riprende i run dall'ultima epoca completa e
salta quelli già completati. Un lock impedisce due supervisori sulla stessa
coda; un errore ferma la coda e viene registrato, senza lanciare i run seguenti.
Per i singoli run, mantenere gli argomenti di `run_config.json` e aggiungere
`--resume .../checkpoint_latest.pt`. Le directory generate sono ignorate da
Git. La coda è stata lanciata come processo separato dalla chat; comando e PID
sono in `launcher.json`.

Nel notebook, immediatamente prima di «Ricostruzione VideoMAE», tre celle
«Monitoraggio della coda e loss per epoca» leggono JSON e CSV locali senza
caricare modelli sulla GPU. Mostrano avanzamento dei quattro run, training MSE,
validation MSE e validation RMSE in Kelvin. Usare **Aggiorna** oppure attivare
**Auto (30 s)** per rileggere lo stato; disattivarlo per fermare il refresh.
Le celle sono autonome e possono essere eseguite senza i player precedenti.

## Campionamento spaziale senza sovrapposizione: preparazione soltanto

Su richiesta dell'utente, il successivo training automatico da 40.000 crop è
**disabilitato**. `expansion_policy.json` con `mode: prepare_only` prevale sul
criterio originale del 5%; dopo i tre run sul masking la coda termina. Il
supervisore è stato sostituito adottando il processo di training già attivo,
senza riavviarlo né perdere l'epoca in corso. `--adopt-training-pid` e
`--adopt-run` servono soltanto per questo passaggio; per una successiva ripresa
normale usare il comando senza questi due argomenti.

Il nuovo sampler è ispirato a
[Demetra/random_tiles.ipynb](https://github.com/dandarm/Demetra/blob/main/notebooks/random_tiles.ipynb)
e alla funzione `_sample_spatial_offset_with_constraints` in
[dataset/build_dataset.py](https://github.com/dandarm/Demetra/blob/main/moduli/videomae/dataset/build_dataset.py).
Riprende il limite di IoU fra eventi temporalmente attivi, il criterio di
copertura storica e la preferenza per i bordi. Il notebook di riferimento usa
IoU massimo 0,20; questa prova usa **0,00**. L'implementazione di riferimento
può ripiegare sul candidato meno sovrapposto quando nessuno è ammissibile;
qui si rifiuta il candidato. Si conservano i frame RSS consecutivi a cinque
minuti, senza importare criteri temporali permissivi o sorgenti di Demetra.

Il dominio locale misura 763×1.762 pixel: 1.344.406 posizioni per frame e
canale, 9.410.842 valori per frame a sette canali. Ogni crop occupa 50.176
posizioni spaziali. Una griglia regolare senza overlap contiene 3×7=21 crop;
il limite superiore basato soltanto sull'area è 26 crop per frame, prima del
rifiuto dei bordi con pixel non finiti. Con 106 partenze di training, il
vincolo stretto produce meno delle precedenti 4.096 clip; non si può ottenere
un aumento a 40.000 clip realmente separate con questo archivio.

```bash
.venv/bin/python scripts/build_spaced_seviri_manifest.py \
  --zarr-root /media/fenrir/disk1/Datasets/MSG-SEVIRI \
  --starts-file results/videomae_pretraining_smoke_masking_sweep/train_starts.csv \
  --validation-manifest results/videomae_pretraining_smoke_masking_sweep/validation.csv \
  --output-dir results/videomae_pretraining_smoke_masking_sweep/spaced_samples \
  --max-count 40000 --max-iou-active 0 --sampling-attempts 2048
```

Parametri aggiuntivi: `--crops-per-start`, `--coverage-weight`,
`--border-boost`, `--min-gap-pixels` e `--center-bounds LON_MIN LON_MAX LAT_MIN
LAT_MAX` (vincolo sul centro geografico del crop, non su tutta la sua area).
Una clip è attiva fino all'ultimo frame incluso, a 75 minuti dalla partenza.
Crop separati possono toccarsi sul bordo, ma non condividere pixel nello
stesso frame. La ricerca casuale è riproducibile, vettorizzata e limitata:
non garantisce un packing ottimale o una saturazione dell'archivio.

La preparazione salva `train_spaced_all.csv`, `train_spaced_small.csv`
(sottoinsieme di circa un decimo, distribuito sulle giornate quando possibile),
statistiche globali separate calcolate su tutti i pixel dei rispettivi
training manifest, `dataset_report.json` con verifica esatta delle IoU e
`sampling_diagnostics.png`. Il grafico usa indici nativi esplicitamente
etichettati; i tensori caricati per il training restano north-up/west-left.
Nessun sample viene materializzato e nessun nuovo training viene lanciato.
Il piccolo sottoinsieme e il manifest completo consentiranno eventualmente
un confronto sull'aumento del numero di clip con lo stesso campionamento.

Preparazione completata: 172 clip valide e sottoinsieme di 17 (10,12×).
Verificate 1.403 coppie concorrenti: IoU massima zero. Il confronto geometrico
con le 4.096 clip casuali misura un riutilizzo del 91,53% dei contributi
pixel/frame; nel manifest separato è zero. Questo non significa che il 91,53%
delle clip sia identico: conta osservazioni pixel/frame ripetute. Il manifest
casuale copre circa 278,58 milioni di posizioni pixel/frame uniche, quello
separato circa 138,08 milioni; il vincolo stretto e la ricerca non ottimale
riducono anche la copertura complessiva. JSON e grafici di confronto sono in
`overlap_comparison.*`, prodotti da `scripts/analyze_seviri_clip_overlap.py`.
Il notebook contiene una sezione autonoma per leggere queste diagnostiche.
Nel monitor la tabella mostra soltanto Stato ed Epoche, con il run come indice;
i titoli di validation esplicitano «90% per tutti».

## Validation al proprio masking e confronto fra maschere

Il monitor del notebook mostra ora **un solo grafico di loss**: training in
linea continua e validation in tratteggio, stesso colore per modello. Default
«Proprio masking»; il selettore consente anche 50%, 75% o 90% per tutti, con
sample e seed di mascheramento identici fra modelli. La tabella conserva solo
Stato/Epoche. Il grafico RMSE è stato rimosso perché è una trasformazione della
stessa MSE, a statistiche condivise fra questi tre run.

Per i run conclusi è disponibile la curva di validation al 90% per tutte le
150 epoche, ma sono stati conservati soltanto checkpoint best e final. Le
validation al 50% e 75% sono quindi misurate esclusivamente alle loro epoche,
mostrate come cerchi collegati in tratteggio, senza inventare curve storiche.
`validation_mask_sweep.csv` contiene 18 valutazioni: 3 modelli × 2 checkpoint
× 3 masking, sulle stesse 512 clip, bf16 e seed 314159. Il best checkpoint
rimane quello scelto mediante validation al 90%, non il best retrospettivo
per ogni nuova percentuale. Riprodurre/riprendere la valutazione con:

```bash
.venv/bin/python scripts/evaluate_seviri_masking_checkpoints.py \
  --queue-dir results/videomae_pretraining_smoke_masking_sweep
```

I training futuri salvano anche `validation_own_mse` a ogni epoca. Se si
riprende un run precedente, le epoche storiche mantengono quel campo vuoto;
non vengono riempite artificialmente. Il criterio di best checkpoint continua
a usare la validation comune specificata da `--validation-mask-ratio`.

Sui checkpoint finali di queste prove, il modello allenato al 50% ha MSE più
bassa nella validation comune al 50% e 75%, quello al 75% nella validation
comune al 90%. Il modello al 90% è peggiore in tutte e tre le prove. Il suo
best al 90% cade all'epoca 76, non alla 150: aggiungere epoche non garantisce
un miglioramento di generalizzazione. Le loss «al proprio masking» misurano
compiti con difficoltà diversa e servono soprattutto a leggere il divario
training/validation; non costituiscono da sole una graduatoria fra modelli.

## Acquisizione RSS per tutto il 2020

Avviato `scripts/acquire_seviri_year.py` sull'archivio esistente
`/media/fenrir/disk1/Datasets/MSG-SEVIRI`, usando il file locale `env`.
Le credenziali sono caricate nel processo, non incluse nel comando/log; il file
`env`, come `.env`, è escluso da Git. La collezione è
[Rapid Scan High Rate SEVIRI L1.5](https://api.eumetsat.int/data/browse/collections/EO%3AEUM%3ADAT%3AMSG%3AMSG15-RSS?format=html),
la stessa delle clip correnti, con cadenza RSS a cinque minuti.

Un anno di file nativi supera lo spazio libero corrente. Il job usa un buffer
privato per giornata, scarica quattro prodotti in parallelo, converte in
brightness temperature a sette canali sul dominio invariato e pubblica gli
Zarr verificati. Rimuove soltanto i nuovi nativi del buffer dopo la verifica;
non elimina i file originari dell'utente. Riprende saltando giornate completate
e frame già presenti; i nuovi giorni includono un report dei gap realmente
presenti nel catalogo, senza interpolare o sostituire frame mancanti.
I file intermedi non finiscono fra gli store pubblici `*.zarr`.

```bash
.venv/bin/python scripts/acquire_seviri_year.py \
  --zarr-root /media/fenrir/disk1/Datasets/MSG-SEVIRI \
  --year 2020 --env-file /media/fenrir/disk1/danieleda/deep_convection/env \
  --workers 4 --crop-batch-size 4
```

Monitoraggio e comando di ripresa in
`MSG-SEVIRI/_acquisition/2020/{progress.json,acquisition.log,launcher.json}`;
`completed_days/*.json` conserva copertura e verifica per ogni giornata.
Il job protegge il percorso con un lock e si ferma sotto 50 GiB liberi o dopo
cinque errori sulla stessa giornata, conservando dati e buffer per la ripresa.
Un primo prodotto del 1 gennaio è stato scaricato (97,5 MiB) e letto con
Satpy: forma `(7,763,1762)`, Kelvin, timestamp nominale corretto. La copia
staged conserva gli store giornalieri preesistenti e aggiunge i frame mancanti
prima della pubblicazione con backup per recuperare un'interruzione.

`MSG-SEVIRI-samples-4096` è invece la cache lossless di videocrop preparata
per il benchmark I/O precedente (42,88 GiB), non la destinazione del download.
L'aggiunta di frame all'archivio non modifica automaticamente i manifest e le
statistiche dei tre esperimenti già conclusi; i futuri split estesi andranno
costruiti e validati separatamente.

## Coda V2 Small: masking e scala 2.000/4.000/8.000 (3 ottobre 2026)

`scripts/run_seviri_v2_experiments.py` prepara e lancia cinque run **da zero**,
150 epoche ciascuno: encoder masking 50/75/90% sulle stesse 2.000 clip, quindi
4.000 e 8.000 clip con il masking selezionato sulla validation comune. Il
decoder masking è sempre 50%, running cell su celle 2×2 di patch: la fase
ruota tra tubelet temporali, come nel sampler ufficiale. Il decoder riceve
tutti i token visibili dell'encoder più i token segnaposto selezionati per
ricostruzione. La loss usa soltanto le posizioni selezionate che l'encoder
non ha visto; non include i target visibili. A encoder masking 50%, le
duplicazioni visibili fanno sì che la sequenza decoder non sia più corta
di V1; il risparmio cresce con encoder masking più alto.

I dati provengono da uno snapshot dei giorni dell'acquisizione verificati e
pubblicati prima della preparazione. Il download successivo non cambia lo
split. Gli ultimi tre giorni con finestre complete sono holdout; l'ultimo
frame del training deve precederli di oltre 24 ore. Il primo snapshot contiene
61 giornate verificate (incluse giornate vuote nel catalogo), 9.503 frame:
30 giorni con partenze di training, fino al 26 febbraio; validation dal 28
febbraio al 1 marzo. Non si usa il vecchio holdout estivo.

Si scelgono finestre complete distanti almeno 80 minuti e una griglia di crop
224×224 con offset casuale per finestra, senza pixel condivisi tra clip attive.
Il pool viene mescolato prima della lettura parallela; i crop non finiti sono
rifiutati. I manifest 2.000 e 4.000 sono prefissi del manifest 8.000, distribuiti
sui giorni disponibili. La verifica delle IoU e le diagnostiche sono in
`dataset_ready.json`; nessun sample viene materializzato. Ogni manifest ha
una coppia globale di statistiche calcolata su **tutti** i suoi pixel/canali.

Validation: 256 clip fisse, encoder masking comune 90% e decoder masking 50%,
con entrambe le maschere riproducibili per indice del sample. Ogni epoca
registra anche la validation al proprio encoder masking, stesso decoder50%.
Best e selezione del masking usano la validation comune; in parità prevale
50/75/90. La curva per dimensione usa MSE Kelvin² del best checkpoint,
poiché le statistiche dei training manifest differiscono. A 150 epoche, i
run 4.000/8.000 hanno circa 2×/4× gli aggiornamenti: non sono compute-matched.
L'holdout usato per selezione non sostituisce un test scientifico indipendente.

```bash
.venv/bin/python scripts/run_seviri_v2_experiments.py \
  --zarr-root /media/fenrir/disk1/Datasets/MSG-SEVIRI \
  --output-dir results/videomae_v2_small_scaling --epochs 150
```

`--prepare-only` prepara soltanto i manifest, poi lo stesso comando senza
flag avvia la coda. Un lock protegge da supervisori duplicati. Il comando
salta i run conclusi e riprende gli incompleti da checkpoint atomici, inclusi
optimizer/scheduler/RNG delle due maschere. Un errore ferma i run successivi.
CUDA preflight: confronto bf16/FP32, backward e optimizer a microbatch16 con
accumulo2; picco riservato richiesto sotto 21 GiB. Come nelle prove precedenti,
LR costante 1,5e-4, bf16, batch effettivo32, 16 worker e task I/O da due clip.

Output: `plan.json`, `temporal_split.json`, `dataset_ready.json`, `queue_status.json`,
`queue.log`, `gpu_telemetry.csv`, `mask50/`, `mask75/`, `mask90/`, `large4000/`,
`large8000/`. Ogni run registra loss per step/epoca, validation e checkpoint
best/latest. `selection.json` sceglie il masking; `dataset_comparison.*` e
`loss_vs_samples.png` vengono aggiornati dopo ogni dimensione completata.
Il notebook monitora questa coda di default, con loss in K² e tabella solo
Stato/Epoche. Il player distingue V1/V2 e, per una ricostruzione completa,
disattiva il solo sottocampionamento decoder V2: la relativa diagnostica
non coincide con la loss sul subset running cell del training.

### Ripresa del download e prodotti inconsistenti

L'acquisizione si era fermata il 2 marzo dopo 61 giornate. Il prodotto
`MSG3-SEVI-MSG15-0100-NA-20200302113915.987000000Z-NA.nat` indica slot11:35
nel nome, ma Satpy riporta start/end11:40 e qualità non valida. Non viene
riparato né sostituito: è escluso e registrato in `rejected_products/20200302.json`
e nel report giornaliero come gap. Invarianti spettrali/radiometriche e forma
restano controlli bloccanti. Il download è ripreso, riusando i nativi già
scaricati atomicamente senza richieste seriali ai metadata per nomi RSS standard.

## Ripetizione dei run 4.000/8.000 al 90% (4 ottobre 2026)

La coda V2 originale è completata e aveva selezionato encoder75%. Le nuove
prove ripartono da zero, per 150 epoche, con encoder90% e decoder50%, senza
modificare manifest annidati, statistiche, seed, batch o holdout. I risultati
al 75% restano conservati per il confronto.

```bash
.venv/bin/python scripts/run_seviri_v2_mask90_scaling.py \
  --queue-dir results/videomae_v2_small_scaling \
  --zarr-root /media/fenrir/disk1/Datasets/MSG-SEVIRI --epochs 150
```

Run sequenziali: `large4000_mask90/` e `large8000_mask90/`. Il comando riprende
gli incompleti dai checkpoint latest e salta i conclusi. Stato in
`mask90_scaling_status.json`, log in `mask90_scaling_queue.log`, PID/comando
in `mask90_scaling_launcher.json`. La selezione originale non viene riscritta.
`mask90_scaling_comparison.csv/json` e `mask90_loss_vs_samples.png` includono
le baseline 2.000 al 75/90% e i due run originali 4.000/8.000 al 75%; aggiungono
le nuove misure man mano che terminano. Il notebook monitora tutti i sette
run e mostra le due curve per dimensione in validation MSE Kelvin² comune.
