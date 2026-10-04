# Handoff: pretraining VideoMAE SEVIRI su GPU

## Aggiornamento Linux GPU — 1 ottobre 2026

Il computer GPU usa RTX 3090 (24 GiB), Python 3.11, PyTorch 2.9.1+cu128 e
Transformers 5.18.0. Gli Zarr sono in
`/media/fenrir/disk1/Datasets/MSG-SEVIRI`; le istruzioni Windows seguenti
documentano il trasferimento originale. La suite aggiornata passa 23 test.

È stato avviato il run Zarr da 4.096 clip fino a **150 epoche totali**,
continuando modello e optimizer delle due epoche già completate. Output:
`results/videomae_pretraining_smoke_large4096/train_onfly_150/`.
Batch 32, 16 worker persistenti, task I/O da due clip, prefetch 2, FP32,
LR costante 1,5e-4. I crop aggiuntivi usano sempre le sole 166 partenze
complete disponibili: rimane un esperimento d'integrazione prolungato.

Lo script ora supporta `--checkpoint-every 1` e `--resume`: a fine epoca
salva atomicamente modello, optimizer, scheduler, RNG e stato dei generatori
per sampler e worker. La ripresa è verificata da test CPU e CUDA. Il vecchio
run da due epoche non aveva RNG salvati, quindi la prima continuazione usa
`--warm-start` e li reinizializza; i checkpoint nuovi permettono ripresa completa.
Consultare `launcher.json` per comando/PID, `progress.json` per l'ultima
epoca completa e `checkpoint_latest.pt` per la ripresa. CSV delle loss,
plot per epoca e telemetria GPU vengono aggiornati durante il training.
Le indicazioni sotto sullo script che salva solo alla fine descrivono la
versione precedente. Per dettagli e comando di ripresa aggiornati, leggere
`VIDEOMAE_PRETRAINING.md`.

Questo documento permette di riprendere il task su un altro computer Windows
con Codex e GPU. L'obiettivo immediato è verificare l'intera pipeline su CUDA;
un addestramento lungo deve iniziare soltanto dopo avere ampliato il dataset e
aggiunto checkpoint riprendibili.

## 1. Obiettivo scientifico invariato

Preaddestrare da zero un VideoMAE-Small self-supervised su sequenze SEVIRI RSS:

- 16 frame consecutivi;
- cadenza nativa di 5 minuti;
- crop spaziale 224 x 224;
- sette canali, in questo ordine esatto:
  `WV_062`, `WV_073`, `IR_087`, `IR_097`, `IR_108`, `IR_120`, `IR_134`;
- tensore del singolo campione `(T,C,H,W) = (16,7,224,224)`;
- batch Hugging Face `(B,T,C,H,W)`;
- patch embedding `Conv3d(in_channels=7, kernel_size=(2,16,16), stride=(2,16,16))`;
- target ricostruito per tubelet: `7 x 2 x 16 x 16 = 3584` valori;
- tube masking iniziale al 90%;
- nessuna etichetta EMMA, GPM, OPERA o RDT nel pretraining.

L'esperimento principale usa inizializzazione casuale. L'adattamento di un
checkpoint RGB rimane un baseline separato e non deve essere descritto come
pretraining atmosferico da zero.

## 2. Invarianti fisiche da non cambiare

### Radiometria

Le temperature di brillanza non sono normalizzate separatamente per canale o
per clip. La pipeline calcola una sola media e una sola deviazione standard su
tutti i pixel e tutti i sette canali del solo training split e applica la
stessa trasformazione affine ovunque:

```text
x_normalizzato = (BT_kelvin - media_globale) / deviazione_standard_globale
```

Non usare normalizzazione per patch, per frame, per clip o per banda. Il flag
VideoMAE `norm_pix_loss` resta disabilitato. In questo modo temperatura
assoluta e differenze intercanale restano preservate fino a una trasformazione
affine invertibile.

### Orientamento geografico

Ogni clip consegnata al modello deve avere nord in alto e ovest a sinistra.
Nei 104 Zarr locali verificati, `x` è ordinata da est verso ovest e `y` da sud
verso nord: la lettura ingenua appariva quindi ruotata di 180 gradi. Il loader
ora controlla l'ordine delle coordinate e inverte gli assi necessari prima di
restituire il tensore. Non reintrodurre flip o rotazioni casuali come data
augmentation.

### Tempo

I frame restano sempre ordinati dal più antico al più recente. Non usare time
reversal, frame shuffle, interpolazione di frame mancanti o ripetizione di
frame.

## 3. Stato verificato sul computer di origine

- Sistema locale senza GPU; PyTorch installato: `2.14.1+cpu`.
- Transformers: `5.18.0`.
- Python: `3.12.12`.
- Test: `23 passed` dopo la correzione dell'orientamento.
- VideoMAE-Small a sette canali costruito e provato su CPU.
- Smoke training globale su due clip completato.
- Manifest esplorativo: 100 clip casuali, tutte materialmente validate.
- Seed del manifest: `20261001`.
- La pipeline rifiuta canali riordinati, tempi mancanti, crop fuori griglia e
  valori non finiti.

Archivio locale attuale:

```text
E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N
```

Contiene 104 store giornalieri frammentari, 1.580 frame e soltanto 166 possibili
istanti iniziali completi da 16 frame. Le 100 clip sono distribuite su 14
giornate. Questo è sufficiente per integrazione e profiling, non per un vero
pretraining from scratch.

## 4. File inclusi nella sincronizzazione Git

La sincronizzazione remota di questa fase include i file seguenti. Dopo il
clone o il pull sul computer GPU, verificarne la presenza prima di procedere:

```text
RDT_plan.md
VIDEOMAE_PRETRAINING.md
requirements-training.txt
requirements-notebook.txt
catalogs/seviri_pretraining_smoke.csv
catalogs/seviri_pretraining_random_100.csv
notebooks/visualize_seviri_pretraining_clips.ipynb
scripts/build_random_seviri_clip_manifest.py
scripts/download_msg_seviri.py
scripts/crop_msg_seviri_native.py
scripts/repair_msg_zarr_times.py
scripts/pretrain_videomae.py
scripts/render_seviri_pretraining_clip.py
scripts/render_seviri_mediterranean_videos.py
src/emma_gpm/seviri.py
src/emma_gpm/seviri_clips.py
src/emma_gpm/videomae.py
tests/test_crop_msg_seviri_native.py
tests/test_seviri.py
tests/test_seviri_clips.py
tests/test_videomae.py
```

I dataset Zarr, i checkpoint, gli optimizer e gli MP4 generati rimangono fuori
da Git e devono essere copiati o rigenerati separatamente. Non usare sul nuovo
PC una vecchia versione di `seviri_clips.py`: perderebbe sia i sette canali
corretti sia l'orientamento nord-up.

## 5. Dati da trasferire o ricostruire

Per il test CUDA breve copiare almeno:

```text
processed/msg_seviri/rss_7ch_bt_emma_30N70N/
```

Il manifest contiene solo timestamp e indici dei crop; non contiene i pixel.
Conservare la stessa struttura `anno/YYYYMMDD.zarr` oppure impostare il nuovo
percorso con:

```powershell
$env:SEVIRI_ZARR_ROOT = "E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N"
```

Per il pretraining reale occorre invece acquisire un archivio continuo molto
più ampio e costruire split temporali senza sovrapposizione fra sequenze o
episodi meteorologici. Non usare le sole 100 clip come dataset finale.

## 6. Preparazione del computer GPU

Da PowerShell, nella radice del progetto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

Installare una build CUDA di PyTorch compatibile con driver e GPU del computer,
seguendo il selettore ufficiale PyTorch. Non considerare riuscita
l'installazione finché questo controllo non restituisce `True`:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Verificare quindi codice e dataset:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -c "import pandas as pd; d=pd.read_csv('catalogs/seviri_pretraining_random_100.csv'); print(len(d), d.clip_id.nunique())"
```

Il risultato atteso della suite è almeno `23 passed`.

## 7. Primo test completo su CUDA

Usare inizialmente batch size 1 e il decoder VideoMAE completo:

```powershell
.\.venv\Scripts\python.exe scripts\pretrain_videomae.py `
  --zarr-root $env:SEVIRI_ZARR_ROOT `
  --manifest catalogs\seviri_pretraining_random_100.csv `
  --output-dir results\videomae_gpu_integration_100 `
  --device cuda `
  --epochs 1 `
  --batch-size 1 `
  --initialization scratch `
  --no-cpu-smoke-decoder
```

Questo comando è un test d'integrazione e throughput, non un risultato
scientifico. Controllare:

- `device` uguale a `cuda` nel riepilogo;
- `num_channels = 7`;
- input `(16,7,224,224)`;
- target width `3584`;
- loss sempre finita;
- utilizzo GPU visibile in `nvidia-smi`;
- output `model`, `optimizer.pt`, `channel_stats.json` e
  `smoke_summary.json` completi.

Non lasciare attivo `--cpu-smoke-decoder`: riduce il decoder a un solo layer e
serve esclusivamente al test locale CPU.

## 8. Lavoro necessario prima del pretraining lungo

Lo script corrente è sufficiente per il test CUDA, ma non ancora per un job di
più giorni. Prima del lancio definitivo, Codex sul computer GPU deve:

1. ampliare l'archivio SEVIRI e produrre un manifest di training realmente
   grande;
2. creare split train/validation/test per blocchi temporali o episodi, evitando
   che finestre sovrapposte finiscano in split diversi;
3. calcolare la coppia globale media/deviazione standard una sola volta sul
   training split e salvarla per tutti i run e per il probing;
4. aggiungere mixed precision CUDA (`bf16` se supportato, altrimenti `fp16`),
   senza modificare i dati fisici salvati;
5. aggiungere checkpoint periodici atomici e ripresa completa di modello,
   optimizer, scheduler, epoca, step e stato RNG;
6. aggiungere gradient accumulation e gradient checkpointing se necessari;
7. rendere configurabili `num_workers`, `pin_memory` e prefetch del DataLoader;
8. registrare loss, throughput, memoria GPU, seed, commit Git, manifest e
   statistiche radiometriche;
9. effettuare un breve overfit controllato e un run di alcune centinaia di step
   prima del job completo;
10. conservare come baseline separata l'inizializzazione RGB adattata a sette
    canali.

Non avviare un job lungo senza checkpoint riprendibili. Lo script corrente
salva modello e optimizer soltanto alla conclusione.

## 9. File principali da leggere

```text
RDT_plan.md
src/emma_gpm/seviri_clips.py
src/emma_gpm/videomae.py
scripts/pretrain_videomae.py
scripts/build_random_seviri_clip_manifest.py
tests/test_seviri_clips.py
notebooks/visualize_seviri_pretraining_clips.ipynb
```

Le visualizzazioni mediterranee sono diagnostiche e non aggiungono canali al
modello. Il tensore VideoMAE resta sempre a sette canali.

## 10. Prompt per la nuova task Codex

> Leggi integralmente `HANDOFF_VIDEOMAE_PRETRAINING_GPU.md` e `RDT_plan.md`,
> quindi verifica che tutti i file elencati nella sezione 4 siano presenti e
> che la suite termini con almeno 23 test superati. Controlla con PyTorch che
> CUDA sia realmente disponibile e identifica la GPU. Non usare EMMA, GPM,
> OPERA o RDT nel pretraining. Mantieni 16 frame ordinati a 5 minuti, sette
> canali nell'ordine stabilito, nord in alto e ovest a sinistra, e una sola
> trasformazione affine globale calcolata esclusivamente sul training split.
> Esegui prima il test CUDA da 100 clip con VideoMAE-Small e decoder completo,
> batch size 1 e inizializzazione casuale. Riporta throughput, memoria massima,
> loss e artefatti prodotti. Non descrivere questo test come pretraining
> scientifico. Prima di avviare un job lungo, amplia il dataset, costruisci
> split temporali senza leakage e implementa mixed precision, checkpoint
> atomici riprendibili, scheduler, logging e DataLoader GPU efficiente. Non
> avviare il job lungo finché un'interruzione non può essere ripresa senza
> perdere lo stato.

## Coda automatica di ablazioni e validation — ottobre 2026

Il precedente run Zarr da 150 epoche è completato in 10.659 s (2h 57m 39s).
È stata preparata e avviata una nuova coda in
`results/videomae_pretraining_smoke_masking_sweep/`: train 4.096 crop e
validation 512 crop su date separate, 24 ore di embargo, tre modelli Small
inizializzati da zero con masking 50/75/90% e 150 epoche ciascuno. Usano bf16,
microbatch 16, accumulo 2 (batch effettivo 32) e checkpoint latest/best.
Validation comune al 90% con maschere fisse, metriche Kelvin e curve per epoca.
Il modello precedente non è una baseline valida su questo nuovo holdout.

Se il migliore migliora l'RMSE di meno del 5% rispetto al nuovo run al 90%,
il supervisore genera 40.000 crop di training e lancia automaticamente 150
epoche da zero con il masking selezionato. Con i dati locali aumentiamo i crop,
non gli episodi: restano solo 106 partenze di training e 48 di validation.

Leggere `VIDEOMAE_PRETRAINING.md`, sezione «Coda masking e aumento dei crop»,
per comando, protocollo, limiti e ripresa. Stato in `queue_status.json`, PID
in `launcher.json`, dettagli per run in `mask50/`, `mask75/`, `mask90/`.

Verifica iniziale della nuova coda: **28 test passati**. Preflight bf16/FP32
con scarto relativo della loss 7,63e-6 e picco riservato 5,91 GiB. Primo run
al 50% avviato e prima epoca completata con training MSE 0,41446 e validation
MSE 0,31374 (RMSE 11,26 K). Checkpoint latest/best e plot verificati.
La telemetria della coda è salvata localmente in `gpu_telemetry.csv`.

### Aggiornamento: preparare soltanto crop senza sovrapposizioni

L'utente ha richiesto soltanto manifest e diagnostiche del nuovo campionamento.
`expansion_policy.json` disabilita il passaggio automatico ai 40.000 crop.
I tre run sul masking continuano: il nuovo supervisore ha adottato il trainer
attivo senza riavviarlo. Il comando di ripresa normale non deve contenere
`--adopt-training-pid` / `--adopt-run`, perché il PID potrebbe non esistere più.
La preparazione CPU del sampler ispirato a Demetra è in `spaced_samples/`.
Controlla IoU zero anche fra partenze diverse che condividono frame; il numero
finale di clip è inferiore a 4.096, dato il dominio e l'archivio frammentario.
Consultare l'ultima sezione di `VIDEOMAE_PRETRAINING.md` per parametri e output.

### Validation con masking 50/75/90 e download annuale

I tre training sono conclusi. I checkpoint migliori/finali sono stati valutati
su tutte e tre le percentuali (18 misure in `validation_mask_sweep.csv`). Il
notebook mostra un solo plot, training continuo e validation tratteggiata;
default «Proprio masking», altri confronti selezionabili. Le curve storiche
al 50/75 non esistono: sono mostrati esclusivamente punti ai checkpoint salvati.
I training futuri registrano `validation_own_mse` a ogni epoca. Suite: 37 test.

Download annuale 2020 avviato con `scripts/acquire_seviri_year.py`, credenziali
nel file locale `env` (ignorato da Git), stesso archivio `MSG-SEVIRI`.
Progresso/log/PID in `MSG-SEVIRI/_acquisition/2020/`. Buffer dei soli nativi nuovi
per giornata, conversione in Zarr 7ch Kelvin e pulizia dopo verifica; non serve
conservare circa 10 TiB di nativi su un disco con circa 3 TiB liberi. Leggere
l'ultima sezione di `VIDEOMAE_PRETRAINING.md` per comando e ripresa.

### Selettore ricostruzione e rimozione della cache materializzata

Il player di ricostruzione del notebook permette di scegliere run 50/75/90%,
checkpoint best/finale e masking proprio oppure fisso al 50/75/90%. Default:
run 75%, best, masking proprio. Best resta selezionato sulla validation comune
al 90%. Il run storico da 150 epoche è disponibile separatamente, con il suo
manifest originale. Il cambio modello ricarica manifest e statistiche del run;
maschere riproducibili permettono il confronto a pari clip e masking.
Verificato nel kernel `.venv` con checkpoint e clip reali, inclusi cambio run,
cambio best/finale e conservazione esatta dei pixel visibili.

Su richiesta è stata eliminata soltanto la cache
`/media/fenrir/disk1/Datasets/MSG-SEVIRI-samples-4096` (42,88 GiB).
Manifest, risultati del benchmark e archivio Zarr restano disponibili.
Il vantaggio locale dei sample NPY era circa il 4% (63,44 contro 60,80 clip/s),
con dati che potevano stare nella cache RAM: non dimostra il comportamento HPC.
Prima di scegliere il formato sul cluster confrontare Zarr e sample aggregati
su 1/4/16 GPU, includendo cache fredda/calda e costo di preparazione/staging.
Valutare storage locale al nodo e suddivisione dei dati fra processi; gli Zarr
attuali hanno chunk in file separati, quindi anche il loro carico di metadata
può diventare un limite. Il trainer non è ancora stato verificato multi-GPU.

### Protocollo corrente: V2 Small, cinque run — 3 ottobre 2026

La nuova coda è `results/videomae_v2_small_scaling`, gestita da
`scripts/run_seviri_v2_experiments.py`: 150 epoche da zero per encoder masking
50/75/90% su 2.000 clip, poi 4.000 e 8.000 con il masking migliore sulla
validation comune encoder 90% / decoder 50%. Decoder sempre running cell 50%;
loss solo sui target selezionati invisibili all'encoder. I vecchi run sono V1
e rimangono separati. Componenti HF adattati al dual masking del codice V2
ufficiale, encoder Small 384×12 e decoder 192×4, sette canali e target 3584.

Dataset annidati da uno snapshot delle sole giornate acquisite/verificate,
finestre distanti 80 minuti e griglia spaziale a offset casuale senza overlap.
Holdout 256 su ultimi tre giorni con finestre complete e embargo 24 ore;
statistiche globali del rispettivo training manifest. Il download può
continuare senza cambiare i manifest congelati. Comando e protocollo completo
nell'ultima sezione di `VIDEOMAE_PRETRAINING.md`. Rilanciare il comando della
coda per ripresa automatica dei checkpoint; non usare pesi del vecchio V1.

Il download annuale non è finito: si era fermato al 2 marzo dopo 61 giornate,
per un native con timestamp Satpy incompatibile con il filename. Ora i prodotti
con questa discordanza sono esclusi, documentati in `rejected_products/` e
nei gap del report giornaliero, senza interpolazione/ripetizione. Acquisizione
riavviata; progresso e PID restano sotto `MSG-SEVIRI/_acquisition/2020/`.
Monitor del notebook aggiornato ai cinque run V2; loss comparabili in K²,
tabella solo Stato/Epoche e grafico finale della loss rispetto ai sample.

Lancio verificato: coda in esecuzione su `mask50`, microbatch 16, accumulo 2,
bf16; preflight CUDA riserva 4,96 GiB e scarto relativo bf16/FP32 6,97e-7,
backward finito e optimizer aggiornato. Suite completa: **50 test passati**.
Manifest 2.000/4.000/8.000 coprono tutti 30 giorni e rispettivamente
513/521/521 finestre complete; l'espansione aumenta soprattutto copertura
spaziale, non il numero di episodi. Verificate 58.446 coppie concorrenti sul
manifest 8.000: IoU massima zero. `launcher.json` registra il supervisore
autonomo; il processo continua dopo la chiusura della chat.

Prime tre epoche verificate, con validation e checkpoint best/latest salvati.
Epoca 2: training 34,47 s, 58,02 clip/s; utilizzo GPU medio durante il training
56,3% (34 misure a intervallo di circa 1 s), picco memoria usata osservato
5,65 GiB. La prima epoca include avvio worker e aperture iniziali (82,74 s).
Player verificato nel kernel `.venv` con checkpoint V2 reale e inferenza CPU,
conservazione esatta dei pixel visibili e gestione dei run ancora in attesa.

### Ripetizione della scala con encoder masking 90% — 4 ottobre 2026

La coda originale è conclusa: masking selezionato 75%, anche i run 4.000 e
8.000 al 75% sono completati. Su richiesta parte una nuova sequenza di due
run **da zero**, encoder90% / decoder running cell50%, 150 epoche:
`large4000_mask90/`, poi `large8000_mask90/`, nello stesso output principale.
Manifest, statistiche, validation, seed e batch coincidono con le prove al 75%.
I risultati e la selezione originali sono conservati.

Supervisore: `scripts/run_seviri_v2_mask90_scaling.py --queue-dir
results/videomae_v2_small_scaling --zarr-root /media/fenrir/disk1/Datasets/MSG-SEVIRI
--epochs 150`. Ripresa con lo stesso comando: salta i run conclusi e usa
checkpoint latest per gli incompleti. Lock condiviso con la coda originale.
Stato/log/PID: `mask90_scaling_status.json`, `mask90_scaling_queue.log`,
`mask90_scaling_launcher.json`; telemetria `mask90_gpu_telemetry.csv`.
`mask90_scaling_comparison.*` e `mask90_loss_vs_samples.png` confrontano
le curve 2.000/4.000/8.000 al 75% e 90%, sulla stessa validation comune.
Il monitor del notebook include i due nuovi run e segue lo stato più recente;
il selettore di ricostruzione li rende disponibili dopo il primo checkpoint.
