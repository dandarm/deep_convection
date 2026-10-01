# Handoff: pretraining VideoMAE SEVIRI su GPU

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
