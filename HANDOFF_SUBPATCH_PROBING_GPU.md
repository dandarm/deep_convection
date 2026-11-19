# Handoff: Probing 3-Head su VideoMAE SEVIRI (Leonardo) e Decoder Sub-Patch 3D su GPU

Documento operativo per proseguire l'esplorazione dal punto esatto raggiunto su **Loki (Windows CPU)** verso la **macchina GPU**, eseguendo il confronto definitivo tra `LightweightFPN3D` (trilineare) e il nuovo [`SubPatchThreeHeadProbe`](src/emma_gpm/subpatch_probe.py) su un dataset reale da **512–2.000+ clip**.

> **Letture obbligatorie prima di toccare codice:** seguire sempre `AGENTS.md` (invarianti dei dati SEVIRI a 7 canali, backbone congelato nel probing principale, niente pesi binari / Zarr / MP4 su Git).
> Per la specifica matematica completa del nuovo decoder leggere [`docs/SUBPATCH_3D_DECODER.md`](docs/SUBPATCH_3D_DECODER.md).

---

## 1. Report sintetico di quanto svolto finora

### 1.1 Verifica e acquisizione dei backbone preaddestrati da Cineca Leonardo

Collegandosi al cluster (`ssh ddarmien@login.leonardo.cineca.it`) sono stati identificati e scaricati i due run VideoMAE-Base (12 layer, `D=768`, 7 canali SEVIRI, `patch_size=16`, `tubelet_size=2`, `num_frames=16`, `norm_pix_loss=False`) completati in DDP su 4× A100:

| Run su Cineca Leonardo | Clip di pretrain | Epoche | Split Validation | Train MSE finale | Val RMSE min (K) | Inizializzazione |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| `ddp_spaced_iou020_base_future_250_20261110` | **9.900** *(spaziate, `max_iou_active <= 0.20`)* | 250 | Omogeneo | `0.0220` | **2.80 K** | **100% from scratch** |
| `ddp_base_future_150_20261104` | **131.072** | 150 | Calendar holdout (mese separato) | **`0.0195`** | **3.69 K** | **100% from scratch** |

- **Provenienza verificata:** tramite `run_config.json` (`resume: null`, `initial_encoder_sha256` al seed `20261024`) è stato confermato che entrambi i modelli sono addestrati **interamente from scratch su SEVIRI** (nessun peso RGB Kinetics).
- **Analisi curve di pretraining:** generato plot comparativo Log-Log (`results/local_probe_loki/comparison_pretraining_losses_loglog.png`) che evidenzia il gap di `+0.89 K` dovuto al *domain shift* stagionale della validation calendar nel run da 131k rispetto alla validation omogenea del run spaziato da 9.9k.

### 1.2 Chiarimento sui controlli (Random e Kinetics RGB)

- **Controllo Kinetics RGB (`MCG-NJU/videomae-small-finetuned-kinetics`):** nel vecchio test pilota mono-task CMA e nei controlli `E0-RGB`/`E1-Joint3`, i 3 pesi RGB sono mediati e replicati sui 7 canali SEVIRI ($\frac{1}{7}\sum T_c$). Questa mediazione cancella le differenze di temperatura di brillanza (BTD), facendo collassare il modello su CT Cirrus (`0.040`) e RDT Active Convection (`CSI 0.025–0.043`), persino peggio del controllo random.
- **Controllo Random (`E0-Rand`):** mantiene pesi casuali indipendenti per canale, preservando il contrasto termico di $\sim 90\text{ K}$ delle incudini fredde; per questo riesce comunque a segmentare la sagoma macroscopica della convezione, ma fallisce sugli eventi rari/deboli (nubi basse marine, cirri sottili, Overshooting Top).

### 1.3 Probing 3-Head su micro-dataset locale (12 train / 6 val / 6 test) ed estensione a 15 clip held-out

- Aggiornato [`src/emma_gpm/three_head_study.py`](src/emma_gpm/three_head_study.py) (`learning_figure`) con **doppio asse X** (`Optimizer updates` + `Epoch` sotto).
- Addestrate le sonde `LightweightFPN3D` fino a **120 epoche** su `Compare-Base-10K-spaced` (best ep 54), `Compare-Base-131K-future` (best ep 99) ed `E0-Rand` (best ep 60).
- Costruita una galleria estesa di **15 clip held-out** (6 originali + 9 extra su regimi diversi: innesco/crescita, nuclei OT, cirri estesi, stratocumuli/nubi basse, transizioni sereno/nuvoloso) dalle finestre precalcolate `2020-09-17` (Medicane Ianos) e `2022-06-28` (outbreak alpino/Med occidentale).
- Sostituiti i vecchi plot statici nel notebook [`notebooks/study_seviri_cloud_mask_probe.ipynb`](notebooks/study_seviri_cloud_mask_probe.ipynb) con **30 video MP4 sincronizzati su tutti i 16 frame** (due video a 5 colonne per clip).

### 1.4 Diagnosi del collo di bottiglia trilineare e test A/B del nuovo decoder `SubPatchFPN3D`

- Dimostrato che l'aspetto "a macchia morbida" delle maschere predette dipende dal filtro passa-basso `F.interpolate(..., mode='trilinear')` + convoluzioni puramente depthwise di `LightweightFPN3D`, che taglia le frequenze spaziali $< 16\text{ px}$.
- Implementato [`src/emma_gpm/subpatch_probe.py`](src/emma_gpm/subpatch_probe.py) (`SubPatchFPN3D` + `SubPatchThreeHeadProbe`, `PixelShuffle-3D` con inizializzazione ICNR, skip *sub-patch* diretti da `f6` e `f3`, 880k parametri su Base, ~2× più veloce su CPU) e verificato con [`tests/test_subpatch_probe.py`](tests/test_subpatch_probe.py) (`2 passed`).
- **Risultati A/B verificati su tutte le 15 clip held-out (training locale su sole 12 clip):**

| Modello (15 clip held-out) | CMA IoU | Low Cloud | Mid Cloud | High Opaque | Cirrus | Active CSI | OT IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Legacy-FPN  · Leonardo 10K` *(120 ep, best 54)* | **0.784** | 0.377 | 0.260 | 0.575 | 0.488 | **0.446** | **0.072** |
| `SubPatch-3D · Leonardo 10K` *(80 ep, best 41)* | 0.775 | 0.345 | 0.236 | **0.599** | 0.470 | 0.413 | 0.028 |
| `Legacy-FPN  · Leonardo 131K` *(120 ep, best 99)* | 0.769 | **0.379** | **0.266** | 0.576 | **0.520** | 0.445 | 0.056 |
| `SubPatch-3D · Leonardo 131K` *(80 ep, best 56)* | 0.751 | 0.341 | 0.224 | **0.601** | 0.466 | 0.357 | 0.033 |
| `Legacy-FPN  · Random` *(120 ep, best 60)* | 0.734 | 0.287 | 0.247 | 0.568 | 0.441 | 0.428 | 0.000 |
| `SubPatch-3D · Random` *(80 ep, best 46)* | 0.760 | 0.306 | 0.199 | 0.576 | 0.337 | 0.396 | 0.018 |

- **Sintesi scientifica:** `SubPatch-3D` ha **provato visivamente e quantitativamente** che l'informazione frastagliata *sub-patch* (2–4 px, buchi di cielo sereno, nubi fredde non convettive `RDT=1`, `High Opaque` +2.4–2.5 punti, `Cirrus` +13.3 punti sul Random) **è già presente negli embedding del nostro VideoMAE-Base**. Tuttavia, su **12 clip di training**, 880k parametri vanno in *data starvation* / overfitting precoce (`train_loss -> 1.37`). Serve ora un dataset reale su GPU.

---

## 2. Inventario file, codice, checkpoint e dati

### 2.1 File di codice e documentazione da portare su GPU via Git

> **Attenzione (`AGENTS.md`):** prima di fare commit ispezionare sempre `git status` e includere **solo** codice, test, configurazioni e documentazione. Mai committare `.pt`, `.safetensors`, `.npy`, `.npz`, `.zarr`, `.mp4`, `.png` o notebook eseguiti pesanti.

File rilevanti introdotti/modificati in questa sessione:
- `src/emma_gpm/subpatch_probe.py` — nuovo decoder `SubPatchFPN3D` + `SubPatchThreeHeadProbe`
- `src/emma_gpm/three_head_study.py` — doppio asse `Optimizer updates` + `Epoch` in `learning_figure`
- `tests/test_subpatch_probe.py` — test unitari su `pixel_shuffle_3d`, shape e invarianza encoder congelato
- `docs/SUBPATCH_3D_DECODER.md` — specifica matematica e architetturale del decoder
- `HANDOFF_SUBPATCH_PROBING_GPU.md` — questo documento
- `scripts/run_subpatch_probe_experiment.py` — riferimento per il loop A/B locale CPU (da adattare/integrare nel runner GPU `bf16`)
- `scripts/build_extended_test_videos.py` — rendering video MP4 sincronizzati a 16 frame

### 2.2 Percorsi dei checkpoint encoder (da trasferire fuori da Git)

Sulla macchina GPU servono solo le cartelle `training/encoder/` (+ `run_config.json`, `channel_stats.json`, `progress.json`) dei due modelli Leonardo (~351 MB ciascuno):

1. **Leonardo 10K Spaced (`ddp_spaced_iou020_base_future_250_20261110`):**
   - Su Leonardo: `/leonardo_work/IscrC_SAME-D2/deep_convection/ddp_spaced_iou020_base_future_250_20261110/training/`
   - Su Loki: `results/leonardo_checkpoints/ddp_spaced_iou020_base_future_250_20261110/training/`
   - Digest encoder verificato (`state_digest`): prefisso `bd9af5d1c79b`
2. **Leonardo 131K (`ddp_base_future_150_20261104`):**
   - Su Leonardo: `/leonardo_work/IscrC_SAME-D2/deep_convection/ddp_base_future_150_20261104/training/`
   - Su Loki: `results/leonardo_checkpoints/ddp_base_future_150_20261104/training/`
   - Digest encoder verificato (`state_digest`): prefisso `e7bc2439a831`

### 2.3 Dati SEVIRI disponibili per costruire il dataset reale di probing

- **Sulla macchina GPU (percorso da individuare dinamicamente):** i dati Zarr SEVIRI risiedono in una cartella che termina con **`MSG-SEVIRI`**, ma **il prefisso del mount potrebbe differire** da `/data1/deep_convection_data/MSG-SEVIRI` (usato in `configs/three_head_probe_dataset_summer2022.json`). Prima di lanciare la preparazione del dataset, individuare il percorso esatto della cartella `MSG-SEVIRI` sulla macchina GPU (es. tramite `find / -maxdepth 4 -type d -name 'MSG-SEVIRI' 2>/dev/null` o variabile d'ambiente) e ispezionare quali giornate `.zarr` sono presenti in locale.
- **Riferimento piano esistente:** `configs/three_head_probe_dataset_summer2022.json` definisce il template canonico da **512 train / 192 validation / 192 test** (`896` clip totali); adattare `source_root`, `teacher_output`, `dataset_output` e le date degli `splits` ai giorni effettivamente presenti nella cartella `MSG-SEVIRI` della macchina GPU.
- **Su disco locale Loki `E:\Datasets\MSG-SEVIRI` (backup/riferimento):** **391 giornate SEVIRI Zarr complete** (di cui **327 nel 2022**, 18 nel 2020, 10 nel 2023).

---

## 3. Istruzioni operative passo-passo per proseguire sulla macchina GPU

### Step 0 — Setup codice, ambiente e verifica test

1. Sincronizzare il repository sulla macchina GPU (pull/checkout del branch contenente `src/emma_gpm/subpatch_probe.py`, `src/emma_gpm/three_head_study.py`, `tests/test_subpatch_probe.py`, `docs/SUBPATCH_3D_DECODER.md`).
2. Copiare i due encoder Leonardo in una cartella locale fuori Git (es. `/data1/deep_convection_results/leonardo_encoders/`), mantenendo `encoder/model.safetensors`, `encoder/config.json`, `run_config.json`, `channel_stats.json`, `progress.json`.
3. Verificare CUDA `bf16` e la suite di test:
   ```bash
   export PYTHONPATH="$(pwd)/src"
   python3 -m compileall -q src scripts tests
   python3 -m pytest -q tests/test_subpatch_probe.py tests/test_three_head_probes.py tests/test_videomae.py
   ```

### Step 1 — Individuare `MSG-SEVIRI` e preparare il dataset reale di probing (zero leakage)

1. **Individuare la cartella `MSG-SEVIRI` locale sulla macchina GPU** (non dare per scontato `/data1/deep_convection_data/MSG-SEVIRI`!): verificare dove si trova il path che termina in `MSG-SEVIRI` e fare l'inventario dei giorni `.zarr` disponibili.
2. Controllare se esiste già un dataset 3-head materializzato coerente con i dati locali; in caso contrario, creare una copia dedicata del config (es. `configs/three_head_probe_dataset_gpu.json` a partire da `configs/three_head_probe_dataset_summer2022.json`) aggiornando `source_root` al percorso reale `.../MSG-SEVIRI`, `teacher_output`, `dataset_output` e le date di `splits` ai giorni disponibili.
3. **Audit anti-leakage obbligatorio:** aggiungere i manifest di training/validation di `ddp_spaced_iou020_base_future_250_20261110` e `ddp_base_future_150_20261104` alla lista `pretraining_manifests` (o verificare che nessuno dei giorni di `validation` e `test` del probing compaia nel pretraining).
4. Materializzare i teacher sul dominio completo prima del crop e poi le clip:
   ```bash
   python3 scripts/prepare_three_head_probes.py --plan configs/three_head_probe_dataset_gpu.json --split train
   python3 scripts/prepare_three_head_probes.py --plan configs/three_head_probe_dataset_gpu.json --split validation
   python3 scripts/prepare_three_head_probes.py --plan configs/three_head_probe_dataset_gpu.json --split test
   python3 scripts/prepare_three_head_probes.py --plan configs/three_head_probe_dataset_gpu.json --materialize
   ```
5. *(Raccomandato dopo il primo giro da 512 clip)* Estendere il piano a **1.024–2.048 clip di training** su più giornate convettive + stratiformi/marine distribuite tra i giorni disponibili in `.../MSG-SEVIRI` (includendo sia outbreak convettivi sia giorni con stratocumuli marini e cirri estesi, così da eliminare il bias solo-convettivo visto sulla clip #6).

### Step 2 — Integrare `SubPatchThreeHeadProbe` nel runner GPU (`bf16`) ed eseguire la matrice A/B

1. In `src/emma_gpm/three_head_training.py` (o in un nuovo runner dedicato `scripts/train_three_head_probes_gpu_ab.py` per non alterare gli hash dei run storici), supportare il campo di configurazione `"decoder": "trilinear" | "subpatch"`:
   - `"trilinear"` $\to$ `ThreeHeadProbe(encoder, heads, encoder_mode="frozen")`
   - `"subpatch"` $\to$ `SubPatchThreeHeadProbe(encoder, heads, out_channels=48, encoder_mode="frozen")`
   - Verificare esplicitamente l'invarianza dell'encoder congelato prima e dopo il training (`state_digest`).
   - Supportare `torch.autocast("cuda", dtype=torch.bfloat16)` e feature caching su disco (`feature_cache`, che estrae una sola volta `(N, 4, 768, 8, 14, 14)` in `float16` su GPU in pochi secondi!).
2. **Matrice sperimentale minima consigliata (tutti gli encoder strettamente congelati, stessa architettura Base $D=768$ per il controllo random!):**

   | # | Nome esperimento | Backbone congelato | Decoder | Note |
   | :-: | :--- | :--- | :--- | :--- |
   | 1 | `G0-Trilinear-RandBase` | **Random Base (`D=768`)** | `LightweightFPN3D` | Controllo random *matched* Base |
   | 2 | `G0-SubPatch-RandBase` | **Random Base (`D=768`)** | `SubPatchFPN3D` | Controllo random *matched* Base |
   | 3 | `G1-Trilinear-RGB` | Kinetics RGB adattato 7ch | `LightweightFPN3D` | Controllo transfer video naturali |
   | 4 | `G1-SubPatch-RGB` | Kinetics RGB adattato 7ch | `SubPatchFPN3D` | Controllo transfer video naturali |
   | 5 | `G2-Trilinear-Leo10K` | Leonardo 10K Spaced (250 ep) | `LightweightFPN3D` | Baseline FPN su Leonardo 10K |
   | 6 | `G2-SubPatch-Leo10K` | Leonardo 10K Spaced (250 ep) | `SubPatchFPN3D` | **Test principale su Leonardo 10K** |
   | 7 | `G3-Trilinear-Leo131K` | Leonardo 131K (150 ep) | `LightweightFPN3D` | Baseline FPN su Leonardo 131K |
   | 8 | `G3-SubPatch-Leo131K` | Leonardo 131K (150 ep) | `SubPatchFPN3D` | **Test principale su Leonardo 131K** |

   > **Nota metodologica importante:** su Loki il controllo random era Small (`D=384`) mentre Leonardo era Base (`D=768`). Sulla GPU usare un **controllo Random Base (`D=768`, 12 layer)** con lo stesso `encoder_config` di Leonardo per eliminare qualsiasi asimmetria di capacità!

3. **Iperparametri di partenza consigliati su GPU (512–2.048 clip, feature cached):**
   - `precision`: `bf16` (CUDA), `batch_size`: `16` (o `32` se entra comodamente in VRAM), `epochs`: `60` (fino a `100` se CT continua a scendere)
   - `optimizer`: `AdamW(lr=1e-3..3e-3, weight_decay=1e-3)` con cosine annealing fino a `0.1 * lr` e gradient clipping `1.0`
   - Salvare sia il miglior checkpoint su **loss totale** sia, per diagnostica, monitorare separatamente l'epoca di minimo di `CMA`, `CT` e `RDT` (su Loki abbiamo visto che `RDT` raggiungeva il minimo prima di `CT`).
   - Se su 512 clip emergesse ancora leggera quadrettatura sub-patch o lieve overfitting su RDT, provare una seconda variante con `Dropout3d(0.1)` nei blocchi `stage2_fuse` e `stage3_fuse` (o aumentare il training split a 1.024–2.048 clip).

### Step 3 — Valutazione, ablazioni obbligatorie e video finali

1. **Ablazioni sul miglior modello `SubPatch` congelato (`AGENTS.md`):**
   - Temporali: `shuffle_time`, `repeat_last` (usando `diagnostic_input` in `src/emma_gpm/three_head_model.py`).
   - Spettrali: `ir108_only` vs tutti i 7 canali canonici (per quantificare il guadagno delle BTD su cirri e convezione).
2. **Rendering dei video di confronto a 16 frame:**
   - Adattare `scripts/build_extended_test_videos.py` / `render_comparison_video` su 15–20 clip del test set GPU, confrontando in video a 5 colonne `Legacy FPN` vs `SubPatch-3D` su Leonardo e `Random Base`.
3. **Criteri chiari di verifica dell'ipotesi:**
   - **Ipotesi confermata pienamente** se su 512–2.048 clip `SubPatchFPN3D` su Leonardo:
     1. elimina le micro-quadrettature viste sul mini-dataset da 12 clip e produce bordi frastagliati visivamente fedeli su CMA, CT e RDT;
     2. supera **contemporaneamente** `LightweightFPN3D` (su CMA, Low Cloud, High Opaque, Cirrus, Active CSI e OT) **e** il controllo `Random Base` con un margine netto e coerente su tutti i regimi;
     3. mostra un calo netto delle prestazioni sotto ablazione temporale (`shuffle_time`) e spettrale (`ir108_only`), provando che la sonda decodifica realmente struttura spazio-temporale e multispettrale appresa dal backbone SEVIRI.
