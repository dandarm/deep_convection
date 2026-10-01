# VideoMAE atmosferico su SEVIRI

Questo repository sviluppa un primo approccio sperimentale per verificare se
un VideoMAE preaddestrato direttamente su osservazioni satellitari atmosferiche
apprenda rappresentazioni utili delle dinamiche convettive.

Il disegno sperimentale ha tre fasi:

1. **Pretraining self-supervised SEVIRI.** Un VideoMAE-Small ricostruisce
   tubelet mascherati di sequenze multispettrali non etichettate.
2. **Pseudo-ground truth deterministico RDT.** Un algoritmo riproducibile,
   ispirato a RDT-CW, usa soltanto WV 6.2 µm e IR 10.8 µm per rilevare,
   inseguire e classificare l'evoluzione delle celle convettive.
3. **Probing a backbone congelato.** Un decoder 3D FPN leggero esegue la
   segmentazione volumetrica a cinque classi senza aggiornare l'encoder.

Il test centrale non è semplicemente ottenere una buona segmentazione, ma
capire quanta informazione su evoluzione temporale, morfologia termica,
microfisica e altezza della sommità nuvolosa sia già decodificabile dalle
rappresentazioni self-supervised.

## Pretraining SEVIRI

Ogni campione contiene 16 frame RSS consecutivi a cinque minuti, cioè 75
minuti fra primo e ultimo frame. Il tensore fornito a Hugging Face ha forma:

```text
(B,T,C,H,W) = (B,16,7,224,224)
```

I sette canali, in ordine vincolante, sono:

| Indice | Canale | Informazione principale |
|---:|---|---|
| 0 | WV 6.2 µm | vapore acqueo in alta troposfera |
| 1 | WV 7.3 µm | vapore acqueo medio-troposferico |
| 2 | IR 8.7 µm | fase e microfisica delle nubi |
| 3 | IR 9.7 µm | assorbimento dell'ozono |
| 4 | IR 10.8 µm | temperatura della sommità nuvolosa |
| 5 | IR 12.0 µm | spessore ottico e split-window |
| 6 | IR 13.4 µm | assorbimento CO₂ e altezza della nube |

La patch embedding è una `Conv3d` con sette canali in ingresso, tubelet
temporale 2 e patch spaziale 16×16. Ogni target ricostruito contiene quindi
`7 × 2 × 16 × 16 = 3584` valori.

### Invarianti fisiche

- I frame sono sempre ordinati dal più antico al più recente.
- Le clip hanno nord in alto e ovest a sinistra; il loader deduce le eventuali
  inversioni dalle coordinate Zarr.
- Una sola media e una sola deviazione standard, calcolate sull'intero training
  split e su tutti i canali insieme, definiscono la trasformazione affine
  globale.
- Non sono ammesse normalizzazioni per canale, clip, frame o patch.
- Non si usano time reversal, channel shuffle, rotazioni, riflessioni o RGB
  color jitter.
- Il pretraining non usa etichette RDT né altri dataset meteorologici.

Le clip spaziali non vengono duplicate su disco: i manifest conservano
timestamp e origine del crop, mentre il cubo viene letto dagli Zarr durante il
training.

## Ground truth RDT e probing

Il teacher deterministico produce cinque classi voxel-wise:

0. sfondo;
1. nube fredda non convettiva;
2. cella convettiva in sviluppo;
3. cella matura o in decadimento;
4. overshooting top.

La segmentazione, il tracking, le transizioni di fase e l'override OT devono
essere completamente deterministici e verificabili. Il sistema va indicato
come **RDT-CW-inspired** finché soglie e regole non saranno state validate
contro la specifica operativa autorevole.

Durante il probing il backbone VideoMAE rimane congelato. Il teacher riceve
solo WV 6.2 e IR 10.8, mentre l'encoder riceve l'intero cubo a sette canali.
Il confronto con un encoder casuale, un encoder RGB adattato e le ablazioni
temporali e spettrali è parte necessaria dell'esperimento.

## Stato attuale

- [x] conversione dei prodotti nativi SEVIRI in Zarr a sette canali;
- [x] loader rigoroso per clip complete, ordinate e north-up;
- [x] VideoMAE-Small a sette canali da zero e adattamento opzionale da RGB;
- [x] loss MAE senza normalizzazione locale delle patch;
- [x] smoke test CPU end-to-end;
- [x] manifest riproducibile di 100 clip casuali validate;
- [x] notebook con player dei crop e diagnostiche spettrali;
- [x] visualizzazioni geografiche separate sul dominio 20°W–40°E, 30–60°N;
- [ ] archivio SEVIRI continuo abbastanza grande per il pretraining;
- [ ] training GPU con mixed precision e checkpoint riprendibili;
- [ ] implementazione e validazione completa del teacher RDT;
- [ ] decoder 3D FPN e protocollo di probing congelato.

L'archivio locale usato per l'integrazione contiene soltanto 1.580 frame e 166
possibili partenze complete. Le 100 clip correnti servono a collaudo e
visualizzazione, non costituiscono un dataset scientificamente sufficiente per
il pretraining from scratch.

## Documentazione da leggere

La specifica scientifica e tecnica autorevole è distribuita in questi file:

1. [`RDT_plan.md`](RDT_plan.md): obiettivo, teacher deterministico, probing,
   controlli e stima della scala dati;
2. [`VIDEOMAE_PRETRAINING.md`](VIDEOMAE_PRETRAINING.md): pipeline implementata,
   smoke test e visualizzazione;
3. [`HANDOFF_VIDEOMAE_PRETRAINING_GPU.md`](HANDOFF_VIDEOMAE_PRETRAINING_GPU.md):
   trasferimento e ripresa del lavoro sul computer GPU;
4. [`AGENTS.md`](AGENTS.md): regole operative per Codex e altri agenti.

## Ambiente

Esempio Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
$env:PYTHONPATH = (Resolve-Path "src").Path
.\.venv\Scripts\python.exe -m pytest -q
```

Impostare la radice degli Zarr senza modificare il codice:

```powershell
$env:SEVIRI_ZARR_ROOT = "E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N"
```

Il nome storico del percorso può contenere `emma`, ma gli Zarr usati nel
pretraining contengono esclusivamente brightness temperature SEVIRI.

## Smoke test CPU

```powershell
.\.venv\Scripts\python.exe scripts\pretrain_videomae.py `
  --device cpu `
  --epochs 1 `
  --batch-size 1 `
  --initialization scratch `
  --no-cpu-smoke-decoder `
  --output-dir results\videomae_pretraining_smoke_full_decoder
```

Per il primo test CUDA e per i requisiti necessari prima di un job lungo,
seguire l'handoff GPU anziché riutilizzare alla cieca il comando CPU.

## Notebook

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m jupyter lab `
  notebooks\visualize_seviri_pretraining_clips.ipynb
```

I dataset, i prodotti nativi, i checkpoint, gli optimizer, i log, le figure e
i video generati restano locali e non devono essere aggiunti al repository.
