# VideoMAE atmosferico su SEVIRI

Questo repository sviluppa un primo approccio sperimentale per verificare se
un VideoMAE preaddestrato direttamente su osservazioni satellitari atmosferiche
apprenda rappresentazioni utili delle dinamiche convettive.

Il disegno sperimentale ha tre fasi:

1. **Pretraining self-supervised SEVIRI.** Un VideoMAE-Small ricostruisce
   tubelet mascherati di sequenze multispettrali non etichettate.
2. **Pseudo-ground truth deterministico RDT.** Un algoritmo riproducibile,
   ispirato a RDT-CW, usa i sette canali SEVIRI per una proxy radiometrica
   UTLS/incudine e IR 10.8 µm per rilevare e inseguire le celle.
3. **Probing a backbone congelato.** Un decoder 3D FPN leggero esegue la
   segmentazione volumetrica a cinque classi senza aggiornare l'encoder.

Il test centrale non è semplicemente ottenere una buona segmentazione, ma
capire quanta informazione su evoluzione temporale, morfologia termica,
microfisica e altezza della sommità nuvolosa sia già decodificabile dalle
rappresentazioni self-supervised.

## Estensione: confronto degli obiettivi SSL

Il progetto descritto in [`handoff_codex_ssl_seviri.md`](handoff_codex_ssl_seviri.md)
aggiunge previsione causale a +15/+30 minuti, joint masked modeling + future
prediction e previsione del residuo advettivo. La pipeline iniziale, i comandi
Windows/WSL per trasferire gli Zarr e i test sono documentati in
[`SSL_SEVIRI.md`](SSL_SEVIRI.md). Il nuovo trainer è
`scripts/pretrain_seviri_ssl.py`; il trainer VideoMAE storico resta separato.

## Pretraining SEVIRI

Il protocollo corrente usa **VideoMAE V2 Small** a sette canali, con masking
separato dell'encoder e del decoder (running cell 50%). I training precedenti
erano V1. La coda `scripts/run_seviri_v2_experiments.py` confronta encoder
masking 50/75/90% su 2.000 clip, poi il masking selezionato su 4.000 e 8.000
clip annidate, 150 epoche da zero per ciascun run. Protocollo e ripresa
nella sezione «Coda V2 Small» di `VIDEOMAE_PRETRAINING.md`; risultati locali in
`results/videomae_v2_small_scaling/` e monitor nel notebook.

La diagnostica degli encoder congelati confronta ora descrittori atmosferici
primitivi e clustering gerarchici sui tre checkpoint finali da 2.000 clip:
`scripts/analyze_seviri_embeddings.py`, dipendenze in
`requirements-analysis.txt`, risultati in `results/seviri_embedding_clustering/`.
Il notebook mostra metriche e clip rappresentative con Play. Le giornate del
nuovo holdout diagnostico sono registrate in `clustering_reserved_days.json`
e vanno escluse da futuri ampliamenti del pretraining.

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

I manifest conservano timestamp e origine del crop. Il loader di base legge
i cubi dagli Zarr; gli esperimenti SSL ripetuti usano file locali verificati,
con clip compresse per Small e frame giornalieri condivisi per Base.

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

Durante il probing il backbone VideoMAE rimane congelato. Il teacher principale
usa i sette canali in Kelvin sul dominio completo: watershed di IR108,
plateau multispettrale, proxy UTLS v0 e profondità termica relativa D.
La proxy non è una temperatura di tropopausa WMO. Il teacher fisso a due
canali resta disponibile come baseline; nessuna pseudo-label entra nel pretraining.
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
4. [`AGENTS.md`](AGENTS.md): regole operative per Codex e altri agenti;
5. [`HANDOFF_RDT_TEACHER_IMPLEMENTATION.md`](HANDOFF_RDT_TEACHER_IMPLEMENTATION.md):
   aggiornamento del teacher multispettrale e limiti della proxy UTLS.

Per scaricare altri anni e monitorare acquisizione, errori e ripresa sul PC GPU,
vedere [`download_data.md`](download_data.md), anche per installare la sola
acquisizione su un altro PC Linux/WSL senza GPU (`requirements-download.txt`).

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

[`notebooks/study_seviri_ssl_training.ipynb`](notebooks/study_seviri_ssl_training.ipynb)
riunisce i risultati di tutti i 22 training conclusi al 28 ottobre:
MVM, Future, Joint, Residual e cinque probe su 4.096 e 8.192 clip Small;
Future e Joint su 16.384 clip Small e 32.768 clip Base. La validation comune
contiene 512 clip del 17 settembre. Le dimensioni del dataset compaiono negli
stessi confronti, con architettura indicata nelle legende e nei titoli.
Le curve mostrano i quattro obiettivi insieme rispetto a epoche e aggiornamenti;
Joint compare una volta con la somma delle sue loss. Le sezioni successive
confrontano errori per canale/orizzonte e variabilità, gradienti, ricostruzioni
storiche, previsioni salvate e rappresentazioni congelate.
A 38.400 aggiornamenti, da 8.192 a 16.384 clip Future passa da 3,77 a 3,67 K
di RMSE e Joint da 3,99 a 3,92 K, conservando l'architettura Small. Base a
32.768 clip termina 150 epoche e 153.600 aggiornamenti con 3,78 K per Future
e 3,70 K per Joint; il confronto a pari aggiornamenti distingue il budget
comune dal risultato dell'intero training.
Il testo chiarisce il riferimento deterministico Advection, i dati e la testa
convoluzionale dei probe, le regressioni ridge e l'interpretazione dei grafici.
La visualizzazione corrente usa assi logaritmici per loss, epoche e aggiornamenti,
barre per canale con tutti i training affiancati e grafici delle variazioni
di RMSE a budget uguale. Le tabelle sono opzionali (`SHOW_TABLES = True`);
valori completi, figure e copia eseguita sono in
`results/notebook_revision_20261028_plots/`. La revisione precedente rimane
in `results/notebook_revision_20261028/`.
Le valutazioni sono lette dai report delle epoche completate; copertura dei
confronti e analisi ancora attese sono esplicite. Le conclusioni si aggiornano
dai risultati disponibili. Il kernel è `Python (deep_convection)`.
Il testo scientifico in inglese definisce metodi, metriche e nomi delle legende.
La modalità predefinita produce figure dirette; `USE_WIDGETS = True` abilita
i selettori per il confronto dei canali. Rieseguire le celle dall'inizio
aggiorna lo snapshot. Protocollo e ripresa sono in `SSL_SEVIRI.md`.

[`notebooks/study_seviri_ssl_scaling.ipynb`](notebooks/study_seviri_ssl_scaling.ipynb)
contiene il rimando allo studio unificato. La versione precedente è archiviata
in `results/ssl_scaling_20261025/notebook_review_20261025/`, insieme alla versione
precedente del notebook di training. I risultati storici del pilot restano
in `results/ssl_pilot_20261024/`.

Per controllare la serie dal terminale, anche attraverso SSH, eseguire
`python3 /home/isac/deep_convection/scripts/status_seviri_ssl.py`.
Il riepilogo verifica i processi, mostra epoca e batch correnti, gli esperimenti
previsti nella serie selezionata e il tempo residuo stimato del run attivo. La serie
predefinita segue `results/ssl_active_series.json` quando presente; `--series`
permette di selezionare esplicitamente una serie storica. `--watch 15` aggiorna ogni
15 secondi; `--json` restituisce lo stato strutturato. Il comando legge i log
e i metadati dei processi e può essere chiuso indipendentemente dal training.
Gli esempi per PowerShell e bridge SSH sono in `SSL_SEVIRI.md`.

L'estensione `results/ssl_extension_20261026` comprende Future e Joint su
16.384 clip, con 75 epoche e batch effettivo 32: 38.400 aggiornamenti per modello,
come i precedenti training da 8.192 clip a 150 epoche. Il nuovo dataset conserva
le 8.192 clip precedenti e la stessa validation. I file delle clip condivise sono
collegati alla nuova raccolta tramite hard link; i metadati delle raccolte sono separati. Dal
27 ottobre le due raccolte di clip Small risiedono fisicamente in
`/data1/deep_convection_data/` su `/dev/sdb`, mantenendo i vecchi percorsi
tramite collegamenti simbolici. La destinazione per riprendere l'upload Zarr è
`/data1/deep_convection_data/MSG-SEVIRI/`. La cache del training Base e gli
originali Zarr del 2020 rimangono fisicamente nei percorsi sotto `/home/isac/`
usati dai training in corso; la sottodirectory originale `MSG-SEVIRI/2022`
è un collegamento alla raccolta su `/data1`. Le verifiche delle due operazioni
sono in `results/storage_migration_20261027/` e
`results/archive_migration_20261027/`.

La serie `results/ssl_base_32768_20261026` usa VideoMAE-Base a sette canali,
con encoder da 768 componenti, 12 blocchi e circa 88 milioni di parametri.
Future e Joint hanno batch 32, 1.024 aggiornamenti per epoca e 150 epoche.
Il dataset da 32.768 clip include tutte le precedenti 16.384; la validation
conserva le stesse 512 clip. I frame giornalieri float16 in Kelvin permettono
di condividere le osservazioni tra finestre sovrapposte. La cache occupa circa
20 GiB e riutilizza circa 10 GiB già presenti tramite hard link.
Il notebook confronta anche i checkpoint Small e Base a pari aggiornamenti;
il nuovo budget completo misura l'effetto combinato di capacità, campionamento
e durata dell'ottimizzazione. Il protocollo e il comando di lancio sono in
`SSL_SEVIRI.md`.

La matrice `results/ssl_factorial_20261028` completa i confronti con Small su
32.768 e 65.536 clip e Base su 16.384 e 65.536 clip. Ogni combinazione comprende
Future e Joint da zero per 150 epoche, con batch 32 e la stessa validation da
512 clip del 17 settembre 2020. L'estensione a 65.536 conserva le 32.768 clip
precedenti e aggiunge finestre del 12 ottobre 2010, 13 aprile 2012,
30 settembre 2015 e 28 febbraio 2016. Il confronto misura insieme l'aumento
dei campioni e la diversità delle date osservate. Il monitor SSH distingue
esplicitamente Small e Base; configurazione, ripresa e percorso dei dati sono
documentati in `SSL_SEVIRI.md`.

[`notebooks/profile_seviri_ssl_performance.ipynb`](notebooks/profile_seviri_ssl_performance.ipynb)
analizza i tempi di lettura e preparazione dei batch, forward, backward,
aggiornamento dei pesi, validation e checkpoint. Confronta batch size,
memoria GPU e attività CUDA usando misure isolate e verifiche su epoche complete.
Le configurazioni e i risultati sono in
`results/ssl_scaling_20261025/performance_review/`; lo script riproducibile è
`scripts/profile_seviri_ssl.py`.

L'audit delle ottimizzazioni PyTorch riutilizzate da Demetra, i riferimenti al
codice HPC e il protocollo dei benchmark sono descritti in
[`docs/SEVIRI_GPU_RUNTIME.md`](docs/SEVIRI_GPU_RUNTIME.md). Il trainer accetta
`--cuda-runtime CONFIG.json` per selezionare compilazione Inductor, AdamW fused,
autotuning cuDNN e buffer delle posizioni fisse. La continuazione di una matrice
esistente usa `scripts/run_seviri_ssl_runtime_revision.py`, che archivia il
checkpoint di partenza e registra le impostazioni di esecuzione mantenendo i
piani scientifici originali.

Il confronto include una cache lossless di frame nativi in Kelvin,
costruita da `scripts/build_seviri_frame_cache.py`, e un percorso sperimentale
con batch float16, normalizzazione e verifica dei valori finiti sulla GPU.
La versione adottata nei training Small usa clip complete compresse senza perdita,
preparate da `scripts/materialize_ssl_clip_files.py`. La ripresa con batch 32,
quattro thread e normalizzazione globale sulla GPU è gestita da
`scripts/run_seviri_ssl_series.py`, con revisione esplicita della configurazione
esecutiva e conservazione del protocollo scientifico originale. I dettagli e
le misure prima/dopo sono in `SSL_SEVIRI.md` e nel notebook delle prestazioni.

[`notebooks/visualize_seviri_rdt_rules.ipynb`](notebooks/visualize_seviri_rdt_rules.ipynb)
esplora l'archivio scaricato con clip, Play sincronizzato e pseudo-classi
deterministiche **RDT-CW-inspired**. Mostra contorni adattivi, plateau, proxy
UTLS, indice D e confidenza, da mappe calcolate prima sul dominio completo. Non carica VideoMAE e non modifica
il pretraining; regole e limiti nella sezione di implementazione di RDT_plan.md.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m jupyter lab `
  notebooks\visualize_seviri_pretraining_clips.ipynb
```

I dataset, i prodotti nativi, i checkpoint, gli optimizer, i log, le figure e
i video generati restano locali e non devono essere aggiunti al repository.

## Teacher sul dominio completo

Installare `requirements-rdt.txt`, poi avviare una giornata:

```bash
.venv/bin/python scripts/build_rdt_pseudolabels.py --start-date 2020-07-11
```

Output incrementali in `results/rdt_preview/domain_utls_july2020/`; aggiungere
`--end-date 2020-07-17` per estendere alla settimana preservando lo stato.
Il notebook legge crop successivi delle mappe pubblicate e non ricalcola il
teacher per clip. Regole e valori v0 sono esplorativi, ancora da validare.

## Cloud-mask probing degli encoder congelati

Il nuovo probe usa il target binario del modulo `CMA_original` per predire la
copertura nuvolosa dell'ultimo frame osservato. Il dataset downstream ha giornate
separate dal pretraining; una testa con due layer Linear viene addestrata sopra
ciascun encoder congelato e sopra controlli casuali appaiati. La serie viene
eseguita su CPU mentre prosegue il pretraining GPU.

Protocollo e comandi: [SEVIRI_CLOUD_MASK_PROBE.md](docs/SEVIRI_CLOUD_MASK_PROBE.md).
Confronti e mappe: [study_seviri_cloud_mask_probe.ipynb](notebooks/study_seviri_cloud_mask_probe.ipynb).
Avanzamento da SSH: `python3 scripts/status_seviri_cloud_probes.py`.
