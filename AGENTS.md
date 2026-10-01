# AGENTS.md

Queste istruzioni valgono per l'intero repository. Il progetto attivo è il
pretraining self-supervised di VideoMAE su SEVIRI seguito dal probing delle
dinamiche convettive mediante pseudo-label deterministiche RDT-CW-inspired.

## Letture obbligatorie

Prima di modificare codice, configurazioni o documentazione, leggere
integralmente e in quest'ordine:

1. `README.md`
2. `RDT_plan.md`
3. `VIDEOMAE_PRETRAINING.md`
4. `HANDOFF_VIDEOMAE_PRETRAINING_GPU.md` per attività di trasferimento,
   configurazione CUDA o training GPU

Consultare inoltre direttamente i file che definiscono il comportamento
eseguibile:

- `src/emma_gpm/seviri.py`
- `src/emma_gpm/seviri_clips.py`
- `src/emma_gpm/videomae.py`
- `scripts/pretrain_videomae.py`
- `tests/test_seviri.py`
- `tests/test_seviri_clips.py`
- `tests/test_videomae.py`

I documenti storici relativi ad altri pilot non definiscono l'approccio
corrente e non devono essere usati per cambiare obiettivo, dominio o sorgenti
del pretraining, salvo richiesta esplicita dell'utente.

## Obiettivo canonico

Il flusso previsto è:

```text
SEVIRI non etichettato
    -> pretraining VideoMAE self-supervised a 7 canali
    -> backbone congelato
    -> decoder 3D FPN leggero
    -> segmentazione a 5 classi costruite dal teacher RDT deterministico
```

Il pretraining e il probing sono fasi separate. Le pseudo-label RDT non devono
entrare nel pretraining, direttamente o attraverso il campionamento.

## Invarianti dei dati

- Ordine dei canali obbligatorio:
  `WV_062`, `WV_073`, `IR_087`, `IR_097`, `IR_108`, `IR_120`, `IR_134`.
- Forma del campione: `(T,C,H,W) = (16,7,224,224)`.
- Cadenza: 5 minuti; 16 frame coprono 75 minuti fra primo e ultimo.
- Tempo sempre crescente. Vietati shuffle, inversione, interpolazione e
  ripetizione di frame mancanti.
- Orientamento sempre north-up e west-left. Il loader deve dedurlo dalle
  coordinate, non assumere l'ordine degli assi né applicare flip ciechi.
- Valori in brightness temperature Kelvin prima della trasformazione globale.
- Usare una sola media e una sola deviazione standard calcolate su tutti i
  pixel e tutti i canali del training split.
- Vietate normalizzazioni per canale, clip, frame o patch.
- Vietati channel shuffle, rotazioni, riflessioni e RGB color jitter.
- Una clip incompleta, riordinata, non finita o con canali mancanti deve essere
  rifiutata, mai riparata tramite sostituzione silenziosa.

## Vincoli del modello

- Backbone principale: VideoMAE-Small.
- `num_channels=7`, `num_frames=16`, `tubelet_size=2`, `patch_size=16`.
- Patch embedding `Conv3d` con sette canali in ingresso.
- Larghezza del target MAE: 3584 valori per tubelet.
- `norm_pix_loss=False`.
- Masking iniziale: tube masking al 90%.
- Il run scientifico principale parte da pesi casuali.
- Un checkpoint RGB adattato è soltanto un controllo separato; non chiamarlo
  pretraining atmosferico from scratch.

## Teacher RDT e interpretazione

Il teacher usa esclusivamente WV 6.2 µm e IR 10.8 µm e produce le classi:
sfondo, nube fredda non convettiva, sviluppo, maturità/decadimento e
overshooting top.

Usare la dicitura `RDT-CW-inspired` finché soglie, tracking, gestione di
split/merge e transizioni del ciclo di vita non saranno validati contro la
specifica operativa autorevole. Una buona prestazione sulle pseudo-label mostra
che il teacher è decodificabile dalla rappresentazione; da sola non dimostra
che il backbone abbia appreso la termodinamica atmosferica.

Il backbone deve rimanere congelato nel probing principale. Confronti con
encoder casuale, encoder RGB adattato e ablazioni temporali e spettrali sono
obbligatori per sostenere la conclusione scientifica.

## Stato e limiti correnti

Il manifest da 100 clip e quello da due clip servono soltanto a integrazione,
visualizzazione e smoke test. Non avviare o descrivere come pretraining
scientifico un run basato soltanto su questi manifest.

Prima di un job GPU lungo servono almeno:

- archivio SEVIRI continuo e split temporali senza leakage;
- statistiche globali persistenti calcolate sul solo training split;
- mixed precision verificata;
- checkpoint atomici e ripresa di modello, optimizer, scheduler e RNG;
- logging di loss, throughput, memoria, seed, commit e manifest;
- validation e brevi run di overfit/integrazione.

## Verifica minima

Da PowerShell nella radice del repository:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe -m pytest -q
```

La suite corrente contiene almeno 23 test. Aggiungere test quando si modifica
ordine dei canali, orientamento, normalizzazione, mascheramento, patchification
o costruzione del modello.

## Dati e Git

Non aggiungere al repository:

- prodotti nativi o Zarr SEVIRI;
- dataset intermedi;
- checkpoint e optimizer;
- log di training;
- figure, PNG, MP4 o notebook eseguiti;
- credenziali o file `.env`.

I manifest piccoli, le configurazioni, il codice, i test e la documentazione
sono versionabili. Prima di un commit, ispezionare sempre `git status` perché il
worktree può contenere modifiche appartenenti ad altre task. Preparare commit
mirati e non includere file estranei senza una richiesta esplicita.
