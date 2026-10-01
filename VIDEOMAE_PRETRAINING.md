# Pretraining VideoMAE su SEVIRI

Questa fase usa esclusivamente sequenze SEVIRI non etichettate. EMMA, GPM,
OPERA e le pseudo-label RDT non entrano nel dataset, nel campionamento o nella
loss. Il nome storico della directory Zarr contiene ancora `emma`, ma lo store
contiene soltanto brightness temperature SEVIRI calibrate.

## Modello

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

Prima del training GPU occorre costruire un manifest SEVIRI-only esteso,
calcolare le statistiche su tutto il training split e definire anni disgiunti
per training, validation e test. Il primo esperimento principale partirà da
pesi casuali; il checkpoint RGB adattato a sette canali resterà un confronto
controllato separato.
