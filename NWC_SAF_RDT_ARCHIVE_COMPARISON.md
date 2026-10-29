# Confronto con l'archivio pubblico NWC SAF RDT-CW

L'archivio pubblico NWC SAF espone visualizzazioni GIF, non i campi numerici
NetCDF storici. Il confronto implementato qui usa quindi i contorni colorati
ufficiali come riferimento osservabile e non li presenta come una ground truth
pixel-perfect.

Sono incluse soltanto le fasi:

- `triggering` (giallo);
- `growing` (rosso);
- `mature` (viola);
- `decay` (blu).

`triggering from split` (arancione), nube fredda non convettiva e overshooting
top sono esclusi dalle metriche.

## Download delle GIF

```powershell
.\.venv\Scripts\python.exe scripts/download_nwcsaf_rdt_archive.py `
  --start 2026-10-27T00:00:00 `
  --end 2026-10-28T00:00:00 `
  --output-dir E:\Datasets\NWC_SAF_RDT\20261027
```

Lo script prova in ordine MSG4, MSG3, MSG2 e MSG1, valida ogni GIF, pubblica i
file in modo atomico e salva URL, checksum, dimensioni e stato in
`manifest.csv`. Un nuovo avvio riusa soltanto file già validi.

## Preparazione dei dati RSS locali

Per il teacher servono tutti i frame reali a cinque minuti. Un intervallo può
essere scaricato e ridotto ai sette canali Kelvin con gli strumenti esistenti:

```powershell
.\.venv\Scripts\python.exe scripts/download_msg_seviri.py `
  --start 2026-10-27T09:00:00 `
  --end 2026-10-27T13:00:00 `
  --out E:\Datasets\NWC_SAF_RDT\rss_native

.\.venv\Scripts\python.exe scripts/crop_msg_seviri_native.py `
  --input E:\Datasets\NWC_SAF_RDT\rss_native `
  --out E:\Datasets\NWC_SAF_RDT\rss_zarr
```

## Confronto

`--start` include il warm-up causale del tracker. Le metriche iniziano solo da
`--evaluation-start`, che deve essere successivo.

```powershell
.\.venv\Scripts\python.exe scripts/compare_nwcsaf_rdt_archive.py `
  --zarr-root E:\Datasets\NWC_SAF_RDT\rss_zarr `
  --gif-manifest E:\Datasets\NWC_SAF_RDT\20261027\manifest.csv `
  --start 2026-10-27T09:00:00 `
  --evaluation-start 2026-10-27T10:00:00 `
  --end 2026-10-27T13:00:00 `
  --output-dir E:\Datasets\NWC_SAF_RDT\comparison
```

Il confronto viene calcolato sulla griglia RSS north-up/west-left. I contorni
della vista MSG primaria a 0° vengono riportati sulla geometria RSS a 9,5°E
con una trasformazione satellitare 3D e ipotesi di quota nube a 8, 12 e 15 km.
La calibrazione fissa GIF-proiezione è in
`configs/nwcsaf_rdt_gif_v2025_msg.json`; è stata stimata sul fondo IR12.0 dello
stesso slot, senza usare i contorni RDT.

Gli output sono:

- `spatial_metrics.csv`: IoU, precision, recall, rapporto di area e metriche
  tolleranti a 5/10/20 km per fase e quota;
- `object_matches.csv`: distanza dei centroidi, rapporto di area, distanza dei
  bordi e Hausdorff95 degli oggetti abbinati;
- `phase_confusion.csv`: matrice di confusione in pixel e km²;
- `object_counts.csv`: oggetti di riferimento, previsti e abbinati;
- `protocol.json` e `summary.json`: provenienza, geometria e sintesi del run.

## Procedura di verifica riproducibile

La verifica con le immagini pubbliche NWC SAF segue questi passaggi. La
distinzione fra dato osservato, trasformazione geometrica e interpretazione è
obbligatoria: il GIF pubblico è un prodotto grafico RDT-CW, non il campo
numerico operativo originale.

1. **Scoperta e selezione degli episodi.** Lo scanner interroga entrambe le
   versioni dell'archivio, misura l'attività visibile e costruisce finestre di
   sei ore distribuite fra mesi e coorti. La selezione non usa le uscite del
   teacher locale.
2. **Acquisizione verificabile.** Per ogni slot a quindici minuti vengono
   conservati URL, orario, satellite, dimensioni e SHA-256 nel `manifest.csv`.
   File assenti, corrotti o cambiati non sono accettati silenziosamente.
3. **Sequenza locale completa.** Ogni finestra usa un'ora di warm-up seguita da
   sei ore di valutazione: 84 scansioni RSS reali a cinque minuti. Una finestra
   con 83/84 frame viene esclusa; non sono ammessi riempimenti, ripetizioni o
   interpolazioni.
4. **Esecuzione causale.** Il teacher locale elabora tutte le 84 osservazioni in
   ordine crescente. Soltanto le 24 osservazioni co-temporali con l'archivio
   pubblico entrano nelle metriche e nel video.
5. **Allineamento geometrico.** I dati locali restano nella proiezione RSS a
   9,5°E. La vista pubblica MSG a 0° è descritta dalla calibrazione specifica
   della coorte; la trasformazione tiene conto delle due posizioni satellitari
   e valuta la parallasse a 8, 12 e 15 km. Le etichette categoriche sono
   campionate con nearest neighbour.
6. **Confronto semantico.** Si confrontano esclusivamente `triggering`,
   `growing`, `mature` e `decay`. Split-trigger, nube fredda non convettiva e
   overshooting top sono esclusi perché non hanno una corrispondenza diretta
   nel protocollo scelto.
7. **Controllo spaziale e per oggetti.** Oltre a IoU, precision e recall senza
   tolleranza, si riportano overlap con buffer di 5/10/20 km, rapporto di area,
   distanza dei centroidi, Hausdorff95 e conteggi di oggetti associati. La
   sensibilità alla quota nube permette di distinguere un errore di parallasse
   da una differenza del teacher.
8. **Ispezione visuale sincronizzata.** Il video mostra in alto il prodotto
   pubblico e in basso IR 10.8 più classi locali, allo stesso UTC e sullo stesso
   canvas. Alcuni GIF storici variano di 1–2 righe terminali: il renderer
   normalizza soltanto quel bordo sul canvas calibrato, senza riscalare la
   mappa o alterarne la regione comune.

### Come interpretare lo scarto

Un buon overlap della regione convettiva complessiva, accompagnato da una
confusione elevata fra fasi, indica principalmente una differenza nelle regole
di ciclo di vita, nel tracking o nelle transizioni temporali. Una variazione
forte delle metriche fra 8, 12 e 15 km suggerisce invece sensibilità alla
parallasse. Distanze dei centroidi sistematiche e concordi possono indicare una
calibrazione geometrica residua; differenze di area con centroidi vicini sono
più coerenti con soglie di segmentazione differenti.

Il confronto non trasforma il GIF in ground truth numerica e non dimostra, da
solo, equivalenza con l'algoritmo operativo NWC SAF. Serve a verificare in modo
riproducibile dove il teacher `RDT-CW-inspired` concorda con il prodotto
pubblico e dove divergono geometria, segmentazione o fase del ciclo di vita.

## Video sincronizzato

Il video impila il GIF pubblico sopra il nostro IR 10.8 con le maschere
colorate. Il pannello locale viene riportato sulla stessa vista MSG a 0° del
GIF, usando la stessa correzione di parallasse a 12 km del confronto
quantitativo. Il teacher continua a elaborare tutti i frame RSS a cinque
minuti; vengono visualizzati soltanto gli istanti comuni a quindici minuti.

```powershell
.\.venv\Scripts\python.exe scripts/render_nwcsaf_rdt_comparison_video.py `
  --zarr-root E:\Datasets\NWC_SAF_RDT\rss_zarr `
  --gif-manifest E:\Datasets\NWC_SAF_RDT\20261027\manifest.csv `
  --start 2026-10-27T09:00:00 `
  --evaluation-start 2026-10-27T10:00:00 `
  --end 2026-10-27T13:00:00 `
  --cloud-height-km 12 `
  --fps 1 `
  --output E:\Datasets\NWC_SAF_RDT\nwcsaf_vs_local.mp4
```

Accanto all'MP4 vengono scritti un'anteprima PNG, i metadati JSON e il log di
FFmpeg. Il file di destinazione non viene sovrascritto.

## Ricerca dei periodi convettivi

L'indice di scoperta legge i GIF in streaming, conta contorni e componenti
connesse delle quattro fasi richieste e scarta subito le immagini. Gli
split-trigger restano soltanto una diagnostica esclusa. I risultati vengono
committati in SQLite e la stessa invocazione riprende automaticamente gli slot
non ancora catalogati.

Le due versioni devono restare coorti separate: la geometria v2021 è calibrata
in `configs/nwcsaf_rdt_gif_v2021.json`, mentre la v2025 usa
`configs/nwcsaf_rdt_gif_v2025_msg.json`.

```powershell
.\.venv\Scripts\python.exe scripts/scan_nwcsaf_rdt_activity.py `
  --period v2021_pretransition v2021 2024-07-26T00:00:00 2025-07-30T00:00:00 `
  --period v2025_current v2025_msg 2025-07-30T00:00:00 2026-10-27T14:00:00 `
  --output-dir E:\Datasets\NWC_SAF_RDT\activity_hourly `
  --cadence-minutes 60 `
  --workers 8

.\.venv\Scripts\python.exe scripts/rank_nwcsaf_rdt_activity.py `
  --database E:\Datasets\NWC_SAF_RDT\activity_hourly\activity.sqlite `
  --output-dir E:\Datasets\NWC_SAF_RDT\activity_ranking `
  --window-hours 6 `
  --candidate-count 36 `
  --control-count 8
```

Il ranking privilegia numero e persistenza delle celle, ricchezza delle fasi e
supporto dei contorni; seleziona prima un caso per mese e poi completa la lista
con gli episodi più attivi, mantenendo almeno dodici ore fra finestre della
stessa coorte. Il campo `dominant_canvas_zone` è soltanto una coordinata
normalizzata di screening e non sostituisce la geolocalizzazione fisica.

La lista bilanciata può essere raffinata a cadenza di quindici minuti senza
cambiare scanner:

```powershell
.\.venv\Scripts\python.exe scripts/scan_nwcsaf_rdt_activity.py `
  --period-csv E:\Datasets\NWC_SAF_RDT\activity_ranking\balanced_candidates.csv `
  --output-dir E:\Datasets\NWC_SAF_RDT\activity_15min `
  --cadence-minutes 15 `
  --workers 8

.\.venv\Scripts\python.exe scripts/summarize_nwcsaf_rdt_candidates.py `
  --database E:\Datasets\NWC_SAF_RDT\activity_15min\activity.sqlite `
  --output E:\Datasets\NWC_SAF_RDT\activity_15min\candidate_summary.csv
```

Il riepilogo raffinato aggiunge copertura, frazione di slot attivi e massima
persistenza consecutiva: in questo modo un picco isolato non domina una
finestra con attività convettiva sostenuta.

La fonte pubblica è l'[archivio NWC/GEO](https://www.nwcsaf.org/web/guest/nec/geo-geostationary-archive).
La semantica e i limiti del prodotto sono descritti nella
[pagina ufficiale RDT-CW v2025](https://www.nwcsaf.org/rdt_description_2025).
