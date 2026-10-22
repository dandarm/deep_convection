# Audit tecnico: `RDT_original` rispetto al teacher multispettrale UTLS v0

Data dell'audit: 21 ottobre 2026.

## 1. Scopo e conclusione esecutiva

Questo documento confronta tre implementazioni:

1. il teacher attuale del progetto, `multispectral_utls_v0`, in
   `src/emma_gpm/rdt/`;
2. la variante detta **pura**, che usa soltanto osservabili SEVIRI in fase di
   inferenza;
3. la variante detta **full**, che aggiunge ERA5, OPERA e proxy dei prodotti
   ausiliari NWC SAF.

`RDT_original` contiene miglioramenti sostanziali e riutilizzabili. I più
importanti sono la gerarchia termica adattiva a passi di 1 K, i casi morfologici
A/B, il tracking con più stadi di recupero, la gestione esplicita di split e
merge e la costruzione di profili verticali della cella. Non è però ancora un
sostituto affidabile del teacher corrente. Prima di produrre pseudo-label per il
probing occorre risolvere un errore bloccante nello schema semantico, allineare
ATBD e codice, correggere la validazione radar e rendere veritiere le
distinzioni fra `full` e `pure`.

Il verdetto operativo è quindi: **integrare selettivamente le parti forti di
`RDT_original`, senza sostituire in blocco `multispectral_utls_v0`**.

## 2. Riferimento usato per la verifica

La fedeltà all'algoritmo RDT-CW è stata verificata contro le due specifiche
locali ufficiali:

- `docs/NWC-CDOP2-GEO-MFT-SCI-ATBD-Convection_v1.1.pdf`;
- `docs/NWC-CDOP2-GEO-MFT-SCI-ATBD-Convection_v2.1.pdf`.

La specifica descrive, fra gli altri elementi, una segmentazione gerarchica a
passi di 1 K, quattro stadi di tracking, regole specifiche per split/merge, una
discriminazione logistica dipendente da categoria verticale, modo e profondità
della storia, e un rilevatore OT basato sul confronto radiale con 16 pixel
circostanti. La discriminazione operativa non è riproducibile esattamente senza
le matrici di coefficienti associate alla versione del prodotto e al relativo
processo di calibrazione.

Per questo motivo entrambe le nuove varianti devono essere descritte come
**RDT-CW-inspired**. `full` non equivale al prodotto operativo originale e
`pure` non è una derivazione ufficiale prevista dall'ATBD.

## 3. Confronto sintetico

| Componente | UTLS v0 corrente | `RDT_original` puro/full | Valutazione |
|---|---|---|---|
| Dominio | rilevamento e tracking full-domain, crop solo a valle | full-domain negli script settimanali | equivalente come principio |
| Segmentazione | h-minima + watershed, prominenza 6 K | gerarchia 1 K, Case A/B, limiti caldo/freddo | miglioramento reale di `RDT_original` |
| Riferimento UTLS | plateau locale per oggetto, 6 BTD, confidenza e filtro causale | `pure`: scalare globale di scena; `full`: ERA5 | il proxy locale corrente è più coerente col probing; ERA5 è utile come controllo |
| Tracking | sola maschera avvettata + AOR | quattro tentativi, cross-correlazione, vicini, guess, ingrandimento | netto miglioramento di `RDT_original` |
| Split/merge | associazione semplice per area | soglia termica comune e parentela primaria/secondaria | netto miglioramento di `RDT_original` |
| Fase | 5 classi canoniche, cooling e area a 15 min | 5 fasi oggetto più OT, eredità e declassamento | più ricco, ma oggi semanticamente incoerente |
| OT | `D`, firma multispettrale, morfologia e persistenza a 2 frame | algoritmo radiale a 16 direzioni, vicino all'ATBD | confronto/ablazione utile; non sostituzione automatica |
| Cadenza | esattamente 5 min; clip invalida rifiutata | intervalli 0.5--16 min accettati e reset silenzioso | regressione rispetto agli invarianti del progetto |
| Dati mancanti | maschera esplicita; clip finale incompleta rifiutata | spesso trasformati in sfondo semantico con quality bit | regressione per il probing |
| Parallasse | griglia satellitare nativa, allineata all'input | rimappatura delle sole etichette negli output settimanali | rischia disallineamento input/target |
| Validazione | ancora da consolidare | OPERA, ERA5 e confronti estesi | infrastruttura migliore, metriche da correggere |

## 4. Improvement effettivi da adottare

### 4.1 Segmentazione termica gerarchica

La scansione fra limite caldo e freddo a incrementi di 1 K è più vicina alla
costruzione RDT-CW di una cella come componente connessa associata a una
soglia specifica. La distinzione fra torre isolata e sistema multicentro evita
di trattare tutte le depressioni termiche con la stessa regola. Anche i profili
di area, temperatura e forma lungo i livelli termici sono diagnostiche utili
che mancano al watershed corrente.

Questa parte va adottata con una decisione esplicita sulla versione: il codice
usa rapporto di superficie Case B pari a 4, coerente con la scelta dichiarata
v1.1, mentre il testo v2.1 riporta 3. Non va chiamata contemporaneamente
implementazione esatta v2.1.

### 4.2 Tracking a più stadi

È l'avanzamento più importante. Il recupero mediante:

1. avvezione in avanti;
2. correlazione/ricerca nei vicini e posizione prevista;
3. verifica all'indietro;
4. espansione delle celle piccole;

è molto più robusto della sola soglia AOR usata dal teacher corrente. Riduce le
interruzioni spurie dovute a crescita rapida, errori di moto e piccoli shift.
La scelta del ramo principale alla soglia termica comune è inoltre più fisica
della sola area alla soglia propria della cella.

### 4.3 Genealogia e memoria della convezione

La distinzione fra padre principale e secondario convettivo è utile nei merge.
Anche l'eredità dello stato convettivo e il successivo declassamento
deterministico sono concettualmente più completi della macchina a stati v0,
che tende a mantenere una cella nello stato convettivo finché la traccia resta
viva.

La fase `triggering from a split` prevista dall'ATBD è però soppressa nel
codice. Può essere una scelta ragionevole per evitare reset artificiali, ma va
documentata come deviazione e validata, non presentata come fedeltà letterale.

### 4.4 Rilevatore OT radiale

`otd.py` implementa una procedura molto più vicina al rilevatore ufficiale:
soglia fredda dipendente dalla latitudine, confronto WV 6.2 - IR 10.8,
16 direzioni a due distanze, contrasto con l'intorno ed estensione del nucleo.
È una valida **ablazione alternativa** al rilevatore relativo basato su `D`.

Il teacher corrente conserva tuttavia due proprietà importanti: riferimento
locale dell'incudine e conferma temporale su due osservazioni. Il candidato
migliore è quindi un rilevatore ibrido o un confronto pre-registrato fra i due,
non la sostituzione immediata.

### 4.5 Infrastruttura sperimentale

Gli script settimanali, le cache giornaliere, l'integrazione ERA5/OPERA e le
ablazioni su parallasse e dati ausiliari costituiscono una base sperimentale più
ampia del teacher corrente. Sono riutilizzabili dopo la correzione delle
metriche e degli split temporali.

## 5. Problemi bloccanti e regressioni

### 5.1 Schema delle classi incoerente fra configurazione, pipeline e ATBD

È il problema più grave.

- `rdt_nwcsaf/config.py` dichiara le cinque classi di progetto `0..4`.
- `rdt_nwcsaf/discrimination.py` restituisce invece `1` non convettiva,
  `2` triggering, `3` growing, `4` mature e `5` decaying.
- `rdt_nwcsaf/pipeline.py` scrive questi valori e poi sovrascrive i pixel OT
  con `4`. Di conseguenza `4` significa contemporaneamente **mature** e
  **overshooting top**, mentre `5` compare fuori dallo schema dichiarato.
- L'ATBD locale documenta invece sette classi `0..6`, con OT `6`.
- Lo script settimanale usa autonomamente `6` per OT, quindi non ha la stessa
  semantica della pipeline generica.
- Alcuni benchmark assumono `[2,3,4]` come insieme convettivo e interpretano
  `4` come OT: così escludono la classe `5` decadente e confondono maturità e
  OT.

Finché questo non è risolto, i report aggregati non sono semanticamente
riproducibili. La mappatura richiesta dal progetto deve essere unica:

```text
0 -> sfondo
1 -> nube fredda non convettiva
2, 3 -> sviluppo
4, 5 -> maturità/decadimento
6 -> OT
```

La pipeline interna può mantenere una variabile di fase più ricca, ma il campo
`semantic_label` destinato al probing deve restare rigorosamente `0..4`, con
OT rimappato a `4` solo nell'ultimo passaggio.

### 5.2 `full` non esegue una catena full indipendente

Nel run settimanale la segmentazione e il tracking vengono eseguiti una sola
volta usando configurazione e tropopausa della variante pura. Anche il
rilevamento OT di base condivide la stessa geometria. ERA5 entra soprattutto
nella discriminazione successiva. Ne consegue che:

- i contorni `full` non sono stati rilevati con la tropopausa ERA5;
- la genealogia `full` non è indipendente da `pure`;
- l'elevato Dice fra le due uscite è in parte incorporato nel disegno
  sperimentale, non una validazione indipendente.

La storia viene inoltre troncata a 12 slot, cioè 60 minuti, mentre la struttura
di calibrazione prevede fino a 18 slot/90 minuti. ERA5 viene associata a pochi
orari sinottici senza interpolazione temporale completa.

### 5.3 La variante pura usa due concetti diversi di “purezza”

La variante GEN è davvero deterministica all'inferenza, ma usa formule
logistiche e pesi fisici costruiti localmente: non sono i coefficienti
operativi NWC SAF.

La variante CAL usa solo SEVIRI in inferenza, ma le sue funzioni sono state
fit-tate contro ERA5 e OPERA. È quindi **SEVIRI-only in inferenza**, non
indipendente da informazioni esterne nella costruzione del teacher. Questa
distinzione è essenziale se il teacher serve a sostenere una conclusione sulla
termodinamica appresa dal backbone.

### 5.4 Proxy di tropopausa puro fragile e non locale

Nello script settimanale il riferimento è uno scalare per l'intera scena:

```text
T_pure = clip(
    0.6 * P10(WV 6.2 | IR 10.8 < 235 K)
  + 0.4 * P05(IR 10.8 | IR 10.8 < 235 K),
    205 K, 226 K)
```

Questa stima cambia con l'estensione del dominio e con la popolazione di nubi
presente. Può essere dominata da fronti o da altre celle lontane e usa le stesse
nubi profonde che deve poi classificare. Il clipping può mascherare un
fallimento senza produrre un flag di bassa confidenza.

Il proxy UTLS v0 corrente è più articolato ma ha proprietà preferibili per
questo progetto: è locale alla cella, multispettrale, causale, pubblica una
confidenza ed è autorizzato ad astenersi. Conviene conservarlo come riferimento
radiometrico dell'incudine, non chiamarlo tropopausa WMO.

Anche la regressione CAL della tropopausa non è una retrieval completa: mantiene
sei pendenze prefissate e stima soprattutto l'intercetta. Nel report locale
l'errore rispetto a ERA5 è MAE 4.788 K e RMSE 5.773 K; è utile come baseline,
non come verità termodinamica.

### 5.5 Prodotti ausiliari e fedeltà operativa sovrastimati

Il Cloud Type è una classificazione euristica multispettrale, non il prodotto
GEO-CT operativo. Il moto HRW è un block matching locale, non la retrieval HRW
operativa. La maschera di instabilità ERA5 è binaria e usa soglie più permissive
di quelle riportate nella specifica, che prevede stati stabile/incerto/instabile.

Questi proxy sono utili, ma devono essere nominati come tali. La formula “full
originale” non è giustificata senza coefficienti ufficiali, versionamento dei
prodotti ausiliari e verifica end-to-end contro un output NWC SAF autorevole.

### 5.6 Parallasse: funzione incompleta e rischio di disallineamento

`apply_parallax_correction_2d` riceve `t_tropo_k` ma non lo usa nella stima
dell'altezza. L'altezza deriva da `T_surface - IR10.8` diviso per un lapse rate
fisso di 6.5 K/km, limitato a 14.5 km. Il forward splatting crea buchi, poi
riempiti scegliendo la classe massima; ciò può cambiare area, forma e numero di
pixel OT.

Per il probing, spostare soltanto la pseudo-label mentre il cubo SEVIRI resta
nella geometria satellitare produce un target non allineato all'input. La scelta
più sicura è:

- conservare le etichette in griglia SEVIRI nativa per il training;
- usare la parallasse solo nel confronto con radar a terra;
- trattare la correzione come sensibilità di valutazione, con conservazione
  dell'area e mask dei buchi esplicite.

### 5.7 Vincoli temporali meno rigorosi del dataset VideoMAE

La pipeline accetta intervalli fra 0.5 e 16 minuti e, in caso di gap o tempo non
crescente, azzera la storia e continua. Il moto resta espresso in pixel per slot
e non viene scalato coerentemente col `dt`. Lo script settimanale ricostruisce
gli orari con `date_range` e contiene un riordino hard-coded per il 17 settembre
2020.

Questo comportamento è retrogrado rispetto al contratto del progetto: esatti
5 minuti, tempo strettamente crescente, nessuna interpolazione o riparazione
silenziosa, clip incompleta rifiutata.

### 5.8 Gestione dei dati mancanti

In più punti i pixel non validi confluiscono nello sfondo semantico, affidando a
un quality bit la distinzione. Per il probing una maschera osservativa mancante
deve restare separata dallo sfondo fisico; se una clip finale non è completa,
deve essere esclusa.

### 5.9 Deriva fra ATBD di progetto e comportamento eseguibile

L'ATBD locale è utile come documento di progetto, ma non descrive in modo
univoco il software attuale:

- la sezione OT illustra una procedura locale basata su riferimento termico,
  anello e prominenza, mentre `otd.py` esegue il rilevatore radiale a 16
  direzioni;
- l'ATBD presenta `full` come superset della catena pura, ma il run settimanale
  condivide detection, tracking e parte del ramo OT prima di applicare la
  discriminazione diversa;
- la correzione di parallasse è descritta come dipendente dai profili ERA5, ma
  la funzione eseguibile usa un lapse rate costante e non usa l'argomento
  `t_tropo_k`;
- in un punto la discriminazione è riassunta come 24 combinazioni
  categoria/modo; l'ATBD ufficiale contempla anche cinque profondità di storia,
  cioè fino a 6 × 4 × 5 = 120 combinazioni, molte delle quali possono essere
  inattive in una specifica configurazione;
- lo schema semantico documentato non coincide con quello della pipeline
  generica né con quello dello script settimanale.

Prima di considerarlo un ATBD verificabile, ogni formula deve rimandare a una
funzione e a un test, e ogni deviazione dalla specifica ufficiale deve avere una
motivazione, una versione e una conseguenza prevista.

## 6. Audit della validazione

### 6.1 Il benchmark “originale” è un confronto fra teacher

Lo script di confronto può invocare la pipeline senza coefficienti calibrati,
senza ERA5 reale, CT o HRW autorevoli e con una tropopausa fallback. Chiamare
questa uscita “originale” e usarla come verità rende recall e precision una
misura di accordo con un altro teacher, non di accuratezza meteorologica.

### 6.2 La metrica dichiarata pixel-level non è pixel-level

Nel benchmark OPERA vengono calcolate maschere radar a 35 dBZ, ma il conteggio
principale classifica l'oggetto e poi aggiunge tutti i suoi pixel osservati a
TP, FP o FN. Una cella verificata positiva contribuisce quindi con l'intera
area della cella, indipendentemente dalla sovrapposizione spaziale col nucleo
radar. Il risultato è una metrica oggetto pesata per area, non una F1 pixel per
pixel.

Questo spiega perché i valori “pixel” possono apparire molto più alti delle
prestazioni oggetto. Vanno ricalcolati con intersezione esplicita fra maschera
predetta e radar rimappato, riportando anche tolleranza spaziale, qualità radar
e copertura.

### 6.3 Gli split non garantiscono indipendenza meteorologica

Selezionare le prime finestre da una lista di casi random non impedisce che clip
dello stesso temporale, vicine nello spazio e nel tempo, finiscano in train e
test. Lo split deve avvenire per giorno/evento, prima della selezione delle
finestre.

### 6.4 I risultati estesi sono molto meno favorevoli del test piccolo

Il report calibrato piccolo riporta, contro OPERA, F1 oggetto circa 0.365 per
`full` e 0.337 per `pure CAL`. Il test full-domain di sette giorni è più
informativo e mostra, alla soglia 35 dBZ:

- `full`: F1 oggetto 0.0362, precision 0.0202;
- `pure`: F1 oggetto 0.0266, precision 0.0139.

Questi numeri indicano un elevato tasso di falsi allarmi e devono avere maggior
peso delle metriche ottenute su poche finestre selezionate. Il Dice `pure/full`
fra circa 0.90 e 0.95 misura soprattutto l'accordo di due rami che condividono
segmentazione e tracking; non dimostra accuratezza.

## 7. Copertura dei test

I 10 test di `RDT_original` passano, ma sono prevalentemente sintetici. Mancano
test che avrebbero intercettato i problemi principali:

- unicità e range dello schema semantico;
- distinzione fra mature e OT;
- differenza end-to-end fra geometria `full` e `pure`;
- rifiuto di cadenze diverse da 5 minuti;
- continuità e monotonia dei timestamp reali;
- conservazione di area e topologia dopo parallasse;
- assenza di leakage fra eventi;
- correttezza della metrica radar pixel-level;
- riproducibilità dei report a partire dal codice corrente.

## 8. Architettura consigliata per il prossimo teacher

### Conservare dal teacher UTLS v0

- cadenza esatta di 5 minuti e rifiuto delle sequenze incomplete;
- rilevamento/tracking sul dominio completo prima del crop;
- maschera esplicita dei dati mancanti e possibilità di astensione;
- proxy locale multispettrale dell'incudine/UTLS con confidenza;
- profondità relativa `D`;
- cinque classi canoniche del probing;
- persistenza OT a due frame;
- griglia SEVIRI nativa per la loss del probing.

### Importare da `RDT_original`

- gerarchia a passi di 1 K e Case A/B, scegliendo esplicitamente la versione
  della soglia di superficie;
- tracking in quattro stadi;
- confronto di area alla soglia termica comune;
- parentela principale/secondaria e gestione esplicita di split/merge;
- profili verticali e categorie come diagnostiche interne;
- declassamento causale;
- OTD radiale a 16 direzioni come ablation o secondo ramo.

### Rinominare le baseline

Per evitare conclusioni più forti dei metodi:

- `full` → `rdt_cw_inspired_era5_opera_cal`;
- `pure GEN` → `rdt_cw_inspired_seviri_gen`;
- `pure CAL` → `rdt_cw_inspired_seviri_distilled`.

### Ordine di intervento

1. fissare uno schema semantico unico e aggiungere test di contratto;
2. correggere le metriche OPERA e usare split per giorni/eventi;
3. separare davvero le catene `full` e `pure` dall'inizio della detection;
4. portare il tracking multistadio nel teacher principale;
5. confrontare watershed e gerarchia 1 K sugli stessi eventi;
6. confrontare OTD radiale, OTD basato su `D` e loro intersezione/unione;
7. valutare la parallasse solo nel dominio di verifica a terra;
8. congelare versione, soglie, manifest e hash di configurazione prima del
   probing VideoMAE.

## 9. Interpretazione scientifica

Una rete che riproduce queste pseudo-label dimostra che il teacher è
decodificabile dalla rappresentazione congelata. Per sostenere che il backbone
ha appreso dinamica e termodinamica atmosferica servono inoltre encoder casuale,
checkpoint RGB adattato, ablazioni temporali e spettrali, valutazione su eventi
indipendenti e confronto con osservazioni esterne non usate per costruire il
teacher.
