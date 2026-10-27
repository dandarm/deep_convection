# RDT Puro SEVIRI: Architettura, Fisica e Algoritmo
### Presentazione Tecnica delle Fasi Operative dell'Algoritmo `nwcsaf_rdt_seviri_pure_gen`

---

## Slide 1: Obiettivo e Filosofia del Modello Puro
### Un sistema di tracciamento convettivo autonomo e indipendente da modelli esterni

- **Origine scientifica:** Adattamento fedele dell'algoritmo **RDT-CW** (*Rapidly Developing Thunderstorms - Convection Warning*) sviluppato da Météo-France per **EUMETSAT NWC SAF**.
- **Autonomia spettrale al 100%:** L'algoritmo opera esclusivamente sui canali infrarossi di **Meteosat Second Generation (SEVIRI)**.
- **Zero dipendenze esterne:** Non richiede modelli numerici di previsione (NWP come ECMWF/ERA5), non richiede fulminazioni a terra e non richiede radar.
- **Scopo scientifico principale:** Fornire un *teacher* deterministico ad altissima qualità e risoluzione temporale (5 minuti) per guidare il probing delle dinamiche convettive dopo il pretraining self-supervised di **VideoMAE**.

---

## Slide 2: Dati di Ingresso e Griglia Geostazionaria
### Il flusso osservativo di base

- **I 7 Canali Termici SEVIRI (in Kelvin):**
  - Vapore acqueo: `WV_062` (alta troposfera), `WV_073` (media troposfera).
  - Finestra infrarossa e split-window: `IR_087`, `IR_097` (ozono), `IR_108` (canale topologico guida), `IR_120`, `IR_134` ($CO_2$).
- **Dominio Spaziale Mediterraneo:**
  - Griglia geostazionaria ritagliata: $763 \times 1762$ pixel (o $764 \times 1762$ su MSG-2).
  - Risoluzione nominale: $3\text{ km}$ al sub-satellite point, $4\text{–}5\text{ km}$ sull'area mediterranea.
  - Orientamento geodetico rigoroso: *North-Up, West-Left*.
- **Risoluzione Temporale:**
  - Cadenza rapida RSS (*Rapid Scanning Service*): **1 frame ogni 5 minuti** (288 slot al giorno).
  - Vincolo di causalità: tempo strettamente crescente, nessun riempimento sintetico.

---

## Slide 3: Architettura Algoritmica ad Alto Livello
### La catena causale delle 7 fasi operative (senza dipendenze geometriche esterne)

```text
[Input SEVIRI 7 Canali (t)]
       │
       ▼
 [Fase 0] Riferimenti Radiativi: Stima di Tropopausa (T_tropo) e Suolo (T_suolo)
       │
       ▼
 [Fase 1] Ancillari Interni: Cloud Type (CT opaco) & Vento di Guida AMV (HRW)
       │
       ▼
 [Fase 2] Rilevamento Multi-Soglia (1 K) & Isolamento Torri Convettive (>= 6 K)
       │
       ▼
 [Fase 3] Tracking Lagrangiano Multi-Passo & Risoluzione Split / Merge
       │
       ▼
 [Fase 4] Rilevamento Overshooting Tops (OTD) a 2 Stadi
       │
       ▼
 [Fase 5] Discriminazione Convettiva GEN & Indice di Profondità Relativa (D)
       │
       ▼
 [Fase 6] Ciclo di Vita Convettivo a 4 Fasi con Ereditarietà Madre
       │
       ▼
 [Fase 7] Rendering Temporale Fluido (Filtro EMA) & Cache Semantica (.npz)
```

---

## Slide 4: Fase 0 — Riferimenti Termici Radiativi
### Determinazione causale di $T_{\text{tropo}}$ e $T_{\text{suolo}}$ senza sondaggi meteo

- **Problema fisico:** Misurare l'altezza e la profondità delle nubi richiede di conoscere la quota della tropopausa e la temperatura del suolo, che variano continuamente nello spazio e nel tempo.
- **Stima della Tropopausa Radiativa ($T_{\text{tropo}}$):**
  - Isolamento dei pixel freddi della scena ($\text{IR}_{10.8} < 235\text{ K}$).
  - Combinazione lineare dei percentili freddi del vapore acqueo e della finestra IR:
    $$T_{\text{tropo}} = 0.6 \cdot P_{10}(\text{WV}_{6.2}) + 0.4 \cdot P_{5}(\text{IR}_{10.8})$$
  - Bounding fisico di sicurezza: $T_{\text{tropo}} \in [205\text{ K}, 226\text{ K}]$.
- **Stima della Temperatura di Suolo ($T_{\text{suolo}}$):**
  - Calcolata dai pixel caldi e sereni ($P_{95}(\text{IR}_{10.8})$) con valore di riferimento a $292\text{ K}$.
- **Tracciabilità delle variabili introdotte:**
  - $T_{\text{tropo}}$ verrà utilizzata in: **Fase 1** (soglie nubi alte), **Fase 2** (limite $LimitTropo$), **Fase 4** (pre-filtro OT) e **Fase 5** (indice di profondità $D$).
  - $T_{\text{suolo}}$ verrà utilizzata in: **Fase 1** (contrasto suolo-nube) e **Fase 5** (profondità assoluta).

---

## Slide 5: Fase 1A — Classificazione delle Nubi (GEO-CT Interno)
### Filtraggio delle nubi spesse e mascheramento della superficie

- **Obiettivo:** Riconoscere le nubi opache a sviluppo verticale ed escludere il cielo sereno, il suolo caldo e le nubi basse stratiformi prima di avviare il rilevamento delle celle.
- **Discriminatori Spettrali Split-Window:**
  - Differenza spettrale di finestra $\Delta T_{\text{split}} = \text{IR}_{10.8} - \text{IR}_{12.0}$:
    - Valori prossimi a $0\text{ K}$ o negativi $\to$ nubi otticamente spesse e dense (opache).
    - Valori positivi elevati ($> 2\text{–}3\text{ K}$) $\to$ cirri sottili semitrasparenti o vapore acqueo al suolo.
  - Differenza tri-spettrale $\Delta T_{\text{tri}} = \text{IR}_{08.7} - \text{IR}_{10.8}$:
    - Sensibile alla fase termodinamica delle particelle (ghiaccio sommitale vs goccioline d'acqua).
- **Uso dei Riferimenti Radiativi ($T_{\text{suolo}}, T_{\text{tropo}}$):**
  - Contrasto con la superficie: un pixel è nuvoloso se $\text{IR}_{10.8} < T_{\text{suolo}} - 5\text{ K}$.
  - Livello di nube alta: identificato da $\text{IR}_{10.8} \le T_{\text{tropo}} + 25\text{ K}$.
- **Output Prodotto (`ct_cloudy_mask`):**
  - Maschera booleana bidimensionale delle nubi opache.
  - **Uso successivo:** Viene passata alla **Fase 2** come maschera di contenimento per restringere l'estrazione delle torri convettive ed eliminare falsi positivi su nubi basse o mare caldo.

---

## Slide 6: Fase 1B — Campo di Vento di Guida AMV (GEO-HRW Interno)
### Tracciamento del moto atmosferico locale per l'avvezione lagrangiana

- **Obiettivo:** Stimare la velocità e la direzione di spostamento delle masse nuvolose tra due slot temporali consecutivi ($t - 5\text{ min}$ e $t$) senza modelli di vento NWP.
- **Algoritmo di Block-Matching Locale:**
  - Vengono selezionati blocchi di pixel caratteristici sulle nubi tracciabili (`IR_108` e `WV_062`).
  - Ricerca per correlazione incrociata spaziale nella finestra di ricerca massimizzando la somiglianza.
- **Output Prodotto ($\vec{v}_{\text{AMV}}$):**
  - Campo vettoriale continuo di spostamento $(\Delta y, \Delta x)$ espresso in pixel per intervallo di 5 minuti.
  - **Uso successivo:** Viene passato alla **Fase 3** per guidare l'avvezione in avanti delle celle durante il tracciamento lagrangiano.

---

## Slide 7: Fase 2A — Rilevamento Multi-Soglia (Tree Slicing)
### Esplorazione topologica della nube a passi termici discreti di 1 K

- **Approccio Topografico Inverso:**
  - La temperatura di brillanza $\text{IR}_{10.8}$ viene considerata come una quota (più freddo = più alto).
  - Vengono eseguite sezioni orizzontali a passi di **$1\text{ K}$**, partendo dal livello caldo $T_{\text{warm}} = 278.15\text{ K}$ ($+5^\circ\text{C}$) fino al livello freddo $T_{\text{cold}} = 198.15\text{ K}$ ($-75^\circ\text{C}$).
- **Uso di `ct_cloudy_mask` (da Fase 1):**
  - L'affettamento opera esclusivamente sui pixel validati dalla maschera nubi opache.
- **Arresto della Discesa (`LimitTropo`):**
  - Impedisce di scendere nelle quote fredde stratiformi dell'incudine (*anvil*).
  - Definisce il pavimento termico minimo di arresto della torre:
    $$T_{\text{limit}} = \max\left(T_{\text{BTLIMIT}},\, T_{\text{tropo}}\right)$$
    con soglia climatica fissa $T_{\text{BTLIMIT}} = 213.15\text{ K}$ ($-60^\circ\text{C}$) e tropopausa locale $T_{\text{tropo}}$ (da **Fase 0**).
- **Struttura Dati ad Albero (`ThresholdSliceHierarchy`):**
  - Per ciascun livello termico $k$, si estraggono le componenti connesse spaziali e si collegano ai nodi genitori.
  - Filtro dimensionale: superficie minima $S \ge 60\text{ km}^2$ (4–5 pixel SEVIRI), massima $S_{\max} = 400.000\text{ km}^2$.

---

## Slide 8: Fase 2B — Isolamento delle Torri e Definizione di $T_{\text{base}}$
### Separazione della convezione attiva dai cirri stratiformi circostanti

- **Definizione della Base della Torre ($T_{\text{base}}$ o $T_{\text{BT}}$):**
  - È la temperatura $\text{IR}_{10.8}$ dell'isoterma che racchiude il contorno geometrico 2D inferiore della torre convettiva.
- **Criterio di Prominenza Termica ($\Delta T_{\text{tower}}$):**
  - Una torre esiste solo se il nucleo sommitale freddo $T_{\min}$ sporge dalla base di almeno **$6\text{ K}$**:
    $$\Delta T_{\text{tower}} = T_{\text{base}} - T_{\min} \ge 6.0\text{ K}$$
- **Separazione di Torri Coalescenti alla Soglia di Fusione ($T_{\text{merge}}$):**
  - Se due torri distinte si uniscono su un tronco comune all'isoterma $T_{\text{merge}}$, ciascuna torre viene separata e la sua base fissata a:
    $$T_{\text{base}} = T_{\text{merge}} - 1.0\text{ K}$$
- **Raffinamento Morfologico della Base (Anti-Anvil):**
  - *Caso A (Strozzatura verticale):* taglio se il rapporto di allargamento tra livelli adiacenti supera la strozzatura ripida ($S_{n+1}/S_n \le 0.70$).
  - *Caso B (Taglio piramidale):* l'area di base $S(T_{\text{base}})$ non può superare 4 volte l'area sommitale $S_{\text{ST}}$ (misurata a $T_{\min} + 6\text{ K}$):
    $$S(T_{\text{base}}) \le 4 \cdot S_{\text{ST}}$$
    Se eccede, $T_{\text{base}}$ viene alzata (forzata a un'isoterma più fredda) per eliminare l'incudine piatta circostante.
- **Variabili Estratte per Cella:** Minimo $T_{\min}$, base $T_{\text{base}}$, area $S$, centroide e contorno 2D.

---

## Slide 9: Fase 3A — Tracciamento Lagrangiano Multi-Passo
### Connessione temporale robusta contro deformazioni e occlusioni

- **Uso del Campo di Vento $\vec{v}_{\text{AMV}}$ (da Fase 1):**
  - Ogni cella al tempo $t - 1$ viene proiettata in avanti nello spazio:
    $$\vec{x}_{\text{pred}} = \vec{x}_{t-1} + \vec{v}_{\text{cell}} \cdot \Delta t$$
- **Criterio di Sovrapposizione Spaziale (AOR):**
  - Calcolo dell'*Area Overlap Ratio* tra la cella predetta $\mathcal{A}_{\text{pred}}$ e la cella osservata $\mathcal{B}_{\text{curr}}$:
    $$\text{AOR} = \frac{|\mathcal{A}_{\text{pred}} \cap \mathcal{B}_{\text{curr}}|}{\min(|\mathcal{A}_{\text{pred}}|, |\mathcal{B}_{\text{curr}}|)} \ge 15\%$$
- **Concatenazione a 4 Passi:**
  1. *Forward Advection* con vento AMV.
  2. *Local Block Matching* attorno al centroide se l'AOR iniziale è ambiguo.
  3. *Backward Advection* di verifica per confermare il legame.
  4. *Enlargement / Dilatazione* per piccole celle in rapido innesco ($\le 5\text{ pixel}$).

---

## Slide 10: Fase 3B — Gestione di Split, Merge e Traiettoria
### Continuità della storia convettiva nelle fusioni e frammentazioni

- **Identificazione Topologica dei Casi di Tracking:**
  - *Continuazione 1-a-1:* La cella prosegue la sua traiettoria con identico identificatore.
  - *Split (1-a-Molti):* Una cella madre si suddivide in due o più frammenti figli.
  - *Merge (Molti-a-1):* Più celle distinte convergono in un unico sistema.
- **Risoluzione sulla Soglia Termica Comune ($T_{\text{comm}}$):**
  - In caso di split o merge, il "ramo principale" (*primary parent*) è assegnato al frammento che massimizza la somiglianza superficiale alla **temperatura comune condivisa**:
    $$T_{\text{comm}} = \min\left(T_{\text{base}}(t),\, T_{\text{base}}(t - \Delta t)\right)$$
  - Garantisce che i tassi di espansione $\Delta S/\Delta t$ confrontino la medesima isoterma fisica nel tempo.
- **Smussamento Esponenziale della Velocità (EMA):**
  - La velocità di spostamento viene filtrata causalmente per evitare strappi:
    $$\vec{v}_t = (1 - \alpha) \cdot \vec{v}_{t-1} + \alpha \cdot \vec{v}_{\text{inst}}, \quad \text{con } \alpha = 0.5$$
- **Memoria Storica Costruita:** Sequenza temporale degli ultimi 12 frame (1 ora di vita) per il calcolo dei trend.

---

## Slide 11: Fase 4 — Rilevamento Overshooting Tops (OTD)
### Identificazione delle cupole convettive penetranti in stratosfera

- **Cos'è un Overshooting Top (OT):**
  - Una cupola convettiva violenta in cui la corrente ascensionale (*updraft*) sfonda il livello di equilibrio neutro e penetra nella stratosfera.
- **Algoritmo di Rilevamento a Due Stadi (100% SEVIRI):**
  1. **Pre-selezione Termica con $T_{\text{tropo}}$ (da Fase 0):**
     - Il pixel deve essere vicino o più freddo della tropopausa locale:
       $$T_{\text{pixel}} \le T_{\text{tropo}} + 5\text{ K}$$
  2. **Conferma a Doppio Criterio Fisico:**
     - *Prominenza Termica Locale:* Il pixel è più freddo dell'anello circostante dell'incudine di almeno **$3.5\text{ K}$**.
     - *Inversione Strato-Troposferica del Vapore:* A causa dell'inversione termica stratosferica, la sommità della cupola mostra:
       $$\text{BTD}(\text{WV}_{6.2} - \text{IR}_{10.8}) \ge -0.5\text{ K}$$
- **Assegnazione di Classe:** I pixel confermati vengono etichettati come **Classe 6 (Overshooting Top)**.

---

## Slide 12: Fase 5A — Discriminazione Convettiva e Indice di Profondità Relativa $D$
### Separazione fisica tra nubi stratiformi e convezione attiva

- **Formulazione dell'Indice di Profondità Relativa $D$:**
  - Misura la frazione di colonna troposferica occupata dalla torre rispetto all'altezza massima teorica disponibile:
    $$D = \frac{T_{\text{base}} - T_{\min}}{T_{\text{base}} - T_{\text{tropo}}}$$
    con $T_{\text{base}}$ e $T_{\min}$ definiti in **Fase 2B**, e $T_{\text{tropo}}$ calcolata in **Fase 0**.
- **Significato Fisico e Regimi Operativi di $D$:**
  - $D < 0.25$: Torre debole o embrionale (convezione superficiale / cumulo modesto).
  - $0.25 \le D < 0.50$: Convezione moderata a medio sviluppo.
  - $0.50 \le D < 0.75$: Convezione profonda / cumulonembo maturo ben strutturato.
  - $D \ge 0.75$: Convezione violenta a tutto spessore (*severe convection*), sommità a ridosso della tropopausa.
- **Predittori Dinamici Multi-Finestra (15, 30, 45, 60 min):**
  - Tasso di raffreddamento della sommità: $\Delta T_{\min} / \Delta t$ [K/h].
  - Tasso di espansione dell'area: $\Delta S / \Delta t$ [$\text{km}^2$/h].
  - Gradiente perimetrale $\text{GradP}$ (elevato nelle celle vigorose e convesse).
- **Esito Binario:** Se i criteri di $D$, cooling ed espansione sono soddisfatti, la cella è **Convettiva**; altrimenti è declassata a **Nube fredda non convettiva (Classe 1)**.

---

## Slide 13: Fase 5B — Ciclo di Vita Convettivo a 4 Fasi
### Classificazione dello stadio evolutivo e preservazione della memoria madre

Se la cella è convettiva, l'algoritmo determina il suo stadio nel **ciclo di vita a 4 fasi canoniche RDT**:

1. **Triggering (Classe 2 — Giallo Brillante):**
   - Cella giovane in rapido raffreddamento iniziale ($\Delta T/\Delta t \ll 0$), piccola area in espansione, transizione dalla fase calda.
2. **Growing (Classe 3 — Arancione Vivo):**
   - Sviluppo vigoroso: forte espansione superficiale, crescita dell'indice $D$, ascesa continua del top e aumento del gradiente perimetrale.
3. **Mature (Classe 4 — Rosso Vivo):**
   - Massima intensità: indice di profondità elevato ($D \ge 0.50$), $T_{\min}$ stabilizzata ai valori minimi prossimi alla tropopausa, espansione rallentata, presenza frequente di OT.
4. **Decaying (Classe 5 — Rosso Scuro / Bordeaux):**
   - Fase di dissipazione: riscaldamento progressivo della sommità ($\Delta T/\Delta t > 0$), progressivo collasso dell'updraft e riduzione della compattezza.

- **Ereditarietà dello Stato Madre:** Nei frammenti derivanti da *Split* (riconosciuti in **Fase 3B**), la cella figlia non riparte artificiosamente da *Triggering*, ma eredita lo stato evolutivo della cella madre.

---

## Slide 14: Fase 6 — Rendering Fluido e Post-Processing Visivo
### Eliminazione degli artefatti e composizione ad alta leggibilità

- **Smoothing Temporale dei Colori (Filtro EMA a livello di pixel):**
  - Per evitare scatti cromatici quando una cella oscilla sulla soglia tra due classi contigue, i colori RGB vengono filtrati con media mobile esponenziale:
    $$C_t = (1 - \gamma) \cdot C_{t-1} + \gamma \cdot C_{\text{target}}, \quad \text{con } \gamma = 0.38$$
  - Risultato: transizioni cromatiche armoniose e naturali.
- **Eliminazione delle "Collane Bianche" (`fill_internal_mask_holes`):**
  - I pixel non classificati all'interno della maschera convettiva vengono riempiti se hanno dimensione $\le 16\text{ pixel}$, eliminando vuoti visivi artificiali.
- **Overshooting Tops in Evidenza:**
  - I pixel OT (Classe 6) ricevono una micro-dilatazione di 1 pixel e colore Magenta brillante per massimo contrasto visivo rispetto al rosso della fase matura.

---

## Slide 15: Sintesi dei Punti di Forza e Stato Operativo
### Perché il Modello RDT Puro IR è strategico

1. **Robustezza Scientifica e Replicabilità:**
   - Funziona su qualsiasi archivio storico SEVIRI dal 2004 a oggi, senza richiedere modelli numerici o reti radar sincrone.
2. **Tracciabilità Fisica Completa:**
   - Tutte le variabili derivate ($T_{\text{tropo}}, T_{\text{suolo}}, \text{CT}, \vec{v}_{\text{AMV}}, D$) entrano in una catena decisionale causale e verificabile.
3. **Qualità Dimostrata sui Medicanes Storici:**
   - Validato con successo su eventi ciclonici complessi: **Ianos**, **Qendresa**, **Blas**, **Detlef**, **Medicane_1500**, **Medicane_1358**, **Medicane_1283**, **Apollo**.
4. **Ruolo Chiave nel Progetto Deep Convection:**
   - Costituisce il target deterministico ad alta risoluzione ideale per verificare se le rappresentazioni apprese dal backbone **VideoMAE a 7 canali** abbiano interiorizzato la dinamica convettiva 3D dell'atmosfera.
