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

- **I 7 Canali Termici SEVIRI (Kelvin):**
  - Vapore acqueo: `WV_062` (alta troposfera), `WV_073` (media troposfera).
  - Finestra infrarossa e split-window: `IR_087`, `IR_097` (ozono), `IR_108` (canale topologico guida), `IR_120`, `IR_134` ($CO_2$).
- **Dominio Spaziale Mediterraneo:**
  - Griglia geostazionaria ritagliata: $763 \times 1762$ pixel (o $764 \times 1762$ su MSG-2).
  - Risoluzione nominale: $3\text{ km}$ al sub-satellite, $4\text{–}5\text{ km}$ sull'area mediterranea.
  - Orientamento geodetico rigoroso: *North-Up, West-Left*.
- **Risoluzione Temporale:**
  - Cadenza rapida RSS (*Rapid Scanning Service*): **1 frame ogni 5 minuti** (288 slot/giorno).
  - Vincolo di causalità: tempo strettamente crescente, nessun riempimento sintetico.

---

## Slide 3: Architettura Algoritmica ad Alto Livello
### La catena causale delle 7 fasi operative

```text
[Input SEVIRI 7 Canali (t)]
       │
       ▼
 [Fase 0] Stima Radiativa di Tropopausa e Suolo (IR Percentiles)
       │
       ▼
 [Fase 1] Prodotti Ancillari Interni: Cloud Type (CT) & Vento AMV (HRW)
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
 [Fase 5] Discriminazione Convettiva GEN & Ciclo di Vita a 4 Fasi
       │
       ▼
 [Fase 6] Correzione Geometrica della Parallasse 3D (MSG -> Suolo)
       │
       ▼
 [Fase 7] Rendering Temporale Fluido (EMA) & Maschere Semantiche
```

---

## Slide 4: Fase 0 — Riferimenti Termici Radiativi
### Determinazione autonoma della tropopausa senza sondaggi atmosferici

- **Problema fisico:** Riconoscere le torri profonde richiede di sapere dove si trova la tropopausa, che varia con latitudine e stagione.
- **Soluzione Puro-IR (Split-Window Causale):**
  - Isolamento dei pixel freddi della scena: $\text{IR}_{10.8} < 235\text{ K}$.
  - Calcolo della temperatura di tropopausa locale pesando vapore acqueo e finestra IR:
    $$T_{\text{tropo,IR}} = 0.6 \cdot P_{10}(\text{WV}_{6.2}) + 0.4 \cdot P_{5}(\text{IR}_{10.8})$$
  - Bounding fisico di sicurezza: $T_{\text{tropo,IR}} \in [205\text{ K}, 226\text{ K}]$.
- **Temperatura di Superficie Radiativa:**
  - Calcolata dai pixel caldi e sereni ($P_{95}(\text{IR}_{10.8})$) o fissata a valore termodinamico di riferimento ($292\text{ K}$) per il calcolo dei profili verticali.

---

## Slide 5: Fase 1 — Ancillari Interni Pure-IR
### Cloud Type opaco e stima del vento atmosferico di guida

- **GEO-CT Interno (Classificazione Nubi):**
  - Riconosce le nubi spesse e opache separandole da suolo e nubi basse tramite differenze di brillanza:
    $$\Delta T_{\text{split}} = \text{IR}_{10.8} - \text{IR}_{12.0}, \quad \Delta T_{\text{tri}} = \text{IR}_{08.7} - \text{IR}_{10.8}$$
  - Produce la maschera di nubi opache `ct_cloudy_mask` che pre-filtra il dominio di detection.
- **GEO-HRW Interno (Vento di Guida AMV):**
  - *Atmospheric Motion Vectors* ricavati da correlazione spaziale locale (*Block-Matching*) tra slot consecutivi ($t - 5\text{ min}$ e $t$).
  - Traccia il pattern di spostamento sui canali `IR_108` e `WV_062`.
  - Fornisce il campo di velocità vettoriale $(\Delta y, \Delta x)$ necessario per l'avvezione lagrangiana al passo successivo.

---

## Slide 6: Fase 2A — Rilevamento Multi-Soglia (Tree Slicing)
### Esplorazione topologica della nube a passi termici discreti

- **Approccio Topografico Inverso:**
  - La temperatura di brillanza $\text{IR}_{10.8}$ viene considerata come una quota (più freddo = più alto).
  - Vengono eseguite sezioni orizzontali a passi di **$1\text{ K}$**, partendo dal livello caldo $T_{\text{warm}} = 278.15\text{ K}$ ($+5^\circ\text{C}$) fino al livello freddo $T_{\text{cold}} = 198.15\text{ K}$ ($-75^\circ\text{C}$).
- **Struttura Dati ad Albero (`ThresholdSliceHierarchy`):**
  - Per ciascun livello termico $k$, si estraggono le componenti connesse spaziali.
  - Le componenti a livelli successivi vengono collegate ai nodi genitori al livello precedente.
- **Filtri di Superficie Minima:**
  - Ogni nucleo isolato deve soddisfare $S \ge 60\text{ km}^2$ (circa 4–5 pixel SEVIRI) ed essere limitato superiormente a $S_{\max} = 400.000\text{ km}^2$.

---

## Slide 7: Fase 2B — Isolamento delle Torri e Taglio dell'Incudine
### Separazione della vera attività convettiva dai cirri stratiformi

- **Criterio di Torre Convettiva (Prominenza Termica):**
  - Una torre è valida se il suo nucleo freddo $T_{\min}$ è più freddo della base di almeno **$6\text{ K}$**:
    $$\Delta T_{\text{tower}} = T_{\text{base}} - T_{\min} \ge 6\text{ K}$$
- **Separazione dei Massimi Multipli (Biforcazione):**
  - Due torri termiche adiacenti vengono separate alla soglia $T_{\text{merge}} - 1\text{ K}$ in cui i rispettivi rami dell'albero si fondono.
- **Raffinamento Morfologico della Base (Anti-Anvil):**
  - *Caso A (Torre isolata con base sproporzionata):* Se l'area della base supera 4 volte l'area a $T_{\min} + 6\text{ K}$, la base viene riscalata verso l'alto per escludere l'incudine sottile.
  - *Caso B (Torri nidificate su base comune):* Si applica una ripartizione guidata dalla quota dei rispettivi minimi.

---

## Slide 8: Fase 3A — Tracciamento Lagrangiano Multi-Passo
### Connessione temporale robusta contro deformazioni e occlusioni

- **Avvezione Cinematica in Avanti:**
  - Ogni cella al tempo $t - 1$ viene traslata nello spazio secondo il proprio vettore di moto locale stimato dall'AMV:
    $$\vec{x}_{\text{pred}} = \vec{x}_{t-1} + \vec{v}_{\text{cell}} \cdot \Delta t$$
- **Criterio di Sovrapposizione Spaziale (AOR):**
  - Calcolo dell'*Area Overlap Ratio* tra la cella predetta $\mathcal{A}_{\text{pred}}$ e la cella osservata $\mathcal{B}_{\text{curr}}$:
    $$\text{AOR} = \frac{|\mathcal{A}_{\text{pred}} \cap \mathcal{B}_{\text{curr}}|}{\min(|\mathcal{A}_{\text{pred}}|, |\mathcal{B}_{\text{curr}}|)} \ge 15\%$$
- **Concatenazione a 4 Passi:**
  1. *Forward Advection* standard.
  2. *Local Block Matching* attorno al centroide se l'AOR iniziale è ambiguo.
  3. *Backward Advection* di verifica per confermare il legame.
  4. *Enlargement / Dilatazione* per piccole celle in rapido innesco ($\le 5\text{ pixel}$).

---

## Slide 9: Fase 3B — Gestione di Split, Merge e Traiettoria
### Continuità della storia convettiva nelle fusioni e frammentazioni

- **Identificazione Topologica dei Casi di Tracking:**
  - *Continuazione 1-a-1:* La cella continua la sua traiettoria con identico ID.
  - *Split (1-a-Molti):* Una cella madre si suddivide in due o più frammenti figli.
  - *Merge (Molti-a-1):* Più celle distinte convergono in un unico sistema.
- **Risoluzione sulla Soglia Termica Comune ($T_{\text{comm}}$):**
  - In caso di split o merge, il "ramo principale" (*primary parent*) è assegnato al frammento che massimizza la somiglianza superficiale alla **soglia di temperatura condivisa** tra i nodi dell'albero.
- **Smussamento Esponenziale della Velocità (EMA):**
  - La velocità di spostamento viene filtrata causalmente per evitare strappi:
    $$\vec{v}_t = (1 - \alpha) \cdot \vec{v}_{t-1} + \alpha \cdot \vec{v}_{\text{inst}}, \quad \text{con } \alpha = 0.5$$

---

## Slide 10: Fase 4 — Rilevamento Overshooting Tops (OTD)
### Identificazione delle cupole convettive penetranti in stratosfera

- **Cos'è un Overshooting Top (OT):**
  - Una cupola convettiva violenta in cui la corrente ascensionale (*updraft*) supera il livello di equilibrio neutro e penetra nella stratosfera.
- **Algoritmo di Rilevamento a Due Stadi (100% SEVIRI):**
  1. **Pre-selezione Termica:**
     - Il pixel deve essere vicino o più freddo della tropopausa stimata:
       $$T_{\text{pixel}} \le T_{\text{tropo,IR}} + 5\text{ K}$$
  2. **Conferma a Doppio Criterio Fisico:**
     - *Prominenza Termica Locale:* Il pixel è più freddo dell'anello circostante dell'incudine di almeno **$3.5\text{ K}$**.
     - *Inversione Strato-Troposferica del Vapore:* A causa dell'inversione termica stratosferica, la sommità della cupola mostra:
       $$\text{BTD}(\text{WV}_{6.2} - \text{IR}_{10.8}) \ge -0.5\text{ K}$$
- **Assegnazione di Classe:** I pixel confermati vengono etichettati come **Classe 6 (Overshooting Top)**.

---

## Slide 11: Fase 5A — Discriminazione Convettiva (`GEN`)
### Separazione tra nubi non convettive e convezione attiva

- **Schema `GEN` (Generic NWC SAF):**
  - Non richiede modelli di machine learning calibrati su radar locale: si basa su leggi termodinamiche e cinematiche universali.
- **Calcolo dei Trend Temporali Multi-Finestra:**
  - Vengono calcolati tassi di raffreddamento e di espansione su **15, 30, 45 e 60 minuti**:
    $$\frac{\Delta T_{\min}}{\Delta t}\text{ [K/h]}, \quad \frac{\Delta S}{\Delta t}\text{ [km}^2\text{/h]}$$
- **Insieme dei Predittori Chiave:**
  - Temperatura minima assoluta $T_{\min}$ e contrasto rispetto all'ambiente.
  - Gradiente perimetrale $\text{GradP}$ (elevato nelle celle convesse e vigorose).
  - Segnatura di vapore e split-window: $\text{WV}_{6.2} - \text{WV}_{7.3}$ e $\text{IR}_{10.8} - \text{IR}_{12.0}$.
- **Esito Binario:** La cella è classificata come *Convettiva* o *Nube fredda non convettiva* (Classe 1, cirro/incudine passiva).

---

## Slide 12: Fase 5B — Ciclo di Vita Convettivo a 4 Fasi
### Classificazione dello stadio evolutivo e preservazione della memoria

Se la cella è convettiva, l'algoritmo determina il suo stadio nel **ciclo di vita a 4 fasi canoniche**:

1. **Triggering (Classe 2 — Giallo Brillante):**
   - Cella giovane in rapido raffreddamento iniziale ($\Delta T/\Delta t \ll 0$), piccola area in espansione, transizione dalla fase calda.
2. **Growing (Classe 3 — Arancione Vivo):**
   - Fase di sviluppo vigoroso: forte espansione dell'area, continua ascesa della sommità, aumento del gradiente perimetrale.
3. **Mature (Classe 4 — Rosso Vivo):**
   - Massima intensità: $T_{\min}$ stabilizzata ai valori minimi prossimi alla tropopausa, estensione massima della base, presenza frequente di OT.
4. **Decaying (Classe 5 — Rosso Scuro / Bordeaux):**
   - Fase di dissipazione: riscaldamento progressivo della sommità ($\Delta T/\Delta t > 0$), progressivo collasso dell'updraft.

- **Ereditarietà dello Stato Madre:** Nei frammenti derivanti da *Split*, la cella figlia non riparte artificiosamente da *Triggering*, ma eredita lo stato evolutivo della cella madre.

---

## Slide 13: Fase 6 — Correzione Geometrica della Parallasse 3D
### Riallineamento geometrico dal punto di vista satellitare alla superficie terrestre

- **Il Problema Geometrico:**
  - Il satellite geostazionario (Meteosat a $0^\circ$ o $9.5^\circ\text{E}$) osserva il Mediterraneo con un angolo di zenith satellite elevato ($\theta_{\text{sat}} \approx 40^\circ\text{–}50^\circ$).
  - Una torre convettiva alta $12\text{–}14\text{ km}$ appare proiettata diverse decine di chilometri a Nord/Nord-Est rispetto alla sua reale posizione a terra.
- **Modello di De-Parallasse 3D:**
  - Calcolo della quota geometrica del top $H_{\text{cloud}}$ via lapse-rate adiabatico da $T_{\text{surf,IR}}$ e $T_{\text{tropo,IR}}$.
  - Riproiezione cartesiana dei pixel verso il sub-satellite point:
    $$\Delta x = H_{\text{cloud}} \cdot \tan(\theta_{\text{azimuth}}), \quad \Delta y = H_{\text{cloud}} \cdot \tan(\theta_{\text{elevation}})$$
- **Consistenza Spaziale Totale:**
  - La correzione di parallasse viene applicata **sia alla maschera semantica delle celle sia allo sfondo termico IR 10.8**, garantendo sovrapposizione perfetta al sub-pixel.

---

## Slide 14: Fase 7 — Rendering Fluido e Post-Processing
### Eliminazione degli artefatti e composizione video ad alta leggibilità

- **Smoothing Temporale dei Colori (Filtro EMA):**
  - Per evitare scatti cromatici quando una cella oscilla sulla soglia tra due classi contigue, i colori RGB vengono filtrati con media mobile esponenziale su ogni pixel:
    $$C_t = (1 - \gamma) \cdot C_{t-1} + \gamma \cdot C_{\text{target}}, \quad \text{con } \gamma = 0.38$$
  - Risultato: transizioni cromatiche armoniose e naturali.
- **Eliminazione delle "Collane Bianche" (`fill_internal_mask_holes`):**
  - La riproiezione discreta in avanti (splatting di coordinate) può generare pixel isolati non mappati all'interno della maschera convettiva.
  - Un filtro morfologico riempie tutti i buchi interni con dimensione $\le 16\text{ pixel}$, garantendo solidità visiva alla nube.
- **Overshooting Tops in Evidenza:**
  - I pixel OT (Classe 6) ricevono una micro-dilatazione di 1 pixel e colore Magenta brillante per massimo contrasto visivo rispetto al rosso della fase matura.

---

## Slide 15: Sintesi dei Punti di Forza e Stato Operativo
### Perché il Modello RDT Puro IR è strategico

1. **Robustezza Scientifica e Replicabilità:**
   - Funziona su qualsiasi archivio storico SEVIRI dal 2004 a oggi, senza richiedere il reperimento di modelli numerici o reti radar sincrone.
2. **Qualità Dimostrata sui Medicanes Storici:**
   - Validato con successo su eventi ciclonici complessi: **Ianos** (2020), **Qendresa** (2014), **Blas** (2021), **Detlef** (2019), **Medicane_1500** (2015).
3. **Efficienza Computazionale:**
   - Supporta il salvataggio in cache compressa degli array semantici (`.npz`), permettendo re-rendering immediati a oltre 25 fps.
4. **Ruolo Chiave nel Progetto Deep Convection:**
   - Costituisce il riferimento (*ground truth target*) ideale e deterministico per testare quanto le rappresentazioni self-supervised apprese dal backbone **VideoMAE a 7 canali** abbiano interiorizzato la termodinamica atmosferica 3D.
