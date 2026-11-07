# Piano Riassuntivo da Presentazione: Pretraining VideoMAE e Probing Gerarchico a 3 Teste per la Convezione Profonda

Struttura narrativa e tecnica suddivisa in **12 slide** per presentare l'architettura, la fisica spettrale SEVIRI, i teacher deterministici Pure-IR (`CMA`, `CT Piano A`, `RDT`) e il protocollo sperimentale di probing parallelo e sequenziale.

---

## Slide 1 — Titolo e Visione d'Insieme
**Titolo**: *Dall'Infrarosso Termico Geostazionario alla Convezione Severa: Pretraining Self-Supervised VideoMAE e Probing Gerarchico a Tre Teste su SEVIRI*
- **Sottotitolo**: Apprendimento delle dinamiche atmosferiche su 7 canali IR a cadenza rapida (5 min) e decodifica multi-livello (*Cloud Mask $\rightarrow$ Cloud Type $\rightarrow$ RDT Convective Lifecycle*).
- **Punti chiave**:
  - Sfruttare l'enorme archivio non etichettato MSG SEVIRI RSS ($5\text{ min}$, $24/24\text{ h}$ giorno e notte) tramite **VideoMAE-Small** a 7 canali termici.
  - Separazione rigorosa tra **Pretraining Self-Supervised** (ricostruzione di tubelet mascherati al $90\%$) e **Probing Downstream** con teacher fisici deterministici.
  - Superamento della dipendenza dai canali solari/visibili e dai modelli numerici esterni (NWP) grazie a tre processori **Pure-IR autonomi** ispirati a EUMETSAT NWC SAF.

---

## Slide 2 — Il Dominio Osservativo: I 7 Canali Infrarossi SEVIRI
**Titolo**: *Perché 7 Canali Termici? La Tomografia Verticale dell'Atmosfera*
- **Invarianti Fisici del Dato**:
  - Clip spaziotemporali $(T, C, H, W) = (16, 7, 224, 224)$ a cadenza $5\text{ min}$ ($75\text{ min}$ di evoluzione continua), espresse in **Brightness Temperature [K]** con una sola media e una sola deviazione standard globali.
- **Ruolo Fisico di Ciascun Canale**:
  1. `WV_062` ($6.2\text{ µm}$) & `WV_073` ($7.3\text{ µm}$): Vapore acqueo in alta e media troposfera (divergenza in quota, onde di gravità, iniezione stratosferica).
  2. `IR_087` ($8.7\text{ µm}$): Fase termodinamica (transizione acqua liquida $\rightarrow$ ghiaccio all'innesco dell'updraft).
  3. `IR_097` ($9.7\text{ µm}$): Assorbimento dell'ozono (interazione della cupola temporalesca con la bassa stratosfera).
  4. `IR_108` ($10.8\text{ µm}$) & `IR_120` ($12.0\text{ µm}$): Finestra pulita e *Split-Window* (temperatura della sommità nuvolosa, filtraggio dei cirri sottili).
  5. `IR_134` ($13.4\text{ µm}$): Banda della $\text{CO}_2$ (*CO2-slicing* per la quota barica effettiva della nube).

---

## Slide 3 — Pretraining Self-Supervised: VideoMAE-Small Atmosferico
**Titolo**: *Apprendere la Fisica e il Movimento Senza Etichette*
- **Architettura**:
  - Backbone **VideoMAE-Small** (12 blocchi Transformer, dimensione $384$) con patch embedding `Conv3d(7, 384, kernel=(2, 16, 16))`.
  - **Tube Masking al 90%**: solo il $10\%$ dei tubelet spaziotemporali è visibile; il modello deve ricostruire i $3584$ valori in Kelvin per tubelet mascherato (`norm_pix_loss=False` per preservare le ampiezze assolute in Kelvin).
- **Cosa impara l'encoder per costrizione fisica**:
  - Per ricostruire il $90\%$ mancante su 75 minuti, l'attenzione 3D deve apprendere l'**avvezione del vento**, i **tassi di raffreddamento adiabatico** ($\partial T/\partial t$) e le **correlazioni spettrali inter-canale** (BTD).

---

## Slide 4 — Le Cinque Firme Fisiche della Convezione Profonda nell'Infrarosso
**Titolo**: *Come Riconoscere una Cella Convettiva nei 7 Canali SEVIRI*
1. **Raffreddamento Esplosivo Causale ($\Delta T_{10.8} / 15\text{ min}$)**:
   - L'ascesa violenta dell'updraft raffredda la sommità di $-4\text{ K}\dots-15\text{ K}$ in 15 minuti (calcolato su 3 intervalli RSS reali).
2. **Firma Spettrometrica dell'Overshooting Top ($WV_{062} - IR_{108} \ge 0\text{ K}$)**:
   - Quando la torre penetra nella stratosfera (inversione termica), il vapore acqueo emette a temperatura uguale o superiore alla finestra $10.8\text{ µm}$.
3. **Glaciazione Rapida e Spessore Ottico ($IR_{087} - IR_{108}$ e $IR_{108} - IR_{120}$)**:
   - Il congelamento rapido innalza la differenza $8.7 - 10.8\text{ µm}$, mentre lo *split-window* $10.8 - 12.0\text{ µm} \to 0\text{ K}$ attesta un nucleo otticamente saturo ($\tau \gg 10$).
4. **Quota Barica tramite $\text{CO}_2$ e $\text{O}_3$ ($IR_{134} - IR_{108}$, $IR_{097} - IR_{108}$)**:
   - Separa le torri che raggiungono la tropopausa dalle nubi medie e dai veli semitrasparenti.
5. **Onde di Gravità Atmosferiche (*Buoyancy Waves*)**:
   - L'urto dell'updraft contro la tropopausa stabile genera increspature concentriche nel campo `WV_062` catturate dai token spaziotemporali di VideoMAE.

---

## Slide 5 — Dai Prodotti NWC SAF ai Nostri 3 Teacher Pure-IR Indipendenti
**Titolo**: *Tre Teacher Deterministici Pure-IR: CMA, CT e RDT*
- **Risoluzione Geostazionaria Esatta della Linea di Costa**:
  - Il satellite MSG è geostazionario: i nostri processori `CMA` e `CT` integrano la maschera statica terra/mare/costa ($1:50\text{m}$) e calibrano il profilo termico superficiale $T_{\text{surf}}$ separatamente su **Terra emersa**, **Oceano Atlantico/Mare del Nord** e **Bacino Mediterraneo/Mar Nero**.
- **Indipendenza Algoritmica tra `CT` e `RDT`**:
  - **`CMA_original` & `CT_original`**: Classificatori **pixel-per-pixel** basati su soglie termiche rispetto a $T_{\text{surf}}, T_{850}, T_{700}, T_{500}, T_{\text{tropo}}$, contrasti spettrali BTD e test di tessitura locale $3\times 3$.
  - **`RDT_original` (`PureGenUtlsTracker`)**: Algoritmo **a oggetti e tracciamento Lagrangiano** basato sulla proxy causale del plateau $T_{\text{UTLS}}(t)$ (pesata su 6 BTD), slicing morfologico a $1\text{ K}$ su `IR_108` e trend temporali a 15 minuti. **Non condivide codice né soglie con `CMA`/`CT`**.

---

## Slide 6 — Validazione con i Canali Visibili (`VIS006`, `VIS008`, `IR_016`) e Riferimenti NWC SAF
**Titolo**: *Confronto Visivo Diurno (Visibile vs Infrarosso) e Verifica Operativa*
- **Confronto con i Prodotti Ufficiali NWC SAF (`v2025_msg`)**:
  - L'integrazione della linea di costa geostazionaria e della calibrazione per bacino ha ridotto il *False Alarm Rate* su terra serena al **`3.03%`** (accuratezza binaria fino all'**`83.7%`** usando i soli canali IR).
- **Verifica Diretta contro i Canali Solari Visibili (`VIS006` $0.6\text{ µm}$, `VIS008` $0.8\text{ µm}$, RGB Naturale)**:
  - Nelle ore diurne, il confronto diretto con la riflettanza visibile mostra a occhio nudo la corrispondenza tra i corpi nuvolosi riflettenti e la maschera `CMA Pure-IR`, evidenziando anche dove l'infrarosso rileva veli di cirro d'alta quota quasi trasparenti nel visibile.

---

## Slide 7 — Perché Tre Sonde Insieme? Il Principio di Nidificazione Fisica
**Titolo**: *Gerarchia Termodinamica e Nidificazione Spaziale (Spatial Nesting)*
- **Definizione di Nidificazione Fisica**:
  $$\Omega_{\text{OT}} \subset \Omega_{\text{RDT-Conv}} \subset \Omega_{\text{CT-HighOpaque}} \subset \Omega_{\text{CMA-Cloud}} \subset \Omega$$
  - Ogni nucleo convettivo attivo (`RDT Classi 2, 3, 4`) è fisicamente incapsulato dentro un sistema nuvoloso alto e otticamente spesso (`CT Piano A Classe 3: High Opaque`, $P_{\text{top}} < 450\text{ hPa}$), che a sua volta è contenuto nella maschera nuvolosa (`CMA == 1`).
- **Risultato della Verifica Multi-Temporale su $\sim 5$ Milioni di Pixel Convettivi (2024–2026)**:
  - **Recall delle celle convettive RDT in `CMA`**: **`100.00%`**
  - **Recall delle celle convettive RDT in `CT High Opaque`**: **`92.45% – 97.80%`** (media **`95.5%`**)
  - **Falsi negativi in Cielo Sereno / Nube Bassa**: **`0.0127%`**

---

## Slide 8 — Specifiche delle 3 Teste di Probing (Adozione Esclusiva del Piano A)
**Titolo**: *Tassonomia Consolidata per il Decoder 3D FPN Multi-Task*
- **Testa 1 — Cloud Mask (`CMA Pure-IR`)**:
  - Segmentazione binaria (`0: Clear Sky`, `1: Cloud`). Compito: sopprimere il fondo radiometrico terrestre e marino.
- **Testa 2 — Cloud Type Piano A (`5 Macro-Regimi Fisici`)** *(Piano B escluso per evitare ridondanza)*:
  - `0: Clear Sky` | `1: Low Cloud (>650 hPa)` | `2: Mid Cloud (450–650 hPa)` | **`3: High Opaque (<450 hPa)`** | `4: Cirrus & Multilayer`.
  - Compito: isolare l'inviluppo profondo `Classe 3 (High Opaque)` separandolo da nubi basse e cirri sottili.
- **Testa 3 — Dinamica Convettiva (`RDT Pure-IR` a 5 Classi)**:
  - `0: Clear/Non-Cold` | `1: Cold Anvil (Non-Conv)` | `2: Triggering/Growing (Giallo)` | `3: Mature/Decaying (Arancione)` | `4: Severe / Overshooting Top (Rosso/Magenta)`.
  - Compito: discriminare gli updraft convettivi attivi dalle incudini stratiformi passive.

---

## Slide 9 — Architettura del Decoder 3D FPN e Condizionamento Gerarchico
**Titolo**: *Architettura Multi-Head con Condizionamento in Cascata e Loss di Nidificazione*
- **Shared Lightweight 3D FPN**:
  - Estrae i token spaziotemporali dai blocchi 3, 6, 9, 12 di VideoMAE e ricostruisce la risoluzione piena $(16, 128, 224, 224)$.
- **Condizionamento Gerarchico dei Logit (*Cascaded Conditioning*)**:
  - `Testa 1 (CMA)` $\rightarrow$ fornisce la probabilità di nube a `Testa 2 (CT-5)` $\rightarrow$ entrambe forniscono le probabilità di regime nuvoloso a `Testa 3 (RDT-5)`.
- **Loss Multi-Task con Vincolo Differenziabile di Nidificazione ($\mathcal{L}_{\text{nesting}}$)**:
  - Penalizza quadraticamente ogni pixel in cui la probabilità predetta di convezione supera la probabilità di `High Opaque`, imponendo coerenza fisica alle predizioni della rete.

---

## Slide 10 — Costruzione Rigorosa del Dataset di Probing
**Titolo**: *Pipeline Dati Senza Contaminazione Temporale né Artefatti di Bordo*
1. **Calcolo Full-Domain Causale prima del Ritaglio**:
   - I tre teacher girano sull'intero dominio $(763 \times 1762)$ a cadenza $5\text{ min}$ per preservare la topologia delle isoterme e il tracking delle celle; solo dopo si estraggono le clip $(16, 7, 224, 224)$.
2. **Divisione Temporale con Intervallo Cuscinetto di 24 Ore**:
   - Train, Validation e Test sono separati da una **fascia di guardia temporale inutilizzata di almeno 24 ore**, evitando che lo stesso sistema temporalesco compaia sia alla fine del training sia all'inizio del test.
3. **Campionamento Spaziale Stratificato (50% / 30% / 20%)**:
   - **50% Convezione Attiva / OT** + **30% Hard Negatives** (incudini alte `High Opaque` non convettive e cirri) + **20% Nubi basse/medie e coste serene**.

---

## Slide 11 — Protocollo di Verifica: Probing Parallelo vs Curriculum Sequenziale
**Titolo**: *Validazione a Backbone Congelato e Studio di Trasferimento Sequenziale*
- **Protocollo Canonico (Fase 1 — Backbone Congelato)**:
  - Confronta il probing singolo (`Solo RDT`) con il **probing congiunto parallelo a 3 teste (`CMA + CT-5 + RDT-5`)** sul backbone VideoMAE congelato (più controlli Random, RGB adattato e ablazione temporale).
- **Protocollo Sequenziale (Fasi 2 e 3 — Curriculum Fine-Tuning)**:
  - **Fase 2**: Fine-tuning del backbone su `CMA` $\rightarrow$ congelamento $\rightarrow$ re-probing su `CT-5` e `RDT-5`.
  - **Fase 3**: Fine-tuning successivo su `CT-5` $\rightarrow$ congelamento $\rightarrow$ re-probing finale su `RDT-5`.
- **Salvaguardia contro la Perdita di Memoria Temporale (*Temporal Forgetting*)**:
  - Poiché `CMA` e `CT-5` sono task statici mentre `RDT-5` richiede la dinamica sui 16 frame, il fine-tuning sequenziale sblocca solo gli ultimi 4 blocchi ViT con $\eta_{\text{enc}} = 10^{-5}$ e mantiene attive le teste precedenti in modo cumulativo.

---

## Slide 12 — Conclusioni e Prossimi Passi Operativi
**Titolo**: *Sintesi dei Risultati Raggiunti e Avvio del Probing GPU*
- **Risultati consolidati**:
  1. Processori Pure-IR `CMA`, `CT Piano A` e `RDT` allineati, mappati con linea di costa geostazionaria e validati su 7 periodi multi-annuali (2024–2026) e con canali visibili diurni.
  2. Dimostrata la **nidificazione fisica al `95.5%`** delle celle convettive RDT dentro la classe `High Opaque` del Cloud Type indipendente.
  3. Specifiche architetturali e sperimentali complete formalizzate in [`three_heads_probing.md`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/three_heads_probing.md) e [`RDT_original/PHYSICAL_SEVIRI_CONVECTION_PRINCIPLES.md`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/PHYSICAL_SEVIRI_CONVECTION_PRINCIPLES.md).
- **Step successivi**:
  - Materializzazione delle clip stratificate con le 3 etichette allineate e lancio della matrice sperimentale `E0–E4` su GPU.
