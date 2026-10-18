# ATBD: NWC SAF GEO-RDT-CW Puro a 7 Canali SEVIRI (`nwcsaf_rdt_seviri_pure_gen`) vs Modello Completo e Proxy del Repository

**Riferimenti Normativi:**
- EUMETSAT NWC SAF, *Algorithm Theoretical Basis Document for Convection Products (RDT-CW and CI)*, `NWC/CDOP2/GEO/MFT/SCI/ATBD/Convection`, Issue 2, Rev. 1 (v2.1, 2019) & Issue 1, Rev. 1 (v1.1, 2016).
- EUMETSAT NWC SAF, *Product User Manual for Convection Products*, `NWC/CDOP3/GEO/MF-PI/SCI/UM/Convection`, Issue 1, Rev. 0 (v1.0, 2019).

---

## 1. Sintesi Architetturale e Tassonomia dei Modelli Sviluppati

Nella directory [`RDT_original/rdt_nwcsaf/`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf) abbiamo sviluppato l'intero algoritmo operativo **EUMETSAT NWC SAF GEO-RDT-CW v5.0** declinato in tre configurazioni rigorosamente separate per isolare il contributo di ciascuna fonte informativa:

| Configurazione | Fonti Dati in Ingresso | Schema di Discriminazione | Prodotti Ancillari (`GEO-CT`, `GEO-HRW`) | Correzione Parallasse 3D | Coefficienti Calibrati (`OPERA`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. `nwcsaf_rdt_full_era5_cal`** (*RDT Nostro Completo*) | 7 canali SEVIRI + **ERA5 NWP** ($T_{\text{tropo}}$, $LI$, $KI$, $SHW$, $\vec{V}_{700}$, $T_{2m}$) | `CAL` ( Regressione logistica 5ch + ereditarietà) | **Sì** (con vincoli termici da ERA5) | **Sì** (con $T_{\text{surf}}, T_{\text{tropo}}$ ERA5) | **Sì** ($\beta_k$ calibrati su radar OPERA) |
| **2. `nwcsaf_rdt_seviri_only_cal`** (*SEVIRI Calibrato*) | Soli 7 canali termici SEVIRI | `CAL` (Regressione logistica 5ch + surrogati NWP) | **Sì** (derivati da soli 7ch SEVIRI) | **Sì** (da `IR_108` + geometria MSG) | **Sì** ($\beta_k$ calibrati su radar OPERA) |
| **3. `nwcsaf_rdt_seviri_pure_gen`** (*RDT Puro 7ch SEVIRI*) | **Soli 7 canali termici SEVIRI** (Zero NWP, Zero Radar, Zero Fulmini) | **`GEN`** (Regole fisiche/radiative pure ATBD v1.1/v2.1) | **Opzionale / Puro IR** (`GEO-CT` e `GEO-HRW` calcolati dai soli 7 canali IR/WV) | **Sì** (da `IR_108` + lapse rate standard + geometria MSG $9.5^\circ\text{E}$) | **NO** (Zero pesi derivati dal radar) |

---

## 2. Prodotti Ancillari Ricavabili Esclusivamente dai 7 Canali Termici SEVIRI (Senza NWP e Senza Visibile)

Nel pacchetto operativo NWC SAF (`GEO-RDT-CW`), due prodotti ancillari satellitari precedono il rilevamento e il tracciamento:
1. **`GEO-CT` (Cloud Type / Cloud Mask dinamica, ATBD §3.2.1.2.1.1):** serve a scartare i **cirri sottili semitrasparenti** (*thin fractional cirrus*) prima del multi-thresholding, evitando che estesi veli di cirro d'alta quota vengano inclusi nelle gonne delle torri o formino falsi sistemi nuvolosi.
   - **È ricavabile dai soli canali IR/WV SEVIRI?** **Sì, al 100% giorno e notte!** La fisica radiativa che distingue un cirro sottile semitrasparente da un'incudine convettiva spessa (opaca) si basa sulla **differenza di emissività nel ghiaccio tra $10.8\ \mu\text{m}$ e $12.0\ \mu\text{m}$ (*Split-Window* $\text{BTD}_{10.8-12.0} = T_{10.8} - T_{12.0}$)** combinata con la fase termica a $8.7\ \mu\text{m}$ ($\text{BTD}_{8.7-10.8}$) e il contrasto vapore acqueo/finestra ($\text{BTD}_{6.2-10.8}$):
     - Un **cirro sottile semitrasparente** lascia filtrare la radiazione calda della bassa troposfera più a $10.8\ \mu\text{m}$ che a $12.0\ \mu\text{m}$ (dove l'assorbimento del ghiaccio è maggiore), producendo un'elevata differenza split-window positiva ($\text{IR}_{10.8} - \text{IR}_{12.0} > 2.5\text{--}3.5\text{ K}$) e una forte depressione $\text{WV}_{6.2} - \text{IR}_{10.8} < -12\text{ K}$.
     - Una **nube convettiva spessa (cumulonembo / congestus)** si comporta quasi come un corpo nero ($\varepsilon \approx 1$), per cui $|\text{IR}_{10.8} - \text{IR}_{12.0}| \le 1.5\text{ K}$ e $\text{WV}_{6.2} - \text{IR}_{10.8} \ge -4\text{ K}$.
   - In [`compute_nwcsaf_cloud_type`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/ancillary.py#L43-L141) questa classificazione spettrale a 8 classi opera **esclusivamente sui canali termici SEVIRI** (`IR_108`, `IR_120`, `IR_087`, `WV_062`, `WV_073`), stimando internamente la temperatura superficiale dal 92° percentile dei pixel sereni `IR_108` quando ERA5 non è fornito.
   - **E i canali del visibile (`VIS0.6`, `VIS0.8`, `HRV`)?** Oltre a non essere presenti nei tensori a 7 canali del repository, introdurrebbero una **discontinuità giorno/notte** (al crepuscolo l'illuminazione solare scompare mentre la convezione profonda prosegue). Usare i soli 7 canali termici/WV garantisce l'**invarianza diurna/notturna** su tutte le 24 ore.
2. **`GEO-HRW` (High Resolution Winds / Atmospheric Motion Vectors, ATBD §3.2.1.2.2.1):** fornisce il campo di moto di primo tentativo (*first-guess motion field*) per l'avvezione delle celle al Passo 1 e Passo 3 del tracking.
   - **È ricavabile dai soli canali IR/WV SEVIRI?** **Sì!** In [`compute_hrw_amv_guess_field`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/ancillary.py#L144-L227), quando i venti ERA5 $\vec{V}_{700}$ non sono passati (`u700_ms=None, v700_ms=None`), il modulo calcola i vettori di moto atmosferico (AMV) direttamente tramite **block-matching (correlazione incrociata normalizzata di Pearson con raffinamento sub-pixel parabolico)** tra i frame consecutivi a $5\text{--}15\text{ min}$ sui canali `IR_108` (top nuvolosi) e `WV_062` (flusso in media-alta troposfera).

---

## 3. Descrizione Rigorosa (Stile ATBD) dell'Algoritmo `nwcsaf_rdt_seviri_pure_gen` e Confronto coi Proxy del Repository

Perché il nostro modello puro [`nwcsaf_rdt_seviri_pure_gen`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/pipeline.py) ottiene un accordo di **`Dice = 0.8958` (`Recall = 91.32%`, `Precision = 87.90%`, `Overall Accuracy = 97.74%`)** rispetto al modello completo multi-sorgente ERA5+OPERA, mentre i tre proxy preesistenti nel repository (`multispectral_utls_v0`, `refined_v2`, `legacy_fixed_2ch`) si fermano a `Dice = 0.2138`, `0.0021` e `0.0826`?

La differenza risiede in **sei pilastri fisico-algoritmici** dell'ATBD ufficiale NWC SAF che mancavano nei proxy precedenti:

### Pilastro 1: Rilevamento Adattivo Multi-Soglia a Passo di $1\text{ K}$ e Isolamento delle Torri Termiche (ATBD §3.2.1.2.1)
- **Proxy del repository (`multispectral_utls_v0`, `refined_v2`, `legacy_fixed_2ch`):** usavano soltanto **3 o 4 soglie fisse** (es. $240, 230, 220, 210\text{ K}$ o profondità relative fisse $D \ge 0, 10, 20, 30\text{ K}$). Qualsiasi cumulo congestus in fase di *triggering* con $T_{\min} = 252\text{ K}$ (circa $-21^\circ\text{C}$, dove avviene la glaciazione iniziale!) era **totalmente invisibile** ai proxy del repository! Inoltre, nei sistemi maturi estesi (MCS), una soglia fissa fondeva più celle adiacenti in un unico blocco indistinto.
- **Nostro RDT (`detection.py`):** costruisce la gerarchia completa [`ThresholdSliceHierarchy`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/detection.py#L28-L118) da $T_{\text{warm}} = +5^\circ\text{C}$ ($278.15\text{ K}$) fino a $T_{\text{cold}} = -75^\circ\text{C}$ ($198.15\text{ K}$) con passo $\Delta T = 1\text{ K}$ (80 livelli).
  - Ogni **torre termica** è definita da un minimo locale con prominenza verticale $\Delta T_{\text{tower}} = T_{\text{BT}} - T_{\min} \ge 6\text{ K}$ e superficie $S \ge S_{\min} = 60\text{ km}^2$.
  - Quando due torri sorelle emergono dentro lo stesso sistema nuvoloso, l'algoritmo scende lungo l'albero termico e le **separa alla soglia più calda $T_{\text{BT}} = T_{\text{merge}} - 1\text{ K}$** che le isola in componenti connesse disgiunte (ATBD Fig. 10).
  - Per le torri mature ($T_{\min} \le -40^\circ\text{C}$), la discesa si arresta al pavimento della tropopausa ($T_{\text{tropo}}$ stimato radiativamente dai canali `WV_062`/`IR_108` o `BTLIMIT` $= -60^\circ\text{C}$), preservando l'intera incudine fredda del cumulonembo senza frammentarla in decine di piccole torrette interne.

### Pilastro 2: Raffinamento del Contorno di Base (`Case A` e `Case B`, ATBD §3.2.1.2.1.2.2)
- **Proxy del repository:** non avevano alcun raffinamento morfologico del profilo verticale $S(T)$: una torre isolata che spuntava da uno strato di nubi basse ereditava l'intero strato caldo circostante.
- **Nostro RDT (`detection.py`):** definisce il **Top of Tower (`ST`)** a $T_{\text{ST}} \approx T_{\min} + 6\text{ K}$ e raffina la **Base of Tower (`BT`)** tramite le due regole ufficiali ATBD (Fig. 11–12):
  - **Caso A (*Budding from flat cloud deck*):** cerca la massima strozzatura verticale nel profilo delle aree consecutive $r_n = S(T_{n+1}) / S(T_n) \le 0.70$ e taglia via il banco nuvoloso caldo sottostante.
  - **Caso B (*Flat pyramidal system*):** impone che il rapporto tra l'area di base e l'area del top rispetti $S_{\text{BT}} \le 4 \cdot S_{\text{ST}}$, stringendo il contorno fino al piede reale della torre convettiva.

### Pilastro 3: Tracking Lagrangiano a 4 Passi e Confronto a Soglia Comune $S(T_{\text{common}})$ (ATBD §3.2.1.2.2)
- **Proxy del repository:** usavano una semplice sovrapposizione geometrica istantanea o di baricentro senza avvezione incrociata e confrontavano direttamente l'area totale $A(t)$ con $A(t-15\text{ min})$. Ma se una cella cambia soglia di base $T_{\text{BT}}$ tra $t-15\text{ min}$ e $t$ (es. dopo uno split o un raffinamento di base), confrontare le aree a soglie diverse produce falsi tassi di espansione enormi!
- **Nostro RDT (`tracking.py`):** implementa i **4 passi sequenziali NWC SAF** (ATBD Fig. 13–19):
  1. **Passo 1:** avvezione in avanti della cella precedente $C$ di $+\vec{V}(C)\Delta t$ e verifica di overlap $\text{AOR} \ge 15\%$.
  2. **Passo 2:** per le celle non agganciate, stima del vettore di spostamento tramite **correlazione incrociata locale `IR_108`** (accettata se $\rho_{\max} \ge 0.50$), o fallback al vicino tracciato / campo `GEO-HRW`.
  3. **Passo 3:** avvezione all'indietro della nuova cella $C'$ di $-\vec{V}_{\text{est}}(C')\Delta t$ e verifica di overlap.
  4. **Passo 4:** dilatazione morfologica di $1\text{ pixel}$ ($3\times 3$) per le piccole celle nascenti ($\le 5\text{ pixel}$) per non perdere la fase di *triggering*.
  - **Regola della Soglia Comune (`common_threshold_area`):** sia la scelta del padre principale (*Main Father*) in caso di split/merge, sia il tasso di espansione areale $\frac{1}{S}\frac{\Delta S}{\Delta t}$ vengono calcolati **esclusivamente alla soglia termica comune $T_{\text{common}} = \min(T_{\text{BT}}(t), T_{\text{BT}}(t-\Delta t))$** interrogando il profilo verticale `Line H`.

### Pilastro 4: Stratificazione Verticale in 6 Categorie Termiche (ATBD §3.2.1.2.3.1)
- **Proxy del repository:** applicavano un'unica regola uniforme a tutte le nubi, indipendentemente dalla quota della nube.
- **Nostro RDT (`discrimination.py`):** classifica ogni cella in **6 regimi verticali** in funzione di $(T_{\min}, T_{\text{BT}}, \text{età})$:
  - `Cat 0 (Mature)`: $T_{\min} \le -40^\circ\text{C}$ e durata $\ge 30\text{ min}$
  - `Cat 1 (Mature Transition)`: $T_{\min} \le -40^\circ\text{C}$ giovane o $T_{\text{BT}} \le -35^\circ\text{C}$
  - `Cat 2 (Cold Transition)`: $-40^\circ\text{C} < T_{\min} \le -35^\circ\text{C}$
  - `Cat 3 (Warm 1 Transition)`: $-35^\circ\text{C} < T_{\min} \le -25^\circ\text{C}$ con base fredda
  - `Cat 4 (Warm 2 Transition)`: $-25^\circ\text{C} < T_{\min} \le -15^\circ\text{C}$ (fase di glaciazione iniziale)
  - `Cat 5 (Warm)`: $T_{\min} > -15^\circ\text{C}$ (convezione calda nascente, dove contano il tasso di raffreddamento esplosivo $\Delta T_{\min}/\Delta t \le -8\text{ K h}^{-1}$ e i gradienti periferici).

### Pilastro 5: Discriminazione Multispettrale a 5 Canali, Ereditarietà Convettiva e Declassificazione Empirica (ATBD §3.2.1.2.3)
- **Proxy del repository:** decidevano la classe di ogni cella istante per istante; non appena una cella convettiva smetteva di raffreddarsi per un singolo frame da 5 minuti, veniva declassata a nube non convettiva o spariva.
- **Nostro RDT (`discrimination.py`):**
  - Combina i predittori spaziali (`T_min`, `Contrast`, `TowerDepth`, `GradP_p95`, `Volume`), multispettrali (`BTD WV6.2-IR10.8` max/p90/SIG, `WBTD WV6.2-WV7.3`, `BTD IR8.7-IR10.8`, `BTD IR12.0-IR10.8`) e temporali (`CoolingRate`, `ExpansionRate` a soglia comune, `dBTD/dt`).
  - Nel modello puro **`GEN`**, usa le **soglie fisiche tabulate per ciascuna delle 6 categorie verticali e per la profondità storica della traiettoria** (senza alcun coefficiente statistico addestrato sul radar).
  - **Ereditarietà Convettiva (*Convective Inheritance*):** una volta che una traiettoria è diagnosticata come convettiva, **eredita lo stato convettivo** lungo la sua vita finché non viene esplicitamente declassificata dalle **regole empiriche di dissipazione ATBD** (riscaldamento prolungato della sommità su $30\text{--}45\text{ min}$, perdita di spessore ottico $\text{WV}_{6.2}-\text{IR}_{10.8} < -8\text{ K}$, o rilassamento del gradiente termico).

### Pilastro 6: Rilevamento Overshooting Top a Due Stadi (`OTD`, ATBD §3.2.1.2.5)
- Implementa in [`otd.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/otd.py) la pre-detection dei minimi freddi interni alla cella ($T_{\text{OT}} \le T_{\text{tropo}}$) e la conferma radiativo-morfologica tramite prominenza rispetto all'incudine circostante ($\ge 3.5\text{ K}$), inversione stratosferica di vapore acqueo ($\text{WV}_{6.2} - \text{IR}_{10.8} > -0.5\text{ K}$) e compattezza spaziale.
