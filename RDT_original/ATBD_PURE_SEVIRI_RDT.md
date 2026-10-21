# Algorithm Theoretical Basis Document (ATBD): Modello RDT Puro a 7 Canali Termici SEVIRI (`nwcsaf_rdt_seviri_pure_gen`)

**Riferimenti Normativi:**
- EUMETSAT NWC SAF, *Algorithm Theoretical Basis Document for Convection Products (RDT-CW and CI)*, `NWC/CDOP2/GEO/MFT/SCI/ATBD/Convection`, Issue 2, Rev. 1 (v2.1, 2019) & Issue 1, Rev. 1 (v1.1, 2016).
- EUMETSAT NWC SAF, *Product User Manual for Convection Products*, `NWC/CDOP3/GEO/MF-PI/SCI/UM/Convection`, Issue 1, Rev. 0 (v1.0, 2019).

**Moduli Eseguibili di Riferimento:**
- [`RDT_original/rdt_nwcsaf/config.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/config.py)
- [`RDT_original/rdt_nwcsaf/pipeline.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/pipeline.py)
- [`RDT_original/rdt_nwcsaf/ancillary.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/ancillary.py)
- [`RDT_original/rdt_nwcsaf/detection.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/detection.py)
- [`RDT_original/rdt_nwcsaf/tracking.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/tracking.py)
- [`RDT_original/rdt_nwcsaf/otd.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/otd.py)
- [`RDT_original/rdt_nwcsaf/discrimination.py`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/discrimination.py)

---

## 1. Sequenza Algoritmica ad Alto Livello (Sintesi Esecutiva)

Il modello **RDT Puro a 7 Canali SEVIRI** riceve in ingresso una sequenza temporale causale di immagini geostazionarie MSG-SEVIRI con 7 canali nel medio e lontano infrarosso espressi in temperatura di brillanza (Kelvin):
$$\mathbf{X}_t \in \mathbb{R}^{7 \times H \times W}, \qquad \text{canali} = (\text{WV}_{6.2},\ \text{WV}_{7.3},\ \text{IR}_{8.7},\ \text{IR}_{9.7},\ \text{IR}_{10.8},\ \text{IR}_{12.0},\ \text{IR}_{13.4})$$
accompagnata dalle griglie geometriche fisse dell'area geodetica dei pixel $A \in \mathbb{R}^{H \times W}$ ($\text{km}^2$) e delle coordinate geografiche $(\phi, \lambda) \in \mathbb{R}^{H \times W}$ (latitudine Nord, longitudine Est). Non viene utilizzato alcun dato esterno (nessun modello meteorologico NWP, nessun radar, nessuna rete di fulminazione, nessun canale visibile).

Per ciascuno slot temporale $t$, il tracker causale [`NwcSafRdtTracker.step_frame`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/pipeline.py#L79-L410) esegue in ordine rigoroso le **7 fasi** seguenti:

1. **Fase 0 — Validazione della Continuità Temporale e Stima Radiativa della Tropopausa (`pipeline.py`):**
   - Converte il timestamp corrente $t$ in minuti decorsi e calcola il passo temporale $\Delta t = t - t_{\text{prev}}$. Se $\Delta t \notin [0.5,\ 3 \cdot \Delta t_{\text{nom}} + 1.0]\text{ min}$ (ossia oltre $16\text{ min}$ a cadenza nominale di $5\text{ min}$), azzera la memoria delle traiettorie precedenti per impedire associazioni spurie attraverso gap osservativi.
   - Stima il tetto termico locale dell'alta troposfera/tropopausa ($T_{\text{tropo,IR}}$) direttamente dai percentili freddi dei canali `WV_062` e `IR_108` sulle sommità nuvolose opache del frame corrente.
2. **Fase 1 — Generazione Interna dei Prodotti Ancillari Puro-IR (`GEO-CT` e `GEO-HRW`, `ancillary.py`):**
   - **Pre-filtraggio Cloud Type (`compute_nwcsaf_cloud_type`):** tramite le differenze di temperatura di brillanza (*Split-Window* $\text{IR}_{10.8}-\text{IR}_{12.0}$, fase del ghiaccio $\text{IR}_{8.7}-\text{IR}_{10.8}$ e spessore ottico $\text{WV}_{6.2}-\text{IR}_{10.8}$), classifica i pixel nuvolosi e costruisce la maschera delle nubi opache (`opaque_cloud_mask`), scartando il cielo sereno e i cirri sottili semitrasparenti (*thin fractional cirrus*).
   - **Campo di Moto di Primo Tentativo (`compute_hrw_amv_guess_field`):** se è disponibile il frame precedente $t-\Delta t$, stima il campo 2D di spostamento atmosferico $(\Delta y_{\text{guess}}, \Delta x_{\text{guess}})$ tramite correlazione incrociata normalizzata a blocchi (*block-matching*) su una combinazione pesata di `IR_108` e `WV_062`.
3. **Fase 2 — Rilevamento Adattivo Multi-Soglia e Raffinamento Morfologico delle Torri (`detect_rdt_cells`, `detection.py`):**
   - Maschera i pixel non validi, fuori dal disco terrestre o eccessivamente stirati ai bordi del disco ($A > 5 \cdot A_{\text{nom}}$), e interseca con la maschera delle nubi opache `GEO-CT`.
   - Costruisce la gerarchia di affettamento termico `ThresholdSliceHierarchy` sul canale `IR_108` da $T_{\text{warm}} = +5^\circ\text{C}$ ($278.15\text{ K}$) a $T_{\text{cold}} = -75^\circ\text{C}$ ($198.15\text{ K}$) con passo $\Delta T = 1\text{ K}$ (80 livelli).
   - Scende ricorsivamente lungo l'albero delle componenti connesse 8-adiacenti per isolare le singole **torri termiche** ($\Delta T_{\text{tower}} = T_{\text{BT}} - T_{\min} \ge 6\text{ K}$, area $S \ge 60\text{ km}^2$), separando le torri sorelle alla soglia più calda $T_{\text{BT}} = T_{\text{merge}} - 1\text{ K}$ che le disgiunge e arrestando la discesa al pavimento della tropopausa per preservare le incudini mature.
   - Raffina il contorno di base (`Base of Tower`, $T_{\text{BT}}$) rispetto al contorno sommitale (`Top of Tower`, $T_{\text{ST}} \approx T_{\min} + 6\text{ K}$) eliminando i banchi nuvolosi bassi (*Case A: strozzatura verticale* $S_{n+1}/S_n \le 0.70$; *Case B: vincolo piramidale* $S_{\text{BT}} \le 4 \cdot S_{\text{ST}}$).
   - Estrae per ciascuna cella gli attributi geometrici, il profilo verticale delle superfici $S(T)$, i baricentri pesati, i gradienti termici interni/periferici e le statistiche multispettrali (quantili e coerenza spaziale `SIG`) sulle 6 combinazioni `BTD`.
4. **Fase 3 — Tracking Lagrangiano a 4 Passi e Gestione delle Traiettorie (`track_rdt_cells`, `tracking.py`):**
   - Collega le celle di $t-\Delta t$ con quelle di $t$ attraverso 4 passi in cascata: (1) avvezione in avanti $+\vec{V}\Delta t$ e test di sovrapposizione $\text{AOR} \ge 15\%$; (2) stima del moto locale tramite correlazione incrociata `IR_108` ($\rho_{\max} \ge 0.50$) per le celle rimaste orfane; (3) avvezione all'indietro $-\vec{V}_{\text{est}}\Delta t$; (4) dilatazione morfologica $3\times 3$ di $1\text{ pixel}$ per le piccole celle nascenti ($\le 5\text{ pixel}$).
   - Risolve le topologie complesse (*Continuation*, *Split*, *Merge*) selezionando il padre principale (*Main Father*) mediante il confronto delle superfici alla **soglia termica comune** $T_{\text{common}} = \min(T_{\text{BT}}(t), T_{\text{BT}}(t-\Delta t))$, e aggiorna la velocità della traiettoria con smussamento esponenziale ($\alpha = 0.5$).
5. **Fase 4 — Rilevamento a Due Stadi degli Overshooting Tops (`detect_cell_overshooting_tops`, `otd.py`):**
   - All'interno di ciascuna cella, individua i minimi termici candidati che perforano il riferimento della tropopausa (Stadio 1) e li conferma tramite prominenza termica rispetto all'incudine circostante ($\ge 3.5\text{ K}$), inversione di vapore acqueo stratosferico ($\text{WV}_{6.2} - \text{IR}_{10.8} > -0.5\text{ K}$) ed estensione spaziale compatta (Stadio 2).
6. **Fase 5 — Stratificazione Verticale, Discriminazione Fisica `GEN`, Ereditarietà e Fase di Sviluppo (`discriminate_and_classify_cell`, `discrimination.py`):**
   - Calcola i trend Lagrangiani ($\Delta T_{\min}/\Delta t$, espansione areale relativa $\frac{1}{S}\frac{\Delta S}{\Delta t}$ a soglia comune $T_{\text{common}}$, $\Delta\text{BTD}/\Delta t$) lungo la storia della traiettoria ($15\text{--}90\text{ min}$).
   - Assegna la cella a una delle **6 categorie verticali** (`Cat 0: Mature` .. `Cat 5: Warm`) in base a $(T_{\min}, T_{\text{BT}}, \text{età})$.
   - Applica le regole di discriminazione fisica/radiativa pura (`GEN`), l'**ereditarietà convettiva** dal padre convettivo e i **criteri empirici di declassificazione** (dissipazione termica e perdita di spessore ottico).
   - Determina la **Fase di Sviluppo** fedele a RDT Tabella 8 (`Triggering`, `Growing`, `Mature`, `Decaying`), con ereditarietà dello stato madre per i frammenti di split (eliminando la 5ª fase artificiale `Triggering from Split`), la **Severità** (`0..4`) e mappa ogni cella nello schema a **4 classi convettive pure RDT** (+ nube fredda e OT):
     - `Classe 0`: Sfondo / cielo sereno / nubi calde
     - `Classe 1`: Nube fredda non convettiva (Blu acciaio)
     - `Classe 2`: Convezione — **Triggering** (Giallo brillante)
     - `Classe 3`: Convezione — **Growing** (Arancione vivace)
     - `Classe 4`: Convezione — **Mature** (Rosso vivo)
     - `Classe 5`: Convezione — **Decaying** (Rosso mattone / bordeaux)
     - `Classe 6`: **Overshooting Top (OT)** (Magenta brillante)
7. **Fase 6 — Previsione Lagrangiana (`forecast_convective_cells`) e Correzione Geometrica 3D della Parallasse (`apply_parallax_correction_2d`):**
   - Estrae i contorni previsti a $+15, +30, +45, +60\text{ min}$ per traslazione cinematica lungo $\vec{V}(C)$.
   - Corregge lo spostamento apparente di parallasse geostazionaria proiettando ciascun pixel classificato dalla posizione apparente vista dal satellite MSG ($9.5^\circ\text{E}$) alla sua vera verticale al suolo in funzione della quota geometrica $H_{\text{cloud}}(T_{\text{IR10.8}})$.

---

## 2. Sequenza Algoritmica di Basso Livello (Specifica Dettagliata e Pseudocodice)

Di seguito è riportata la specifica matematica e algoritmica passo-passo, aderente alle funzioni implementate nel codice.

### Fase 0 — Continuità Temporale e Stima Radiativa della Tropopausa
**Riferimento codice:** [`NwcSafRdtTracker.step_frame`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/pipeline.py#L110-L128)

```text
PROCEDURA Fase0_Inizializzazione_Slot(frame_7ch, timestamp, stato_precedente, cfg):
    1. Converti timestamp in minuti UTC: t_curr = _to_timestamp_minutes(timestamp)
    2. SE stato_precedente.t_prev è NULLO:
           dt_min = cfg.expected_cadence_minutes   # 5.0 min
           has_valid_prev = FALSO
       ALTRIMENTI:
           dt_min = t_curr - stato_precedente.t_prev
           has_valid_prev = (0.5 <= dt_min <= 3.0 * cfg.expected_cadence_minutes + 1.0)
           SE NON has_valid_prev:
               SVUOTA stato_precedente (prev_labels, prev_cells, prev_hierarchy, track_histories)
               dt_min = cfg.expected_cadence_minutes

    3. # Stima radiativa pura della temperatura di tropopausa (senza NWP)
       maschera_fredda = isfinite(IR_108) AND (IR_108 < 235.0 K)
       SE count(maschera_fredda) >= 50:
           T_tropo_est = 0.6 * Percentile_10(WV_062[maschera_fredda]) + 0.4 * Percentile_5(IR_108[maschera_fredda])
       ALTRIMENTI:
           T_tropo_est = Percentile_5(WV_062)
       T_tropo_k = clip(T_tropo_est, 205.0 K, 226.0 K)
    RITORNA (dt_min, has_valid_prev, T_tropo_k)
```

---

### Fase 1 — Prodotti Ancillari Interni Puro-IR (`GEO-CT` e `GEO-HRW`)
**Riferimento codice:** [`compute_nwcsaf_cloud_type`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/ancillary.py#L43-L141) e [`compute_hrw_amv_guess_field`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/ancillary.py#L144-L259)

#### 1.1 Pre-filtraggio Spettrale Cloud Type (`GEO-CT`)
Per separare le nubi convettive/stratiformi opache dai cirri sottili semitrasparenti (*thin cirrus*) usando i soli canali termici:
- Differenza Split-Window (emissività differenziale del ghiaccio): $\Delta T_{\text{SW}} = \text{IR}_{10.8} - \text{IR}_{12.0}$
- Differenza fase ghiaccio/acqua: $\Delta T_{8.7-10.8} = \text{IR}_{8.7} - \text{IR}_{10.8}$
- Differenza vapore acqueo / finestra: $\Delta T_{\text{WV-IR}} = \text{WV}_{6.2} - \text{IR}_{10.8}$
- Temperatura superficiale di riferimento (senza NWP): $T_{\text{surf,IR}} = \text{clip}(\text{Percentile}_{92}(\text{IR}_{10.8}),\ 280.0\text{ K},\ 308.0\text{ K})$
- Raffreddamento rispetto alla superficie: $\Delta T_{\text{surf}} = T_{\text{surf,IR}} - \text{IR}_{10.8}$

```text
PROCEDURA Fase1A_Compute_NWCSAF_Cloud_Type(IR_108, IR_120, IR_087, WV_062, WV_073, T_tropo_k):
    1. T_surf = clip(Percentile_92(IR_108), 280.0, 308.0)
    2. dT_surf = T_surf - IR_108
       btd_sw = IR_108 - IR_120
       btd_87_108 = IR_087 - IR_108
       btd_wv_ir = WV_062 - IR_108

    3. # Identificazione cielo sereno
       is_clear = (dT_surf < 6.0 K) AND (btd_sw < 2.2 K) AND (btd_wv_ir < -18.0 K)

    4. # Identificazione cirri sottili semitrasparenti (da escludere in RDT)
       # I cirri sottili lasciano trapelare radiazione calda a 10.8 um più che a 12.0 um (btd_sw elevata)
       is_semitransparent_cirrus = (NOT is_clear) AND (IR_108 < 263.15 K) AND (
           (btd_sw > 2.8 K AND btd_wv_ir < -10.0 K) OR
           (btd_sw > 2.0 K AND btd_87_108 > -0.5 K AND IR_108 > 228.0 K AND btd_wv_ir < -12.0 K)
       )

    5. # Nube opaca valida per il rilevamento RDT (dilatata di 1 pixel 3x3 per preservare i bordi delle torri)
       opaque_raw = (NOT is_clear) AND (NOT is_semitransparent_cirrus)
       opaque_cloud_mask = BinaryDilation_3x3(opaque_raw, iterazioni=1) AND (NOT is_clear)
    RITORNA opaque_cloud_mask
```

#### 1.2 Vettori di Moto Atmosferico (`GEO-HRW`) da Block-Matching IR/WV
```text
PROCEDURA Fase1B_Compute_HRW_AMV_Guess_Field(prev_IR108, curr_IR108, prev_WV062, curr_WV062):
    SE prev_IR108 è NULLO:
        RITORNA (zeros(H, W), zeros(H, W))

    1. # Combinazione spettrale per massimizzare la tessitura delle sommità nuvolose
       P = 0.7 * prev_IR108 + 0.3 * prev_WV062
       C = 0.7 * curr_IR108 + 0.3 * curr_WV062

    2. # Griglia rada di blocchi 16x16 pixel con passo stride = 12 pixel e ricerca max_shift = +/- 8 pixel
       PER OGNI blocco (r0, c0) sulla griglia rada:
           SE min(prev_IR108[r0 : r0+16, c0 : c0+16]) > 268.15 K:
               CONTINUA  # salta i blocchi privi di nubi significative
           tpl = P[r0 : r0+16, c0 : c0+16]
           SE std(tpl) < 1.2 K:
               CONTINUA
           # Estrai tutte le (2*max_shift + 1)^2 = 289 finestre candidate in C tramite sliding_window_view
           # Calcola il coefficiente di correlazione di Pearson rho(dy, dx) per tutti gli shift (-8..+8, -8..+8)
           (dy_best, dx_best), rho_max = argmax_{dy, dx} rho(dy, dx)
           SE rho_max >= 0.65:
               amv_dy[ir, ic] = dy_best
               amv_dx[ir, ic] = dx_best
               amv_w[ir, ic]  = clip((rho_max - 0.60) / 0.35, 0.0, 1.0)

    3. # Regolarizzazione Gaussiana (sigma = 1.0) e interpolazione bilineare alla risoluzione nativa (H, W)
       dy_full = Upsample_Bilineare(GaussianFilter(amv_dy * amv_w) / max(GaussianFilter(amv_w), 1e-6))
       dx_full = Upsample_Bilineare(GaussianFilter(amv_dx * amv_w) / max(GaussianFilter(amv_w), 1e-6))
       peso_freddo = clip((258.15 - curr_IR108) / 25.0, 0.0, 1.0)
       alpha = clip(Upsample_Bilineare(GaussianFilter(amv_w)), 0.0, 0.85) * (0.4 + 0.6 * peso_freddo)
    RITORNA (alpha * dy_full, alpha * dx_full)
```

---

### Fase 2 — Rilevamento Adattivo Multi-Soglia e Raffinamento delle Torri
**Riferimento codice:** [`ThresholdSliceHierarchy`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/detection.py#L28-L143) e [`detect_rdt_cells`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/detection.py#L169-L633)

#### Parametri Canonici (ATBD §3.2.1.2.1, UM Tabella 5)
- $T_{\text{warm}} = 278.15\text{ K}$ ($+5^\circ\text{C}$), $T_{\text{cold}} = 198.15\text{ K}$ ($-75^\circ\text{C}$), passo $\Delta T = 1.0\text{ K}$.
- Prominenza minima di torre: $\Delta T_{\text{tower}} = 6.0\text{ K}$.
- Area minima e massima: $S_{\min} = 60.0\text{ km}^2$, $S_{\max} = 400\,000.0\text{ km}^2$.
- Limite superiore di base: $T_{\text{BTLIMIT}} = 213.15\text{ K}$ ($-60^\circ\text{C}$), $T_{\text{mature}} = 233.15\text{ K}$ ($-40^\circ\text{C}$).
- Rapporto di strozzatura (Caso A): $r_{\max} = 0.70$; rapporto Base/Top (Caso B): $R_{\text{BT/ST}} = 4.0$.

```text
PROCEDURA Fase2_Detect_RDT_Cells(frame_7ch, area_km2, opaque_cloud_mask, T_tropo_k, cfg):
    1. # Maschera di validità radiometrica, geometrica e Cloud Type
       valid_mask = isfinite(frame_7ch).all(axis=0) AND (100 K <= frame_7ch <= 400 K).all(axis=0)
                    AND isfinite(area_km2) AND (area_km2 > 0) AND opaque_cloud_mask
       SE cfg.enforce_stretched_mask:
           valid_mask = valid_mask AND (area_km2 <= 5.0 * cfg.nominal_pixel_area_km2)

    2. # Gradiente orizzontale di IR_108 in [K / km]
       (gy_px, gx_px) = Gradient_2D(IR_108 su valid_mask)
       grad_ir_km = hypot(gy_px, gx_px) / sqrt(area_km2)

    3. # Costruzione della gerarchia pigra (lazy) di affettamento a passo di 1 K
       # Per ogni soglia T_k = 198.15 + k * 1.0 K:
       #   slice_at(T_k) etichetta le componenti 8-connesse di (valid_mask AND IR_108 <= T_k),
       #   calcola l'area in km^2 di ogni componente con bincount, il minimo T_min con np.minimum.at
       #   e i bounding box locali con ndimage.find_objects.
       hierarchy = ThresholdSliceHierarchy(IR_108, area_km2, valid_mask, cfg)

    4. # Funzione ricorsiva top-down di discesa lungo l'albero termico
       FUNZIONE _descend(bbox, loc_mask, T_soglia, T_min_comp, is_isolated_single_tower):
           # 4.a Pavimento locale della Base of Tower (BT floor)
           floor_k = cfg.btlimit_k   # 213.15 K (-60 C)
           SE cfg.limittropo:
               floor_k = max(floor_k, mediana(T_tropo_k[bbox][loc_mask]))
           bt_floor_k = max(cfg.tcold_k, floor_k, T_min_comp + 6.0 K)

           # 4.b Scansione verso il basso da (T_soglia - 1 K) fino a bt_floor_k
           PER T_low DA (T_soglia - 1.0 K) A bt_floor_k PASSO -1.0 K:
               (child_labels, child_areas, child_minima, child_bboxes) = hierarchy.slice_at_with_bboxes(T_low)
               ids_sovrapposti = unique(child_labels[bbox][loc_mask]) > 0
               # Filtra i rami figli che costituiscono vere torri termiche (Area >= 60 km^2 e prominenza >= 6 K)
               valid_children = [
                   cid PER cid IN ids_sovrapposti
                   SE child_areas[cid] >= 60.0 km^2 AND (T_low - child_minima[cid]) >= 6.0 K
               ]
               SE len(valid_children) >= 2:
                   # BIFORCAZIONE: due o più torri sorelle si separano!
                   # Ciascuna torre riceve T_low = T_merge - 1 K come soglia isolante iniziale
                   PER OGNI cid IN valid_children:
                       _descend(child_bboxes[cid], child_labels[child_bboxes[cid]] == cid,
                                T_low, child_minima[cid], is_isolated_single_tower = FALSO)
                   RITORNA
               SE len(valid_children) == 1 AND sum(area_km2[bbox][loc_mask]) > 400000.0 km^2:
                   # Il sistema supera ancora S_max: stringi il contorno a T_low
                   cid = valid_children[0]
                   (bbox, loc_mask, T_soglia, T_min_comp) = (child_bboxes[cid], child_labels == cid, T_low, child_minima[cid])

           # 4.c Torre singola (o sistema maturo giunto al pavimento bt_floor_k senza ulteriori biforcazioni)
           seed_yx = argmin_{(y,x) in loc_mask} IR_108[y, x]
           T_min = IR_108[seed_yx]
           T_ST = min(T_soglia, Allinea_Griglia_1K(T_min + 6.0 K))
           S_ST = hierarchy.area_containing_seed(T_ST, seed_yx)
           S_BT = sum(area_km2[bbox][loc_mask])
           T_BT_ref = T_soglia

           # 4.d Raffinamento della Base Eccessiva per Torre Singola (ATBD Sec 3.2.1.2.1.2.2)
           # Per i sistemi maturi (T_min <= 233.15 K), non tagliare mai più in alto del bordo dell'incudine (-40 C)
           single_floor_k = max(T_ST, bt_floor_k, 233.15 K SE T_min <= 233.15 K ALTRIMENTI T_ST)
           SE S_ST >= 60.0 km^2 AND (is_isolated_single_tower OR S_BT > 400000.0 km^2):
               livelli = [T_BT_ref, T_BT_ref - 1 K, ..., single_floor_k]
               aree_liv = [hierarchy.area_containing_seed(lv, seed_yx) PER lv IN livelli]

               # CASO A (Torre che emerge da un banco nuvoloso piatto, per livelli >= 238.15 K = -35 C):
               rapporti = [aree_liv[i+1] / aree_liv[i] PER i TALE CHE livelli[i] >= 238.15 K]
               i_min = argmin(rapporti)
               SE rapporti[i_min] <= 0.70 AND aree_liv[i_min + 1] >= 60.0 km^2:
                   T_BT_ref = livelli[i_min + 1]
                   (_, bbox, loc_mask) = hierarchy.component_containing_seed(T_BT_ref, seed_yx)
                   S_BT = aree_liv[i_min + 1]

               # CASO B (Torre singola piramidale piatta: S_BT <= 4 * S_ST):
               SE S_BT > 4.0 * S_ST OR S_BT > 400000.0 km^2:
                   S_max_consentita = min(400000.0 km^2, 4.0 * S_ST)
                   PRIMO livello lv CON 60.0 km^2 <= area(lv) <= S_max_consentita:
                       T_BT_ref = lv
                       (_, bbox, loc_mask) = hierarchy.component_containing_seed(T_BT_ref, seed_yx)
                       S_BT = area(lv)

           SE 60.0 km^2 <= S_BT <= 400000.0 km^2 AND (T_BT_ref - T_min) >= 6.0 K:
               AGGIUNGI (bbox, loc_mask, T_BT_ref, T_ST, seed_yx) a selected_towers

    5. # Avvia la discesa da tutte le radici a T_warm = 278.15 K (+5 C)
       PER OGNI componente radice comp_id a T_warm CON Area >= 60.0 km^2 AND (278.15 - T_min) >= 6.0 K:
           _descend(root_bbox[comp_id], root_mask[comp_id], 278.15 K, T_min, is_isolated_single_tower = VERO)

    6. # Ordina le torri dalla più fredda alla più calda e assegna etichette disgiunte cell_id = 1..N
       # Per ogni cella calcola:
       #   - T_min, T_mean, T_std, Contrast = T_mean - T_min, TowerDepth = T_BT - T_min
       #   - Baricentro termico pesato: G_w = sum(a_p * (T_BT - T_p) * x_p) / sum(a_p * (T_BT - T_p))
       #   - Volume termico: V_therm = sum(a_p * max(T_BT - T_p, 0))
       #   - Gradiente interno medio e gradiente periferico (media e 95° percentile sul bordo eroso 3x3)
       #   - Ellisse dei momenti secondi (asse maggiore, asse minore, eccentricità, orientamento)
       #   - Statistiche (max, p90, p75, mediana, media, coerenza spaziale SIG) per le differenze spettrali:
       #       * BTD_wv62_ir108 = WV_062 - IR_108   (soglia SIG = -2.0 K)
       #       * WBTD_wv62_wv73 = WV_062 - WV_073   (soglia SIG = -10.0 K)
       #       * BTD_ir87_ir108 = IR_087 - IR_108   (soglia SIG = -1.0 K)
       #       * BTD_ir120_ir108 = IR_120 - IR_108  (soglia SIG = -1.0 K)
       #       * BTD_ir108_ir120 = IR_108 - IR_120  (Split-Window)
    RITORNA (labels_2d, cells_dict, hierarchy, valid_mask)
```

---

### Fase 3 — Tracking Lagrangiano a 4 Passi e Soglia Comune $S(T_{\text{common}})$
**Riferimento codice:** [`track_rdt_cells`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/tracking.py#L140-L530)

```text
PROCEDURA Fase3_Track_RDT_Cells(prev_cells, prev_labels, prev_hierarchy, prev_IR108,
                                curr_cells, curr_labels, curr_hierarchy, curr_IR108, area_km2, guess_motion_yx):
    1. # PASSO 1: Avvezione in avanti delle celle precedenti C di +V(C) * dt
       PER OGNI cella precedente p_id, p_cell IN prev_cells:
           shift_yx = round(p_cell.velocity_yx_px_per_step)   # o guess_motion_yx al baricentro se nuova
           maschera_avvettata = Trasla_Maschera(p_cell.mask, shift_yx)
           PER OGNI cella corrente c_id che interseca maschera_avvettata:
               AOR = Area(maschera_avvettata INTERSECT c_cell.mask) / min(p_cell.area_km2, c_cell.area_km2)
               SE AOR >= 0.15:
                   Registra_Link(p_id -> c_id, step=1, aor=AOR)

    2. # PASSO 2 & PASSO 3: Correlazione incrociata locale IR_108 e avvezione all'indietro per le celle c_id non agganciate
       PER OGNI cella corrente c_id SENZA genitori dopo il Passo 1:
           (disp_yx, rho_max) = CrossCorrelazione_Locale_IR108(prev_IR108, curr_IR108, c_cell.weight_centroid_yx,
                                                               finestra=25 px, max_shift=12 px)
           SE rho_max >= 0.50:
               v_est_yx = disp_yx
           ALTRIMENTI:
               v_est_yx = Velocita_Vicino_Tracciato_Piu_Prossimo_O_GuessField(c_cell, guess_motion_yx)
           # Avvezione all'indietro di -v_est_yx
           maschera_indietro = Trasla_Maschera(c_cell.mask, -round(v_est_yx))
           PER OGNI p_id che interseca maschera_indietro CON AOR >= 0.15:
               Registra_Link(p_id -> c_id, step=3, aor=AOR)

    3. # PASSO 4: Dilatazione morfologica di 1 pixel (3x3) per le piccole celle nascenti (<= 5 pixel) ancora orfane
       PER OGNI cella corrente c_id SENZA genitori E CON c_cell.pixel_count <= 5:
           maschera_dilatata = BinaryDilation_3x3(maschera_indietro, iterazioni=1)
           PER OGNI p_id che interseca maschera_dilatata CON AOR >= 0.15:
               Registra_Link(p_id -> c_id, step=4, aor=AOR)

    4. # Risoluzione di Split, Merge e Link Complessi tramite la Regola della Soglia Comune (ATBD v1.1 p. 22)
       # Per ogni coppia collegata (p_id, c_id), la soglia comune è T_common = min(p_cell.t_bt_k, c_cell.t_bt_k):
       #   - In un MERGE (più padri p_1..p_k -> una figlia c_id):
       #       il Padre Principale (Main Father) è argmax_{p_i} S_{t-dt}(p_i, T_common)
       #   - In uno SPLIT (un padre p_id -> più figlie c_1..c_m):
       #       la Figlia Principale (Main Child, che eredita il track_id) è argmax_{c_j} S_t(c_j, T_common);
       #       le figlie secondarie ricevono evento "split_child" e un nuovo track_id ma ereditano il flag convettivo.
       #   - In una componente connessa bipartite molti-a-molti (len(padri) >= 2 AND len(figlie) >= 2):
       #       Link Complesso -> azzera l'eredità temporale di trend (evento "complex").

    5. # Calcolo della velocità della famiglia collegata e smussamento esponenziale (ATBD Fig. 14)
       # V_raw(c_id) = (Baricentro_Famiglia_Figlie - Baricentro_Famiglia_Padri) / dt
       # V_smoothed(c_id) = 0.5 * V_raw(c_id) + 0.5 * V_precedente(Main_Father)
    RITORNA (parents, children, edges, primary_parent, events, convective_parent)
```

---

### Fase 4 — Rilevamento a Due Stadi degli Overshooting Tops (`OTD`)
**Riferimento codice:** [`detect_cell_overshooting_tops`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/otd.py#L23-L185)

```text
PROCEDURA Fase4_Detect_Overshooting_Tops(cell, frame_7ch, area_km2, T_tropo_k, cfg):
    1. # Stadio 1: Pre-rilevamento dei minimi locali freddi interni alla cella
       T_ref_ot = min(223.15 K, T_tropo_k + 3.0 K)
       SE cell.t_min_k > T_ref_ot:
           RITORNA Nessun_OT

       # Identifica i minimi regionali 8-connessi con IR_108 <= T_ref_ot
       # Ciascun candidato è cresciuto fino a min(T_min_cand + 3.0 K, T_ref_ot) con area <= 150 km^2 (<= 15 pixel)

    2. # Stadio 2: Conferma radiativo-morfologica del candidato OT
       PER OGNI candidato OT c_ot:
           # Calcola la temperatura mediana dell'anello di incudine circostante (raggio 2..5 pixel attorno al candidato)
           T_anvil_ring = mediana(IR_108[anello_incudine])
           prominenza_anvil = T_anvil_ring - c_ot.t_min_k
           btd_wv_ir_ot = max(WV_062 - IR_108 su c_ot)
           btd_87_108_ot = max(IR_087 - IR_108 su c_ot)

           # Un OT è confermato se:
           # (a) buca la tropopausa (T_min <= T_tropo_k + 1.5 K) E ha prominenza sull'incudine >= 3.5 K, OPPURE
           # (b) presenta inversione stratosferica di vapore acqueo (WV_062 - IR_108 >= -0.5 K) con prominenza >= 2.2 K
           SE ((c_ot.t_min_k <= T_tropo_k + 1.5 K) AND (prominenza_anvil >= 3.5 K)) OR
              ((btd_wv_ir_ot >= -0.5 K) AND (prominenza_anvil >= 2.2 K) AND (btd_87_108_ot >= -1.5 K)):
               Marca c_ot come OT_Confermato
    RITORNA (candidate_mask, confirmed_mask, confirmed_count)
```

---

### Fase 5 — Stratificazione Verticale, Discriminazione Fisica `GEN`, Ereditarietà e Ciclo di Vita RDT a 4 Fasi Convettive
**Riferimento codice:** [`discriminate_and_classify_cell`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/discrimination.py#L275-L515)

#### 5.1 Calcolo dei Predittori Temporali a Soglia Comune (`compute_temporal_predictors`)
Per ciascuna finestra temporale storica disponibile $\Delta \tau \in \{15, 30, 45, 60\}\text{ min}$ lungo la traiettoria del *Main Father*:
- **Tasso di raffreddamento:** $\dot{T}_{\min}(\Delta \tau) = \frac{T_{\min}(t) - T_{\min}(t - \Delta \tau)}{\Delta \tau}\ \left[\text{K h}^{-1}\right]$ (negativo quando la nube sale e si raffredda).
- **Tasso di espansione areale relativa a soglia comune:**
  $$T_{\text{common}} = \min\!\big(T_{\text{BT}}(t),\ T_{\text{BT}}(t - \Delta \tau)\big), \qquad \dot{S}_{\text{rel}}(\Delta \tau) = \frac{S_t(T_{\text{common}}) - S_{t-\Delta \tau}(T_{\text{common}})}{\max(S_{t-\Delta \tau}(T_{\text{common}}), 1) \cdot \Delta \tau}\ \left[\text{h}^{-1}\right]$$
- **Trend spettrale:** $\Delta\text{BTD}_{6.2-10.8}(\Delta \tau) = \text{BTD}_{6.2-10.8}^{\text{p90}}(t) - \text{BTD}_{6.2-10.8}^{\text{p90}}(t - \Delta \tau)$.

#### 5.2 Stratificazione Verticale in 6 Categorie (`classify_vertical_category`, ATBD §3.2.1.2.3.1)
Ogni cella viene assegnata a una delle 6 categorie verticali in base a $(T_{\min}, T_{\text{BT}}, \text{età della traiettoria})$:
- **`Cat 0 (Mature)`:** $T_{\min} \le 233.15\text{ K}$ ($-40^\circ\text{C}$) **e** durata tracciata $\ge 30\text{ min}$.
- **`Cat 1 (Mature Transition)`:** ($T_{\min} \le 233.15\text{ K}$ e durata $< 30\text{ min}$) **oppure** ($T_{\min} > 233.15\text{ K}$ e $T_{\text{BT}} \le 238.15\text{ K} = -35^\circ\text{C}$).
- **`Cat 2 (Cold Transition)`:** $233.15\text{ K} < T_{\min} \le 238.15\text{ K}$ ($-35^\circ\text{C}$) e $238.15\text{ K} < T_{\text{BT}} \le 248.15\text{ K}$ ($-25^\circ\text{C}$).
- **`Cat 3 (Warm 1 Transition)`:** ($233.15\text{ K} < T_{\min} \le 238.15\text{ K}$ con $T_{\text{BT}} > 248.15\text{ K}$) **oppure** ($238.15\text{ K} < T_{\min} \le 248.15\text{ K}$ con $T_{\text{BT}} \le 258.15\text{ K} = -15^\circ\text{C}$).
- **`Cat 4 (Warm 2 Transition)`:** ($238.15\text{ K} < T_{\min} \le 248.15\text{ K}$ con $T_{\text{BT}} > 258.15\text{ K}$) **oppure** ($248.15\text{ K} < T_{\min} \le 258.15\text{ K}$).
- **`Cat 5 (Warm)`:** $T_{\min} > 258.15\text{ K}$ ($-15^\circ\text{C}$).

#### 5.3 Discriminazione Fisica Pura (`GEN`), Ereditarietà e Fase del Ciclo di Vita
```text
PROCEDURA Fase5_Discriminate_And_Classify_Pure_GEN(cell, prev_cell, conv_parent_cell, history, ot_res, T_tropo_k):
    1. cat = classify_vertical_category(cell.t_min_k, cell.t_bt_k, eta_traiettoria)
    2. # Valutazione dello score fisico/radiativo GEN per la categoria verticale cat:
       # Combina:
       #   - Raffreddamento attivo della sommità: cooling_rate_k_per_h <= soglia_cooling(cat)  (es. <= -8 K/h in Cat 4-5, <= -4 K/h in Cat 2-3)
       #   - Spessore ottico e iniezione di vapore acqueo: btd_wv62_ir108_p90 >= soglia_btd(cat) E coerenza spaziale SIG >= 0.35
       #   - Contrasto torre/base (T_mean - T_min) e gradiente periferico 95° percentile (GradP_p95 >= 1.8 K/km)
       #   - Espansione areale relativa positiva a soglia comune (dS_rel > 0.15 h^-1)
       #   - Fase ghiaccio in sommità (BTD_87_108 >= -1.5 K) e bassa differenza Split-Window (|IR_108 - IR_120| <= 2.0 K)
       is_conv_gen, score_gen = Valuta_Regole_Fisiche_GEN(cell, temporal_preds, cat, T_tropo_k)

    3. # Ereditarietà Convettiva (Convective Inheritance, ATBD Sec 3.2.1.2.3.2.3)
       inherited_conv = (conv_parent_cell NON È NULLO) AND conv_parent_cell.is_convective
       declassified = FALSO
       SE inherited_conv:
           # Regole empiriche di declassificazione (dissipazione):
           # La cella perde lo stato convettivo ereditato se:
           #   (a) Si riscalda costantemente per >= 30 min (cooling_rate_30m >= +4.0 K/h) E non ha OT confermati, OPPURE
           #   (b) È diventata otticamente sottile/stratiforme (btd_wv62_ir108_p90 < -9.5 K AND GradP_p95 < 1.0 K/km), OPPURE
           #   (c) T_min è risalita sopra -20 C (253.15 K) senza raffreddamento attivo (cooling_rate >= -1.0 K/h)
           SE Criterio_Dissipazione_Verificato(cell, history, temporal_preds, ot_res):
               declassified = VERO
               inherited_conv = FALSO

    4. # Decisione finale di convettività (incluso override fisico se ha un Overshooting Top confermato)
       is_convective = is_conv_gen OR inherited_conv OR (ot_res.confirmed_count > 0)

    5. # Determinazione della Fase di Sviluppo RDT (ATBD Tabella 8, p. 51):
       SE NON is_convective:
           phase_code = -1
           class_id = 1   # Classe 1: Nube fredda non convettiva (Blu acciaio)
       ALTRIMENTI:
           # Gestione ereditarietà da split: il figlio eredita lo stato vitale della madre
           SE cell.event == "split_new":
               phase_code = mother_cell.phase_code SE (mother_cell È PRESENTE) ALTRIMENTI 1
           ALTRIMENTI SE cell.event IN ("new", "complex"):
               phase_code = 0   # Triggering: nuova attivazione
           ALTRIMENTI:
               # Valutazione termodinamica del trend di raffreddamento e di espansione:
               cooling_rate = cell.cooling_rate_15m   # positivo = si raffredda; negativo = si riscalda
               d_area = cell.area_change_15m          # variazione areale [km^2]
               is_mature_cat = (vertical_cat == 0) OR (T_min <= 216.0 K)

               SE cooling_rate > +2.0 K/h:   # Raffreddamento attivo (updraft in ascesa)
                   SE NON is_mature_cat:
                       phase_code = 1   # Growing
                   ALTRIMENTI:
                       phase_code = 1 SE (d_area > 0) ALTRIMENTI 2   # Growing se espande, altrimenti Mature
               ALTRIMENTI SE |cooling_rate| <= 2.0 K/h:   # Temperatura stazionaria al top
                   SE is_mature_cat:
                       phase_code = 2   # Mature
                   ALTRIMENTI SE (d_area > 0 OR is_convective):
                       phase_code = 1   # Growing
                   ALTRIMENTI:
                       phase_code = 3   # Decaying
               ALTRIMENTI:   # Riscaldamento (cooling_rate < -2.0 K/h, correnti discendenti/collasso)
                   warming_rate = -cooling_rate
                   low_warming = (warming_rate <= 5.0 K/h)
                   SE NON is_mature_cat:
                       phase_code = 1 SE (is_convective AND low_warming) ALTRIMENTI 3
                   ALTRIMENTI:
                       phase_code = 2 SE (low_warming OR has_ot) ALTRIMENTI 3   # Decaying

           # Mappatura diretta nelle 4 classi convettive canoniche RDT:
           SE phase_code == 0:
               class_id = 2   # Classe 2: Triggering (Giallo brillante, RGB: 250, 204, 21)
           ALTRIMENTI SE phase_code == 1:
               class_id = 3   # Classe 3: Growing (Arancione vivace, RGB: 249, 115, 22)
           ALTRIMENTI SE phase_code == 2:
               class_id = 4   # Classe 4: Mature (Rosso vivo, RGB: 220, 38, 38)
           ALTRIMENTI:
               class_id = 5   # Classe 5: Decaying (Rosso mattone / bordeaux, RGB: 153, 27, 27)

           # I pixel interni confermati come Overshooting Top ricevono Classe 6 (Magenta, RGB: 232, 121, 249)
    RITORNA (is_convective, phase_code, severity, class_id)
```

---

### Fase 6 — Previsione Lagrangiana e Correzione Geometrica 3D della Parallasse
**Riferimento codice:** [`forecast_convective_cells`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/discrimination.py#L518-L565) e [`apply_parallax_correction_2d`](file:///c:/Users/Daniele.LOKI/Documents/Progetti/deep_convection/RDT_original/rdt_nwcsaf/ancillary.py#L262-L344)

```text
PROCEDURA Fase6_Parallax_Correction_2D(raster_2d, IR_108, lat_deg, lon_deg, sat_lon_deg = 9.5):
    1. # Stima della temperatura superficiale ambiente dall'85° percentile di IR_108
       T_surf = clip(Percentile_85(IR_108), 280.0 K, 305.0 K)
    2. # Quota geometrica della sommità nuvolosa (m) tramite gradiente adiabatico troposferico standard (6.5 K/km)
       H_cloud_m = clip((T_surf - IR_108) / 0.0065, 0.0 m, 14500.0 m)

    3. # Geometria di vista dal satellite geostazionario MSG (r_sat = 42,164 km, R_eq = 6,378.169 km)
       xx = R_eq * cos(rad(lat_deg)) * sin(rad(lon_deg - sat_lon_deg))
       yy = R_eq * sin(rad(lat_deg))
       z_earth = sqrt(max(R_eq^2 - xx^2 - yy^2, (0.28 * R_eq)^2))
       geom_factor = (r_sat / (r_sat - z_earth)) / z_earth

    4. # Spostamento apparente verso l'esterno del disco (m) e traslazione inversa alla vera verticale al suolo
       dx_app_m = H_cloud_m * xx * geom_factor
       dy_app_m = H_cloud_m * yy * geom_factor
       dr = round(-dy_app_m / dy_step_m)
       dc = round(-dx_app_m / dx_step_m)

    5. # Riproiezione ordinata per priorità di classe (le classi convettive 2, 3, 4 sovrascrivono la classe 1)
       PER OGNI pixel nuvoloso (r, c) ordinato per valore crescente di raster_2d[r, c]:
           out[clip(r + dr[r, c], 0, H-1), clip(c + dc[r, c], 0, W-1)] = raster_2d[r, c]
    RITORNA out
```

---

## 3. Capitolo Finale: Cosa perde il Modello Puro SEVIRI rispetto al vero RDT Operativo (secondo la documentazione ufficiale NWC SAF) e cosa mantiene il nostro Modello Completo (`Full`)

Secondo la documentazione ufficiale EUMETSAT NWC SAF (*ATBD v2.1* e *Product User Manual v1.0*), il pacchetto operativo `GEO-RDT-CW` non è un algoritmo puramente satellitare, ma un sistema di fusione multi-sensore. Di seguito elenchiamo **esattamente cosa perde il nostro Modello Puro 7ch SEVIRI (`nwcsaf_rdt_seviri_pure_gen`) rispetto al vero RDT operativo**, e quali di questi elementi sono invece mantenuti nel nostro **Modello Completo (`nwcsaf_rdt_full_era5_cal`)**:

| Componente del Vero RDT Operativo (Specifica Ufficiale NWC SAF) | Cosa perde il **Modello Puro SEVIRI (`pure_gen`)** usando i soli 7 canali IR/WV | È mantenuto nel nostro **Modello Completo (`full_era5_cal`)**? | Impatto Fisico e Operativo della Perdita nel Modello Puro |
| :--- | :--- | :---: | :--- |
| **1. Maschera di Instabilità Termodinamica NWP (`NWPCONV`: $LI \le -3$, $KI \ge 30$, $SHW \le -3$, CAPE)**<br>*(ATBD §3.2.1.2.3.1, p. 35)* | **Perduto:** non conosce la stabilità termodinamica della colonna d'aria sottostante la nube. | **SÌ** (calcolato dai profili verticali completi ERA5 a 37 livelli di pressione) | In situazioni invernali o post-frontali sull'oceano/Nord Europa, spessi sistemi baroclini freddi non convettivi in atmosfera stabile ($LI > 0$) possono presentare sommità fredde e forti gradienti `IR_108`, generando falsi positivi nel modello puro che il filtro `NWPCONV` del modello completo invece blocca. |
| **2. Temperatura e Pressione della Tropopausa Termica WMO da Profilo NWP ($T_{\text{tropo,WMO}}$)**<br>*(ATBD §3.2.1.2.1.2.1 & §3.2.1.2.5)* | **Sostituito da stima radiativa IR/WV:** il modello puro stima $T_{\text{tropo}}$ dal 5°–10° percentile delle nubi fredde `WV_062`/`IR_108` presenti nella scena. | **SÌ** (calcolato col criterio del lapse-rate WMO $\Gamma \le 2\text{ K km}^{-1}$ di Reichler et al., 2003 su ERA5) | Se in una scena c'è soltanto convezione giovane/moderata che non ha ancora raggiunto la tropopausa, oppure in presenza di forti anomalie di tropopausa (es. *tropopause folding* associato a jet streak o cicloni mediterranei), la stima puramente radiativa ha un'incertezza di $\pm 3\text{--}5\text{ K}$, influenzando la soglia `LIMITTROPO` e la sensibilità agli *Overshooting Tops*. |
| **3. Coefficienti Statistici Calibrati su Verità Terrestre (`CAL` $\beta_k$ per $6\text{ categorie} \times 4\text{ regimi storici}$)**<br>*(ATBD §3.2.1.2.3.2.2, Tab. 2–4)* | **Perduto per scelta di purezza scientifica:** il modello puro usa lo schema fisico generico non calibrato (`GEN`), per non introdurre pesi appresi dal radar. | **SÌ** (`CalibratedRdtModel` addestrato mediante regressione logistica sulle riflettività radar OPERA) | I pesi statistici $\beta_k$ pesano in modo ottimale il contributo congiunto delle 5 bande spettrali e dei trend temporali riducendo i falsi allarmi su incudini stratiformi (portando la *Precision* rispetto al modello completo dal `94.57%` del puro con ancillari al `98.39%` del calibrato). |
| **4. Venti Orizzontali NWP a $700\text{ hPa}$ ($\vec{V}_{700}$) e Temperatura Superficiale NWP ($T_{\text{surf}}$)**<br>*(ATBD §3.2.1.2.2.1)* | **Sostituito da soli dati IR/WV:** il modello puro usa solo il block-matching `IR_108`/`WV_062` per il moto e il 92° percentile `IR_108` per $T_{\text{surf}}$. | **SÌ** ($u_{700}, v_{700}, T_{\text{skin}}$ da ERA5 fusi con gli AMV satellitari) | Per le piccole celle isolate appena nate (*triggering* al primo frame, dove non esiste una storia precedente della cella né tessitura nuvolosa estesa intorno), il vento di guida a $700\text{ hPa}$ aiuta l'avvezione al primo passo; inoltre su domini totalmente coperti da nubi estese il 92° percentile `IR_108` sottostima la vera $T_{\text{surf}}$. |
| **5. Dati in Tempo Reale dalle Reti di Fulminazione a Terra (`LGH`: ATDnet / EUCLID / Météorage)**<br>*(ATBD §3.2.1.2.3.2.4, p. 41)* | **Perduto:** i fulmini non sono osservabili dai canali termici SEVIRI. | **NON presente** (le reti fulmini europee sono proprietarie e a pagamento) | Nel vero RDT operativo, la presenza di $\ge 1\text{--}2$ fulmini nube-suolo forza immediatamente la classificazione convettiva (`LGH override`) anche per piccole celle calde appena nate prima ancora che raggiungano forti raffreddamenti IR, e contribuisce all'indice di *Severity*. |
| **6. Canali SEVIRI nel Visibile (`VIS0.6`, `VIS0.8`, `HRV` a $1\text{ km}$) per `GEO-CT` e `OTD` diurno**<br>*(ATBD §3.2.1.2.5)* | **Non usati (nemmeno di giorno):** usiamo esclusivamente i 7 canali termici/WV per preservare l'omogeneità giorno/notte. | **NON usati** (per coerenza con i tensori a 7 canali termici del progetto) | Di giorno, la riflettanza visibile e le ombre proiettate sul canale `HRV` a $1\text{ km}$ aumentano l'accuratezza nella conferma degli *Overshooting Tops* più piccoli e nella separazione tra nubi basse calde e suolo caldo, ma al prezzo di una discontinuità giorno/notte al crepuscolo. |
