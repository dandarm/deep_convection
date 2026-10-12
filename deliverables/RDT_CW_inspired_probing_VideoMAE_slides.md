# RDT-CW-inspired come probing di VideoMAE

Testo pronto per una mini-presentazione di **due slide**. Il teacher mantiene
le cinque classi e la logica temporale del piano iniziale, ma sostituisce le
soglie termiche assolute con una segmentazione adattiva e con un riferimento
UTLS derivato dai canali SEVIRI.

Il metodo deve essere indicato come **RDT-CW-inspired** finché segmentazione,
tracking, split/merge e transizioni del ciclo di vita non saranno validati
contro la specifica operativa autorevole.

---

## Slide 1 — Celle convettive con soglie adattive

**Sottotitolo**

La geometria nasce da IR 10.8 µm; gli altri canali termici stimano un
riferimento radiometrico della regione UTLS.

### 1. Segmentazione adattiva in IR 10.8 µm

- Individuazione dei minimi locali di brightness temperature.
- Crescita delle regioni verso temperature più calde tramite component tree,
  equivalente a una watershed della topografia termica.
- Per ogni torre, il bordo è il contorno più caldo che la mantiene separata
  dai minimi vicini: la soglia `T_edge` è quindi specifica della cella.
- Componenti con area **< 100 km²** vengono eliminate nella configurazione
  iniziale.

Questo passaggio riprende il principio RDT-CW di adaptive temperature
thresholding.[^R1] Non usa più la scala fissa 240, 230, 220 e 210 K.

### 2. Proxy SEVIRI della temperatura UTLS

**UTLS = Upper Troposphere–Lower Stratosphere**, cioè alta troposfera e bassa
stratosfera.

- Si seleziona il plateau esteso e uniforme dell'incudine, escludendo i minimi
  piccoli e molto freddi che potrebbero essere Overshooting Top.
- IR 8.7, IR 10.8 e IR 12.0 verificano fase del ghiaccio, opacità e uniformità.
- WV 6.2, WV 7.3, IR 9.7 e IR 13.4 forniscono sensibilità complementare
  all'alta troposfera, all'ozono e all'assorbimento della CO₂.[^R2]
- La mediana pesata di IR 10.8 sul plateau, corretta con le differenze
  multispettrali, produce `T_UTLS_proxy`.
- Il campo viene stimato su un contesto mesoscalare, filtrato nel tempo e
  accompagnato da un indicatore di confidenza.

`T_UTLS_proxy` è un riferimento radiometrico vicino al livello di equilibrio
dell'incudine. Non viene presentato come misura diretta della tropopausa
termica WMO.

### 3. Profondità termica normalizzata

```text
D = (T_edge - IR10.8) / (T_edge - T_UTLS_proxy)
```

- `D ≈ 0`: bordo della torre.
- `0 < D < 1`: nube progressivamente più profonda.
- `D ≈ 1`: sommità vicina al riferimento UTLS.
- `D > 1`: pixel più freddo dell'incudine, candidato OT.

### 4. Tracking e diagnostiche, invariati

- Associazione tra frame con
  **AOR = |A ∩ B| / min(|A|, |B|)** e soglia iniziale **AOR ≥ 0,15**.
- Gestione esplicita di split e merge.
- Contrasto `T_mean − T_min`, cooling e variazione dell'area lungo la traccia.

**Nota da dire a voce**

Il proxy è una nostra sintesi metodologica, non una formula già pubblicata.
Combina il principio di segmentazione adattiva di RDT-CW con diagnostiche
spettrali e morfologiche già usate per nubi alte e Overshooting Top.[^R3][^R4][^R5]

---

## Slide 2 — Cinque classi e probing del backbone congelato

**Sottotitolo**

La semantica delle classi rimane invariata; cambiano il bordo radiometrico
della cella e il criterio assoluto usato per l'OT.

### Passo 1 — Sfondo

- Pixel esterno alle celle ottenute dalla segmentazione adattiva:
  **Classe 0, sfondo**.
- Pixel interno a una cella: passaggio alla discriminazione convettiva.

La condizione fissa `IR 10.8 > 240 K` viene sostituita dal bordo `T_edge`
specifico della cella.

### Passo 2 — Discriminazione convettiva

La cella è candidata convettiva se soddisfa i criteri iniziali:

- **T_mean − T_min ≥ 10 K**;
- cooling **≤ −2,5 K / 15 min**, oppure stato convettivo ereditato per una
  cella già matura;
- espansione dell'area **> 0**.

Se fallisce: **Classe 1, nube fredda non convettiva**.

### Passo 3 — Fase della cella

- Cooling rapido e crescita: **Classe 2, cella in sviluppo**.
- Cooling rallentato, temperatura stabile o warming:
  **Classe 3, cella matura o in decadimento**.

Tracking, cooling, espansione e significato delle Classi 1–3 restano quelli
del piano iniziale.

### Passo 4 — Overshooting Top, override a livello di pixel

Un pixel di Classe 2 o 3 diventa **Classe 4, Overshooting Top** se combina:

- `D > 1`, cioè temperatura inferiore al riferimento dell'incudine;
- contrasto locale rispetto all'anvil circostante;
- firma coerente nelle differenze WV 6.2–IR 10.8, IR 9.7–IR 10.8 e
  IR 13.4–IR 10.8;[^R3]
- dimensione, morfologia e persistenza temporale compatibili con un OT.
  Il confronto con l'incudine riprende la famiglia di metodi IRW-texture.[^R4][^R5]

La soglia assoluta `IR 10.8 ≤ 215 K` viene quindi eliminata. La soglia iniziale
`WV 6.2 − IR 10.8 ≥ 1 K` resta un parametro candidato da validare, non una
regola già confermata.

### Probing

```text
Teacher deterministico multispettrale -> pseudo-label a 5 classi

VideoMAE congelato, 7 canali -> decoder 3D FPN -> predizione a 5 classi
```

I sette canali, nello stesso ordine del VideoMAE, sono:
**WV 6.2, WV 7.3, IR 8.7, IR 9.7, IR 10.8, IR 12.0 e IR 13.4 µm**.

**Messaggio scientifico da evidenziare**

Una buona prestazione mostra che il backbone congelato rende decodificabili
la geometria termica e l'evoluzione definite dal teacher. La conclusione
richiede confronti con encoder casuale e RGB adattato, oltre alle ablazioni
temporali e spettrali.

---

## Riferimenti scientifici e ruolo nel metodo

[^R1]: **NWC SAF, Algorithm Theoretical Basis Document for Convection
    Products, RDT-CW.** Descrive le celle tramite soglie adattive specifiche
    per oggetto, la selezione del contorno più caldo che separa minimi vicini,
    il tracking e la gestione di split e merge.
    <https://www.nwcsaf.org/AemetWebContents/ScientificDocumentation/Documentation/GEO/v2016/NWC-CDOP2-GEO-MFT-SCI-ATBD-Convection_v1.1.pdf>

[^R2]: **ESA, The MSG System.** Motiva il ruolo fisico dei canali SEVIRI:
    WV 6.2 e WV 7.3 per differenti strati troposferici, IR 9.7 per ozono e
    monitoraggio della tropopausa, IR 13.4 per sounding termico e altezza
    delle nubi. <https://www.esa.int/esapub/bulletin/bullet111/chapter3_bul111.pdf>

[^R3]: **Mikuš e Strelec Mahović, 2013, Satellite-based overshooting top
    detection methods and an analysis of correlated weather conditions.**
    Confronta differenze di brightness temperature basate su SEVIRI 6.2, 9.7
    e 13.4 µm rispetto a IR 10.8; la combinazione 6.2–10.8 e 9.7–10.8 ottiene
    i risultati migliori nello studio.
    <https://doi.org/10.1016/j.atmosres.2012.09.001>

[^R4]: **Bedka et al., 2010, Objective Satellite-Based Detection of
    Overshooting Tops Using Infrared Window Channel Brightness Temperature
    Gradients.** È il riferimento per contrasto locale, tessitura IR e
    confronto fra un minimo freddo e l'incudine circostante.
    <https://doi.org/10.1175/2009JAMC2286.1>

[^R5]: **Bedka, 2011, Overshooting cloud top detections using MSG SEVIRI
    infrared brightness temperatures and their relationship to severe weather
    over Europe.** Applica e valuta le firme IR degli OT su MSG SEVIRI.
    <https://doi.org/10.1016/j.atmosres.2010.10.001>

### Provenienza della formula proposta

Nessuno di questi riferimenti definisce direttamente
`T_UTLS_proxy = mediana pesata dell'incudine + correzione multispettrale` né la
normalizzazione `D`. Queste due formule sono una proposta originale del
progetto. La letteratura giustifica i segnali in ingresso e i vincoli fisici;
la forma definitiva, i pesi e le soglie dovranno essere calibrati e validati.

## Indicazione grafica facoltativa

- Colori delle classi: 0 grigio, 1 verde acqua, 2 azzurro, 3 arancio, 4 rosso.
- Evidenziare `T_edge`, il plateau dell'incudine e il minimo OT su un profilo
  termico schematico.
- Inserire il richiamo **“Proxy SEVIRI da validare”** vicino a
  `T_UTLS_proxy`.

Fonti interne: `RDT_plan.md`, `README.md`, `VIDEOMAE_PRETRAINING.md`.
