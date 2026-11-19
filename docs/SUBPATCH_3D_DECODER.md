# Sub-Patch 3D Decoder (`SubPatchFPN3D` & `SubPatchThreeHeadProbe`)

Specifica tecnica e razionale scientifico del decoder deterministico *sub-patch* introdotto per il probing a 3 teste (CMA, CT, RDT) su encoder VideoMAE SEVIRI congelati.

- **Implementazione:** [`src/emma_gpm/subpatch_probe.py`](../src/emma_gpm/subpatch_probe.py)
- **Test unitari:** [`tests/test_subpatch_probe.py`](../tests/test_subpatch_probe.py)
- **Script A/B locale:** [`scripts/run_subpatch_probe_experiment.py`](../scripts/run_subpatch_probe_experiment.py)

---

## 1. Diagnosi del collo di bottiglia nel decoder precedente (`LightweightFPN3D`)

Il backbone VideoMAE canonico elabora clip SEVIRI di forma $(T, C, H, W) = (16, 7, 224, 224)$ tramite un embedding a tubelet `Conv3d(7, D, kernel=(2, 16, 16), stride=(2, 16, 16))` (`patch_size=16`, `tubelet_size=2`). Ogni blocco del Transformer produce una griglia di token di dimensione:

$$
(T', H', W') = (8, 14, 14) = 1.568 \text{ token}, \quad \text{ciascuno in } \mathbb{R}^{D} \; (D=768 \text{ per Base}, \; D=384 \text{ per Small}).
$$

In principio, l'informazione spaziotemporale ad alta frequenza *interna alla patch* $2 \times 16 \times 16$ (bordi frastagliati delle nubi, tasche di cielo sereno, piccoli Overshooting Top) **può essere preservata dentro il vettore latente di dimensione $D$**, poiché la convoluzione iniziale proietta i $2 \times 16 \times 16 \times 7 = 3.584$ valori di brillanza su $D$ basi spaziotemporali (e i 7 canali IR hanno un rango spettrale effettivo $\approx 2.5\text{--}3$, quindi $\sim 1.200\text{--}1.500$ gradi di libertà reali).

Tuttavia, il decoder originale [`LightweightFPN3D`](../src/emma_gpm/three_head_model.py) presenta **tre limiti strutturali** che impediscono matematicamente la decodifica *sub-patch*:

1. **Upsampling puramente trilineare (`F.interpolate(..., mode='trilinear')`):**
   Le feature dei blocchi `[3, 6, 9, 12]` vengono proiettate a 128 canali sulla griglia $(8, 14, 14)$ e immediatamente interpolate trilinearmente a $(8, 28, 28) \to (8, 56, 56) \to (8, 112, 112) \to (16, 224, 224)$. L'interpolazione trilineare è un **filtro passa-basso**: tratta ogni token come un singolo campione puntuale ogni 16 pixel e traccia rampe lineari lisce tra centri patch adiacenti, senza mai convertire i canali latenti in coordinate *sub-patch*.
2. **Nessun mescolamento tra canali nel raffinamento:**
   Ogni blocco `self.refine[index]` usa solo una convoluzione *depthwise* `Conv3d(128, 128, 3, padding=1, groups=128)` seguita da `GroupNorm + GELU`. Senza convoluzioni *pointwise* ($1\times 1\times 1$) ad alta risoluzione, i canali non possono ricombinarsi per sintetizzare contorni netti; le non-linearità piegano semplicemente rampe lisce, generando maschere tondeggianti ("a macchia morbida").
3. **Nessuna decodifica sub-temporale:**
   L'asse temporale resta compresso a $T'=8$ lungo tutta la piramide e viene interpolato linearmente a $T=16$ soltanto sui logit finali, sfocando l'evoluzione tra frame consecutivi a 5 minuti.

---

## 2. Perché un decoder deterministico sub-patch e NON una Diffusion Head

Per il task canonico di **probing delle rappresentazioni congelate** su maschere categoriali deterministiche (CMA, CT, RDT), una testa generativa a diffusione è metodologicamente inadatta:

1. **Confusione tra decodificabilità e allucinazione generativa:** il probing misura quanta fisica atmosferica è linearmente o semi-linearmente *decodificabile* dal backbone congelato. Una diffusion head ad alta capacità allucinerebbe bordi frastagliati visivamente plausibili ma non necessariamente ancorati agli embedding del backbone (peggiorando IoU/CSI sui bordi ambigui e invalidando il test scientifico).
2. **Calibrazione probabilistica ed efficienza:** le teste deterministiche allenate con Focal Cross-Entropy + Dice stimano direttamente le probabilità a posteriori calibrate $P(y_{t,y,x} \mid x)$ in un **singolo forward pass**, mentre la diffusione richiede 20–100 step di denoising iterativi e campionamenti d'ensemble per ottenere probabilità.
3. **Separazione dei ruoli nel progetto:** il decoder deterministico *sub-patch* risolve la fedeltà spaziotemporale della **segmentazione diagnostica al tempo presente**; una testa a diffusione condizionata dagli stessi embedding VideoMAE potrà eventualmente essere introdotta in una fase successiva (Obiettivo C) solo per il **nowcasting generativo probabilistico futuro** (+1h / +2h), dove l'evoluzione atmosferica è intrinsecamente stocastica.

---

## 3. Architettura matematica di `SubPatchFPN3D` e `SubPatchThreeHeadProbe`

### 3.1 Operatore `pixel_shuffle_3d`

Data una mappa di feature $X \in \mathbb{R}^{B \times (C \cdot r_t \cdot r_h \cdot r_w) \times T \times H \times W}$, l'operatore deterministico `pixel_shuffle_3d(X, rt, rh, rw)` riorganizza la dimensione dei canali nelle coordinate spazio-temporali *sub-voxel*:

$$
(B, \, C \cdot r_t r_h r_w, \, T, \, H, \, W)
\;\xrightarrow{\text{view + permute}}\;
(B, \, C, \, T \cdot r_t, \, H \cdot r_h, \, W \cdot r_w).
$$

Questo sostituisce al 100% `F.interpolate(..., mode='trilinear')`: ogni posizione interna alla patch $2 \times 16 \times 16$ riceve la propria combinazione lineare appresa dei canali dell'encoder.

### 3.2 Inizializzazione ICNR anti-scacchiera

Ogni strato `Conv3d` che precede un `pixel_shuffle_3d` con fattore $F = r_t r_h r_w$ è inizializzato con **ICNR** (*Initialized to Convolution NN Resize*, Aitken et al. 2017): un singolo kernel Kaiming per ciascuno dei $C$ canali di uscita viene replicato identicamente su tutti gli $F$ slot *sub-voxel* (bias a zero). All'epoca 0 il decoder si comporta quindi come un upsampling costante privo di rumore ad alta frequenza a scacchiera, e durante il training i pesi dei sotto-voxel divergono solo dove il gradiente dei bordi reali delle nubi lo richiede.

### 3.3 Blocco di cucitura e mescolamento (`DepthwiseSeparableRes3D`)

Per eliminare le discontinuità lungo i confini della griglia $16\times 16$ (tipiche dei decoder per-patch puri come `Linear(D, 256)`) e reintrodurre il mescolamento tra canali ad alta risoluzione, ogni stadio applica un blocco residuo leggero:

$$
y = x + \text{GELU}\!\left(\text{GroupNorm}_8\!\left(\text{Conv3d}_{1\times 1\times 1}\!\left(\text{DWConv3d}_{3\times 3\times 3}(x)\right)\right)\right).
$$

### 3.4 Flusso multi-scala a 4 stadi

Input: feature dai blocchi Transformer $[f_3, f_6, f_9, f_{12}]$, ciascuno $(B, D, 8, 14, 14)$.

| Stadio | Ingressi | Operazione di spacchettamento | Risoluzione uscita $(T, H, W)$ | Stride spaziale | Canali |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Stage 1** | $\text{cat}(f_{12}, f_9)$ | `Conv 1x1x1 (2D -> 192)` $\to$ `PixelShuffle3D(1, 2, 2)` + Res3D | $(8, 28, 28)$ | 8 px | 96 |
| **Stage 2** | Stage 1 + skip $f_6$ | Stage 1: `PixelShuffle3D(1, 2, 2)` (64 ch)<br>$f_6$: `Conv(D -> 128)` $\to$ `PixelShuffle3D(1, 4, 4)` (32 ch)<br>Fuse `96 -> 64` + Res3D | $(8, 56, 56)$ | 4 px | 64 |
| **Stage 3** | Stage 2 + skip $f_3$ | Stage 2: `PixelShuffle3D(2, 2, 2)` (32 ch)<br>$f_3$: `Conv(D -> 96)` $\to$ `PixelShuffle3D(2, 8, 8)` (16 ch, 128 sub-voxel/tubelet)<br>Fuse `48 -> 48` + Res3D | $(16, 112, 112)$ | 2 px (**$T=16$**) | 48 |
| **Stage 4 (Teste)** | Feature condivise (48 ch) | **CMA:** `Conv(48 -> 4)` $\to$ `PixelShuffle3D(1, 2, 2)` $\to$ $(1, 16, 224, 224)$<br>**CT:** `Conv(48 -> 20)` $\to$ `PixelShuffle3D(1, 2, 2)` + cond. `Conv(1 -> 5)` su $\sigma(\text{CMA})$<br>**RDT:** `Conv(48 -> 20)` $\to$ `PixelShuffle3D(1, 2, 2)` + cond. `Conv(6 -> 5)` su $[\sigma(\text{CMA}), \text{softmax}(\text{CT})]$ | $(16, 224, 224)$ | **1 px** | 1 / 5 / 5 |

### 3.5 Complessità e velocità

- **Parametri addestrabili (encoder congelato):**
  - Su **VideoMAE-Base ($D=768$):** **880.575** parametri (~1.0% del backbone).
  - Su **VideoMAE-Small ($D=384$):** **647.103** parametri.
- **Efficienza computazionale:** poiché la risoluzione piena $(16, 224, 224)$ è raggiunta solo sui logit finali (1 o 5 canali, mai su 128 canali), su CPU `SubPatchThreeHeadProbe` è **~2× più veloce** di `LightweightFPN3D` (~20 s/epoca vs ~30 s/epoca su 12 clip).

---

## 4. Evidenze dal test A/B locale (12 clip train, 15 clip held-out)

Sul micro-dataset locale di Loki (12 clip di training, 6 validation, valutato su 15 clip held-out da Medicane Ianos `2020-09-17` e outbreak `2022-06-28`), il test A/B controllato (stesse feature cached, stesso seed `20261109`, stessa `probe_loss`) ha dimostrato:

1. **Conferma dell'ipotesi sub-patch:** `SubPatchFPN3D` estrae immediatamente bordi frastagliati a scala 2–4 px, tasche di cielo sereno e nubi fredde non convettive (`RDT=1`) che la vecchia FPN trilineare cancellava. Le metriche **CT High Opaque** salgono subito (`0.575 -> 0.599` su Leonardo 10K, `0.576 -> 0.601` su Leonardo 131K) e il divario su **CT Cirrus** rispetto al Random raddoppia (`0.470` vs `0.337`, **+13.3 punti IoU**).
2. **Regime di *data starvation* su 12 clip:** con 880k parametri su sole 12 clip di training, la train loss precipita a `1.37` con overfitting precoce (epoca 41 su 10K, epoca 56 su 131K) e lieve perdita di generalizzazione su regimi rari nel training (es. stratocumuli marini della clip #6).
3. **Implicazione per la fase GPU:** su un dataset reale da **512–2.000+ clip**, il nuovo decoder può esprimere la propria capacità *sub-patch* senza andare in *data starvation*; si consiglia inoltre di valutare una leggera regolarizzazione (`weight_decay=1e-3`, opzionale `Dropout3d(0.1)` nei blocchi di fusione) nel confronto diretto contro `LightweightFPN3D`.
