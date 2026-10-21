Project Specification: RDT-CW Reproduction and VideoMAE Probing
1. Project Objective
The primary objective of this project is to implement a deterministic RDT-CW-inspired reproduction (strictly focusing on convective cells and their thermodynamic evolution) using seven raw-Kelvin SEVIRI channels to generate pseudo-ground truth. IR 10.8 controls full-domain watershed geometry; all six brightness-temperature differences select a coherent anvil plateau for a causal UTLS/anvil temperature proxy v0. It may be described as an exact RDT-CW reproduction only after every threshold, tracking rule, and lifecycle transition has been validated against the authoritative operational specification. The heuristic outputs from this classical physical baseline will serve as dense, voxel-wise pseudo-ground-truth annotations across temporal image sequences.

Subsequently, we leverage a pre-trained VideoMAE (Video Masked Autoencoder) backbone to conduct spatiotemporal semantic segmentation probing. The main teacher and VideoMAE both receive seven channels; a fixed two-channel teacher is retained as a separate baseline. VideoMAE ingests a seven-channel multispectral input cube spanning from WV 6.2 µm to the CO2 absorption band at IR 13.4 µm. By attaching a lightweight 3D Feature Pyramid Network (FPN) probing decoder, we aim to evaluate the capacity of self-supervised 3D representations to implicitly learn and generalize complex convective dynamics, thermal morphology, cloud-top height, microphysics, and rapid temporal evolution.

2. Current RDT-CW-inspired Rules: Multispectral UTLS v0

This section supersedes the original fixed-temperature rules. The explicit
implementation handoff is `HANDOFF_RDT_TEACHER_IMPLEMENTATION.md`.

Phase A: Full-domain detection

- IR108 topology alone receives configurable Gaussian smoothing (sigma 0.7
  pixels); all radiometric diagnostics use unchanged raw Kelvin values.
- Significant regional minima (h-minima prominence 6 K) seed an eight-connected
  watershed. Each tower stops at the discrete level immediately below its
  first neighbouring merge saddle, with 0.5 K levels relative to its minimum.
- A single isolated basin must have a closed thermal rim with relative
  multispectral support. The v0 support heuristic maximizes median rim gradient
  times spectral-coherent fraction times area^0.5. Rim gradient must be at
  least 0.4 K/pixel; coherence means robust six-BTD distance <=4.5 MAD, with a
  0.75 K scale floor. This is not a calibrated ice/cloud classifier. Its
  dependence on surface patterns, view angle and boundary context needs
  validation. Unresolved isolated support is rejected, not capped at a fixed
  brightness temperature. Cells below 100 km² are discarded.
- Invalid observations and invalid geodesic geometry remain masked. Objects
  touching these pixels or the outer domain edge have uncertain history.

Phase B: Proxy and normalized depth

Select an extended connected plateau excluding the coldest decile (retain
quantile ties so a flat anvil is not removed wholesale) and the 1 K edge band.
Prefer gradients below the cell's 60th percentile; reject robust multiband
outliers. When history exists, prefer low Eulerian five-minute changes at the
same observed pixels. Require >=20 pixels and >=200 km². The weighted median
IR108 is T_anvil; reliability weights decrease with gradient and temporal
change and are bounded before geodesic-area weighting.

T_UTLS_proxy_v0 = T_anvil, with zero spectral correction and a three-frame
causal median along the primary track. Save instantaneous and filtered values.
Confidence is a heuristic quality index, not a probability: area, sample size,
thermal/spectral dispersion, proxy jumps, contour support and denominator
separation all contribute. Missing plateau gives NaN, never the cold minimum.

D = (T_edge - IR108) / (T_edge - T_UTLS_proxy).

A denominator below 2 K makes D invalid and prevents OT. The proxy is named
**SEVIRI-derived UTLS/anvil temperature proxy**, not WMO tropopause temperature.
No absolute 240/230/220/210/215 K segmentation/OT cutoff is used.

Phase C: Tracking and lifecycle

Area-weighted AOR >=0.15 links compact object masks, advected using the primary
track's estimated displacement. Simple merges/splits retain the largest
physical-area main branch; all edges are retained. Many-to-many families
reset. This v0 dominance uses detected areas, not operational common-level
redefinition. No global motion field is imposed; orphan local-correlation
recovery from the previous crop-based refinement is not yet included here.
State continues across midnight; actual missing five-minute observations reset
history while track IDs remain unique across the scan.

Cooling and area change compare connected objects THREE intervals earlier:
actual 15-minute differences, not scaled five-minute derivatives. The first
three frames lack tendencies. Split/merge transitions invalidate area change
until three ordinary links are again available; simple primary paths can
retain cooling and convective state. Development requires contrast >=10 K,
cooling <=-2.5 K/15min and positive area change. Convective inheritance makes
maturity/decay reachable without renewed rapid cooling. Phase uncertainty is
kept separate from retained convective state.

3. Semantic Routing and OT

Outside detected cells: 0. Detected unconfirmed/nonconvective cells: 1.
Developing cells: 2. Inherited mature/decaying cells: 3. These are heuristic
pseudo-classes, not independently verified meteorology.

OT overrides only classes 2/3 and requires valid proxy confidence >=0.35,
D>1, >=3 K contrast to the anvil, a local cold nucleus with >=1 K neighbourhood
contrast, WV062-IR108 >=1 K and robust six-BTD coherence (including O3/CO2).
The latter is an outlier guard, not a calibrated O3/CO2 sign model. Components
must satisfy configured physical area (0..1000 km²) and axis ratio <=5;
the first candidate observation is provisional. Class 4 requires two
consecutive spatially associated candidate observations within an ordinary
primary continuation. All these values remain explicit v0 hypotheses.

4. VideoMAE Probing Strategy
To probe whether self-supervised VideoMAE features capture temporal thermodynamics and microphysics, we freeze the encoder and train a lightweight 3D Feature Pyramid Network (3D FPN) for dense volumetric segmentation.
Probing Setup & Multispectral Input Pipeline
Input Tensor Construction: The Zarr archive stores each clip as $(C,T,H,W)$, while the Hugging Face VideoMAE interface receives batched tensors $\mathbf{X} \in \mathbb{R}^{B \times T \times C \times H \times W}$. The patch-embedding layer internally permutes these tensors to $(B,C,T,H,W)$ before applying its 3D convolution.
Channels (C=7): Unlike the heuristic GT generator which only uses 2 channels, the VideoMAE receives a rich multispectral input to implicitly deduce microphysics, optical thickness, and cloud-top height:
WV 6.2 µm (High-level water vapor)
WV 7.3 µm (Mid-level water vapor)
IR 8.7 µm (Cloud phase / microphysics)
IR 9.7 µm (Ozone)
IR 10.8 µm (Cloud top temperature / window channel)
IR 12.0 µm (Cloud optical thickness)
IR 13.4 µm (CO2 absorption / cloud-top height information)
Temporal Window: $T=16$ consecutive RSS timesteps at $\Delta t = 5\text{ min}$, spanning 75 minutes from the first to the last observation. A 15-minute subsampling of the same physical interval is retained as an ablation, not as the primary input.
VideoMAE Input Layer: The patch embedding is `Conv3d(in_channels=7, out_channels=D, kernel_size=(2,16,16), stride=(2,16,16))`. For masked pretraining, each reconstructed tubelet contains $7 \times 2 \times 16^2 = 3584$ scalar values.
Decoder Architecture & Loss
3D FPN Decoder: Uses $1 \times 1 \times 1$ 3D convolutions for lateral connections from intermediate transformer layers (e.g., layers 3, 6, 9, 12), and $3 \times 3 \times 3$ transposed convolutions for top-down spatial-temporal upsampling.
Segmentation Head: Projects to the 5 semantic classes.
Loss Function: A combination of Focal Loss (to handle extreme class imbalance, especially for Class 4 OT pixels) and Soft Dice Loss.

5. Preliminary Atmospheric Pretraining Phase
The primary backbone is a VideoMAE V2 Small with a seven-channel patch embedding. It uses encoder tube masking and independent decoder running-cell masking (50%); reconstruction loss is computed only on decoder-selected targets hidden from the encoder. Prior integration runs used V1, without decoder masking. Pretraining is self-supervised and must not use RDT-CW, EMMA, OPERA, or any other downstream label. Complete 16-frame RSS windows are sampled from the regional SEVIRI archive, and 224 x 224 spatial crops are drawn on the fly rather than materialized as duplicated clip files.

The initial masking strategy is 90% tube masking. A single global mean and standard deviation, computed exclusively over all pixels and all seven channels in the pretraining split, define one affine transform shared by every channel and clip. Per-channel, per-clip, and per-patch normalization are disabled because absolute brightness temperature and inter-channel brightness-temperature differences are physical signals needed by the downstream task. Time reversal, channel permutation, and RGB color jitter are not valid augmentations for this experiment.

Every tensor is oriented north-up and west-left at the Zarr loader boundary. The loader infers any required reversal from the monotonic `x` and `y` coordinates; it does not rely on a hard-coded image flip. Random rotations and spatial reflections are disabled because they would alter the fixed geographic orientation and atmospheric advection geometry.

Four initialization controls are retained:

1. Random seven-channel VideoMAE-Small without atmospheric pretraining.
2. RGB VideoMAE checkpoint adapted from 3 to 7 input channels, without SEVIRI pretraining.
3. RGB checkpoint adapted to seven channels and then continually pre-trained on SEVIRI.
4. Seven-channel VideoMAE-Small pre-trained from scratch on SEVIRI.

For RGB adaptation, the three input kernels are averaged, repeated seven times, and scaled by 3/7. All compatible Transformer weights are retained; the seven-channel patch projection is inflated and the MAE prediction head is initialized anew at 3584 outputs per tubelet. This is a valid initialization strategy but is not described as atmospheric pretraining from scratch.

The storage estimate is based on the local acquisition measured in this project: approximately 97.75 MiB per native L1.5 scene and 5.72 MiB per processed seven-channel regional frame at 30-70 N, 20 W-40 E. One May-September season contains 44,064 nominal RSS scenes. With temporal starts every four frames (20 minutes) and eight random spatial crops per start, it yields approximately 88,104 candidate clips before quality rejection.

| MJJAS seasons | Native transfer | Processed seven-channel Zarr | Candidate clips |
|---:|---:|---:|---:|
| 1 | 4.11 TiB | 0.24 TiB | 88,104 |
| 5 | 20.54 TiB | 1.20 TiB | 440,520 |
| 10 | 41.08 TiB | 2.40 TiB | 881,040 |
| 17 (2008-2024) | 69.83 TiB | 4.09 TiB | 1,497,768 |

The recommended staged design is: one season for engineering validation; five non-adjacent seasons as the minimum credible from-scratch experiment; ten seasons for the main from-scratch result; and the complete 2008-2024 archive only if the ten-season learning curve has not saturated. Native files are processed day by day and may be removed only after Zarr verification, while spatial clips are always generated at training time. Downstream validation and test years must be excluded from self-supervised pretraining to avoid transductive temporal leakage.

6. Required Controls for the Scientific Claim
Success on the deterministic RDT-CW-inspired labels shows that the frozen representation exposes the structures used by that teacher; by itself, it does not prove that the encoder learned atmospheric thermodynamics. The main comparison must therefore include a random frozen encoder, an adapted RGB encoder, a SEVIRI continually pre-trained encoder, and a SEVIRI-from-scratch encoder under the same decoder and split.

Temporal and spectral controls are mandatory: shuffled frame order, a repeated single-frame input, the two teacher channels alone, all seven channels, and all channels except IR 13.4 µm. These experiments test whether performance comes from temporal evolution, added spectral information, or a static IR 10.8 threshold shortcut. At least one evaluation independent of the RDT pseudo-labels (for example OPERA radar evolution or the existing GPM/CloudSat probes) is required before claiming that the backbone learned convective dynamics or thermodynamic state.

7. Exploratory Python Implementation (12 October 2026)

`src/emma_gpm/rdt_teacher.py` and
`notebooks/visualize_seviri_rdt_rules.ipynb` implement a deterministic preview,
not a validated operational RDT-CW teacher. No model is trained or evaluated
against these labels yet; RDT-based example selection is diagnostic only and
must never feed self-supervised pretraining manifests.

- Four fixed threshold levels are segmented with eight-connectivity and the
  100 km² minimum area. Disjoint 240 K envelopes define target objects;
  colder levels show nested cores, without automatically dividing an envelope
  into multiple convective cells or claiming operational adaptive thresholds.
- Physical area uses ellipsoidal quadrilateral pixel footprints inferred from
  projected grid centres/corners and the actual Zarr projection. AOR uses
  physical intersection divided by the smaller physical object area.
- One-to-one links retain IDs. Split/merge events create new IDs, preserve all
  parent edges and reset derivatives/state; no motion compensation is applied.
- Valid non-border one-to-one links use five-minute minimum-temperature and
  area differences, multiplied by three for K/15min and km²/15min. This is
  not an independent 15-minute estimate or an advection-corrected tendency.
- Developing objects require contrast, cooling and growth thresholds together.
  Previously convective objects with valid history and sufficient contrast
  may become mature/decaying without continued positive growth. This resolves
  the provisional lifecycle rule, pending validation against the specification.
- OT is a pixel override only inside confirmed-by-rule convective objects.
  New objects, ambiguous links and crop-edge objects lack valid temporal
  classification; they remain cold/unconfirmed and have a separate uncertainty
  mask. Class 1 therefore does not prove absence of convection.

The player loads 15 minutes of prior context plus the 16 displayed frames,
rejecting missing/nonfinite context rather than padding. It displays cold
envelopes, nested cores, object IDs, pseudo-classes, uncertainty and rule
metrics with physical units. A seeded month-balanced catalog of 48 candidate
crops is saved locally, including rejections and selection provenance; the
ordering by candidate activity is for visual inspection, not prevalence
estimation. Export is optional and disabled by default. Production labels
will require tracking on a wider domain before cropping, validation of
thresholds and transitions, and a separate probing split protocol.

8. Larger Diagnostic Dataset

`scripts/build_seviri_rdt_dataset.py --count 8192 --workers 8` prepares a
month-balanced diagnostic manifest from the published archive, then applies
the same teacher thresholds in parallel. A random-offset 224-pixel lattice
has 21 spatial crops per complete window on the current 763×1762 domain.
Windows are separated by 95 minutes, including three preceding context
frames; no crops in the pool share pixels at common timestamps. The current
snapshot yields 4,779 windows and 100,359 geometric candidate crops before
pixel-quality rejection. Eight thousand clips are an initial sample of this
pool, not 8,192 independent weather systems.

Output under `results/rdt_preview/annual8192/`: `manifest.csv`, `protocol.json`,
`inventory.csv`, native-grid physical pixel areas, per-clip compressed masks
(classes, tracks, uncertainty), object/lineage tables and checkpointed records.
`catalog.csv` and `progress.json` publish every 128 completed candidates;
`summary.json` is written at completion. A process lock prevents duplicate
scans. Resume requires compatible source hashes and snapshot identity; cached
mask files are checksum-verified. No raw clip materialization is performed.

Metrics distinguish clips containing candidates, local candidate track IDs,
mean occupancy over all 16 frames, maximum per-frame occupancy, physical
peak area, OT occupancy and temporal uncertainty. The known-cold denominator
is also saved. Neither occupancy nor clip frequency is detection confidence;
month-balanced frequencies conditional on finite sampled crops are not annual
meteorological prevalence. Sampling more clips finds more examples without
making the teacher more accurate. Cold-envelope border censorship and
split/merge resets can leave much of the cloud area temporally unassessed;
retain the uncertainty mask and do not treat such pixels as confirmed
nonconvective negatives in a future probing loss.

The notebook now reads the larger scan (including partial results) rather
than rebuilding 48 examples. Default player has only IR108, WV062 and classes;
contours, IDs, criteria and track evolution are separate views. Explanations
are kept in a closed guide and this document. Sorting by peak occupancy
surfaces more visible examples, without altering the scan distribution or
teacher thresholds. Wider-domain tracking before cropping is still required
before producing scientifically validated pseudo-labels.

9. SEVIRI-only Refinement (13 October 2026)

`rdt_refined.py` implements a bounded refinement following the
[NWC SAF ATBD v1.1, sections 3.2.1.2.1–2](https://www.nwcsaf.org/AemetWebContents/ScientificDocumentation/Documentation/GEO/v2016/NWC-CDOP2-GEO-MFT-SCI-ATBD-Convection_v1.1.pdf):
adaptive contours for significant thermal towers, movement-aware overlap,
correlation recovery and a main trajectory through simple splits/merges.
Many-to-many families still reset their trajectories. Weighted centres of
linked families estimate movement; the mask enlargement recovery applies
only to tiny orphan objects and never changes satellite pixels or targets.
This does not implement the full operational statistical discrimination,
cloud-type filtering, all contour refinements or the NWP/HRW motion field.

No NWP, lightning or external labels are used. The two input channels remain
WV062 and IR108. Local explicit choices: 240 K warm cap retained for the
project's target definition, 198 K cold limit, 1 K level step, 6 K tower
depth, base/top physical-area ratio cap 4, motion search ±12 pixels and
one-pixel association-only recovery for objects of at most five pixels.
Temporal motion smoothing is fixed at 0.5. These numerical choices remain
part of a reproducible RDT-CW-inspired approximation, not an operational
equivalence claim. The original contrast/cooling/growth/OT thresholds stay
unchanged; changed contours also change their measured values.

The teacher works on a square context with a requested 64-pixel margin,
clamped at the archive boundary, then crops outputs to the original 224×224.
This reduces artificial crop-edge truncation without claiming domain-wide
tracking. Nonfinite expanded context is rejected, never filled; paired
comparisons therefore use only clips valid in both versions. Context halos
may overlap between neighbouring crops; the central input samples retain
the original nonoverlapping manifest. Areas/temperatures in object tables
refer to full context objects; `visible_area_km2` records their central part.

Simple primary branches preserve identity and cooling. Their area jumps
are not automatically interpreted as growth: `growth_valid=False` and the
growth derivative is NaN on split/merge transitions. Convective inheritance
uses the primary parent only; `phase_valid`/`phase_uncertain` distinguish
retained convective state from an assessed phase. Keep those flags for future
probing loss masks. Growth for ordinary links compares physical contours
at a common thermal threshold.

Rerun the **same** manifest with:

```bash
.venv/bin/python scripts/build_seviri_rdt_dataset.py \
  --teacher refined_v2 --halo 64 --count 8192 --workers 8 \
  --manifest-from results/rdt_preview/annual8192 \
  --output-dir results/rdt_preview/annual8192_refined
```

The baseline stays in `annual8192`; new masks and records are separate.
At completion, `paired_comparison.csv/json` summarize common-valid clips,
single-point track fraction, visible track length, temporal evaluability
and candidate occupancy. They measure continuity, not detection accuracy.
The notebook retains its layout and reads the refined directory; backend
version binding prevents a refined catalog from displaying legacy masks.

10. Full-domain incremental implementation (13 October 2026)

`src/emma_gpm/rdt/` implements the updated handoff. The legacy fixed teacher
and the earlier 64-pixel-halo refinement remain separate local baselines.
The primary job is `scripts/build_rdt_pseudolabels.py`: framewise full-domain
analysis, compact per-object masks, causal state, atomic publication of each
frame directory, Zarr diagnostic fields, object/lineage CSV tables and two
latest state checkpoints. Chunk/table checksums, configuration/source hashes,
metadata snapshots and the last raw-input hash guard resume. Existing source
archives are assumed immutable; the last-frame input check does not verify
all historical raw chunks against replacement with identical metadata.

```bash
.venv/bin/python -m pip install -r requirements-rdt.txt
.venv/bin/python scripts/build_rdt_pseudolabels.py --start-date 2020-07-11
# Same output directory/config/code; extend rather than rerun:
.venv/bin/python scripts/build_rdt_pseudolabels.py --start-date 2020-07-11 --end-date 2020-07-17
```

Default output: `results/rdt_preview/domain_utls_july2020/`. Three preceding
frames are processed when available; no frames are invented when absent.
`progress.json` and `frame_metrics.csv` update after each committed frame.
Complete 16-frame windows on a 224-pixel lattice publish `catalog.csv`
incrementally. This reticle is diagnostic, not a pretraining manifest. The
notebook loads the original finite seven-channel crop and a crop of the
PRECOMPUTED domain labels: it never recalculates contours/proxies per clip.
Native-index origins are mapped to north-up/west-left label coordinates.
Global track IDs do not restart per clip; cell component IDs are per frame.

The two-channel baseline is preserved and callable via the public API mode
`legacy_fixed_2ch`; no additional comparative scan is scheduled in this
stage, following the user's request to focus on the adaptive teacher.
The proxy/isolated support must be inspected before scaling to the year or
using these pseudo-labels for frozen-backbone probing. No NWP, radar,
lightning or external pseudo-labels are used. No pretraining behavior changes.

11. Whole July diagnostic scan (13 October 2026)

The 11 July proof-of-integration finished in 1,737.84 seconds (~29 minutes).
The month-wide job uses the identical teacher configuration and scientific
sources on 1–31 July, starting a separate causal trajectory from the beginning
of the month. The daily experiment remains unchanged; its labels cannot be
spliced into the monthly trajectory because their earlier context and global
IDs differ.

```bash
.venv/bin/python scripts/build_rdt_pseudolabels.py \
  --start-date 2020-07-01 --end-date 2020-07-31 \
  --output-dir results/rdt_preview/domain_utls_july2020_month
```

Available July observations: 8,922 of 8,928 expected, across all 31 days.
22/23/24 July each have two missing slots; no filling is performed. Up to
three preceding June frames provide warmup, outside the July metrics.
Progress, logs, checkpoints and partial crop catalog are in the monthly
output directory; the notebook now reads that directory by default.
The runtime estimate from the daily test is 14–16 hours, subject to changing
cell density, storage throughput and final table assembly. Resume uses the
same command. No algorithm, masking, or pretraining manifest is changed.

12. Read-only full-domain day player

The final cells in `visualize_seviri_rdt_rules.ipynb` use
`rdt_domain_dashboard.py` (outside the running teacher's source package).
Default source is the completed 11 July experiment: 288 native five-minute
frames across 763×1762 pixels. The monthly archive is a separate selectable
source, with committed days/frame counts refreshed on demand. Frames are
loaded individually; a two-frame cache retains only display bands and selected
saved diagnostic maps. No teacher is rerun and no full-day radiance cube is
materialized. Coordinate-derived orientation and geometry fingerprints bind
raw satellite and label maps; missing slots remain explicit gaps.

The synchronized Play/slider shows satellite and classes/contours/plateau/
proxy/D/confidence/OT across the same entire projected domain. Geographic
reference lines and uncertain-phase stripes are optional. The day overview
reads existing frame metrics (convective objects and persistent OT pixels),
with direct jumps to their peaks. These are per-frame candidate counts, not
independent event counts or detection confidence. Playback strides of 15 or
30 minutes only skip displayed slots; scientific data cadence stays five
minutes. The monthly scan, algorithms and source fingerprints are unchanged.

13. Radiometric quality-mask recovery

The monthly run stopped before 2020-07-05 23:05 UTC: two otherwise finite
WV073 observations were -0.434326171875 K. The old global Kelvin sanity guard
rejected the complete domain and described the error as normalized input.
1,432 committed frames through 23:00 remained intact.

The same existing [100,400] K sanity limits now contribute to the per-pixel
`valid_observation` mask in domain mode, together with finite seven-channel
observations and physical geometry. `invalid_radiometry` records finite,
Earth-visible pixels failing the limits. Their raw radiances are unchanged;
they receive no object/OT/depth assignment, and uncertainty is explicit.
Strict clip mode still rejects ANY invalid observation. Globally out-of-range
input still raises, so normalized inputs are not accepted silently. The
monthly summary reports the number of rejected radiometric pixel samples.
These are data-quality limits, not new cloud/convection thresholds.

Source compatibility was audited against the previous 22:55 checkpoint:
old/new code produces exactly equal common fields, object tables and links
for the good 23:00 frame; the 23:05 frame reproduces the old failure and the
new domain code excludes exactly two pixels without altering raw values.
Every previously committed frame already passed the old global range guard,
so its existing valid-observation mask and scientific outputs remain valid.
Old/new source fingerprints, original protocol/status and input details are
retained under `recovery/radiometry_20200705T2305/`. The source-pinned
`--quality-mask-upgrade` option allows only this documented compatibility
transition; changed configuration/geometry/other sources still fail resume.
No previous label chunks are overwritten or regenerated. Earlier frames
have an identically-false radiometric-invalid flag derived at crop-read time.

Checkpoint verification has a distinct progress state; error reports retain
the actual requested frame count instead of resetting it to zero. The
pretraining loader, normalization and source Zarr products are unchanged.

## 14. Scansione annuale parallela del dominio (14 ottobre 2026)

`build_rdt_year_parallel.py` congela l’inventario e divide **solo ai gap reali**
delle osservazioni RSS. Dentro ciascuna sequenza il teacher resta seriale e
causale, anche attraversando la mezzanotte. Le soglie e le regole fisiche non
cambiano. Ogni processo usa un thread numerico; il limite iniziale è 20 processi
sui 20 core fisici. I 40 thread logici non sono automaticamente più veloci.

Lo snapshot corrente contiene 90.960 frame su 317 giorni, dal 1 gennaio al
9 dicembre 2020, divisi in 20 sequenze continue. Non è un archivio annuale
completo: restano gap e la parte finale di dicembre non è scaricata. Una
sequenza lunga impone un limite alla scalabilità; i processi attivi diminuiranno
quando le altre sequenze termineranno. L’ultimo timestamp pubblicato non
indica che tutti i giorni precedenti siano già completi.

I campi mantengono gli stessi chunk Zarr v2 compressi senza perdita, raccolti
in un solo `fields.zarr.zip` per frame. I lettori accettano anche le vecchie
directory Zarr. Oggetti e lineage restano partizionati per frame per evitare
una concatenazione annuale in RAM. Directory temporanee e rename atomici
proteggono ogni commit; due checkpoint locali per sequenza consentono la
ripresa. Gli ID sono int64: `(sequence_id+1)*2**32 + local_track_id`, univoci
anche tra processi, stabili al variare del numero di worker. Sono identificatori,
non conteggi di eventi. Le clip diagnostiche vengono ritagliate dopo il calcolo;
non influenzano i manifest SSL.

```bash
.venv/bin/python scripts/build_rdt_year_parallel.py --year 2020 --workers 20
```

Output: `results/rdt_preview/domain_utls_2020_parallel/`. Il notebook RDT usa
questa scansione nel monitor e la aggiunge al player dell’intero dominio,
conservando le prove di luglio. `progress.json` riporta attività e throughput;
`sequence_status.json` mostra ciascuna sequenza; `parallel_plan.json` conserva
inventario, partizioni e hash del codice; `scan.log` e `segments/NNN/training.log`
riportano gli errori. Lo stesso comando riprende i processi incompleti. Non
modificare i sorgenti del teacher durante il job: una ripresa con fingerprint
differente viene rifiutata. I benchmark con limiti di frame rimangono marcati
`partial` e separati dai risultati annuali. Le ottimizzazioni delle statistiche
BTD e dell’associazione sono state confrontate su quattro frame reali: campi,
oggetti e lineage identici, accelerazione del solo calcolo 1,38×.

Benchmark end-to-end sugli stessi 40 frame (due per ciascuna delle 20 sequenze),
comprensivo di avvio dei processi e scrittura: 243,71 s con un processo,
42,31 s con otto, 19,08 s con venti. Tutti i chunk Zarr e le tabelle degli
oggetti/collegamenti sono identici fra le tre prove. Il test breve non misura
il rallentamento finale quando resta una sola sequenza né il costo dei cataloghi
di clip, pubblicati dopo 16 frame e poi ogni 288 frame per processo. Non
estrapolare un’accelerazione annuale lineare di 12,8×. I file persistenti del
benchmark occupano mediamente 7,16 MB per frame: circa 606 GiB sull’inventario
corrente, oltre ai pochi checkpoint mantenuti e alle tabelle diagnostiche.
Il PC ha oltre 2 TiB liberi; ciascun worker si ferma sotto 50 GiB senza eliminare
dati. I report dettagliati sono in `results/rdt_preview/parallel_benchmark/`.

La lettura CSV tratta gli hash come testo: è stato riprodotto e corretto un
crash del parser nativo pandas provocato da un hash con prefisso numerico
`7e252472...`, interpretato erroneamente come esponente. La regressione è
coperta da un test, senza cambiare alcuna regola del teacher.

## 15. Ricerca annuale e zoom di studio (14 ottobre 2026)

`index_rdt_domain_events.py --label-root OUTPUT --workers 4 --follow` costruisce
un indice separato, leggendo soltanto i frame già pubblicati. Non modifica
teacher, checkpoint, soglie o manifest SSL. L’indice SQLite atomico conserva
percentuali di pixel validi convettivi (classi 2/3/4) e OT (classe 4), più il
picco per traccia: area km² delle celle convettive o pixel OT persistenti.
Per lo studio si escludono gli oggetti al bordo e senza supporto della proxy;
la convezione richiede anche storia e fase valutabili. I valori sono
candidati del teacher, non una misura validata dell’intensità meteorologica.
Split/merge possono generare più tracce dello stesso sistema atmosferico.

Il viewer del dominio offre ricerca giornaliera o su tutte le giornate
calcolate dell’archivio scelto, menu dei picchi percentuali separati di almeno
90 minuti e zoom 224×224 con 16 frame consecutivi attorno al picco. Il crop
è fisso nello spazio, centrato sulla traccia al picco e può tagliare sistemi
più estesi. Gap e osservazioni invalide vengono rifiutati, mai sostituiti.
Durante il job il ranking resta provvisorio, con copertura indicizzata
visibile. Usare Rileggi giornate per aggiornare le graduatorie. L’indice
segue il teacher e termina quando tutti i frame pubblicati sono indicizzati.
Le prove precedenti di luglio conservano i propri risultati separati.

Le clip del catalogo generale sono su reticolo 224×224 senza overlap, con
finestre di 16 frame a passo 80 minuti ancorate all’inizio di ogni sequenza
continua. Non sono il random sampling del pretraining. Gli zoom selezionati
sulle pseudo-classi sono destinati esclusivamente allo studio, mai al SSL.

La BTD WV6.2−IR10.8 ha scala grafica fissa −80/+10 K con bianco a zero,
blu negativo e rosso positivo (`TwoSlopeNorm`, lineare separatamente sui
lati dello zero). Il precedente −40/+10 K era un limite grafico, non una
statistica dell’archivio. Un campione con seed 20261014, tre frame interi
per mese e una clip casuale completa per mese (36 frame e 12 clip), mostra
mediane −41,25 K sul dominio e −42,125 K sui crop; positivi rispettivamente
0,201% e 0,801%. Il precedente minimo grafico saturava oltre metà dei pixel
campionati. Questi risultati sono campionari, non estremi o frequenze
esaustive dell’anno. `sample_seviri_btd_statistics.py` riproduce la misura;
report e indici sono in `OUTPUT/exploration/`, esclusi da Git.

## 16. Solo dominio completo; crop su richiesta (14 ottobre 2026)

Su richiesta dell’utente il catalogo automatico su reticolo è disattivato:
`build_rdt_year_parallel.py` usa ora il dominio completo come default. Il flag
`--crop-catalog` riattiva esplicitamente la diagnostica storica; il runner
seriale supporta `--no-crop-catalog`. Non vengono più lette e concatenate le
vecchie tabelle di crop nel supervisore annuale. I file precedenti restano
conservati. Rilevamento, proxy, tracking, classi e salvataggi full-domain non
cambiano. Zoom e indice degli eventi restano disponibili separatamente.

Il job è stato fermato e ripreso dai checkpoint, conservando 30.249 frame
committati. La migrazione documentata in `recovery/disable_catalog/` conserva
sorgenti precedenti, protocolli, hash e conteggi per sequenza; soltanto i due
script di orchestrazione sono cambiati, senza modificare i sorgenti scientifici,
i dati o le soglie. Ogni worker riprende dopo la verifica dei checksum locali.
Il test del runner senza catalogo verifica 16 frame full-domain, assenza di
chiamate al campionamento e ripresa senza modifiche ai frame già salvati.

Gli zoom sono strumenti di ispezione e selezione, non costituiscono da soli
un benchmark di probing. Il probing principale dovrà includere anche sfondi
e nubi non convettive, split temporali indipendenti e backbone congelato.
La scelta degli eventi tramite il teacher non deve influenzare i manifest SSL.

All’arresto il producer RDT occupava circa 5,9 GiB e l’indice circa 0,9 GiB.
Il consumo principale della macchina era un kernel Spyder preesistente
(circa 127 GiB), lasciato attivo perché esterno a questi job. Il riavvio RDT
non elimina lo stato di altre applicazioni né forza lo svuotamento della
cache del sistema operativo.

## 17. Ripresa su cella con supporto 1×20 pixel (15 ottobre 2026)

La scansione si è fermata a 69.245 frame: nel frame 7 settembre 2020 00:00 UTC,
una cella aveva supporto locale di una sola riga e venti colonne. Superava
il minimo di venti pixel, ma `np.gradient` richiede almeno due campioni per
ciascun asse. `estimate_proxy` ora restituisce esplicitamente
`insufficient_spatial_support`, proxy NaN e confidenza zero per queste forme,
senza introdurre gradienti nulli, temperature sostitutive o frame saltati.
Rilevamento e tracking della cella continuano; la proxy non è valutabile.

Due regressioni coprono supporti 1×24 e 24×1. Il caso reale è stato riprodotto;
il precedente frame 6 settembre 23:55 produce campi, oggetti e lineage identici
prima/dopo la correzione. I tre frame 7 settembre 00:00/00:05/00:10 completano
il calcolo; soltanto il primo contiene una cella con questa astensione.
La suite mirata (multispettrale, storage e controllo catalogo) passa 14 test.

La migrazione in `recovery/thin_proxy/` conserva i sorgenti/protocolli precedenti,
il report del caso e la verifica di compatibilità: tutti i frame già pubblicati
avevano completato il vecchio calcolo del gradiente, quindi non ricadono nel
nuovo ramo, posto dopo il controllo preesistente del numero minimo di pixel.
Solo `utls_proxy.py` è cambiato; soglie, dati e configurazione restano invariati.
Il job riprende i tre segmenti incompleti dopo verifica dei checkpoint,
conservando i 17 completati e il catalogo dei crop disattivato.

## 18. Risorse nella coda finale (15 ottobre 2026)

Con 17 sequenze completate rimangono tre task seriali causali: aumentare il
limite globale da 20 non crea ulteriori sequenze indipendenti. Il monitor
separa ora numero di processi, sequenze incomplete e limite configurato.
`frames_per_second` del producer è una media storica di tutte le fasi;
il notebook misura anche il rapporto tra nuovi frame e tempo tra due letture,
con intervallo minimo di 30 secondi. Non è la velocità di un singolo worker.

Un profilo reale intorno al 12 settembre mostra il peso di detection e proxy
per oggetto. Il benchmark isolato di scrittura confronta concorrenza Zarr
2/8/16, pool 1/4/8 e Blosc 1/4 thread. I chunk compressi risultano identici,
ma i miglioramenti temporali non sono stabili nelle ripetizioni: non vengono
applicati al job corrente. Il disco dati è meccanico, ma nel campione osservato
non risultava saturo. Una pipeline che prepari il rilevamento dei frame futuri
indipendentemente e mantenga seriali associazione e storia temporale è un
possibile intervento successivo, da misurare end-to-end e verificare rispetto
alla versione seriale. Non spezzare arbitrariamente le sequenze per occupare
più core. Report locali in `recovery/resource_benchmark/`.

## 19. Pipeline della coda annuale (15 ottobre 2026)

Le tre sequenze ancora incomplete possono usare `--detector-workers 2
--async-save` nel supervisor annuale. Ogni sequenza mantiene un solo teacher
causale: solo il rilevamento indipendente sul frame corrente viene preparato
in due processi aggiuntivi. I risultati sono consumati in ordine FIFO, con al
massimo due task per detector. Non si introducono confini artificiali o frame
futuri nella storia delle celle. Input, geometria e configurazione del risultato
preparato sono verificati prima di usarlo; il risultato è monouso.

Un solo thread salva il frame precedente mentre il teacher calcola quello
successivo. Il checkpoint viene serializzato prima che lo stato vivo avanzi;
la pubblicazione rimane atomica e ordinata. Al massimo un salvataggio è in
corso. Un errore nel successivo input non perde il commit precedente. La
ripresa verifica ancora tutti i checksum degli output già pubblicati.

Su 12 frame reali consecutivi del 13 settembre, dallo stesso checkpoint,
lettura, calcolo e salvataggio richiedono 55,90 s in seriale e 37,79 s con la
pipeline: throughput 1,48 volte maggiore, tempo ridotto del 32%. Tutti i chunk
Zarr compressi, le tabelle di oggetti e collegamenti e lo stato finale del
tracking coincidono. Il risultato riguarda questo campione e il carico del PC
al momento della misura; la velocità dell'intero anno dipende anche dalle celle
presenti e dalla concorrenza con la conversione dei download. Report e migrazione
con provenienza dei checkpoint in `recovery/pipelining/`.

I flag sono disattivati per default: due detector **per sequenza** sono utili
nella coda di tre task, mentre venti sequenze con due detector ciascuna possono
sovraccaricare il PC. Il monitor distingue worker di tracking e processi extra
di rilevamento. Per valutare il guadagno usare frame/s su intervalli recenti;
la durata di un singolo salvataggio e la media storica non misurano il throughput
di una pipeline sovrapposta. I test coprono uguaglianza di campi/tabelle e ID,
gap, input invalidi, checkpoint ordinati, errore della sorgente, CLI e ripresa.

Per riprendere la stessa coda dopo un arresto, mantenere i flag di esecuzione:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python scripts/build_rdt_year_parallel.py \
  --year 2020 --workers 20 \
  --output-dir results/rdt_preview/domain_utls_2020_parallel \
  --detector-workers 2 --async-save
```

Il comando attivo e il PID vengono conservati nel `launcher.json` locale.
Non lanciare una seconda istanza mentre il supervisor è già attivo.

## 20. Zoom di studio vicino al bordo osservabile (17 ottobre 2026)

La traccia 68721044871 al picco del 18 settembre 2020 13:50 UTC genera un
crop centrato che include il bordo curvo fuori disco: 9.656 pixel non osservati
su 50.176 in ciascuno dei 16 frame 13:10–14:25, NaN in tutti i sette canali.
Il dominio completo li maschera; il lettore rigoroso delle clip li rifiuta.

Il solo viewer di studio interseca ora le mappe `valid_observation` salvate
nei 16 timestamp effettivi e cerca il crop 224×224 interamente valido più
vicino al crop centrato, contenente pixel della traccia al picco. La ricerca
è deterministica: distanza quadratica fra origini north-up, parità risolte
per riga e colonna. Le coordinate vengono poi convertite negli indici nativi.
Lo spostamento è dichiarato nel viewer; una cella estesa può restare tagliata.
Se non esiste un crop valido contenente la traccia, lo zoom resta rifiutato.

Nel caso verificato l'origine nativa passa da (539,1302) a (470,1275): tutti
i valori e le maschere dei 16 frame sono validi e 1.230 pixel della traccia
sono visibili al picco. Non si riempiono NaN, non si cambiano frame, maschere
o teacher, né il lettore delle clip o il campionamento del pretraining.
