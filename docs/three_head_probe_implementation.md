# Three-head SEVIRI probing

This implementation follows `three_heads_probing.md`, including the exploratory
revision approved on 9 November 2026. The frozen-encoder comparison and sequential
fine-tuning experiments have separate records in
`configs/three_head_probe_experiments.json`: 21 runs, 60 epochs each, batch size 16.
All experiments share the downstream dataset, training-only scalar normalization,
validation selection and held-out test observations.

## Approved exploratory dataset

The active plan is `configs/three_head_probe_dataset_exploratory2022.json`.
Training uses 1–2 July 2022 (512 clips), validation 4 July (192 clips), and test
7 July (192 clips). Dates touched by pretraining or pretraining validation are
excluded. At least one unused day separates successive downstream periods.
The full 224 × 224 crop lies within 20°W–40°E, 30°N–70°N, with origins selected
at individual-pixel resolution. Each input has sixteen five-minute frames and
all seven SEVIRI IR channels. One scalar mean and standard deviation are fitted
on all pixels and channels of the downstream training clips.

The following approved criteria govern publication:

- At least 99.5% of observed RDT convective pixels must lie inside CMA clouds.
- Fewer than 0.1% must belong to CT clear-sky or low-cloud classes.
- The fraction in CT high opaque clouds is reported as a diagnostic, with the
  original 90% reference retained. July gives 87.1%, 86.9%, and 84.1% in the
  training, validation and test periods. Both required checks pass in all three.

These fractions use full-domain observations after warm-up, before cell-history
exclusions. The manifest preserves the original `teacher_quality_gates` and
stores a separate `teacher_acceptance` under the explicit
`ct_high_opaque_diagnostic` policy. The original strict gate remains available.
The teacher's pixel labels retain their corrected v3 definitions.

Training samples have 50%, 30%, and 20% probability mass in groups A, B, and C.
Group B requires at least 25% high opaque/cirrus CT pixels and 1% cold
non-convective RDT pixels among valid targets. Under `cold_cloud_centre`, a
mixed scene qualifies if its last-frame central pixel is RDT cold non-convective
and CT high opaque/cirrus. Scenes completely free of active convection also
qualify when they meet the two coverage fractions. Group B takes priority over
A; the remaining convective clips belong to A. Group C contains remaining clips
with less than 50% high-cloud coverage. OT clips receive triple weight only
within A. Segmentation targets retain all pixel classes, including convection
elsewhere in B clips. Validation and test use uniform spatial/temporal candidates.
The manifest records sampling roles, centre labels, coverage, support and input
rejections. Counts of overlapping crops describe correlated observations.

## Teacher correction and verification

`scripts/prepare_three_head_probes.py` runs CMA/CT and the Pure-GEN/UTLS RDT
teacher on complete, north-up and west-left satellite frames before cropping.
State continues across midnight and resets at a genuine cadence gap. The first
three observations provide causal warm-up. A detected cell contributes targets
when its trajectory contains four exact five-minute timestamps and the teacher
computes a finite temperature change from the actual fifteen-minute reference.
Background pixels require four valid observations. The resulting common validity
mask applies to all heads; native quality flags remain stored for diagnostics.

The corrected RDT version,
`rdt_cw_inspired_pure_gen_utls_v3_observation_support`, distinguishes observed
radiometry from the cloud prefilter selecting cell candidates. Finite observed
pixels outside the prefilter contribute to spatial gradients. Missing observations
and stretched satellite pixels retain their exclusion flags. Tracking events
are retained so continuous trajectories can support fifteen-minute area changes.
Passing `separate_observation_support=False` reproduces legacy v2 behavior and
its configuration hash; existing serialized v2 trackers retain their behavior.

Tests verify detected contours, actual missing-data flags, version identity and
track history. The complete July comparison covers 1,152 full-domain frames:
cold-cell footprints and CT maps match exactly between versions. Common-mask
cold-cell target coverage increases from 2.5% to 68.6% for training, 2.2% to 76.4%
for validation and 3.6% to 71.9% for test. The previous preparation used native
quality flags; the revised preparation verifies the actual fifteen-minute
trajectory. `scripts/verify_three_head_teacher_revision.py` records the paired
class transitions, coverage and geometry checks during preparation.

`scripts/audit_three_head_sampling.py --check-inputs` checks all seven channels
and sixteen frames for finite radiometry in 100–400 K. The July training audit
covers 501,046 spatial origins in each of 35 time windows. It finds zero crops
meeting the original convection-free group-B definition and 21,911 complete
cold-centred candidates under the approved revision. These are correlated
position/window combinations. The active sampler draws a reproducible random
subset from this pool using 8,192 candidate origins per window.

Earlier diagnostics remain available for February, April, May and autumn 2021.
All nine corrected-teacher periods examined fall below the original 90% CT
reference. February also fails the retained clear/low criterion (0.51%). April
has 11,337 complete strict group-B candidates; its alternative plan remains an
archived proposal. The active exploratory experiments use the approved July
plan. Phase-conditioned plots explain CT agreement by native RDT phase; OT
statistics have particularly small support, including repeated tracked pixels.

Native CT classes 1–4, 5/6/10, 7, 8/9 and 11–15 map to the five requested regimes.
Native CT class 0 and fill 255 are excluded. Native RDT classes 0, 1, 2/3, 4/5
and 6 map to background, cold non-convective cloud, developing convection,
mature/decaying convection, and persistent overshooting top.

GPU integration restores the original MVM decoder strictly and verifies gradients
in the last four encoder blocks during cumulative transfer. A separate 64-update
one-clip integration reduces joint loss from 4.43 to 1.56 with an unchanged frozen
encoder. These checks establish executable integration before the full probe runs.

## Encoder and decoder

The main encoder is Small MVM pretrained on 65,536 clips. Source weights are
selected by the original SSL validation loss. The learned weights load strictly;
fixed sinusoidal positions are rebuilt from ten context frames to sixteen probe
frames. Small Future/Joint/Residual and Leonardo Base Future 131,072 provide
additional three-head transfer comparisons on identical downstream targets.

The FPN combines blocks 3/6/9/12 with width 128, lateral pointwise projections,
depthwise 3D refinement and progressive interpolation. CMA probabilities
condition CT; CMA and CT probabilities condition RDT. Pointwise projection is
performed before the final linear upsampling, preserving the full-resolution
prediction function while reducing large intermediate allocations. Tests compare
this computation against explicit full-resolution shared-feature construction.

The RGB control uses `MCG-NJU/videomae-small-finetuned-kinetics`: its provenance
includes natural-video MAE pretraining and supervised Kinetics fine-tuning.
The three-channel projection is averaged, repeated over seven channels and
scaled by 3/7. The random control uses the matched Small architecture.

## Experiment matrix and selection

- E0-Rand and E0-RGB: frozen encoders, shared FPN and three conditioned heads.
- E1-Single-cma/ct/rdt: three independently initialized single-task decoders.
- E1-Joint3: frozen SSL encoder and the joint conditioned decoder.
- E2-selective and E2-cumulative: adapt the last four encoder blocks to CMA,
  then freeze and train fresh CT and RDT decoders.
- E3-selective and E3-cumulative: adapt the corresponding CMA encoder to CT,
  then freeze and train a fresh RDT decoder.
- E4-JointFT: jointly fine-tune all encoder blocks and the three heads.

Selective adaptation uses encoder/head learning rates 0.00001/0.001 and a
coefficient-one mean squared displacement penalty from the original SSL encoder.
Cumulative CMA adaptation retains the original masked reconstruction decoder
with coefficient 0.1, 90% tube masking and 50% decoder masking. Cumulative CT
adaptation retains the learned CMA head and FPN. Both sequential variants unfreeze
the same four encoder blocks, making their retention strategy explicit.

CMA combines BCE and Dice; CT combines focal CE (gamma 2) and Dice; RDT combines
training-frequency-weighted focal CE and generalized Dice. Active task losses
have unit coefficients. The joint nesting penalty has coefficient 0.1. All
terms exclude target value 255. AdamW, bf16, gradient clipping and a cosine
learning-rate schedule are used consistently. Checkpoint selection uses the
validation task loss. Test metrics are evaluated at that selected checkpoint.

Saved metrics include pixel-pooled class CSI/IoU, F1, confusion counts, support,
multiclass Brier and binary active-convection Brier/CSI. Temporal permutation,
repetition of the last frame, two spectral subsets and removal of IR 13.4 are
evaluation-only interventions. Their results describe sensitivity including
input distribution shift. The notebook states the limited weather-period
coverage and distinguishes teacher agreement from independent physical evidence.

## Execution and monitoring

From the repository root, use its virtual environment:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_three_head_probes.py \
  --plan configs/three_head_probe_dataset_summer2022.json --split train
# Run the same command with --split validation and --split test.
PYTHONPATH=src .venv/bin/python scripts/prepare_three_head_probes.py \
  --plan configs/three_head_probe_dataset_exploratory2022.json --materialize
PYTHONPATH=src .venv/bin/python scripts/train_three_head_probes.py
.venv/bin/python scripts/status_three_head_probes.py
.venv/bin/python scripts/refresh_three_head_probe_notebook.py --watch
```

The preparation command streams each continuous split in order. Separate split
processes can run concurrently. The training supervisor serializes GPU runs and
resumes completed epoch checkpoints. Its state includes weights, optimizer,
scheduler, sampling generator, Python/NumPy/Torch/CUDA random state and protocol
identity. A committed latest checkpoint is published after its best checkpoint.
Resume tests require bitwise-identical final weights and optimizer state.

Frozen features are extracted once per encoder/data/normalization identity.
Reusing a cache reads its completion metadata; training does not repeat file-copy
checksums or RAM staging audits. Run heartbeats distinguish feature extraction,
training, validation and diagnostics. A stale heartbeat is reported as a
suspected stall, rather than assumed proof of a deadlock.

`notebooks/study_seviri_cloud_mask_probe.ipynb` retains the earlier cloud-mask
results and adds the new protocol, teacher acceptance, class metrics, sequential
transfer, interventions and aligned target/prediction images. It reads saved
reports and examples, so refreshing its figures performs no model inference.
The optional watcher refreshes ongoing loss curves every five minutes and
refreshes test comparisons after each completed experiment. It stops after
the suite's final update. Atomic notebook publication preserves user edits
detected during execution. Training runs use a separate source snapshot under
the result directory, so notebook edits leave their resume identity unchanged.
