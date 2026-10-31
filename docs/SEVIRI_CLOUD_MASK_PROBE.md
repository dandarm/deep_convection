# Frozen-encoder cloud-mask probe

The downstream target is the binary `cma` produced by the repository's
`CMA_original/cma_nwcsaf` implementation: clear = 0, cloudy = 1, missing = 255.
The probe predicts cloud probability at every pixel of the **last observed
frame**, index 9 of a canonical 16-frame SEVIRI clip. Only frames 0–9 enter
the encoder. The previous forecasting probes remain available.

## Data and label generation

The initial dataset contains 448 training, 112 validation and 112 test clips.
Training uses 13–14 December 2020, validation 16 December, and test 26 December.
These dates are disjoint from every encoder's pretraining and validation dates;
the runner checks the original manifests before extracting features. At least
one unused calendar day separates downstream splits. Each selected timestamp
has 14 spatially disjoint crops. Clip starts are 80 minutes apart in training
and 160 minutes apart in validation/test. Clips from the same day describe
correlated atmospheric structures; the experiment currently has one validation
day and one test day.

CMA runs causally on the **complete 763 × 1762 archive rectangle**, at the actual
five-minute cadence, before cropping. Its state starts at the first observation
of each day; the first selected target follows 60 minutes of observations.
The label at index 9 is saved before frames 10–15 are processed. Those six
later frames are retained only to preserve canonical clip storage.

Inputs are seven-channel brightness temperatures in kelvin, stored losslessly
as float16 NPY clips and oriented from the actual x/y coordinates. Invalid
radiometry stays missing in the domain processor. Incomplete final clips fail
validation. The binary `cma` field is used directly; other products such as
`cma_cover` have different class semantics.

Geolocation comes from the source projection. A Natural Earth 1:10m land
polygon mask supplies land/sea classification at pixel centres. The repository's
Pure-IR defaults supply zero elevation, absent snow/ice ancillary, and IR-only
solar geometry; the atmospheric and surface references are estimated from
SEVIRI. These choices and the complete CMA configuration are recorded. Winter
snow and complex terrain remain relevant limitations of these generated labels.
The measured scores quantify agreement with this particular CMA implementation.
External cloud observations would support a separate assessment of physical
cloud-detection accuracy.

## Frozen encoder and trainable output

Each completed run supplies its final exported encoder, its original global
mean and standard deviation, and its architecture. Encoder weights have
`requires_grad=False`, evaluation mode is permanent, and inference uses CPU
BF16 autocast. Cached features are stored as float32. A SHA-256 of all encoder
parameters and buffers is checked before and after extraction.

The feature at each of the last temporal plane's 14 × 14 spatial positions
integrates the observed history through the encoder's self-attention. A shared
head applies LayerNorm, Linear(D, 128), GELU and Linear(128, 256), then assembles
the 16 × 16 pixel predictions into the 224 × 224 output. The head has two Linear
layers. Only the head is optimized: 30 epochs, batch size 16, AdamW, learning
rate 0.001 and weight decay 0.0001. Every encoder uses identical downstream
splits, head seed and minibatch order. The head initialization is matched for
encoders of the same architecture; Small and Base have different input widths.

A random encoder control is paired with each distinct encoder configuration,
pretraining normalization and initialization seed. Its architecture and initial
weights follow the corresponding original run. This permits a comparison
between pretrained and random representations with the same head procedure.
The `random_*` directory name contains a hash of those control settings.

## Selection and metrics

Training minimizes binary cross-entropy over valid labelled pixels. Every epoch
is evaluated on validation. The checkpoint with greatest validation mean IoU
is selected, using the first epoch for ties. The fixed cloud-probability
threshold is 0.5. Test is evaluated after checkpoint selection.

Cloud IoU measures overlapping predicted and target cloud area relative to
their combined area. Dice also measures spatial overlap, with twice the common
area in the numerator. Precision measures the fraction of predicted cloud
pixels labelled cloudy by CMA; recall measures the fraction of CMA cloud
pixels recovered. Clear IoU and mean IoU expose behaviour on the clear class.
Pixel confusion counts are pooled across all test clips; per-clip scores are
also saved. They represent projected-image pixel counts. Four displayed examples
are selected at fixed positions in the ordered test set.

The notebook reports paired IoU differences in percentage points. Comparisons
retain the pretraining epoch budget: Small/16,384 has 75 epochs, while the other
listed runs have 150. Results from one held-out day support an initial
cloud-segmentation comparison. Additional days, seeds, spectral/temporal
ablations and an adapted RGB encoder control are needed for broader conclusions
about representation quality and atmospheric dynamics.

## Execution and resumption

From the repository root, using the existing Python environment:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python scripts/build_seviri_cloud_probe_dataset.py \
  --zarr-root /data1/deep_convection_data/MSG-SEVIRI \
  --output-dir /data1/deep_convection_data/cloud_mask_probe_20261031

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  nice -n 15 .venv/bin/python scripts/run_seviri_cloud_probes.py \
  --dataset /data1/deep_convection_data/cloud_mask_probe_20261031

python3 scripts/status_seviri_cloud_probes.py
```

The queue starts available completed encoders, then waits for the remaining
exports to be completed by the pretraining runner. It also waits for dataset
construction when launched concurrently. It never loads a checkpoint while
pretraining is writing it. Stop with SIGTERM and rerun the same command to
resume: dataset generation resumes by complete day, feature extraction by
committed blocks of 16 clips, and head training by complete epoch with optimizer
state. Source, configuration, dataset, feature and weight checksums guard reuse.

All large artifacts live outside Git. Inputs reside on `/data1`; probe results
and feature caches are in `results/cloud_mask_probe_20261031`. `queue.log` records
exceptions. The status command reports a dead process or a possible stall after
180 seconds without a heartbeat. The dataset builder is monitored separately
while the queue waits for its manifest. The queue uses two CPU threads and
can run at reduced scheduling priority while the GPU continues pretraining.

`notebooks/study_seviri_cloud_mask_probe.ipynb` reads completed result files on
every execution. It contains class balance, model comparisons, paired random
controls, learning curves and spatial examples. Its committed version contains
code and explanations; execution outputs remain local.
