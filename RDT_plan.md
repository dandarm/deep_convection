Project Specification: RDT-CW Reproduction and VideoMAE Probing
1. Project Objective
The primary objective of this project is to implement a deterministic RDT-CW-inspired reproduction (strictly focusing on convective cells and their thermodynamic evolution) utilizing geostationary Infrared (IR, 10.8 µm) and Water Vapor (WV, 6.2 µm) channels to generate pseudo-ground truth. It may be described as an exact RDT-CW reproduction only after every threshold, tracking rule, and lifecycle transition has been validated against the authoritative operational specification. The heuristic outputs from this classical physical baseline will serve as dense, voxel-wise pseudo-ground-truth annotations across temporal image sequences.

Subsequently, we leverage a pre-trained VideoMAE (Video Masked Autoencoder) backbone to conduct spatiotemporal semantic segmentation probing. While the heuristic GT is generated using only two channels, the VideoMAE will ingest a seven-channel multispectral input cube spanning from WV 6.2 µm to the CO2 absorption band at IR 13.4 µm. By attaching a lightweight 3D Feature Pyramid Network (FPN) probing decoder, we aim to evaluate the capacity of self-supervised 3D representations to implicitly learn and generalize complex convective dynamics, thermal morphology, cloud-top height, microphysics, and rapid temporal evolution.

2. RDT-CW Algorithmic Rules
The classical RDT-CW framework processes consecutive geostationary satellite frames to detect, track, and classify convective cloud cells through four deterministic processing phases.
Phase A: Detection via Adaptive Thresholding
Cloud systems are detected at $T(t)$ using multi-threshold contour analysis on the IR 10.8 µm channel:

Adaptive Temperature Thresholds: Dynamic thresholds $T_{thresh} \in {240\text{K}, 230\text{K}, 220\text{K}, 210\text{K}}$ isolate cloud cores and anvils.
Spatial Connectivity: 8-connectivity connected-component labeling groups contiguous pixels where $T_{IR} \le T_{thresh}$.
Morphological Filtering: Components with surface area $A < 100\text{ km}^2$ are discarded as noise.
Phase B: Geometric & Overlap Tracking
Tracking establishes temporal continuity between components at frame $t-1$ and frame $t$:

Area Overlap Ratio (AOR): Evaluates the geometric intersection over the minimum area of the two candidate cells.
Association Metric: A candidate pair is linked if $AOR \ge 0.15$.
Lineage Handling: Explicit tracking of splits (one-to-many) and merges (many-to-one) establishes parent-child relationships across successive time steps.
Phase C: Convection Discrimination & Growth Metrics
Every tracked cloud cell is evaluated for convective activity based on structural and dynamic indices:

Spatial Thermal Gradient: $T_{mean} - T_{min}$
Cooling Rate (Thermal Tendency): $(T_{min}(t) - T_{min}(t-\Delta t)) / \Delta t$
Area Expansion Rate: $(Area(t) - Area(t-\Delta t)) / \Delta t$
Phase D: Overshooting Top (OT) Detection
Overshooting Tops represent severe convective cores breaching the tropopause:

Brightness Temperature Difference (BTD): $T_{WV6.2} - T_{IR10.8}$
OT Condition: A pixel within a convective cell is flagged as an OT if $BTD \ge 1.0\text{ K}$ and $T_{IR10.8} \le 215\text{ K}$.

3. Ground Truth Dataset Construction & Logical Routing
To provide clear instructions for an agentic coding LLM, the classification follows a strict sequential decision matrix for each segmented cloud cell rather than a visual tree. Each pixel is ultimately mapped to one of 5 mutually exclusive semantic classes.

Step 1: Background Separation

Condition: Is the pixel unsegmented or is the IR 10.8 µm temperature > 240 K?
Action:
If YES -> Assign Class 0 (Background).
If NO -> Proceed to Step 2.

Step 2: Convective Discrimination

Condition: Evaluate the cell against convective criteria:
Spatial Thermal Gradient $\ge 10\text{ K}$.
Cooling Rate $\le -2.5\text{ K} / 15\text{ min}$ (or inherited if previously convective and currently mature).
Area Expansion Rate $> 0$.
Action:
If the cell FAILS to meet these criteria -> Assign Class 1 (Non-Convective Cold Cloud).
If the cell MEETS the criteria -> Proceed to Step 3.

Step 3: Convective Phase Assignment

Condition: Check the current Cooling Rate ($dT/dt$).
Action:
If $dT/dt \le -2.5\text{ K} / 15\text{ min}$ (rapidly cooling and growing) -> Assign Class 2 (Developing Convective Cell).
If $dT/dt > -2.5\text{ K} / 15\text{ min}$ (cooling has slowed, stabilized, or warming) -> Assign Class 3 (Mature / Decaying Convective Cell).

Step 4: Overshooting Top (OT) Override (Pixel-Level)

Condition: For individual pixels within Class 2 or Class 3 cells, evaluate the Brightness Temperature Difference.
Action:
If $BTD \ge 1.0\text{ K}$ AND $T_{IR10.8} \le 215\text{ K}$ -> Override the cell class and assign Class 4 (Overshooting Top) to those specific pixels.

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
The primary backbone is a VideoMAE-Small with a seven-channel patch embedding. Pretraining is self-supervised and must not use RDT-CW, EMMA, OPERA, or any other downstream label. Complete 16-frame RSS windows are sampled from the regional SEVIRI archive, and 224 x 224 spatial crops are drawn on the fly rather than materialized as duplicated clip files.

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
