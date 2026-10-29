# SEVIRI GPU runtime: audit of Demetra and controlled benchmarks

The source of the reused execution controls is **Demetra commit
`b0576746fd85e69db403ccab5759128734714424`** (29 October 2026).
`git fetch origin` followed by `git rev-list --left-right --count HEAD...origin/main`
returned `0 0`. The local changes in the copied Demetra checkout were retained.
The search covered Python, Slurm launch scripts, configuration files and code
cells of notebooks. A machine-readable line inventory is stored locally in
`results/gpu_runtime_20261029/demetra_inventory.json`.

## Located implementations and their applicability

| Mechanism | Demetra implementation | SEVIRI before this revision |
|---|---|---|
| BF16/FP16 autocast | `moduli/videomae/utils.py:794`, `engine_for_pretraining.py:44`, `engine_for_tracking.py:158` | BF16 enabled; FP32 optimizer state; no FP16 scaler needed for this configuration |
| TF32 matrix multiplication and convolutions | `moduli/videomae/utils.py:686`, `utils.py:772` | FP32 matmul precision `highest`; cuDNN TF32 already enabled |
| cuDNN convolution autotuning | `moduli/videomae/utils.py:694` | Disabled |
| SDPA and selectable attention kernel | `moduli/videomae/models/modeling_finetune.py:201`, `utils.py:698` | Transformers selects SDPA in encoder and decoder; actual kernel is hardware-dependent |
| Inductor compilation and mode selection | `moduli/videomae/utils.py:766` | Eager execution |
| CUDA Graphs control | `moduli/videomae/utils.py:725` | No graph capture configured; Demetra's `INDUCTOR_DISABLE_CUDAGRAPHS` variable is not read by installed PyTorch 2.14 |
| Fused optimizer | `moduli/videomae/optim_factory.py:203` | Native AdamW with CUDA foreach selected automatically; Apex fused alternatives exist in Demetra, while its default optimizer is plain AdamW |
| Fixed positional encodings as nonpersistent buffers | `moduli/videomae/models/modeling_pretrain.py:74`, `models/modeling_finetune.py:398` | Hugging Face positional tensors reside on CPU and are converted in each embedding call |
| Constant zero attention bias as a buffer | `moduli/videomae/models/modeling_finetune.py:169` | Hugging Face uses separate query/key/value linear projections |
| Optional activation recomputation | `moduli/videomae/models/modeling_pretrain.py:142`, `models/modeling_finetune.py:478` | Disabled, as in Demetra's specialization/tracking performance profiles |
| Pinned memory, nonblocking H2D, persistent workers | `moduli/videomae/dataset/data_manager.py:112`, `engine_for_pretraining.py:76` | Pinned float16 buffers, thread pool, four prefetched batches, GPU global normalization |
| Gradient accumulation and clearing with `set_to_none` | `moduli/videomae/engine_for_tracking.py:189`, `moduli/firstpass/src/cyclone_locator/train.py:1171` | Supported; active effective batch 32, accumulation 1 |
| Selective synchronization for profiling | `moduli/videomae/engine_for_pretraining.py:56` | Explicit device synchronization at epoch boundaries; integrity/finite checks also create synchronization points |
| Optional expensive diagnostic logging | `moduli/videomae/arguments.py:194` | Per-step loss and periodic progress reporting; no weight-drift scan |
| DDP static graph, gradient bucket views, 64 MiB buckets, unused-parameter detection and buffer broadcast controls | `moduli/videomae/utils.py:738` | Single GPU instance; requires a distributed trainer and multi-GPU validation |
| Compile/DDP compatibility override | `moduli/videomae/utils.py:866` sets `dynamo.config.optimize_ddp=False` in `get_model` | Single-process compilation; this override is reserved for checking on the cluster |
| DDP accumulation without intermediate all-reduce | `moduli/firstpass/src/cyclone_locator/train.py:1185` | Single GPU instance |
| Distributed sampler, epoch-dependent shuffling | `moduli/videomae/specialization.py:431` | Random sampler reshuffles clips each epoch; temporal order within clips is preserved |
| Rank discovery for torchrun/MPI/PMI/Slurm | `moduli/videomae/utils.py:567` | Single process |
| CPU socket binding, thread limits, NCCL setup | `moduli/videomae/sbatch_specialization.sh:17` | Four PyTorch CPU threads and four loader threads; no cluster process binding |
| Optional DeepSpeed setup | `moduli/videomae/utils.py:1160`, `arguments.py:131` | Disabled; the supplied Demetra configuration defines FP16/Adam settings |

Demetra's `COMMON_RUNTIME_ARGS` uses the `eager` compiler backend; specialization
and tracking override this with `inductor`, BF16, SDPA and disabled activation
checkpointing. The selected profile determines which paths actually execute.
The cluster specialization launcher requests eight nodes with four GPUs each,
five CPU cores per process, and one BLAS/OpenMP thread per library. Reusing those
resource counts requires checking the target partition and allocation.

## Implementation

`src/emma_gpm/ssl_runtime.py` adapts Demetra's runtime configuration and compilation
helpers to the SEVIRI Hugging Face models. Native `torch.optim.AdamW(fused=True)`
provides the fused optimizer without introducing an Apex dependency.

Compilation targets the dense transformer stacks and the forecasting head.
Mask construction, mask validation, the causal history/target split and the
radiometric finite-value checks retain their original execution paths.
`nn.Module.compile` preserves the model's parameter names for checkpoint loading
and encoder export. The shared encoder receives both full historical token
sequences and visible-token sequences in Joint, which require separate compiled
specializations. Validation is another specialization because gradients are off.

The runtime JSON is stored in the run identity. An execution revision binds the
new source hashes, exact parent checkpoint hashes, runtime configuration and
unchanged original experiment plans. Loading an old AdamW checkpoint restores
its parameter-group flags, so the requested fused setting must be reapplied
after loading. Optimizer moments, update counts, scheduler and random states
remain in the checkpoint. The numerical rounding of different kernels can
change the subsequent trajectory; the revision records its epoch boundary.

For a headless installation, Inductor requires a compiler and Python development
headers. This host uses Ubuntu's `libpython3.12-dev` package extracted under
`~/.local/lib/python-dev`, without a system installation. The runtime helper
uses those headers when the system include directory lacks `Python.h`.

## Measurement protocol

The controlled benchmarks use real clips from the frozen 16,384-clip manifest,
the production daily-frame file cache, batch 32, BF16, four reader threads and
four prefetched batches. Each candidate begins from the same seed, with the
same sample and mask order. Six warmup updates precede three measurements of
48 updates. Warmup and compilation time are recorded separately. Every update
includes data transfer and normalization, window validation, forward, loss,
backward, finite-gradient checking and optimizer step. Benchmark scalar logging
uses a flushed CSV stream; production retains its existing per-step logger.
Full training epochs provide the final throughput check.

The isolated GPU is an NVIDIA H200 MIG instance with **32 SMs**, rather than the
full physical accelerator. `nvidia-smi` reports GPU utilization as unavailable
for this allocation. Elapsed training time and clips per second are directly
measurable; profiler kernel activity describes the fraction of wall time spent
executing kernels. SM occupancy requires hardware counters.

Raw benchmark results, numerical checks and the deployment record live under
`results/gpu_runtime_20261029/`; these generated artifacts remain outside Git.

References: [PyTorch compilation modes](https://docs.pytorch.org/docs/stable/generated/torch.compile),
[AdamW execution variants](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

## Reproducing measurements and resuming an existing matrix

An example benchmark command is:

```bash
TORCHINDUCTOR_COMPILE_THREADS=4 .venv/bin/python scripts/benchmark_seviri_ssl_runtime.py \
  --dataset results/ssl_factorial_20261028/base_n16384/dataset \
  --frame-cache /home/isac/deep_convection_data/cache_native_seviri_20261026_n32768 \
  --model-size base --objective future --runtime configs/ssl_cuda_h200_mig.json \
  --output-dir results/my_runtime_benchmark
```

Use a new output directory for each measurement and arrange exclusive GPU use.
`--trace` adds an untimed CPU/CUDA profiler capture after the throughput repeats.

At a saved epoch boundary, a frozen matrix can continue with:

```bash
TORCHINDUCTOR_COMPILE_THREADS=4 .venv/bin/python scripts/run_seviri_ssl_runtime_revision.py \
  --output-dir results/ssl_factorial_20261028 \
  --revision-dir results/ssl_factorial_20261028/runtime_revision_20261029 \
  --cuda-runtime configs/ssl_cuda_h200_mig.json
```

The runner takes the same process locks as the original queue, preserves the
original plans, archives parent checkpoints and original run configuration,
and skips completed experiments. The revision directory subsequently contains
the exact runtime JSON used by every remaining trainer. A fresh trainer can
receive `--cuda-runtime configs/ssl_cuda_h200_mig.json` directly. A fresh matrix
can be prepared with the ordinary matrix runner's `--prepare-only` option and
started with this execution-revision runner.

## Results on the current H200 MIG instance

The final profile uses BF16 with Inductor `default`, fused AdamW, cuDNN
autotuning, TF32 for remaining FP32 matrix multiplication and fixed positional
buffers on the GPU. SDPA keeps automatic backend selection. The profiler found
`aten::_scaled_dot_product_cudnn_attention` and its backward in both the baseline
and the optimized training update. Explicit Flash Attention and `reduce-overhead`
CUDA Graphs produced essentially the same elapsed times as this selected profile.

| Model / objective | Before (ms/update) | After (ms/update) | Reduction in time | Allocated peak before → after (GiB) |
|---|---:|---:|---:|---:|
| Small / Future | 164 | 128 | 22% | 7.6 → 6.4 |
| Small / Joint | 241 | 192 | 20% | 9.2 → 7.9 |
| Base / Future | 317 | 254 | 20% | 13.9 → 11.6 |
| Base / Joint | 443 | 360 | 19% | 16.7 → 14.3 |

Batch size remains 32. Base/16,384 uses 512 updates per epoch; the 32,768- and
65,536-clip datasets use 1,024 and 2,048 updates respectively. These counts are
preserved. The host wait for the prepared batch is about 2 ms per update.
This interval covers the part of loading visible to the training loop; data
preparation overlaps GPU computation in the worker threads.

The initial compilation took about one minute in the first Base/Joint test.
Subsequent processes reused the disk compilation cache. Training and validation
compile distinct graphs, and the first full epoch includes this startup cost.

A frozen-weight comparison used the **epoch 114 Base/Future checkpoint** and all
512 validation clips. Both executions report **4.60 K RMSE**; the largest RMSE
difference across channel/horizon pairs is about **0.001 K**. For the first two
validation clips, the RMS difference between the two predicted pixel arrays is
**0.09 K**. These are numerical differences between executions of the same weights.
All restored optimizer moments and step counters matched exactly before the
resumed update, and the compiled backward/fused update passed finite-gradient
checks. This validates execution compatibility; subsequent training trajectories
can differ because floating-point rounding differs.

The full CPU regression suite passed (323 tests, with one CUDA-only test skipped)
before the final migration guard test was added. All 15 targeted runtime,
checkpoint-migration and SSL-training tests then passed. Real CUDA checks covered
Small and Base, Future and Joint, full validation, and an update from restored
optimizer state. Compilation and `git diff --check` also passed.

Additional GPU-related code in Demetra includes deterministic/cuDNN switches in
`classification.py:55` and `inference_classification.py:63`, and one-time CUDA
allocator cache clearing in `specialization.py:399` and inference entry points.
Those settings belong to their respective execution paths. The selected SEVIRI
profile keeps the measured convolution backend and normal allocator reuse.

## Applied execution revision and complete epochs

The active matrix now uses `runtime_revision_20261029`, resumed from epoch 114.
The original plans and checkpoint copies are preserved; 33 execution source
files are checked before every stage. The remaining Small/Base Future/Joint
experiments inherit the same runtime profile.

On Base/Future with 16,384 clips, training decreased from about **164 to
131 seconds per epoch**. A complete epoch including validation and
checkpoint writing decreased from about **173 to 140 seconds**.
The first revised epoch (115), including compilation startup, took 154 seconds.
The steady measurement uses epoch 116, 117.
Batch size remains 32, with 512 updates per epoch. The allocated GPU peak for
the complete steady epoch is about **11.7 GiB**, versus 13.9 GiB
before the revision. These are allocator measurements within this process.
