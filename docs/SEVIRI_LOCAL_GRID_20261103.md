# Local completion of the SSL experiment grid

The audit covers Small and Base at 4,096, 8,192, 16,384, 32,768 and
65,536 training clips, for MVM, Future, Joint and Residual. Twenty combinations
have completed training and twenty additional combinations are scheduled.

| Training clips | Additional Small experiments | Additional Base experiments |
| --- | --- | --- |
| 4,096 | — | MVM, Future, Joint, Residual |
| 8,192 | — | MVM, Future, Joint, Residual |
| 16,384 | MVM, Residual | MVM, Residual |
| 32,768 | MVM, Residual | MVM, Residual |
| 65,536 | MVM, Residual | MVM, Residual |

Each new experiment starts from random weights with seed 20261024 and runs
150 epochs. The effective batch remains 32 clips, with one optimizer update
per batch. This gives respectively 128, 256, 512, 1,024 and 2,048 updates per
epoch. Learning rate is 0.00015 and weight decay is 0.05. MVM uses tube masking
at 90%. All experiments retain the original training manifests, corresponding
global training statistics, seven channels, ten historical input frames and
the common 512-clip validation manifest. Future targets are at 15 and 30 minutes.

Existing Small MVM results at 4,096 and 8,192 already contain 150 epochs and
remain the reference for those combinations. Existing Small Future and Joint
at 16,384 contain 75 epochs. Their shorter training budget remains explicit in
the audit and must be considered when comparing their best validation errors.

Execution proceeds by increasing dataset size, with MVM first within each size.
It uses one local GPU, BF16, pinned batch transfers, four loading threads,
shuffling between epochs, verified native NPY frame files, TF32, cuDNN tuning,
fused AdamW, static positional buffers and compiled transformer stacks.
The batch size preserves the historical update budget; the separate Leonardo
experiments measure the effect of larger batches. The 131,072-clip experiments
continue to belong to the HPC series.

`configs/ssl_grid_completion_20261103.json` defines the selected combinations and
their complete execution order. `configs/ssl_local_optimized_runtime.json`
defines CUDA settings. A frozen runtime copy protects the running sequence
from later edits in the working checkout. Each stage saves its model,
optimizer and RNG state after every epoch; restarting the same frozen launcher
resumes an interrupted stage and skips completed stages.

Results are stored at
`/data1/deep_convection_results/ssl_grid_completion_20261103`, linked from
`results/ssl_grid_completion_20261103`. The audit is `grid_audit.json`; stage
logs, `loss_epochs.csv`, `progress.json` and checkpoints use the existing matrix
layout. The supervisor stops and reports a failed stage if its process exits
unsuccessfully. Subsequent stages remain queued for investigation and resume.

Read current progress from this machine or an SSH shell:

```bash
cd /home/isac/deep_convection
.venv/bin/python scripts/status_seviri_ssl.py \
  --series results/ssl_grid_completion_20261103
```

The launcher command and PID are recorded in `launcher.json`. The preparation
log and subsequent stage transitions appear in `queue.log`. First-start source
checks and compilation contribute initialization time; steady epoch timings
are recorded separately in each training history.
