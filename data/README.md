# External data directory

Raw and intermediate datasets are intentionally excluded from Git.

Set the external location before running the pipeline:

```bash
export EMMA_GPM_DATA_ROOT=/home/daniele/Datasets/Deep_convection
```

The expected subdirectories are `raw/` and `interim/`. Download and preparation
scripts create them when needed. Repository outputs remain under `catalogs/`,
`results/` and `figures/`.

