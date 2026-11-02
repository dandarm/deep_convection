#!/bin/bash
# Adapted from Demetra/moduli/videomae/sbatch_specialization.sh.
#SBATCH --job-name=seviri_ssl
#SBATCH --account=iscrc_same-d2
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-5%4
#SBATCH --time=23:50:00
#SBATCH --signal=B:USR1@3600
#SBATCH --requeue
#SBATCH --output=seviri_ssl_%A_%a.out
#SBATCH --error=seviri_ssl_%A_%a.err

set -euo pipefail
module load python/3.11.7
SEVIRI_HPC_ROOT=${SEVIRI_HPC_ROOT:-"$WORK/deep_convection/ssl_131072_20261102"}
SEVIRI_VENV=${SEVIRI_VENV:-"$WORK/deep_convection/venv"}
source "$SEVIRI_VENV/bin/activate"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=4
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
export HF_HOME="$WORK/deep_convection/huggingface"
export TORCHINDUCTOR_CACHE_DIR="$WORK/deep_convection/inductor/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$WORK/deep_convection/triton/${SLURM_JOB_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
test -f "$SEVIRI_HPC_ROOT/hpc_plan.json"
stop_file="$SEVIRI_HPC_ROOT/stop_requests/${SLURM_ARRAY_TASK_ID}.stop"
trap 'touch "$stop_file"' USR1

# Each GPU runs one complete experiment with the original effective batch 32.
# The frozen copy remains stable when the repository in HOME is updated.
srun --ntasks=1 python -u "$SEVIRI_HPC_ROOT/source_archive/scripts/run_seviri_ssl_hpc.py" \
    --plan "$SEVIRI_HPC_ROOT/hpc_plan.json" --index "$SLURM_ARRAY_TASK_ID" &
worker=$!
set +e
while true; do
    wait "$worker"
    result=$?
    if ! kill -0 "$worker" 2>/dev/null; then break; fi
done
set -e
if [[ "$result" == 75 ]]; then
    # Only a checkpointed, deliberately paused run asks to be queued again.
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
exit "$result"
