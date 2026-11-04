#!/bin/bash
# Single-node layout and DDP settings adapted from Demetra specialization.
#SBATCH --job-name=seviri_ddp
#SBATCH --account=iscrc_same-d2
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=23:50:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --output=seviri_ddp_%j.out
#SBATCH --error=seviri_ddp_%j.err

set -euo pipefail
module load python/3.11.7
unset PYTHONPATH
source "$WORK/deep_convection/venv/bin/activate"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=2
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
export HF_HOME="$WORK/deep_convection/huggingface"
SEVIRI_DDP_ROOT=${SEVIRI_DDP_ROOT:-"$WORK/deep_convection/ddp_4gpu_20261102"}
SEVIRI_DDP_STAGE=${SEVIRI_DDP_STAGE:-pilots}
SEVIRI_GLOBAL_BATCH=${SEVIRI_GLOBAL_BATCH:-32}
export TORCHINDUCTOR_CACHE_DIR="$SEVIRI_DDP_ROOT/inductor"
export TRITON_CACHE_DIR="$SEVIRI_DDP_ROOT/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
trap 'touch "$SEVIRI_DDP_ROOT/stop_after_epoch"' USR1
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv -l 2 > "$SEVIRI_DDP_ROOT/gpu_${SLURM_JOB_ID}.csv" &
monitor=$!
cleanup() {
    kill "$monitor" 2>/dev/null || true
    if [[ "$SEVIRI_DDP_STAGE" == ram-pilots || "$SEVIRI_DDP_STAGE" == full ]]; then
        python "$SEVIRI_DDP_ROOT/source_archive/scripts/stage_seviri_ssl_ram.py" \
            --root "$SEVIRI_DDP_ROOT" --cleanup
    fi
}
trap cleanup EXIT
srun --ntasks=1 --cpus-per-task=32 --cpu-bind=none python -u \
    "$SEVIRI_DDP_ROOT/source_archive/scripts/run_seviri_ssl_ddp_trials.py" \
    --root "$SEVIRI_DDP_ROOT" --stage "$SEVIRI_DDP_STAGE" \
    --baseline-global-batch "$SEVIRI_GLOBAL_BATCH" &
worker=$!
set +e
while true; do
    wait "$worker"
    result=$?
    if ! kill -0 "$worker" 2>/dev/null; then break; fi
done
set -e
if [[ "$result" == 75 ]]; then
    # Clear only this job's wall-time request; resume starts at the saved epoch.
    rm -f "$SEVIRI_DDP_ROOT/stop_after_epoch"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$result"
