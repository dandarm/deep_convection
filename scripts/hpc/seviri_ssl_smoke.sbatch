#!/bin/bash
# Short synthetic integration test; results are engineering checks.
#SBATCH --job-name=seviri_smoke
#SBATCH --account=iscrc_same-d2
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=seviri_smoke_%j.out
#SBATCH --error=seviri_smoke_%j.err
set -euo pipefail
module load python/3.11.7
unset PYTHONPATH
source "$WORK/deep_convection/venv/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=4 MPLBACKEND=Agg HF_HUB_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/deep_convection/inductor/smoke_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$WORK/deep_convection/triton/smoke_${SLURM_JOB_ID}"
cd "$HOME/deep_convection"
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
for objective in mvm future joint; do
    srun python -u scripts/pretrain_seviri_ssl.py --synthetic-smoke \
        --objective "$objective" --device cuda --precision bf16 --batch-size 2 \
        --epochs 1 --cpu-threads 4 --num-workers 0 \
        --cuda-runtime configs/ssl_cuda_h200_mig.json \
        --output-dir "$WORK/deep_convection/smoke_${SLURM_JOB_ID}/${objective}"
done
