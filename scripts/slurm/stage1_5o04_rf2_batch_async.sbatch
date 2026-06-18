#!/bin/bash -l
#SBATCH --job-name=noro_rf2_batch_async
#SBATCH --partition=short_gpu
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT=/rhome/xli616/Norovirus
cd "${PROJECT_ROOT}"

: "${RF2_WORKER_MANIFEST:?RF2_WORKER_MANIFEST is required}"

module load gcc/12.2.0 || true
module load cuda/11.8 || true
export DGLBACKEND=pytorch
export PATH=/rhome/xli616/Norovirus/data/framework/external/RFantibody/.venv/bin:${PATH}
export LD_LIBRARY_PATH=/rhome/xli616/.conda/envs/noro_rfantibody/lib:${LD_LIBRARY_PATH:-}

echo "RF2_ASYNC_WORKER_START $(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "RF2_WORKER_MANIFEST=${RF2_WORKER_MANIFEST}"
nvidia-smi || true

/rhome/xli616/.conda/envs/noro_rfantibody/bin/python scripts/run_stage1_5o04_campaign.py \
  --execute \
  --rf2-worker-manifest "${RF2_WORKER_MANIFEST}"

echo "RF2_ASYNC_WORKER_END $(date --iso-8601=seconds)"
