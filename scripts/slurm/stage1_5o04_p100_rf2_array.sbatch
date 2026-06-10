#!/bin/bash -l
#SBATCH --job-name=noro_5o04_p100_rf2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-49%1
#SBATCH --output=/rhome/xli616/Norovirus/outputs/stage1_5O04_hotspot_transfer_rf2_only/logs/p100_rf2_%A_%a.out
#SBATCH --error=/rhome/xli616/Norovirus/outputs/stage1_5O04_hotspot_transfer_rf2_only/logs/p100_rf2_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/rhome/xli616/Norovirus
OUT_ROOT=${PROJECT_ROOT}/outputs/stage1_5O04_hotspot_transfer_rf2_only
mkdir -p "${OUT_ROOT}/logs"

cd "${PROJECT_ROOT}"

module load gcc/12.2.0 || true
module load cuda/11.8 || true
export DGLBACKEND=pytorch
export PATH=/rhome/xli616/Norovirus/data/framework/external/RFantibody/.venv/bin:${PATH}
export LD_LIBRARY_PATH=/rhome/xli616/.conda/envs/noro_rfantibody/lib:${LD_LIBRARY_PATH:-}

echo "JOB_START $(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

/rhome/xli616/.conda/envs/noro_rfantibody/bin/python scripts/run_stage1_5o04_campaign.py \
  --execute \
  --condition-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root outputs/stage1_5O04_hotspot_transfer_rf2_only \
  --pipeline-config data/configs/stage1_5o04/pipeline.stage1_5o04.rf2_only.yaml \
  --tooling-config data/configs/stage1_5o04/tooling.hpcc.yaml \
  --resolved-inputs data/configs/stage1_5o04/resolved_inputs.hpcc.yaml \
  --resolved-targets data/configs/stage1_5o04/resolved_targets.full_target.yaml

echo "JOB_END $(date --iso-8601=seconds)"
