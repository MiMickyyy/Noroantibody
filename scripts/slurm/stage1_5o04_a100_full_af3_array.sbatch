#!/bin/bash -l
#SBATCH --job-name=noro_5o04_a100_full_af3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-49%1
#SBATCH --output=/rhome/xli616/Norovirus/outputs/stage1_5O04_hotspot_transfer_full_af3/logs/a100_full_%A_%a.out
#SBATCH --error=/rhome/xli616/Norovirus/outputs/stage1_5O04_hotspot_transfer_full_af3/logs/a100_full_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/rhome/xli616/Norovirus
OUT_ROOT=${PROJECT_ROOT}/outputs/stage1_5O04_hotspot_transfer_full_af3
mkdir -p "${OUT_ROOT}/logs"

cd "${PROJECT_ROOT}"

module load gcc/12.2.0 || true
module load cuda/11.8 || true
export DGLBACKEND=pytorch
export PATH=/rhome/xli616/Norovirus/data/framework/external/RFantibody/.venv/bin:${PATH}
export LD_LIBRARY_PATH=/rhome/xli616/.conda/envs/noro_rfantibody/lib:${LD_LIBRARY_PATH:-}

export AF3SCORE_PARTITION=${AF3SCORE_PARTITION:-gpu}
export AF3SCORE_GRES=${AF3SCORE_GRES:-gpu:a100:1}
export AF3SCORE_CPUS=${AF3SCORE_CPUS:-8}
export AF3SCORE_MEM=${AF3SCORE_MEM:-48G}
export AF3SCORE_CUDA_MODULE=${AF3SCORE_CUDA_MODULE:-cuda/12.4}
export AF3SCORE_FLASH_ATTENTION=${AF3SCORE_FLASH_ATTENTION:-triton}
export AF3SCORE_DB_DIR=${AF3SCORE_DB_DIR:-/bigdata/operations/pkgadmin/srv/projects/db/alphafold/3.0.0}
export AF3SCORE_MODEL_DIR=${AF3SCORE_MODEL_DIR:-${AF3SCORE_DB_DIR}/model}

echo "JOB_START $(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "AF3SCORE_PARTITION=${AF3SCORE_PARTITION}"
echo "AF3SCORE_GRES=${AF3SCORE_GRES}"
nvidia-smi || true

/rhome/xli616/.conda/envs/noro_rfantibody/bin/python scripts/run_stage1_5o04_campaign.py \
  --execute \
  --condition-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root outputs/stage1_5O04_hotspot_transfer_full_af3 \
  --pipeline-config data/configs/stage1_5o04/pipeline.stage1_5o04.yaml \
  --tooling-config data/configs/stage1_5o04/tooling.hpcc.yaml \
  --resolved-inputs data/configs/stage1_5o04/resolved_inputs.hpcc.yaml \
  --resolved-targets data/configs/stage1_5o04/resolved_targets.full_target.yaml

echo "JOB_END $(date --iso-8601=seconds)"
