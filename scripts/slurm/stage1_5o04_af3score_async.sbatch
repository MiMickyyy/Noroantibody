#!/bin/bash -l
#SBATCH --job-name=noro_5o04_af3_async
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

set -euo pipefail

PROJECT_ROOT=/rhome/xli616/Norovirus
cd "${PROJECT_ROOT}"

: "${AF3_ASYNC_INPUT_DIR:?AF3_ASYNC_INPUT_DIR is required}"
: "${AF3_ASYNC_OUTPUT_DIR:?AF3_ASYNC_OUTPUT_DIR is required}"
AF3_ASYNC_NUM_JOBS=${AF3_ASYNC_NUM_JOBS:-1}
AF3_ASYNC_CANDIDATE_ID=${AF3_ASYNC_CANDIDATE_ID:-unknown_candidate}

module load gcc/12.2.0 || true
module load "${AF3SCORE_CUDA_MODULE:-cuda/12.8}" || true

export AF3SCORE_PARTITION=${AF3SCORE_PARTITION:-short_gpu}
export AF3SCORE_QOS=${AF3SCORE_QOS:-short_gpu}
export AF3SCORE_TIME=${AF3SCORE_TIME:-02:00:00}
export AF3SCORE_GRES=${AF3SCORE_GRES:-gpu:a100:1}
export AF3SCORE_CPUS=${AF3SCORE_CPUS:-8}
export AF3SCORE_MEM=${AF3SCORE_MEM:-48G}
export AF3SCORE_CUDA_MODULE=${AF3SCORE_CUDA_MODULE:-cuda/12.8}
export AF3SCORE_FLASH_ATTENTION=${AF3SCORE_FLASH_ATTENTION:-xla}
export AF3SCORE_DB_DIR=${AF3SCORE_DB_DIR:-/bigdata/operations/pkgadmin/srv/projects/db/alphafold/3.0.0}
export AF3SCORE_MODEL_DIR=${AF3SCORE_MODEL_DIR:-${AF3SCORE_DB_DIR}/model}

mkdir -p "${AF3_ASYNC_OUTPUT_DIR}"

echo "AF3_ASYNC_PARENT_START $(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "AF3_ASYNC_CANDIDATE_ID=${AF3_ASYNC_CANDIDATE_ID}"
echo "AF3_ASYNC_INPUT_DIR=${AF3_ASYNC_INPUT_DIR}"
echo "AF3_ASYNC_OUTPUT_DIR=${AF3_ASYNC_OUTPUT_DIR}"
echo "AF3SCORE_PARTITION=${AF3SCORE_PARTITION}"
echo "AF3SCORE_QOS=${AF3SCORE_QOS}"
echo "AF3SCORE_TIME=${AF3SCORE_TIME}"
echo "AF3SCORE_GRES=${AF3SCORE_GRES}"

bash scripts/hpcc/run_af3score_hpcc.sh "${AF3_ASYNC_INPUT_DIR}" "${AF3_ASYNC_OUTPUT_DIR}" "${AF3_ASYNC_NUM_JOBS}"

echo "AF3_ASYNC_PARENT_END $(date --iso-8601=seconds)"
