#!/bin/bash -l
#SBATCH --job-name=noro_5o04_af3_smoke
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/rhome/xli616/Norovirus/outputs/stage1_5O04_hotspot_transfer_af3_smoke/logs/af3score_parent_%j.out
#SBATCH --error=/rhome/xli616/Norovirus/outputs/stage1_5O04_hotspot_transfer_af3_smoke/logs/af3score_parent_%j.err

set -euo pipefail

PROJECT_ROOT=/rhome/xli616/Norovirus
OUT_ROOT=${PROJECT_ROOT}/outputs/stage1_5O04_hotspot_transfer_af3_smoke
SOURCE_PDB=${AF3_SMOKE_SOURCE_PDB:-${PROJECT_ROOT}/outputs/stage1_5O04_hotspot_transfer/conditions/5O04_cdr13_only_H1L10_H3L08/mpnn_aux/5O04_cdr13_only_H1L10_H3L08_bb001/5O04_cdr13_only_H1L10_H3L08_bb001_outputs/5O04_cdr13_only_H1L10_H3L08_bb001_dldesign_0.pdb}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/input_pdb"
cd "${PROJECT_ROOT}"

module load gcc/12.2.0 || true
module load cuda/12.4 || true

test -r "${SOURCE_PDB}"
rm -rf "${OUT_ROOT}/af3score_run"
mkdir -p "${OUT_ROOT}/input_pdb"
cp "${SOURCE_PDB}" "${OUT_ROOT}/input_pdb/af3_smoke.pdb"

export AF3SCORE_PARTITION=${AF3SCORE_PARTITION:-short_gpu}
export AF3SCORE_GRES=${AF3SCORE_GRES:-gpu:blackwell6000:1}
export AF3SCORE_CPUS=${AF3SCORE_CPUS:-8}
export AF3SCORE_MEM=${AF3SCORE_MEM:-48G}
export AF3SCORE_CUDA_MODULE=${AF3SCORE_CUDA_MODULE:-cuda/12.8}
export AF3SCORE_USE_PIP_NVCC=${AF3SCORE_USE_PIP_NVCC:-0}
export AF3SCORE_FLASH_ATTENTION=${AF3SCORE_FLASH_ATTENTION:-xla}
export AF3SCORE_DB_DIR=${AF3SCORE_DB_DIR:-/bigdata/operations/pkgadmin/srv/projects/db/alphafold/3.0.0}
export AF3SCORE_MODEL_DIR=${AF3SCORE_MODEL_DIR:-${AF3SCORE_DB_DIR}/model}

echo "AF3_SMOKE_PARENT_START $(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "SOURCE_PDB=${SOURCE_PDB}"
echo "AF3SCORE_PARTITION=${AF3SCORE_PARTITION}"
echo "AF3SCORE_GRES=${AF3SCORE_GRES}"
echo "AF3SCORE_MODEL_DIR=${AF3SCORE_MODEL_DIR}"

bash scripts/hpcc/run_af3score_hpcc.sh "${OUT_ROOT}/input_pdb" "${OUT_ROOT}/af3score_run" 1

test -s "${OUT_ROOT}/af3score_run/af3score_metrics.csv"
metric_rows=$(tail -n +2 "${OUT_ROOT}/af3score_run/af3score_metrics.csv" | wc -l)
if [[ "${metric_rows}" -lt 1 ]]; then
  echo "AF3Score smoke failed: metrics CSV contains no candidate rows." >&2
  exit 1
fi
echo "AF3Score metrics:"
cat "${OUT_ROOT}/af3score_run/af3score_metrics.csv"
echo "AF3_SMOKE_PARENT_END $(date --iso-8601=seconds)"
