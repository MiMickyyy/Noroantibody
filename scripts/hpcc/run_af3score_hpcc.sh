#!/bin/bash -l
# HPCC adapter for Mingchenchen/AF3Score.
#
# Usage is intentionally identical to AF3score_pipeline.sh:
#   run_af3score_hpcc.sh <input_pdb_dir> <output_dir> <num_jobs>
#
# The upstream AF3Score scripts contain site-specific Python, Slurm, CUDA, and
# model/database paths. This wrapper creates a runtime copy and patches only
# that copy from environment variables, keeping repository configs portable.

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 <input_pdb_dir> <output_dir> <num_jobs>

Required runtime resources:
  AF3SCORE_DIR        AF3Score source checkout
  AF3SCORE_PYTHON     Python 3.11 executable with AF3Score dependencies
  AF3SCORE_MODEL_DIR  AlphaFold3 model-parameter directory
  AF3SCORE_DB_DIR     AlphaFold3 database directory

Optional Slurm controls:
  AF3SCORE_PARTITION  Default: gpu
  AF3SCORE_NODELIST   Default: empty
  AF3SCORE_GRES       Default: gpu:1
  AF3SCORE_CPUS       Default: 8
  AF3SCORE_MEM        Default: 48G
  AF3SCORE_CUDA_MODULE Default: cuda/12.4
EOF
  exit 2
}

[[ $# -ge 3 ]] || usage

input_pdb_dir="$(realpath "$1")"
output_dir="$(realpath "$2")"
num_jobs="$3"

AF3SCORE_DIR="${AF3SCORE_DIR:-/rhome/xli616/Norovirus/external/AF3Score}"
AF3SCORE_PYTHON="${AF3SCORE_PYTHON:-/rhome/xli616/.conda/envs/noro_af3score/bin/python}"
AF3SCORE_PARTITION="${AF3SCORE_PARTITION:-gpu}"
AF3SCORE_NODELIST="${AF3SCORE_NODELIST:-}"
AF3SCORE_GRES="${AF3SCORE_GRES:-gpu:1}"
AF3SCORE_CPUS="${AF3SCORE_CPUS:-8}"
AF3SCORE_MEM="${AF3SCORE_MEM:-48G}"
AF3SCORE_CUDA_MODULE="${AF3SCORE_CUDA_MODULE:-cuda/12.4}"
AF3SCORE_DB_DIR="${AF3SCORE_DB_DIR:-${ALPHAFOLD_DB:-/bigdata/operations/pkgadmin/srv/projects/db/alphafold/3.0.0}}"
AF3SCORE_MODEL_DIR="${AF3SCORE_MODEL_DIR:-/rhome/xli616/models}"
AF3SCORE_FLASH_ATTENTION="${AF3SCORE_FLASH_ATTENTION:-triton}"
AF3SCORE_PREPARE_WORKERS="${AF3SCORE_PREPARE_WORKERS:-6}"
AF3SCORE_JAX_MEM_FRACTION="${AF3SCORE_JAX_MEM_FRACTION:-0.80}"

fail() {
  echo "[AF3Score HPCC adapter] ERROR: $*" >&2
  exit 1
}

[[ -d "$AF3SCORE_DIR" ]] || fail "AF3SCORE_DIR not found: $AF3SCORE_DIR"
[[ -x "$AF3SCORE_PYTHON" ]] || fail "AF3SCORE_PYTHON is not executable: $AF3SCORE_PYTHON"
[[ -d "$AF3SCORE_DB_DIR" ]] || fail "AF3SCORE_DB_DIR not found: $AF3SCORE_DB_DIR"
[[ -d "$AF3SCORE_MODEL_DIR" ]] || fail "AF3SCORE_MODEL_DIR not found: $AF3SCORE_MODEL_DIR"
find "$AF3SCORE_MODEL_DIR" -maxdepth 2 -type f | grep -q . || fail "AF3SCORE_MODEL_DIR has no model files: $AF3SCORE_MODEL_DIR"

"$AF3SCORE_PYTHON" - <<'PY'
import importlib
missing = []
for module in ("alphafold3", "Bio", "h5py", "jax", "numpy", "pandas", "tqdm"):
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")
if missing:
    raise SystemExit("Missing AF3Score Python dependencies:\n" + "\n".join(missing))
PY

runtime_dir="$output_dir/_af3score_hpcc_runtime"
rm -rf "$runtime_dir"
mkdir -p "$runtime_dir"

for name in \
  AF3score_pipeline.sh \
  functions.sh \
  01_prepare_get_json.py \
  02_prepare_pdb2jax.py \
  02_submit_prepare_jax.sh \
  03_submit_af3score.sh \
  04_get_metrics.py \
  ipsae_calculator.py \
  run_af3score.py \
  src
do
  cp -a "$AF3SCORE_DIR/$name" "$runtime_dir/"
done

sed -i \
  -e "s|^PYTHON_EXEC=.*|PYTHON_EXEC=\"$AF3SCORE_PYTHON\"|" \
  -e "s|^slurm_partition=.*|slurm_partition=\"$AF3SCORE_PARTITION\"|" \
  -e "s|^slurm_nodelist=.*|slurm_nodelist=\"$AF3SCORE_NODELIST\"|" \
  "$runtime_dir/AF3score_pipeline.sh"

cat > "$runtime_dir/functions.sh" <<'EOF'
#!/bin/bash

submit_job() {
  local partition="$1"
  local nodelist="$2"
  local script="$3"
  local log_file="$4"
  shift 4
  local job_output
  if [[ -n "$nodelist" ]]; then
    job_output=$(sbatch --partition="$partition" --nodelist="$nodelist" --output="$log_file" "$script" "$@")
  else
    job_output=$(sbatch --partition="$partition" --output="$log_file" "$script" "$@")
  fi
  if [[ "$job_output" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
      echo "${BASH_REMATCH[1]}"
  else
      echo "Submission failed: $job_output" >&2
      echo "Command: sbatch --partition=$partition --nodelist=$nodelist --output=$log_file $script $*" >&2
      exit 1
  fi
}

wait_for_jobs() {
  local description="$1"
  shift
  local job_ids=("$@")
  if [[ ${#job_ids[@]} -eq 0 ]]; then
    echo "No $description jobs to wait for."
    return 0
  fi
  echo "Waiting for all $description jobs to complete (Total: ${#job_ids[@]})..."
  while true; do
    local unfinished=0
    local squeue_output
    if ! squeue_output=$(squeue -u "$USER" -h -o "%i" 2>/dev/null); then
      echo "Warning: squeue failed, retrying in 60 seconds..."
      sleep 60
      continue
    fi
    for job_id in "${job_ids[@]}"; do
      if printf '%s\n' "$squeue_output" | grep -qx "$job_id"; then
        unfinished=$((unfinished + 1))
      fi
    done
    if [[ "$unfinished" -eq 0 ]]; then
      echo "All $description jobs completed"
      break
    fi
    echo "[$(date '+%H:%M:%S')] $unfinished $description jobs still pending/running..."
    sleep 60
  done
}

log_step() {
    echo "========================================================================================"
    echo "========== [Step $1] $2"
    echo "========================================================================================"
}

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S')  $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S')  $1" >&2
    exit 1
}
EOF

cat > "$runtime_dir/02_submit_prepare_jax.sh" <<EOF
#!/bin/bash -l
#SBATCH --gres=${AF3SCORE_GRES}
#SBATCH -N 1
#SBATCH --cpus-per-task=${AF3SCORE_CPUS}
#SBATCH --mem=${AF3SCORE_MEM}
#SBATCH -J af3_prep_jax

set -euo pipefail
module load ${AF3SCORE_CUDA_MODULE} || true
export XLA_PYTHON_CLIENT_MEM_FRACTION=${AF3SCORE_JAX_MEM_FRACTION}
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

"\$4" "\$3/02_prepare_pdb2jax.py" \\
  --pdb_folder "\$1" \\
  --output_folder "\$2" \\
  --num_workers "${AF3SCORE_PREPARE_WORKERS}"
EOF

cat > "$runtime_dir/03_submit_af3score.sh" <<EOF
#!/bin/bash -l
#SBATCH --gres=${AF3SCORE_GRES}
#SBATCH -N 1
#SBATCH --cpus-per-task=${AF3SCORE_CPUS}
#SBATCH --mem=${AF3SCORE_MEM}
#SBATCH -J af3score

set -euo pipefail
echo "========== Job started at: \$(date) =========="
start_time=\$(date +%s)

module load ${AF3SCORE_CUDA_MODULE} || true
export XLA_FLAGS="\${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${AF3SCORE_JAX_MEM_FRACTION}
export XLA_CLIENT_MEM_FRACTION=${AF3SCORE_JAX_MEM_FRACTION}
if [[ -n "\${AF3SCORE_HMMER_PATH:-}" ]]; then
  export PATH="\${AF3SCORE_HMMER_PATH}:\$PATH"
fi
export PATH="\$("\$4" -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):\$PATH"

batch_json_dir="\$1"
batch_h5_dir="\$2"
output_dir="\$3"
buckets=\$(basename "\$batch_json_dir" | grep -oE '[0-9]+$' || true)
if [[ -z "\$buckets" ]]; then
  buckets=512
fi

echo "Running AF3Score on: \$batch_json_dir  \$batch_h5_dir  buckets=\$buckets -> \$output_dir"

"\$4" "\$5/run_af3score.py" \\
  --db_dir="${AF3SCORE_DB_DIR}" \\
  --model_dir="${AF3SCORE_MODEL_DIR}" \\
  --batch_json_dir="\$batch_json_dir" \\
  --batch_h5_dir="\$batch_h5_dir" \\
  --output_dir="\$output_dir" \\
  --run_data_pipeline=False \\
  --run_inference=true \\
  --init_guess=true \\
  --num_samples=1 \\
  --buckets="\$buckets" \\
  --flash_attention_implementation="${AF3SCORE_FLASH_ATTENTION}" \\
  --write_cif_model=False \\
  --write_summary_confidences=true \\
  --write_full_confidences=true \\
  --write_best_model_root=false \\
  --write_ranking_scores_csv=false \\
  --write_terms_of_use_file=false \\
  --write_fold_input_json_file=false

end_time=\$(date +%s)
elapsed=\$((end_time - start_time))
echo "========== Job finished at: \$(date) =========="
echo "========== Total runtime: \${elapsed} seconds =========="
EOF

chmod +x "$runtime_dir/"*.sh

cat > "$output_dir/af3score_hpcc_runtime_config.txt" <<EOF
AF3SCORE_DIR=$AF3SCORE_DIR
AF3SCORE_PYTHON=$AF3SCORE_PYTHON
AF3SCORE_PARTITION=$AF3SCORE_PARTITION
AF3SCORE_NODELIST=$AF3SCORE_NODELIST
AF3SCORE_GRES=$AF3SCORE_GRES
AF3SCORE_CPUS=$AF3SCORE_CPUS
AF3SCORE_MEM=$AF3SCORE_MEM
AF3SCORE_CUDA_MODULE=$AF3SCORE_CUDA_MODULE
AF3SCORE_DB_DIR=$AF3SCORE_DB_DIR
AF3SCORE_MODEL_DIR=$AF3SCORE_MODEL_DIR
AF3SCORE_FLASH_ATTENTION=$AF3SCORE_FLASH_ATTENTION
EOF

exec "$runtime_dir/AF3score_pipeline.sh" "$input_pdb_dir" "$output_dir" "$num_jobs"
