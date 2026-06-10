#!/bin/bash -l
# Build the HPCC Python environment needed by Mingchenchen/AF3Score.

set -euo pipefail

module load gcc/12.2.0 || true

AF3SCORE_DIR="${AF3SCORE_DIR:-/rhome/xli616/Norovirus/external/AF3Score}"
AF3SCORE_ENV_PREFIX="${AF3SCORE_ENV_PREFIX:-/rhome/xli616/.conda/envs/noro_af3score}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/rhome/xli616/.conda/pkgs}"
MAMBA_EXE="${MAMBA_EXE:-$(command -v mamba || true)}"
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
export CONDA_PKGS_DIRS
mkdir -p "$CONDA_PKGS_DIRS"

fail() {
  echo "[AF3Score bootstrap] ERROR: $*" >&2
  exit 1
}

[[ -d "$AF3SCORE_DIR" ]] || fail "AF3SCORE_DIR not found: $AF3SCORE_DIR"

if [[ ! -x "$AF3SCORE_ENV_PREFIX/bin/python" ]]; then
  if [[ -n "$MAMBA_EXE" ]]; then
    "$MAMBA_EXE" create -y -p "$AF3SCORE_ENV_PREFIX" python=3.11 pip cmake ninja
  elif [[ -n "$CONDA_EXE" ]]; then
    "$CONDA_EXE" create -y -p "$AF3SCORE_ENV_PREFIX" python=3.11 pip cmake ninja
  else
    fail "Neither mamba nor conda is available."
  fi
fi

PYTHON="$AF3SCORE_ENV_PREFIX/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel

cd "$AF3SCORE_DIR"
"$PYTHON" -m pip install -r dev-requirements.txt
"$PYTHON" -m pip install .
"$PYTHON" -m pip install biopython h5py pandas tqdm
"$AF3SCORE_ENV_PREFIX/bin/build_data"

"$PYTHON" - <<'PY'
import importlib
for module in ("alphafold3", "Bio", "h5py", "jax", "numpy", "pandas", "tqdm"):
    importlib.import_module(module)
print("AF3Score Python environment OK")
PY
