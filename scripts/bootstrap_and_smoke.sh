#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Preparing input aliases and resolved input metadata"
python scripts/prepare_inputs.py

echo "[2/4] Preparing target structures, crops, and residue maps"
python scripts/prepare_targets.py \
  --pipeline-config data/configs/pipeline.yaml \
  --campaign-config data/configs/hotspot_campaigns.yaml

echo "[3/4] Validating runtime/tooling detection"
python scripts/autodetect_runtime_and_tooling.py --strict

echo "[4/4] Running phase0 smoke dry-run"
python scripts/run_pipeline.py --phase phase0_smoke --dry-run --limit-per-combination 1

echo "Bootstrap-and-smoke completed successfully."
