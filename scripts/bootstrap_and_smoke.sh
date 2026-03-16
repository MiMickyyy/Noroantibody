#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_inputs.py
python scripts/prepare_targets.py --pipeline-config data/configs/pipeline.yaml --campaign-config data/configs/hotspot_campaigns.yaml
python scripts/run_pipeline.py --phase phase0_smoke --dry-run
