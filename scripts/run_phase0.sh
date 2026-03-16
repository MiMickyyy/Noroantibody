#!/usr/bin/env bash
set -euo pipefail
python scripts/run_pipeline.py --phase phase0_smoke "$@"
