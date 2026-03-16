#!/usr/bin/env bash
set -euo pipefail
python scripts/run_pipeline.py --phase phase4_h2_refine "$@"
