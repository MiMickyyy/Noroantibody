#!/usr/bin/env python3
"""Small dry-run validation helper for end-to-end smoke checks."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    root = Path(".").resolve()
    run(["python", "scripts/prepare_inputs.py"])
    run([
        "python",
        "scripts/prepare_targets.py",
        "--pipeline-config",
        "data/configs/pipeline.yaml",
        "--campaign-config",
        "data/configs/hotspot_campaigns.yaml",
    ])
    run(["python", "scripts/run_pipeline.py", "--phase", "phase0_smoke", "--dry-run", "--limit-per-combination", "1"])
    print("Dry-run validation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
