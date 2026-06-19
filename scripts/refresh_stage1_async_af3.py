#!/usr/bin/env python3
"""Refresh Stage1 async AF3Score metrics into condition summaries.

The Stage1 persistent worker submits AF3Score as independent Slurm jobs. Those
jobs can finish after the condition CSV has already been written with
``submitted_async`` placeholders. This helper reads ``af3score_async_jobs.csv``,
parses completed AF3Score metrics, updates each condition CSV in place, then
optionally runs the normal Stage1 merge step.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_common import read_yaml, resolve_path  # noqa: E402
from run_pipeline import af3score_validation_pass  # noqa: E402
from tool_wrappers import _parse_af3score_metric_csv, _safe_float  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", required=True)
    p.add_argument("--pipeline-config", default="data/configs/stage1_5o04/pipeline.stage1_baker_hotspot_2700.yaml")
    p.add_argument("--tooling-config", default="data/configs/stage1_5o04/tooling.hpcc.yaml")
    p.add_argument("--resolved-inputs", default="data/configs/stage1_5o04/resolved_inputs.hpcc.yaml")
    p.add_argument("--resolved-targets", default="data/configs/stage1_5o04/resolved_targets.full_target.yaml")
    p.add_argument("--cdr-config", default="data/configs/cdr_boundaries.yaml")
    p.add_argument("--no-merge", action="store_true")
    return p.parse_args()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    fields = list(fieldnames)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    tmp.replace(path)


def is_completed_metric(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return any(csv.DictReader(handle))
    except Exception:
        return False


def combined_score(rf2_score: float, af3_score: float, af3_cfg: dict) -> float:
    if math.isnan(af3_score):
        return round(float(rf2_score), 6)
    weights = af3_cfg.get("ranking_weights", {}) or {}
    rf2_weight = float(weights.get("rf2", 0.6))
    af3_weight = float(weights.get("af3score", 0.4))
    denom = rf2_weight + af3_weight
    if denom <= 0:
        rf2_weight, af3_weight, denom = 1.0, 0.0, 1.0
    return round(float((rf2_weight * rf2_score + af3_weight * af3_score) / denom), 6)


def update_condition_csv(condition_csv: Path, candidate_id: str, metrics: dict, af3_cfg: dict) -> bool:
    fields, rows = read_csv_rows(condition_csv)
    if not rows:
        return False
    changed = False
    for row in rows:
        if str(row.get("candidate_id", "")) != candidate_id:
            continue
        rf2_score = _safe_float(row.get("ranking_score"), _safe_float(row.get("combined_ranking_score"), 0.0))
        af3_score = _safe_float(metrics.get("af3score_rank_score"), float("nan"))
        metrics["rf2_relaxed_pass"] = int(float(row.get("rf2_relaxed_pass") or 0))
        metrics["af3score_validation_pass"] = af3score_validation_pass(metrics, af3_cfg)
        metrics["combined_ranking_score"] = combined_score(rf2_score, af3_score, af3_cfg)
        for key, value in metrics.items():
            row[key] = value
            if key not in fields:
                fields.append(key)
        changed = True
    if changed:
        write_csv_rows(condition_csv, fields, rows)
    return changed


def refresh(root: Path, args: argparse.Namespace) -> int:
    out_root = resolve_path(root, args.output_root)
    manifest = out_root / "af3score_async_jobs.csv"
    pipeline_cfg = read_yaml(resolve_path(root, args.pipeline_config))
    af3_cfg = pipeline_cfg.get("af3score", {}) or {}
    _, jobs = read_csv_rows(manifest)
    updated = 0
    completed = 0
    pending = 0
    for job in jobs:
        metrics_csv = Path(str(job.get("af3score_metric_csv", "")))
        if not metrics_csv.is_absolute():
            metrics_csv = root / metrics_csv
        if not is_completed_metric(metrics_csv):
            pending += 1
            continue
        completed += 1
        candidate_id = str(job.get("candidate_id", "")).strip()
        condition_name = str(job.get("condition_name", "")).strip()
        expected_description = Path(str(job.get("af3score_output_dir", candidate_id))).name
        metrics = _parse_af3score_metric_csv(metrics_csv, expected_description)
        metrics["af3score_input_pdb"] = str(job.get("af3score_input_pdb", ""))
        metrics["af3score_output_dir"] = str(job.get("af3score_output_dir", ""))
        metrics["af3score_job_id"] = str(job.get("af3score_job_id", ""))
        condition_csv = out_root / "conditions" / condition_name / "condition_summary_compact.csv"
        if update_condition_csv(condition_csv, candidate_id, metrics, af3_cfg):
            updated += 1

    print(f"AF3 async refresh: completed_metrics={completed} pending_metrics={pending} updated_rows={updated}")
    if updated and not args.no_merge:
        cmd = [
            sys.executable,
            "scripts/run_stage1_5o04_campaign.py",
            "--merge-only",
            "--output-root",
            str(out_root),
            "--pipeline-config",
            args.pipeline_config,
            "--tooling-config",
            args.tooling_config,
            "--resolved-inputs",
            args.resolved_inputs,
            "--resolved-targets",
            args.resolved_targets,
            "--cdr-config",
            args.cdr_config,
        ]
        subprocess.run(cmd, cwd=root, check=False)
    return updated


def main() -> int:
    args = parse_args()
    root = Path(".").resolve()
    refresh(root, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
