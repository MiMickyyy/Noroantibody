#!/usr/bin/env python3
"""Summarize CDR1 rescue Phase5/Phase6 outputs into easy-to-review tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import PipelineError, atomic_write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CDR1 rescue pilot/main outputs.")
    parser.add_argument(
        "--phase5-ranking-csv",
        default="results/summaries/phase5_cdr1_rescue_condition_ranking.csv",
    )
    parser.add_argument(
        "--phase6-ranked-csv",
        default="results/summaries/phase6_cdr1_rescue_final_ranked_candidates.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="results/summaries",
    )
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(".").resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    phase5_path = (root / args.phase5_ranking_csv).resolve()
    phase6_path = (root / args.phase6_ranked_csv).resolve()

    if not phase5_path.exists():
        raise PipelineError(f"Missing phase5 ranking CSV: {phase5_path}")
    if not phase6_path.exists():
        raise PipelineError(f"Missing phase6 ranked CSV: {phase6_path}")

    phase5_df = pd.read_csv(phase5_path)
    phase6_df = pd.read_csv(phase6_path)
    if phase5_df.empty:
        raise PipelineError(f"Phase5 ranking CSV is empty: {phase5_path}")
    if phase6_df.empty:
        raise PipelineError(f"Phase6 ranked CSV is empty: {phase6_path}")

    phase5_out = out_dir / "phase5_cdr1_rescue_rank_summary.csv"
    phase6_cond_out = out_dir / "phase6_cdr1_rescue_condition_summary.csv"
    phase6_top_out = out_dir / "phase6_cdr1_rescue_top_candidates_preview.csv"

    phase5_cols = [
        "rank",
        "selected_for_phase6",
        "condition_id",
        "parent_candidate_id",
        "hotspot_set_name",
        "editable_cdr1_positions",
        "total_designs",
        "strict_pass_count",
        "relaxed_pass_count",
        "strict_pass_rate",
        "relaxed_pass_rate",
        "mean_ranking_score",
        "mean_rf2_pae",
        "mean_design_rf2_rmsd",
        "unique_sequence_count",
        "unique_backbone_count",
    ]
    phase5_export = phase5_df[[c for c in phase5_cols if c in phase5_df.columns]].copy()
    atomic_write_csv(phase5_out, phase5_export.to_dict(orient="records"), list(phase5_export.columns))

    for col in ("strict_pass", "relaxed_pass", "rf2_pae", "design_rf2_rmsd", "ranking_score"):
        if col in phase6_df.columns:
            phase6_df[col] = pd.to_numeric(phase6_df[col], errors="coerce")

    cond_summary = (
        phase6_df.groupby(["condition_id", "parent_candidate_id", "hotspot_set_name"], as_index=False)
        .agg(
            total_candidates=("candidate_id", "count"),
            strict_pass_count=("strict_pass", "sum"),
            relaxed_pass_count=("relaxed_pass", "sum"),
            mean_ranking_score=("ranking_score", "mean"),
            mean_rf2_pae=("rf2_pae", "mean"),
            mean_design_rf2_rmsd=("design_rf2_rmsd", "mean"),
            unique_sequence_count=("full_sequence", "nunique"),
            unique_backbone_count=("backbone_id", "nunique"),
        )
        .sort_values(
            ["strict_pass_count", "relaxed_pass_count", "mean_ranking_score", "mean_design_rf2_rmsd", "mean_rf2_pae"],
            ascending=[False, False, False, True, True],
        )
    )
    atomic_write_csv(phase6_cond_out, cond_summary.to_dict(orient="records"), list(cond_summary.columns))

    top_df = phase6_df.sort_values(
        ["strict_pass", "relaxed_pass", "ranking_score", "design_rf2_rmsd", "rf2_pae"],
        ascending=[False, False, False, True, True],
    ).head(int(args.top_n))
    top_cols = [
        "candidate_id",
        "condition_id",
        "parent_candidate_id",
        "hotspot_set_name",
        "strict_pass",
        "relaxed_pass",
        "rf2_pae",
        "design_rf2_rmsd",
        "ranking_score",
        "cdr1_edited_positions",
        "h1_sequence",
        "full_sequence",
    ]
    top_export = top_df[[c for c in top_cols if c in top_df.columns]].copy()
    atomic_write_csv(phase6_top_out, top_export.to_dict(orient="records"), list(top_export.columns))

    print(f"Wrote: {phase5_out}")
    print(f"Wrote: {phase6_cond_out}")
    print(f"Wrote: {phase6_top_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
