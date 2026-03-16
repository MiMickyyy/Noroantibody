#!/usr/bin/env python3
"""Parse candidate outputs and regenerate ranking/selection tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import PipelineError, atomic_write_csv, greedy_sequence_dedup, read_yaml
from tool_wrappers import combine_weighted_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-parse and rank candidate outputs by phase.")
    parser.add_argument("--phase", required=True, choices=["phase1_coarse_pilot", "phase2_focused_pilot", "phase3_main_campaign"])
    parser.add_argument("--pipeline-config", default="data/configs/pipeline.yaml")
    parser.add_argument("--top-combinations", type=int, default=None)
    parser.add_argument("--top-candidates", type=int, default=25)
    return parser.parse_args()


def collect_candidates(phase_dir: Path) -> pd.DataFrame:
    rows = []
    combos = phase_dir / "combinations"
    if not combos.exists():
        return pd.DataFrame()
    for cdir in combos.iterdir():
        cfile = cdir / "candidates.csv"
        if cfile.exists():
            rows.append(pd.read_csv(cfile))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    args = parse_args()
    root = Path(".").resolve()
    pipeline_cfg = read_yaml(root / args.pipeline_config)
    filters = pipeline_cfg.get("filters", {})
    th = filters.get("hard_thresholds", {})
    weights = filters.get("ranking_weights", {})

    df = collect_candidates(root / args.phase)
    if df.empty:
        raise PipelineError(f"No candidate files found for {args.phase}")

    df["hard_filter_pass"] = (
        (df["rf2_pae"].astype(float) < float(th.get("rf2_pae_max", 10.0)))
        & (df["design_rf2_rmsd"].astype(float) < float(th.get("design_rf2_rmsd_max", 2.0)))
    ).astype(int)

    scores = []
    for _, row in df.iterrows():
        scores.append(
            combine_weighted_score(
                {
                    "rf2_pae": row["rf2_pae"],
                    "design_rf2_rmsd": row["design_rf2_rmsd"],
                    "hotspot_agreement": row.get("hotspot_agreement", 0),
                    "groove_localization": row.get("groove_localization", 0),
                    "h1_h3_role_consistency": row.get("h1_h3_role_consistency", 0),
                    "structural_plausibility": row.get("structural_plausibility", 0),
                },
                weights,
            )
        )
    df["ranking_score"] = scores

    summary = (
        df.groupby(["combination_id", "campaign_name", "h1_length", "h3_length"], as_index=False)
        .agg(
            total_candidates=("candidate_id", "count"),
            hard_pass_candidates=("hard_filter_pass", "sum"),
            best_ranking_score=("ranking_score", "max"),
            mean_ranking_score=("ranking_score", "mean"),
        )
        .sort_values(["hard_pass_candidates", "best_ranking_score", "mean_ranking_score"], ascending=False)
    )

    out_summary = root / "results/summaries" / f"{args.phase}_summary_reparsed.csv"
    summary.to_csv(out_summary, index=False)

    if args.phase in {"phase1_coarse_pilot", "phase2_focused_pilot"}:
        top_n = int(args.top_combinations or (8 if args.phase == "phase1_coarse_pilot" else 2))
        top = summary.head(top_n).copy()
        top.insert(0, "rank", range(1, top.shape[0] + 1))
        out_top = root / "results/summaries" / (
            "phase1_top8_combinations.csv" if args.phase == "phase1_coarse_pilot" else "phase2_top2_combinations.csv"
        )
        top.to_csv(out_top, index=False)

    if args.phase == "phase3_main_campaign":
        passed = df[df["hard_filter_pass"] == 1].copy()
        if passed.empty:
            passed = df.copy()
        rows = passed.to_dict(orient="records")
        deduped = greedy_sequence_dedup(
            rows=rows,
            sequence_key="full_sequence",
            score_key="ranking_score",
            identity_threshold=float(pipeline_cfg.get("postprocess", {}).get("sequence_dedup_identity_threshold", 0.95)),
        )
        top = sorted(deduped, key=lambda x: float(x.get("ranking_score", 0.0)), reverse=True)[: int(args.top_candidates)]
        atomic_write_csv(
            root / "results/summaries/phase3_top25_pre_h2.csv",
            top,
            list(df.columns),
        )

    print(f"Re-parsing complete: {out_summary}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
