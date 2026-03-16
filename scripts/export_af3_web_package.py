#!/usr/bin/env python3
"""Standalone AF3-web export helper for final 25 candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import PipelineError, SAFETY_ETHICS_STATEMENT, atomic_write_csv, read_sequence_file, read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AF3-web-ready package from final candidate table.")
    parser.add_argument("--final-table", default="results/summaries/final25_h2_optimized_candidates.csv")
    parser.add_argument("--resolved-inputs", default="data/processed/resolved_inputs.yaml")
    parser.add_argument("--resolved-targets", default="data/processed/resolved_targets.yaml")
    parser.add_argument("--outdir", default="results/af3_web_exports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(".").resolve()
    final_table = root / args.final_table
    if not final_table.exists():
        raise PipelineError(f"Missing final table: {final_table}")

    resolved_inputs = read_yaml(root / args.resolved_inputs).get("resolved_inputs", {})
    resolved_targets = read_yaml(root / args.resolved_targets)

    df = pd.read_csv(final_table)
    if df.empty:
        raise PipelineError("Final table is empty.")

    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    submission_csv = outdir / "af3_web_submission_table.csv"
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "candidate_id": r.get("candidate_id", ""),
                "campaign_name": r.get("campaign_name", ""),
                "combination_id": r.get("combination_id", ""),
                "h1_length": r.get("h1_length", ""),
                "h3_length": r.get("h3_length", ""),
                "h1_sequence": r.get("h1_sequence", ""),
                "h2_sequence": r.get("h2_sequence", ""),
                "h3_sequence": r.get("h3_sequence", ""),
                "full_nanobody_sequence": r.get("full_sequence", ""),
                "rf2_pae": r.get("rf2_pae", ""),
                "design_rf2_rmsd": r.get("design_rf2_rmsd", ""),
                "ranking_score": r.get("ranking_score", ""),
                "parent_backbone_id": r.get("parent_backbone_id", ""),
                "parent_backbone_pdb": r.get("parent_backbone_pdb", r.get("backbone_pdb", "")),
                "rf2_best_pdb": r.get("rf2_best_pdb", ""),
                "notes": r.get("warning", ""),
            }
        )

    atomic_write_csv(
        submission_csv,
        rows,
        [
            "candidate_id",
            "campaign_name",
            "combination_id",
            "h1_length",
            "h3_length",
            "h1_sequence",
            "h2_sequence",
            "h3_sequence",
            "full_nanobody_sequence",
            "rf2_pae",
            "design_rf2_rmsd",
            "ranking_score",
            "parent_backbone_id",
            "parent_backbone_pdb",
            "rf2_best_pdb",
            "notes",
        ],
    )

    fasta_out = outdir / "af3_final25_nanobody.fasta"
    with fasta_out.open("w", encoding="utf-8") as handle:
        for _, r in df.iterrows():
            handle.write(f">{r['candidate_id']}\n{r['full_sequence']}\n")

    vp1_seq = read_sequence_file(Path(resolved_inputs["vp1_sequence_file"]))[0][1]
    pdom_seq = read_sequence_file(Path(resolved_inputs["p_domain_dimer_sequence_file"]))[0][1]

    context_txt = outdir / "af3_antigen_context.txt"
    context_txt.write_text(
        "\n".join(
            [
                "AF3 manual submission context",
                "============================",
                f"Full cleaned antigen target: {resolved_targets.get('full_cleaned_target', '')}",
                f"Cropped design target: {resolved_targets.get('cropped_design_target', '')}",
                f"Residue mapping table: {resolved_targets.get('mapping_table', '')}",
                f"VP1 length: {len(vp1_seq)}",
                f"P-domain length: {len(pdom_seq)}",
                "No local AF3 execution performed by this pipeline.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = outdir / "af3_handoff_summary.md"
    summary.write_text(
        "\n".join(
            [
                SAFETY_ETHICS_STATEMENT,
                "",
                "# AF3 Web Export Summary",
                "",
                f"- Input final table: `{final_table}`",
                f"- Submission CSV: `{submission_csv}`",
                f"- FASTA: `{fasta_out}`",
                f"- Antigen context: `{context_txt}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"AF3 export written to: {outdir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
