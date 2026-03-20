#!/usr/bin/env python3
"""Check per-combination diversity for a pipeline phase."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check candidate diversity per combination.")
    p.add_argument("--phase", required=True, help="Phase directory name, e.g. phase2_focused_pilot")
    p.add_argument("--root", default=".", help="Project root directory")
    p.add_argument(
        "--out-csv",
        default="",
        help="Optional output CSV path. Default: results/summaries/<phase>_diversity_check.csv",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    combo_root = root / args.phase / "combinations"
    if not combo_root.exists():
        raise SystemExit(f"Missing phase combinations directory: {combo_root}")

    rows = []
    for combo_dir in sorted(p for p in combo_root.iterdir() if p.is_dir()):
        cfile = combo_dir / "candidates.csv"
        if not cfile.exists():
            continue
        df = pd.read_csv(cfile)
        if df.empty:
            continue

        seq_col = df["full_sequence"].fillna("").astype(str).str.strip() if "full_sequence" in df.columns else pd.Series([], dtype=str)
        rank_col = df["ranking_score"] if "ranking_score" in df.columns else pd.Series([], dtype=float)
        sig_col = (
            df["backbone_signature"].fillna("").astype(str).str.strip()
            if "backbone_signature" in df.columns
            else pd.Series([""] * len(df), dtype=str)
        )
        hard_col = pd.to_numeric(df.get("hard_filter_pass", 0), errors="coerce").fillna(0)

        known_sig = int(sig_col.ne("").sum())
        uniq_sig = int(sig_col[sig_col.ne("")].nunique())
        if int(df.shape[0]) <= 1:
            status = "ok"
        elif known_sig == 0:
            status = "no_signature_data"
        elif uniq_sig <= 1:
            status = "warn_backbone_diversity"
        else:
            status = "ok"
        rows.append(
            {
                "combination_id": combo_dir.name,
                "rows": int(df.shape[0]),
                "uniq_seq": int(seq_col[seq_col.ne("")].nunique()),
                "uniq_score": int(pd.to_numeric(rank_col, errors="coerce").nunique(dropna=True)),
                "known_backbone_signature_rows": known_sig,
                "uniq_backbone_signature": uniq_sig,
                "hard_pass": int(hard_col.sum()),
                "status": status,
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise SystemExit("No candidates.csv found under phase combinations.")

    out_df = out_df.sort_values(
        by=["status", "hard_pass", "uniq_backbone_signature", "uniq_seq", "uniq_score", "combination_id"],
        ascending=[True, False, False, False, False, True],
    )

    out_csv = (
        Path(args.out_csv).resolve()
        if args.out_csv
        else (root / "results" / "summaries" / f"{args.phase}_diversity_check.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"Wrote: {out_csv}")
    print(out_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
