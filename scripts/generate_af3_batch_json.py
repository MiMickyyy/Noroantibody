#!/usr/bin/env python3
"""Generate AlphaFold Server batch JSON (alphafoldserver dialect) for nanobody-antigen jobs.

Default behavior:
- take design nanobody sequences from af3_final25_nanobody.fasta
- if fewer than target count, supplement from final25_h2_guardrailed.csv
- add one wild-type nanobody sequence
- pair each nanobody with P-domain monomer sequence at count=2 (dimer)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header = None
    seq_parts: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                seq = "".join(seq_parts).upper()
                if seq:
                    records.append((header, seq))
            header = line[1:].strip() or f"seq_{len(records)+1}"
            seq_parts = []
        else:
            seq_parts.append(re.sub(r"[^A-Za-z]", "", line))
    if header is not None:
        seq = "".join(seq_parts).upper()
        if seq:
            records.append((header, seq))
    return records


def extract_longest_aa_run(path: Path, min_len: int = 30) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blob = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in text)
    candidates = re.findall(r"[ACDEFGHIKLMNPQRSTVWY*]{%d,}" % min_len, blob)
    if not candidates:
        raise ValueError(f"Could not extract sequence from {path}")
    seq = max(candidates, key=len).replace("*", "")
    return seq.upper()


def validate_aa_sequence(seq: str, label: str):
    if not AA_RE.match(seq):
        bad = sorted(set(re.sub(r"[ACDEFGHIKLMNPQRSTVWY]", "", seq)))
        raise ValueError(f"{label} contains non-standard amino-acid letters: {bad}")


def sanitize_name(text: str, max_len: int = 90) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    if not name:
        name = "job"
    return name[:max_len]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AF3 alphafoldserver JSON batch.")
    parser.add_argument("--design-fasta", default="af3_final25_nanobody.fasta")
    parser.add_argument("--wildtype-fasta", default="Nanobody.fa")
    parser.add_argument("--antigen-seq-file", default="P-domain dimer.prot")
    parser.add_argument("--supplement-csv", default="results/summaries/final25_h2_guardrailed.csv")
    parser.add_argument("--target-design-count", type=int, default=25)
    parser.add_argument("--out-json", default="results/af3_web_exports/af3_26_jobs.json")
    parser.add_argument("--out-map-csv", default="results/af3_web_exports/af3_26_jobs_map.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(".").resolve()

    design_fasta = (root / args.design_fasta).resolve()
    wt_fasta = (root / args.wildtype_fasta).resolve()
    antigen_file = (root / args.antigen_seq_file).resolve()
    supplement_csv = (root / args.supplement_csv).resolve()
    out_json = (root / args.out_json).resolve()
    out_map = (root / args.out_map_csv).resolve()

    if not design_fasta.exists():
        raise FileNotFoundError(f"Missing design FASTA: {design_fasta}")
    if not wt_fasta.exists():
        raise FileNotFoundError(f"Missing wildtype FASTA: {wt_fasta}")
    if not antigen_file.exists():
        raise FileNotFoundError(f"Missing antigen sequence file: {antigen_file}")

    design_records = read_fasta(design_fasta)
    if not design_records:
        raise ValueError(f"No sequences found in {design_fasta}")

    selected: List[Dict[str, str]] = []
    seen_ids = set()
    seen_seqs = set()

    for cid, seq in design_records:
        seq = seq.upper()
        validate_aa_sequence(seq, f"design:{cid}")
        if cid in seen_ids:
            continue
        selected.append(
            {"candidate_id": cid, "sequence": seq, "source": f"design_fasta:{design_fasta.name}"}
        )
        seen_ids.add(cid)
        seen_seqs.add(seq)
        if len(selected) >= args.target_design_count:
            break

    if len(selected) < args.target_design_count and supplement_csv.exists():
        df = pd.read_csv(supplement_csv)
        req_cols = {"candidate_id", "full_sequence"}
        if req_cols.issubset(df.columns):
            for _, row in df.iterrows():
                cid = str(row["candidate_id"]).strip()
                seq = str(row["full_sequence"]).strip().upper()
                if not cid or not seq:
                    continue
                if cid in seen_ids or seq in seen_seqs:
                    continue
                validate_aa_sequence(seq, f"supplement:{cid}")
                selected.append(
                    {"candidate_id": cid, "sequence": seq, "source": f"supplement_csv:{supplement_csv.name}"}
                )
                seen_ids.add(cid)
                seen_seqs.add(seq)
                if len(selected) >= args.target_design_count:
                    break

    if len(selected) < args.target_design_count:
        raise ValueError(
            f"Need {args.target_design_count} design sequences, got {len(selected)} "
            f"(from {design_fasta.name} + supplement)."
        )

    wt_records = read_fasta(wt_fasta)
    if not wt_records:
        raise ValueError(f"No wildtype sequence found in {wt_fasta}")
    wt_id, wt_seq = wt_records[0]
    wt_seq = wt_seq.upper()
    validate_aa_sequence(wt_seq, "wildtype")

    antigen_seq = extract_longest_aa_run(antigen_file, min_len=30)
    validate_aa_sequence(antigen_seq, "antigen")

    jobs: List[dict] = []
    mapping_rows: List[dict] = []

    for idx, rec in enumerate(selected, start=1):
        cid = rec["candidate_id"]
        seq = rec["sequence"]
        job_name = sanitize_name(f"{idx:02d}_{cid}")
        jobs.append(
            {
                "name": job_name,
                "modelSeeds": [],
                "sequences": [
                    {"proteinChain": {"sequence": seq, "count": 1}},
                    {"proteinChain": {"sequence": antigen_seq, "count": 2}},
                ],
                "dialect": "alphafoldserver",
                "version": 1,
            }
        )
        mapping_rows.append(
            {
                "job_name": job_name,
                "candidate_id": cid,
                "source": rec["source"],
                "nanobody_length": len(seq),
            }
        )

    wt_job_name = sanitize_name(f"{len(selected)+1:02d}_wildtype_{wt_id}")
    jobs.append(
        {
            "name": wt_job_name,
            "modelSeeds": [],
            "sequences": [
                {"proteinChain": {"sequence": wt_seq, "count": 1}},
                {"proteinChain": {"sequence": antigen_seq, "count": 2}},
            ],
            "dialect": "alphafoldserver",
            "version": 1,
        }
    )
    mapping_rows.append(
        {
            "job_name": wt_job_name,
            "candidate_id": "wildtype",
            "source": f"wildtype_fasta:{wt_fasta.name}",
            "nanobody_length": len(wt_seq),
        }
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_map.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(jobs, indent=2), encoding="utf-8")

    with out_map.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["job_name", "candidate_id", "source", "nanobody_length"],
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    print(f"Wrote AF3 batch JSON: {out_json}")
    print(f"Wrote AF3 job map: {out_map}")
    print(f"Design sequences used: {args.target_design_count}")
    print(f"Wildtype added: 1")
    print(f"Total jobs: {len(jobs)}")
    print(f"Antigen sequence length (monomer): {len(antigen_seq)}; count=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

