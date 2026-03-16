#!/usr/bin/env python3
"""Lookup utility for structure / P-domain / full-length residue numbering."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import PipelineError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map residues between structure, P-domain, and full-length numbering.")
    parser.add_argument("--map", default="data/maps/residue_mapping_table.csv")
    parser.add_argument("--chain", default=None)
    parser.add_argument("--structure-resnum", type=int, default=None)
    parser.add_argument("--full-length-resnum", type=int, default=None)
    parser.add_argument("--p-domain-resnum", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    map_path = Path(args.map)
    if not map_path.exists():
        raise PipelineError(f"Mapping table not found: {map_path}")

    df = pd.read_csv(map_path)
    q = df
    if args.chain is not None:
        q = q[q["structure_chain"].astype(str) == str(args.chain)]
    if args.structure_resnum is not None:
        q = q[q["structure_resnum"].astype(int) == int(args.structure_resnum)]
    if args.full_length_resnum is not None:
        q = q[q["full_length_resnum"].fillna(-1).astype(int) == int(args.full_length_resnum)]
    if args.p_domain_resnum is not None:
        q = q[q["p_domain_resnum"].fillna(-1).astype(int) == int(args.p_domain_resnum)]

    if q.empty:
        print("No mapping rows matched query.")
        return 0

    cols = [
        "structure_chain",
        "structure_resnum",
        "structure_icode",
        "resname",
        "p_domain_resnum",
        "full_length_resnum",
        "in_cropped_target",
    ]
    print(q[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
