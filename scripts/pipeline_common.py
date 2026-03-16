#!/usr/bin/env python3
"""Shared utilities for the Norovirus RFantibody pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from Bio.Align import PairwiseAligner

SAFETY_ETHICS_STATEMENT = (
    "Safety and Ethics Statement:\n"
    "This study is a computational structural modeling and protein design project focused on "
    "nanobody–Norovirus interactions. The work uses Virus-Like Particle (VLP)-related structural "
    "information only and does not involve infectious virus, viral propagation, animal experiments, "
    "human subjects, clinical samples, or wet-lab experimental procedures. All project activities are "
    "conducted under institutional safety and ethics oversight at the University of California, Riverside."
)


class PipelineError(RuntimeError):
    """Raised when required inputs/config are invalid for pipeline execution."""


@dataclass(frozen=True)
class CDRBoundaries:
    h1: Tuple[int, int]
    h2: Tuple[int, int]
    h3: Tuple[int, int]
    chain_id: str

    @property
    def h1_len(self) -> int:
        return self.h1[1] - self.h1[0] + 1

    @property
    def h2_len(self) -> int:
        return self.h2[1] - self.h2[0] + 1

    @property
    def h3_len(self) -> int:
        return self.h3[1] - self.h3[0] + 1


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{now_str()}] {msg}")


def read_yaml(path: Path) -> dict:
    if not path.exists():
        raise PipelineError(f"Missing YAML config: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def write_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(paths: Sequence[Path]):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    text = text.strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "file"


def find_first_existing(base_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        if not name:
            continue
        path = base_dir / name
        if path.exists():
            return path
    return None


def read_sequence_file(path: Path) -> List[Tuple[str, str]]:
    """Reads FASTA or SnapGene-like .prot payload and returns [(id, sequence), ...]."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # FASTA parser
    if lines and lines[0].startswith(">"):
        records = []
        header = None
        seq_parts: List[str] = []
        for line in lines:
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts).upper().replace("*", "")))
                header = line[1:].strip() or f"seq_{len(records)+1}"
                seq_parts = []
            else:
                seq_parts.append(re.sub(r"[^A-Za-z*]", "", line))
        if header is not None:
            records.append((header, "".join(seq_parts).upper().replace("*", "")))
        records = [(h, s) for h, s in records if s]
        if records:
            return records

    # SnapGene/proprietary text payload extraction: pick long AA stretches
    blob = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in text)
    candidates = re.findall(r"[ACDEFGHIKLMNPQRSTVWY*]{30,}", blob)
    if candidates:
        candidates = sorted(candidates, key=len, reverse=True)
        return [(path.stem, candidates[0].replace("*", ""))]

    raise PipelineError(f"Could not extract protein sequence from file: {path}")


def sequence_identity(seq_a: str, seq_b: str, aligner: Optional[PairwiseAligner] = None) -> float:
    if not seq_a or not seq_b:
        return 0.0
    if aligner is None:
        aligner = PairwiseAligner(mode="global")
        aligner.match_score = 1.0
        aligner.mismatch_score = 0.0
        aligner.open_gap_score = 0.0
        aligner.extend_gap_score = 0.0
    score = aligner.score(seq_a, seq_b)
    return float(score) / float(max(len(seq_a), len(seq_b)))


def greedy_sequence_dedup(
    rows: List[dict],
    sequence_key: str,
    score_key: str,
    identity_threshold: float,
) -> List[dict]:
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda x: float(x.get(score_key, 0.0)), reverse=True)
    keep: List[dict] = []
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = 0.0
    aligner.extend_gap_score = 0.0

    for row in rows_sorted:
        seq = row.get(sequence_key, "")
        if not seq:
            continue
        too_close = False
        for selected in keep:
            sid = sequence_identity(seq, selected[sequence_key], aligner=aligner)
            if sid >= identity_threshold:
                too_close = True
                break
        if not too_close:
            keep.append(row)
    return keep


def deterministic_rng(seed_base: int, key: str) -> random.Random:
    key_hash = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16)
    return random.Random(seed_base + key_hash)


def run_command(
    cmd: Sequence[str],
    log_path: Path,
    cwd: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    """Run command with robust logging; supports dry-run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now_str()}] CMD: {' '.join(shlex.quote(x) for x in cmd)}\n")
        if dry_run:
            handle.write("DRY_RUN=1, command not executed.\n")
            return 0
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        handle.write(f"Exit code: {proc.returncode}\n")
        return int(proc.returncode)


def sanitize_pdb_for_rfantibody(src: Path, dst: Path) -> Dict[str, int]:
    """Sanitize PDB atom records for RFantibody parsers.

    Rules:
    - Keep `ATOM` records only (drop `HETATM` and `ANISOU`).
    - Keep altloc `' '` and `'A'`; normalize kept altloc to `' '`.
    - De-duplicate by (chain, resseq, icode, atom name), keeping first entry.
    - Preserve non-coordinate records (including REMARK/TER/END) as-is.
    """
    if not src.exists():
        raise PipelineError(f"Cannot sanitize missing PDB file: {src}")

    lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
    out_lines: List[str] = []
    seen_atom_keys = set()
    backbone_atoms = defaultdict(set)  # key=(chain,resseq,icode) -> {"N","CA","C"}
    residue_keys = set()

    stats = {
        "atoms_in": 0,
        "atoms_kept": 0,
        "dropped_altloc": 0,
        "dropped_duplicate_atom_records": 0,
        "dropped_hetatm": 0,
        "dropped_anisou": 0,
        "dropped_malformed": 0,
        "residues_total": 0,
        "residues_missing_backbone": 0,
    }

    for raw in lines:
        line = raw.rstrip("\n")
        rec = line[:6]

        if rec.startswith("ATOM"):
            stats["atoms_in"] += 1
            if len(line) < 54:
                stats["dropped_malformed"] += 1
                continue

            # PDB fixed columns: altLoc is column 17 (0-based index 16).
            altloc = line[16] if len(line) > 16 else " "
            if altloc not in (" ", "A"):
                stats["dropped_altloc"] += 1
                continue

            chain = line[21] if len(line) > 21 else " "
            resseq = line[22:26] if len(line) > 25 else "    "
            icode = line[26] if len(line) > 26 else " "
            atom_name = line[12:16] if len(line) > 15 else "    "
            atom_key = (chain, resseq, icode, atom_name)

            if atom_key in seen_atom_keys:
                stats["dropped_duplicate_atom_records"] += 1
                continue
            seen_atom_keys.add(atom_key)

            residue_key = (chain, resseq, icode)
            residue_keys.add(residue_key)
            atom_trim = atom_name.strip().upper()
            if atom_trim in {"N", "CA", "C"}:
                backbone_atoms[residue_key].add(atom_trim)

            # Normalize altLoc for kept atoms.
            norm = line if len(line) >= 80 else line.ljust(80)
            if altloc != " ":
                norm = norm[:16] + " " + norm[17:]

            out_lines.append(norm.rstrip())
            stats["atoms_kept"] += 1
            continue

        if rec.startswith("HETATM"):
            stats["dropped_hetatm"] += 1
            continue

        if rec.startswith("ANISOU"):
            stats["dropped_anisou"] += 1
            continue

        out_lines.append(line)

    missing_backbone = [
        key for key in residue_keys if not {"N", "CA", "C"}.issubset(backbone_atoms.get(key, set()))
    ]
    stats["residues_total"] = len(residue_keys)
    stats["residues_missing_backbone"] = len(missing_backbone)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    tmp_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    tmp_path.replace(dst)
    return stats


def atomic_write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(path)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_int_pair(value, name: str) -> Tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise PipelineError(f"{name} must be [start, end]")
    a, b = value
    if a is None or b is None:
        raise PipelineError(f"{name} is missing start/end")
    a = int(a)
    b = int(b)
    if a > b:
        a, b = b, a
    return a, b


def load_cdr_boundaries(path: Path) -> CDRBoundaries:
    cfg = read_yaml(path)
    cdr = cfg.get("cdr_boundaries", {})
    chain = str(cfg.get("nanobody_chain_id", "")).strip()
    if not chain:
        raise PipelineError("Missing nanobody_chain_id in cdr_boundaries.yaml")
    h1 = safe_int_pair(cdr.get("H1"), "cdr_boundaries.H1")
    h2 = safe_int_pair(cdr.get("H2"), "cdr_boundaries.H2")
    h3 = safe_int_pair(cdr.get("H3"), "cdr_boundaries.H3")
    if not (h1[1] < h2[0] < h2[1] < h3[0]):
        raise PipelineError("CDR boundaries must satisfy H1 < H2 < H3 in sequence order")
    return CDRBoundaries(h1=h1, h2=h2, h3=h3, chain_id=chain)


def write_status(path: Path, data: dict):
    write_json(path, data)


def read_status(path: Path) -> dict:
    return read_json(path, default={})
