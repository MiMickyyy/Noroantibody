#!/usr/bin/env python3
"""Small regression test for PDB sanitization used by RFantibody wrappers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_common import sanitize_pdb_for_rfantibody


def _write(path: Path, text: str):
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_sanitize_altloc_duplicate_ca():
    raw = """
ATOM      1  N   GLY A 350      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA AGLY A 350      11.000  10.000  10.000  0.50 20.00           C
ATOM      3  CA BGLY A 350      11.100  10.100  10.100  0.50 20.00           C
ATOM      4  C   GLY A 350      12.000  10.000  10.000  1.00 20.00           C
ATOM      5  O   GLY A 350      13.000  10.000  10.000  1.00 20.00           O
TER
END
"""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / "input.pdb"
        dst = td_path / "output.pdb"
        _write(src, raw)
        stats = sanitize_pdb_for_rfantibody(src, dst)
        out = dst.read_text(encoding="utf-8")

        assert stats["dropped_altloc"] == 1
        assert stats["dropped_duplicate_atom_records"] == 0
        assert stats["residues_missing_backbone"] == 0
        assert out.count(" CA ") == 1
        assert " CA BGLY " not in out


if __name__ == "__main__":
    test_sanitize_altloc_duplicate_ca()
    print("test_pdb_sanitizer.py: OK")
