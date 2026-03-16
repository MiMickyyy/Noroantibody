#!/usr/bin/env python3
"""Resolve and sanitize user-provided input filenames for robust CLI usage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline_common import PipelineError, log, read_yaml, write_json, write_yaml, slugify, find_first_existing


DEFAULT_CANDIDATES = {
    "vp1_sequence_file": ["VP1.prot", "VP1.fasta", "VP1.fa", "VP1.faa"],
    "p_domain_dimer_sequence_file": [
        "P-domain dimer.fasta",
        "P-domain dimer.fa",
        "P-domain dimer.prot",
        "P_domain_dimer.fasta",
        "P_domain_dimer.fa",
        "P_domain_dimer.prot",
    ],
    "nanobody_sequence_file": ["Nanobody.fasta", "Nanobody.fa", "Nanobody.prot", "nanobody.fasta", "nanobody.fa"],
}

OPTIONAL_CANDIDATES = {
    "nanobody_framework_pdb_file": [
        "Nanobody_framework.pdb",
        "Nanobody-framework.pdb",
        "nanobody_framework.pdb",
        "nanobody-framework.pdb",
        "framework.pdb",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve and sanitize input files.")
    parser.add_argument("--pipeline-config", default="data/configs/pipeline.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--copy-instead-of-symlink", action="store_true")
    return parser.parse_args()


def ensure_alias(src: Path, dst: Path, copy_mode: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_mode:
        dst.write_bytes(src.read_bytes())
    else:
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    cfg_path = root / args.pipeline_config
    cfg = read_yaml(cfg_path)

    inputs_cfg = cfg.get("inputs", {})
    sanitized_dir = root / "data/raw/sanitized"
    sanitized_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_root": str(root),
        "copy_mode": bool(args.copy_instead_of_symlink),
        "aliases": {},
    }
    resolved = {}

    for key, fallback_names in DEFAULT_CANDIDATES.items():
        configured = str(inputs_cfg.get(key, "")).strip()
        candidates = [configured] + fallback_names
        src = find_first_existing(root, candidates)
        if src is None:
            raise PipelineError(
                f"Could not resolve required input for '{key}'. "
                f"Checked: {', '.join([x for x in candidates if x])}"
            )
        alias_name = slugify(src.stem) + src.suffix
        alias_path = sanitized_dir / alias_name
        ensure_alias(src, alias_path, copy_mode=args.copy_instead_of_symlink)

        manifest["aliases"][key] = {
            "original_path": str(src),
            "alias_path": str(alias_path),
            "was_renamed": src.name != alias_name,
        }
        resolved[key] = str(alias_path)
        log(f"Input resolved: {key} -> {alias_path.name} (source: {src.name})")

    for key, fallback_names in OPTIONAL_CANDIDATES.items():
        configured = str(inputs_cfg.get(key, "")).strip()
        candidates = [configured] + fallback_names
        src = find_first_existing(root, candidates)
        if src is None:
            continue

        alias_name = slugify(src.stem) + src.suffix
        alias_path = sanitized_dir / alias_name
        ensure_alias(src, alias_path, copy_mode=args.copy_instead_of_symlink)

        manifest["aliases"][key] = {
            "original_path": str(src),
            "alias_path": str(alias_path),
            "was_renamed": src.name != alias_name,
        }
        resolved[key] = str(alias_path)
        log(f"Optional input resolved: {key} -> {alias_path.name} (source: {src.name})")

    out_manifest = sanitized_dir / "input_aliases.json"
    out_resolved = root / "data/processed/resolved_inputs.yaml"
    write_json(out_manifest, manifest)
    write_yaml(out_resolved, {"resolved_inputs": resolved})

    print(f"Wrote alias manifest: {out_manifest}")
    print(f"Wrote resolved input config: {out_resolved}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
