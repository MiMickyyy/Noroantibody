#!/usr/bin/env python3
"""Prepare antigen targets and residue mapping for RFantibody pipeline."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser, PDBIO, PDBParser
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1

from pipeline_common import (
    PipelineError,
    ensure_dirs,
    load_csv,
    log,
    read_json,
    read_sequence_file,
    read_yaml,
    write_json,
    write_yaml,
)


@dataclass
class ResidueRecord:
    chain: str
    resnum: int
    icode: str
    resname: str
    residue_obj: object
    full_length_resnum: Optional[int] = None
    p_domain_resnum: Optional[int] = None
    in_crop: bool = False


class ResidueSelect:
    def __init__(self, allowed_chains: Sequence[str], allowed_keys: Optional[set] = None):
        self.allowed_chains = set(allowed_chains)
        self.allowed_keys = allowed_keys

    def accept_model(self, model):
        return True

    def accept_chain(self, chain):
        return chain.id in self.allowed_chains

    def accept_residue(self, residue):
        if residue.get_parent().id not in self.allowed_chains:
            return False
        if residue.id[0] != " ":
            return False
        if not is_aa(residue, standard=False):
            return False
        if self.allowed_keys is None:
            return True
        key = (
            residue.get_parent().id,
            int(residue.id[1]),
            (residue.id[2] or "").strip(),
        )
        return key in self.allowed_keys

    def accept_atom(self, atom):
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare full/cropped antigen targets and numbering maps.")
    parser.add_argument("--pipeline-config", default="data/configs/pipeline.yaml")
    parser.add_argument("--campaign-config", default="data/configs/hotspot_campaigns.yaml")
    parser.add_argument("--resolved-inputs", default="data/processed/resolved_inputs.yaml")
    parser.add_argument("--force-redownload", action="store_true")
    return parser.parse_args()


def parse_structure(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("antigen", str(path))
    return next(structure.get_models())


def write_structure_subset(model, out_pdb: Path, chains: Sequence[str], allowed_keys: Optional[set] = None):
    io_obj = PDBIO()
    io_obj.set_structure(model)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    io_obj.save(str(out_pdb), ResidueSelect(chains, allowed_keys))


def chain_residue_records(model, chain_ids: Sequence[str]) -> Dict[str, List[ResidueRecord]]:
    out = {}
    for chain_id in chain_ids:
        chain = None
        for c in model.get_chains():
            if c.id == chain_id:
                chain = c
                break
        if chain is None:
            raise PipelineError(f"Chain {chain_id} not found in antigen structure")
        rows = []
        for residue in chain.get_residues():
            if residue.id[0] != " " or not is_aa(residue, standard=False):
                continue
            icode = (residue.id[2] or "").strip()
            rows.append(
                ResidueRecord(
                    chain=chain_id,
                    resnum=int(residue.id[1]),
                    icode=icode,
                    resname=residue.get_resname().upper(),
                    residue_obj=residue,
                )
            )
        out[chain_id] = rows
    return out


def chain_sequence(records: List[ResidueRecord]) -> str:
    return "".join(protein_letters_3to1.get(r.resname, "X") for r in records)


def local_alignment_map(target_seq: str, query_seq: str) -> Dict[int, int]:
    aligner = PairwiseAligner(mode="local")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aln = aligner.align(target_seq, query_seq)[0]
    mapping: Dict[int, int] = {}
    for (t0, t1), (q0, q1) in zip(aln.aligned[0], aln.aligned[1]):
        n = min(t1 - t0, q1 - q0)
        for i in range(n):
            mapping[q0 + i] = t0 + i
    return mapping


def infer_pdomain_monomer(seq_records: List[Tuple[str, str]]) -> Tuple[str, str, List[str]]:
    warnings = []
    if not seq_records:
        raise PipelineError("No sequence records found for P-domain input")
    if len(seq_records) > 1:
        return seq_records[0][0], seq_records[0][1], warnings

    name, seq = seq_records[0]
    if len(seq) % 2 == 0:
        half = len(seq) // 2
        if seq[:half] == seq[half:]:
            warnings.append("P-domain input appears to be duplicated dimer sequence; inferred monomer by halving.")
            return f"{name}_monomer", seq[:half], warnings
    return name, seq, warnings


def min_residue_distance(res_a, res_b) -> float:
    best = 1e9
    for atom_a in res_a.get_atoms():
        if atom_a.element == "H":
            continue
        for atom_b in res_b.get_atoms():
            if atom_b.element == "H":
                continue
            d = atom_a - atom_b
            if d < best:
                best = d
    return float(best)


def merge_segments(values: List[int], gap_merge_max: int, pad: int) -> List[Tuple[int, int]]:
    if not values:
        return []
    vals = sorted(set(values))
    segs = []
    s = vals[0]
    prev = vals[0]
    for v in vals[1:]:
        if v - prev <= gap_merge_max + 1:
            prev = v
            continue
        segs.append((s, prev))
        s = prev = v
    segs.append((s, prev))
    return [(a - pad, b + pad) for a, b in segs]


def keep_by_segments(records: List[ResidueRecord], segments: List[Tuple[int, int]]) -> set:
    keys = set()
    for r in records:
        for a, b in segments:
            if a <= r.resnum <= b:
                keys.add((r.chain, r.resnum, r.icode))
                break
    return keys


def download_5iyn_bio1(out_path: Path, force: bool = False) -> dict:
    if out_path.exists() and not force:
        return {
            "source": "local_cache",
            "path": str(out_path),
            "downloaded": False,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://files.rcsb.org/download/5IYN.pdb1.gz"
    log(f"Downloading 5IYN biological assembly 1 from {url}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with gzip.GzipFile(fileobj=io.BytesIO(response.content), mode="rb") as gz:
        data = gz.read()
    out_path.write_bytes(data)

    return {
        "source": "rcsb_download",
        "url": url,
        "path": str(out_path),
        "downloaded": True,
    }


def main() -> int:
    args = parse_args()
    root = Path(".").resolve()

    pipeline_cfg = read_yaml(root / args.pipeline_config)
    campaign_cfg = read_yaml(root / args.campaign_config)
    resolved_cfg = read_yaml(root / args.resolved_inputs)

    ensure_dirs(
        [
            root / "data/raw",
            root / "data/target",
            root / "data/maps",
            root / "data/processed",
        ]
    )

    warnings: List[str] = []

    local_structure_file = str(pipeline_cfg.get("inputs", {}).get("local_antigen_structure_file", "")).strip()
    if local_structure_file:
        structure_path = root / local_structure_file
        if not structure_path.exists():
            raise PipelineError(f"Configured local antigen structure does not exist: {structure_path}")
        provenance = {
            "source": "user_local_structure",
            "path": str(structure_path),
            "downloaded": False,
        }
    else:
        structure_path = root / "data/raw/5IYN_bio1.pdb"
        provenance = download_5iyn_bio1(structure_path, force=args.force_redownload)

    ext_log_path = root / "data/raw/external_sources.json"
    ext_log = read_json(ext_log_path, default={})
    ext_log["antigen_structure"] = provenance
    write_json(ext_log_path, ext_log)

    model = parse_structure(structure_path)
    chain_ids = pipeline_cfg.get("target_prep", {}).get("antigen_chain_ids", ["A", "B"])

    full_cleaned_path = root / "data/target/antigen_full_cleaned_AB.pdb"
    write_structure_subset(model, full_cleaned_path, chains=chain_ids)
    clean_model = parse_structure(full_cleaned_path)

    records_by_chain = chain_residue_records(clean_model, chain_ids)

    resolved_inputs = resolved_cfg.get("resolved_inputs", {})
    vp1_path = Path(resolved_inputs.get("vp1_sequence_file", ""))
    pdom_path = Path(resolved_inputs.get("p_domain_dimer_sequence_file", ""))
    if not vp1_path.exists() or not pdom_path.exists():
        raise PipelineError(
            "Resolved input aliases are missing. Run scripts/prepare_inputs.py before prepare_targets.py"
        )

    vp1_seq = read_sequence_file(vp1_path)[0][1]
    pdom_records = read_sequence_file(pdom_path)
    pdom_name, pdom_monomer, pdom_warnings = infer_pdomain_monomer(pdom_records)
    warnings.extend(pdom_warnings)

    # Map full-length -> p-domain numbering
    full_to_pdom: Dict[int, int] = {}
    pdom_map = local_alignment_map(vp1_seq, pdom_monomer)
    coverage = len(pdom_map) / max(len(pdom_monomer), 1)
    if coverage < 0.7:
        warnings.append(
            f"P-domain to VP1 mapping low coverage ({coverage:.3f}); p_domain_resnum fields may be blank."
        )
    for q_idx, t_idx in pdom_map.items():
        full_to_pdom[t_idx + 1] = q_idx + 1

    # Map structure residues -> full-length numbering per chain
    for chain_id, rows in records_by_chain.items():
        seq = chain_sequence(rows)
        c_map = local_alignment_map(vp1_seq, seq)
        if len(c_map) / max(len(seq), 1) < 0.7:
            warnings.append(
                f"Structure chain {chain_id} to VP1 mapping coverage is low; numbering may be incomplete."
            )
        for q_idx, rec in enumerate(rows):
            if q_idx in c_map:
                rec.full_length_resnum = int(c_map[q_idx] + 1)
                rec.p_domain_resnum = full_to_pdom.get(rec.full_length_resnum)

    campaigns = campaign_cfg.get("campaigns", {})
    hotspots_full = sorted(
        {
            int(x)
            for _, info in campaigns.items()
            for x in info.get("hotspot_full_length_residues", [])
        }
    )

    # map hotspots to structure residues on each chain
    hotspot_records_by_chain: Dict[str, List[ResidueRecord]] = defaultdict(list)
    for chain_id, rows in records_by_chain.items():
        for rec in rows:
            if rec.full_length_resnum in hotspots_full:
                hotspot_records_by_chain[chain_id].append(rec)

    crop_start, crop_end = pipeline_cfg.get("target_prep", {}).get("crop_window_full_length", [285, 445])
    radius = float(pipeline_cfg.get("target_prep", {}).get("hotspot_context_radius_angstrom", 10.0))
    gap_merge_max = int(pipeline_cfg.get("target_prep", {}).get("crop_gap_merge_max", 2))
    pad = int(pipeline_cfg.get("target_prep", {}).get("crop_segment_padding", 2))

    allowed_crop_keys = set()
    crop_ranges = {}
    for chain_id, rows in records_by_chain.items():
        initial = [r.resnum for r in rows if r.full_length_resnum and crop_start <= r.full_length_resnum <= crop_end]

        anchors = hotspot_records_by_chain.get(chain_id, [])
        if not anchors:
            warnings.append(f"No hotspot residues mapped on chain {chain_id}; crop will use window only.")

        context = []
        if anchors:
            for rec in rows:
                for anchor in anchors:
                    if min_residue_distance(rec.residue_obj, anchor.residue_obj) <= radius:
                        context.append(rec.resnum)
                        break

        merged = merge_segments(initial + context, gap_merge_max=gap_merge_max, pad=pad)
        keys = keep_by_segments(rows, merged)
        for rec in rows:
            if (rec.chain, rec.resnum, rec.icode) in keys:
                rec.in_crop = True
        allowed_crop_keys.update(keys)
        crop_ranges[chain_id] = merged

    cropped_path = root / "data/target/antigen_top_cap_cropped_AB.pdb"
    write_structure_subset(clean_model, cropped_path, chains=chain_ids, allowed_keys=allowed_crop_keys)

    map_rows = []
    for chain_id in chain_ids:
        for rec in records_by_chain[chain_id]:
            map_rows.append(
                {
                    "structure_chain": rec.chain,
                    "structure_resnum": rec.resnum,
                    "structure_icode": rec.icode,
                    "resname": rec.resname,
                    "full_length_resnum": rec.full_length_resnum if rec.full_length_resnum is not None else "",
                    "p_domain_resnum": rec.p_domain_resnum if rec.p_domain_resnum is not None else "",
                    "in_cropped_target": int(rec.in_crop),
                }
            )

    map_csv = root / "data/maps/residue_mapping_table.csv"
    with map_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "structure_chain",
                "structure_resnum",
                "structure_icode",
                "resname",
                "full_length_resnum",
                "p_domain_resnum",
                "in_cropped_target",
            ],
        )
        writer.writeheader()
        for row in map_rows:
            writer.writerow(row)

    crop_report = {
        "full_cleaned_target": str(full_cleaned_path),
        "cropped_target": str(cropped_path),
        "chain_ids": chain_ids,
        "crop_window_full_length": [crop_start, crop_end],
        "hotspot_context_radius_angstrom": radius,
        "crop_ranges_structure_resnum": crop_ranges,
        "hotspot_full_length_union": hotspots_full,
        "mapping_rows": len(map_rows),
        "warnings": warnings,
        "p_domain_sequence_name": pdom_name,
        "p_domain_length": len(pdom_monomer),
        "vp1_length": len(vp1_seq),
    }
    write_json(root / "data/target/target_prep_report.json", crop_report)

    text_lines = [
        "Target Preparation Report",
        "========================",
        "",
        f"Input structure: {structure_path}",
        f"Full cleaned target: {full_cleaned_path}",
        f"Cropped target: {cropped_path}",
        f"Crop full-length window: {crop_start}-{crop_end}",
        f"Hotspot context radius: {radius} A",
        "",
        "Final kept residue ranges (structure numbering):",
    ]
    for chain_id in chain_ids:
        segs = crop_ranges.get(chain_id, [])
        seg_txt = ", ".join([f"{a}-{b}" for a, b in segs]) if segs else "none"
        text_lines.append(f"- Chain {chain_id}: {seg_txt}")

    if warnings:
        text_lines.extend(["", "Warnings:"])
        for w in warnings:
            text_lines.append(f"- {w}")

    (root / "data/target/target_prep_report.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    # expose a simplified resolved target config for downstream phases
    target_cfg = {
        "full_cleaned_target": str(full_cleaned_path),
        "cropped_design_target": str(cropped_path),
        "mapping_table": str(map_csv),
        "crop_report": str(root / "data/target/target_prep_report.json"),
    }
    write_yaml(root / "data/processed/resolved_targets.yaml", target_cfg)

    log("Target preparation complete.")
    print(f"- {full_cleaned_path}")
    print(f"- {cropped_path}")
    print(f"- {map_csv}")
    print(f"- {root / 'data/target/target_prep_report.txt'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
