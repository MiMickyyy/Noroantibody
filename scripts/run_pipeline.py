#!/usr/bin/env python3
"""Master orchestrator for Norovirus CHDC2094 nanobody redesign pipeline."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa

from pipeline_common import (
    CDRBoundaries,
    PipelineError,
    SAFETY_ETHICS_STATEMENT,
    atomic_write_csv,
    deterministic_rng,
    ensure_dirs,
    greedy_sequence_dedup,
    load_cdr_boundaries,
    log,
    now_str,
    read_json,
    read_sequence_file,
    read_status,
    read_yaml,
    slugify,
    write_json,
    write_status,
)
from tool_wrappers import (
    combine_weighted_score,
    load_tool_config,
    mutate_h2_only,
    random_loop,
    run_proteinmpnn_sequence_design,
    run_rfdiffusion_backbone,
    run_rf2_filter,
    thread_sequence_on_backbone_pose,
)


@dataclass
class Combination:
    combination_id: str
    campaign_name: str
    hotspot_full_length_residues: List[int]
    h1_delta: int
    h3_delta: int
    h1_length: int
    h3_length: int


@dataclass
class RescueCondition:
    condition_id: str
    phase_condition_id: str
    parent_candidate_id: str
    parent_combination_id: str
    parent_campaign_name: str
    parent_h1_length: int
    parent_h2_length: int
    parent_h3_length: int
    parent_full_sequence: str
    parent_h1_sequence: str
    parent_h2_sequence: str
    parent_h3_sequence: str
    parent_structure_pdb: str
    hotspot_set_name: str
    hotspot_residues: List[int]
    hotspot_tokens: List[str]


@dataclass
class LocalMaturationBranch:
    branch_name: str
    phase_branch_id: str
    editable_positions: List[int]


DEFAULT_TEST1_REAL_CANDIDATE_ID = (
    "campaign_C_A_plus_pocket_rim_HBGA_adjacent_H113_H311_"
    "bb062_s01_H2v03_Set_1_polar_anchor_bb029_s01"
)
DEFAULT_TEST1_FULL_SEQUENCE = (
    "QVQLQESGGGLVQPGGSLRLSCTYTPKPAHNVVAYGWYRQAPEKQRELVATGAVGGNVG"
    "YADSVKGRFTISRDNAKRTVYLQMNDLKPEDAAVYYCNIIDDSAVRYEGQGTQVTVSSHHHHHH"
)


def parse_model(structure_path: Path):
    sfx = structure_path.suffix.lower()
    if sfx in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", str(structure_path))
    return next(structure.get_models())


def contiguous_segments(values: List[int]) -> List[List[int]]:
    if not values:
        return []
    vals = sorted(set(int(x) for x in values))
    out: List[List[int]] = []
    start = vals[0]
    prev = vals[0]
    for v in vals[1:]:
        if v == prev + 1:
            prev = v
            continue
        out.append([start, prev])
        start = prev = v
    out.append([start, prev])
    return out


def target_chain_segments(cropped_target: Path) -> Dict[str, List[List[int]]]:
    model = parse_model(cropped_target)
    out: Dict[str, List[int]] = {}
    for chain in model.get_chains():
        numbers = []
        for residue in chain.get_residues():
            if residue.id[0] != " " or not is_aa(residue, standard=False):
                continue
            numbers.append(int(residue.id[1]))
        if numbers:
            out[str(chain.id)] = numbers
    return {k: contiguous_segments(v) for k, v in out.items()}


def build_target_contig(chain_segs: Dict[str, List[List[int]]], chain_order: List[str]) -> str:
    parts = []
    for chain_id in chain_order:
        for seg in chain_segs.get(chain_id, []):
            parts.append(f"{chain_id}{seg[0]}-{seg[1]}")
    if not parts:
        raise PipelineError("No protein residues detected in cropped target for RFdiffusion contig.")
    # Keep target segments as separate chains in the contig.
    return "/0 ".join(parts)


def build_hotspot_tokens_per_campaign(campaign_cfg: dict, mapping_csv: Path) -> Dict[str, List[str]]:
    if not mapping_csv.exists():
        raise PipelineError(f"Missing mapping table for hotspot conversion: {mapping_csv}")
    df = pd.read_csv(mapping_csv)
    if "in_cropped_target" in df.columns:
        df = df[df["in_cropped_target"].astype(int) == 1]

    out: Dict[str, List[str]] = {}
    for cname, info in campaign_cfg.get("campaigns", {}).items():
        tokens: List[str] = []
        for full in info.get("hotspot_full_length_residues", []):
            q = df[df["full_length_resnum"].fillna(-1).astype(int) == int(full)]
            if q.empty:
                log(f"[WARN] Hotspot full-length {full} has no mapping in cropped target for {cname}.")
                continue
            for _, row in q.iterrows():
                chain = str(row["structure_chain"])
                resnum = int(row["structure_resnum"])
                tokens.append(f"{chain}{resnum}")
        # de-dup preserve order
        uniq = []
        seen = set()
        for t in tokens:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        out[cname] = uniq
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phased RFantibody pipeline with resume and dry-run support.")
    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "phase0_smoke",
            "phase1_coarse_pilot",
            "phase2_focused_pilot",
            "phase3_main_campaign",
            "phase4_h2_refine",
            "phase5_cdr1_rescue_pilot",
            "phase6_cdr1_rescue_main",
            "phase_next_test1_local_maturation",
            "phase_next_champion_narrow50",
            "all",
        ],
    )
    parser.add_argument("--pipeline-config", default="data/configs/pipeline.yaml")
    parser.add_argument("--campaign-config", default="data/configs/hotspot_campaigns.yaml")
    parser.add_argument("--design-config", default="data/configs/design_matrix.yaml")
    parser.add_argument("--phases-config", default="data/configs/phases.yaml")
    parser.add_argument("--tooling-config", default="data/configs/tooling.yaml")
    parser.add_argument("--cdr-config", default="data/configs/cdr_boundaries.yaml")
    parser.add_argument("--resolved-inputs", default="data/processed/resolved_inputs.yaml")
    parser.add_argument("--resolved-targets", default="data/processed/resolved_targets.yaml")
    parser.add_argument("--cdr1-rescue-config", default="data/configs/cdr1_rescue_phase.yaml")
    parser.add_argument("--cdr1-rescue-hotspots", default="data/configs/cdr1_rescue_hotspots.yaml")
    parser.add_argument(
        "--test1-local-maturation-config",
        default="data/configs/test1_local_maturation_phase.yaml",
    )
    parser.add_argument(
        "--test1-local-maturation-hotspots",
        default="data/configs/test1_local_maturation_hotspots.yaml",
    )
    parser.add_argument(
        "--champion-narrow50-config",
        default="data/configs/champion_narrow50_phase.yaml",
    )
    parser.add_argument(
        "--champion-narrow50-hotspots",
        default="data/configs/champion_narrow50_hotspots.yaml",
    )
    parser.add_argument(
        "--phase2-selection-config",
        default="data/configs/phase2_selected_combinations.yaml",
        help=(
            "Optional manual phase2 selection YAML. If present and enabled, "
            "selected_combination_ids overrides phase1_top8_combinations.csv."
        ),
    )
    parser.add_argument(
        "--phase3-selection-config",
        default="data/configs/phase3_selected_combinations.yaml",
        help=(
            "Optional manual phase3 selection YAML. If present and enabled, "
            "selected_combination_ids overrides phase2_top2_combinations.csv."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Force real execution mode (overrides dry_run_default in config).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume behavior even if resume_default=true.",
    )
    parser.add_argument("--limit-per-combination", type=int, default=None)
    parser.add_argument("--max-combinations", type=int, default=None)
    parser.add_argument(
        "--phase4-input-csv",
        default="__AUTO__",
        help=(
            "Input CSV for phase4 H2 optimization. "
            "If omitted, auto-detect in order: "
            "phase3_selected.csv (project root) -> "
            "results/summaries/phase3_selected.csv -> "
            "results/summaries/phase3_top25_pre_h2.csv. "
            "Custom table must contain at least candidate_id."
        ),
    )
    return parser.parse_args()


def require_file(path: Path, hint: str):
    if not path.exists():
        raise PipelineError(f"Missing required file: {path}. {hint}")


def resolve_path_like(root: Path, value: str) -> Optional[Path]:
    token = str(value).strip()
    if not token:
        return None
    p = Path(token).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def load_base_context(args: argparse.Namespace) -> dict:
    root = Path(".").resolve()
    pipeline_cfg = read_yaml(root / args.pipeline_config)
    campaign_cfg = read_yaml(root / args.campaign_config)
    design_cfg = read_yaml(root / args.design_config)
    phases_cfg = read_yaml(root / args.phases_config)
    tool_cfg = load_tool_config(root / args.tooling_config)

    resolved_inputs_path = root / args.resolved_inputs
    resolved_targets_path = root / args.resolved_targets
    require_file(resolved_inputs_path, "Run: python scripts/prepare_inputs.py")
    require_file(resolved_targets_path, "Run: python scripts/prepare_targets.py")

    resolved_inputs = read_yaml(resolved_inputs_path).get("resolved_inputs", {})
    resolved_targets = read_yaml(resolved_targets_path)

    phase2_selection_path = root / args.phase2_selection_config
    phase2_manual_selection_ids: List[str] = []
    if phase2_selection_path.exists():
        phase2_cfg = read_yaml(phase2_selection_path)
        enabled = bool(phase2_cfg.get("enabled", True))
        ids = phase2_cfg.get("selected_combination_ids", [])
        if enabled and isinstance(ids, list):
            phase2_manual_selection_ids = [str(x).strip() for x in ids if str(x).strip()]

    phase3_selection_path = root / args.phase3_selection_config
    phase3_manual_selection_ids: List[str] = []
    if phase3_selection_path.exists():
        phase3_cfg = read_yaml(phase3_selection_path)
        enabled = bool(phase3_cfg.get("enabled", True))
        ids = phase3_cfg.get("selected_combination_ids", [])
        if enabled and isinstance(ids, list):
            phase3_manual_selection_ids = [str(x).strip() for x in ids if str(x).strip()]

    cdr1_rescue_cfg_path = root / args.cdr1_rescue_config
    cdr1_rescue_hotspot_path = root / args.cdr1_rescue_hotspots
    cdr1_rescue_cfg = read_yaml(cdr1_rescue_cfg_path) if cdr1_rescue_cfg_path.exists() else {}
    cdr1_rescue_hotspots = (
        read_yaml(cdr1_rescue_hotspot_path) if cdr1_rescue_hotspot_path.exists() else {}
    )
    test1_local_cfg_path = root / args.test1_local_maturation_config
    test1_local_hotspot_path = root / args.test1_local_maturation_hotspots
    test1_local_cfg = read_yaml(test1_local_cfg_path) if test1_local_cfg_path.exists() else {}
    test1_local_hotspots = (
        read_yaml(test1_local_hotspot_path) if test1_local_hotspot_path.exists() else {}
    )
    champion_narrow_cfg_path = root / args.champion_narrow50_config
    champion_narrow_hotspot_path = root / args.champion_narrow50_hotspots
    champion_narrow_cfg = (
        read_yaml(champion_narrow_cfg_path) if champion_narrow_cfg_path.exists() else {}
    )
    champion_narrow_hotspots = (
        read_yaml(champion_narrow_hotspot_path) if champion_narrow_hotspot_path.exists() else {}
    )

    # Normalize known path-like fields to absolute paths for robust execution
    for key in (
        "vp1_sequence_file",
        "p_domain_dimer_sequence_file",
        "nanobody_sequence_file",
        "nanobody_framework_pdb_file",
    ):
        val = str(resolved_inputs.get(key, "")).strip()
        if val:
            resolved = resolve_path_like(root, val)
            if resolved is not None:
                resolved_inputs[key] = str(resolved)
    for key in ("full_cleaned_target", "cropped_design_target", "mapping_table", "crop_report"):
        val = str(resolved_targets.get(key, "")).strip()
        if val:
            resolved = resolve_path_like(root, val)
            if resolved is not None:
                resolved_targets[key] = str(resolved)

    cdr = load_cdr_boundaries(root / args.cdr_config)

    nanobody_path = resolve_path_like(root, resolved_inputs.get("nanobody_sequence_file", ""))
    if nanobody_path is None or not nanobody_path.exists():
        raise PipelineError(
            f"Resolved nanobody sequence alias missing: {nanobody_path}. "
            "Run scripts/prepare_inputs.py and check aliases."
        )
    nanobody_seq = read_sequence_file(nanobody_path)[0][1]

    framework_cfg = str(
        resolved_inputs.get("nanobody_framework_pdb_file", "")
        or pipeline_cfg.get("inputs", {}).get("nanobody_framework_pdb_file", "")
    ).strip()
    framework_pdb: Optional[Path] = None
    if framework_cfg:
        framework_pdb = resolve_path_like(root, framework_cfg)
        if framework_pdb is None or not framework_pdb.exists():
            raise PipelineError(
                f"Configured nanobody framework PDB does not exist: {framework_pdb}. "
                "Please set inputs.nanobody_framework_pdb_file to a valid structure."
            )

    cropped_target = resolve_path_like(root, resolved_targets.get("cropped_design_target", ""))
    mapping_table = resolve_path_like(root, resolved_targets.get("mapping_table", ""))
    if cropped_target is None or not cropped_target.exists():
        raise PipelineError(
            f"Resolved cropped target missing: {cropped_target}. Run scripts/prepare_targets.py first."
        )
    if mapping_table is None or not mapping_table.exists():
        raise PipelineError(
            f"Resolved mapping table missing: {mapping_table}. Run scripts/prepare_targets.py first."
        )
    chain_order = list(pipeline_cfg.get("target_prep", {}).get("antigen_chain_ids", ["A", "B"]))
    chain_segs = target_chain_segments(cropped_target)
    target_contig = build_target_contig(chain_segs, chain_order=chain_order)
    campaign_hotspots = build_hotspot_tokens_per_campaign(campaign_cfg, mapping_csv=mapping_table)

    return {
        "root": root,
        "pipeline_cfg": pipeline_cfg,
        "campaign_cfg": campaign_cfg,
        "design_cfg": design_cfg,
        "phases_cfg": phases_cfg,
        "tool_cfg": tool_cfg,
        "resolved_inputs": resolved_inputs,
        "resolved_targets": resolved_targets,
        "cdr": cdr,
        "nanobody_seq": nanobody_seq,
        "framework_pdb": str(framework_pdb) if framework_pdb else "",
        "rfdiffusion_target_contig": target_contig,
        "campaign_hotspot_tokens": campaign_hotspots,
        "phase2_selection_path": str(phase2_selection_path),
        "phase2_manual_selection_ids": phase2_manual_selection_ids,
        "phase3_selection_path": str(phase3_selection_path),
        "phase3_manual_selection_ids": phase3_manual_selection_ids,
        "cdr1_rescue_cfg_path": str(cdr1_rescue_cfg_path),
        "cdr1_rescue_hotspot_path": str(cdr1_rescue_hotspot_path),
        "cdr1_rescue_cfg": cdr1_rescue_cfg,
        "cdr1_rescue_hotspots": cdr1_rescue_hotspots,
        "test1_local_cfg_path": str(test1_local_cfg_path),
        "test1_local_hotspot_path": str(test1_local_hotspot_path),
        "test1_local_cfg": test1_local_cfg,
        "test1_local_hotspots": test1_local_hotspots,
        "champion_narrow_cfg_path": str(champion_narrow_cfg_path),
        "champion_narrow_hotspot_path": str(champion_narrow_hotspot_path),
        "champion_narrow_cfg": champion_narrow_cfg,
        "champion_narrow_hotspots": champion_narrow_hotspots,
    }


def split_framework_and_cdr(seq: str, cdr: CDRBoundaries) -> dict:
    n = len(seq)
    if cdr.h3[1] > n:
        raise PipelineError(
            f"CDR H3 end ({cdr.h3[1]}) exceeds nanobody sequence length ({n})."
        )

    # 1-based indexing in config
    h1s, h1e = cdr.h1
    h2s, h2e = cdr.h2
    h3s, h3e = cdr.h3

    # Convert to 0-based slices
    f0 = seq[: h1s - 1]
    h1 = seq[h1s - 1 : h1e]
    f1 = seq[h1e:h2s - 1]
    h2 = seq[h2s - 1 : h2e]
    f2 = seq[h2e:h3s - 1]
    h3 = seq[h3s - 1 : h3e]
    f3 = seq[h3e:]

    if not all([h1, h2, h3]):
        raise PipelineError("At least one CDR segment extracted empty; verify CDR boundaries.")

    return {
        "framework_prefix": f0,
        "framework_between_h1_h2": f1,
        "framework_between_h2_h3": f2,
        "framework_suffix": f3,
        "h1_native": h1,
        "h2_native": h2,
        "h3_native": h3,
    }


def compose_nanobody_sequence(parts: dict, h1_seq: str, h2_seq: str, h3_seq: str) -> str:
    return (
        parts["framework_prefix"]
        + h1_seq
        + parts["framework_between_h1_h2"]
        + h2_seq
        + parts["framework_between_h2_h3"]
        + h3_seq
        + parts["framework_suffix"]
    )


def split_designed_sequence(parts: dict, full_seq: str, h1_len: int, h3_len: int) -> Tuple[str, str, str]:
    f0_len = len(parts["framework_prefix"])
    f1_len = len(parts["framework_between_h1_h2"])
    h2_len = len(parts["h2_native"])
    f2_len = len(parts["framework_between_h2_h3"])
    f3_len = len(parts["framework_suffix"])

    expected = f0_len + h1_len + f1_len + h2_len + f2_len + h3_len + f3_len
    if len(full_seq) != expected:
        raise PipelineError(
            f"Designed sequence length mismatch: got {len(full_seq)}, expected {expected} "
            f"(H1={h1_len}, H2={h2_len}, H3={h3_len})."
        )

    i = f0_len
    h1_seq = full_seq[i : i + h1_len]
    i += h1_len + f1_len
    h2_seq = full_seq[i : i + h2_len]
    i += h2_len + f2_len
    h3_seq = full_seq[i : i + h3_len]

    if not (h1_seq and h2_seq and h3_seq):
        raise PipelineError("Failed to split designed sequence into H1/H2/H3 segments.")
    return h1_seq, h2_seq, h3_seq


def hotspot_numbers_from_tokens(tokens: Sequence[str]) -> List[int]:
    nums = []
    for t in tokens:
        m = re.search(r"(-?\\d+)", str(t))
        if m:
            nums.append(int(m.group(1)))
    return sorted(set(nums))


def residue_has_contact(res_a, res_b, cutoff: float) -> bool:
    for atom_a in res_a.get_atoms():
        if atom_a.element == "H":
            continue
        for atom_b in res_b.get_atoms():
            if atom_b.element == "H":
                continue
            if atom_a - atom_b <= cutoff:
                return True
    return False


def compute_backbone_signature(pdb_path: Path) -> str:
    """Return a stable content hash for a generated backbone PDB."""
    if not pdb_path.exists():
        return ""
    h = hashlib.md5()
    try:
        with pdb_path.open("rb") as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
    except Exception:
        return ""
    return h.hexdigest()


def compute_interface_heuristics(
    pdb_path: Path,
    parts: dict,
    h1_len: int,
    h3_len: int,
    hotspot_tokens: Sequence[str],
    cutoff: float = 5.0,
) -> dict:
    metrics = {
        "hotspot_agreement": 0.0,
        "groove_localization": 0.0,
        "h1_h3_role_consistency": 0.0,
        "target_contact_residue_count": 0,
        "h1_target_contact_residue_count": 0,
        "h2_target_contact_residue_count": 0,
        "h3_target_contact_residue_count": 0,
        "hotspot_overlap_count": 0,
    }
    if not pdb_path.exists():
        return metrics

    parser = PDBParser(QUIET=True)
    try:
        model = next(parser.get_structure("candidate", str(pdb_path)).get_models())
    except Exception:
        return metrics

    chain_residues: Dict[str, List] = {}
    for chain in model.get_chains():
        residues = [r for r in chain.get_residues() if r.id[0] == " " and is_aa(r, standard=False)]
        if residues:
            chain_residues[str(chain.id)] = residues
    if not chain_residues:
        return metrics

    if "H" in chain_residues:
        binder_chain = "H"
    elif "C" in chain_residues:
        binder_chain = "C"
    else:
        binder_chain = sorted(chain_residues.items(), key=lambda x: len(x[1]))[0][0]

    binder_res = chain_residues[binder_chain]
    target_chains = [cid for cid in chain_residues if cid != binder_chain]
    if not target_chains:
        return metrics
    target_res = [r for cid in target_chains for r in chain_residues[cid]]

    f0_len = len(parts["framework_prefix"])
    f1_len = len(parts["framework_between_h1_h2"])
    h2_len = len(parts["h2_native"])
    f2_len = len(parts["framework_between_h2_h3"])

    h1_start = f0_len
    h1_end = h1_start + h1_len
    h2_start = h1_end + f1_len
    h2_end = h2_start + h2_len
    h3_start = h2_end + f2_len
    h3_end = h3_start + h3_len

    h1_indices = {i for i in range(h1_start, min(h1_end, len(binder_res)))}
    h2_indices = {i for i in range(h2_start, min(h2_end, len(binder_res)))}
    h3_indices = {i for i in range(h3_start, min(h3_end, len(binder_res)))}

    h1_contact_targets = set()
    h2_contact_targets = set()
    h3_contact_targets = set()
    all_contact_targets = set()

    for tidx, tres in enumerate(target_res):
        tid = (str(tres.get_parent().id), int(tres.id[1]))
        contacted_h1 = False
        contacted_h2 = False
        contacted_h3 = False
        for bidx, bres in enumerate(binder_res):
            if not (bidx in h1_indices or bidx in h2_indices or bidx in h3_indices):
                continue
            if residue_has_contact(bres, tres, cutoff=cutoff):
                if bidx in h1_indices:
                    contacted_h1 = True
                if bidx in h2_indices:
                    contacted_h2 = True
                if bidx in h3_indices:
                    contacted_h3 = True
        if contacted_h1 or contacted_h2 or contacted_h3:
            all_contact_targets.add(tid)
        if contacted_h1:
            h1_contact_targets.add(tid)
        if contacted_h2:
            h2_contact_targets.add(tid)
        if contacted_h3:
            h3_contact_targets.add(tid)

    hotspot_nums = set(hotspot_numbers_from_tokens(hotspot_tokens))
    contacted_nums = {resnum for _, resnum in all_contact_targets}
    overlap_nums = sorted(contacted_nums & hotspot_nums)
    hotspot_agreement = (len(overlap_nums) / len(hotspot_nums)) if hotspot_nums else 0.0

    h1_n = len(h1_contact_targets)
    h2_n = len(h2_contact_targets)
    h3_n = len(h3_contact_targets)
    total_n = len(all_contact_targets)
    role_consistency = ((h1_n + h3_n) / max(1, h1_n + h2_n + h3_n))
    groove_localization = min(1.0, 0.7 * hotspot_agreement + 0.3 * min(1.0, total_n / 12.0))

    metrics.update(
        {
            "hotspot_agreement": round(float(hotspot_agreement), 4),
            "groove_localization": round(float(groove_localization), 4),
            "h1_h3_role_consistency": round(float(role_consistency), 4),
            "target_contact_residue_count": int(total_n),
            "h1_target_contact_residue_count": int(h1_n),
            "h2_target_contact_residue_count": int(h2_n),
            "h3_target_contact_residue_count": int(h3_n),
            "hotspot_overlap_count": int(len(overlap_nums)),
        }
    )
    return metrics


def generate_all_combinations(campaign_cfg: dict, design_cfg: dict, native_h1: int, native_h3: int) -> List[Combination]:
    campaigns = campaign_cfg.get("campaigns", {})
    h1_deltas = list(design_cfg.get("loop_design", {}).get("h1_length_deltas", [-1, 0, 1]))
    h3_deltas = list(design_cfg.get("loop_design", {}).get("h3_length_deltas", [-2, -1, 0, 1, 2]))

    combos = []
    for campaign_name, info in campaigns.items():
        hotspots = [int(x) for x in info.get("hotspot_full_length_residues", [])]
        for d1 in h1_deltas:
            for d3 in h3_deltas:
                h1_len = native_h1 + int(d1)
                h3_len = native_h3 + int(d3)
                if h1_len <= 0 or h3_len <= 0:
                    continue
                cid = slugify(f"{campaign_name}__H1{h1_len}__H3{h3_len}")
                combos.append(
                    Combination(
                        combination_id=cid,
                        campaign_name=campaign_name,
                        hotspot_full_length_residues=hotspots,
                        h1_delta=int(d1),
                        h3_delta=int(d3),
                        h1_length=int(h1_len),
                        h3_length=int(h3_len),
                    )
                )
    return combos


def combos_for_phase(
    phase_name: str,
    phase_cfg: dict,
    all_combos: List[Combination],
    root: Path,
    phase2_manual_ids: Optional[List[str]] = None,
    phase2_selection_path: Optional[Path] = None,
    phase3_manual_ids: Optional[List[str]] = None,
    phase3_selection_path: Optional[Path] = None,
) -> List[Combination]:
    if phase_name == "phase0_smoke":
        campaigns = sorted(set(c.campaign_name for c in all_combos))
        if not campaigns:
            return []
        selected_campaign = campaigns[0]
        cands = [c for c in all_combos if c.campaign_name == selected_campaign and c.h1_delta == 0 and c.h3_delta == 0]
        if not cands:
            cands = [c for c in all_combos if c.campaign_name == selected_campaign][:1]
        return cands[:1]

    if phase_name == "phase1_coarse_pilot":
        return all_combos

    if phase_name == "phase2_focused_pilot":
        if phase2_manual_ids:
            unique_ids: List[str] = []
            seen = set()
            for cid in phase2_manual_ids:
                if cid not in seen:
                    unique_ids.append(cid)
                    seen.add(cid)

            combo_map = {c.combination_id: c for c in all_combos}
            missing = [cid for cid in unique_ids if cid not in combo_map]
            if missing:
                source = phase2_selection_path if phase2_selection_path else Path("<manual>")
                raise PipelineError(
                    f"Manual phase2 selection contains unknown combination IDs in {source}: {missing}"
                )
            if len(unique_ids) != 8:
                source = phase2_selection_path if phase2_selection_path else Path("<manual>")
                log(
                    f"[WARN] Manual phase2 selection has {len(unique_ids)} combinations in {source}; "
                    "the approved plan expects 8."
                )
            return [combo_map[cid] for cid in unique_ids]

        prev = root / "results/summaries/phase1_top8_combinations.csv"
        require_file(prev, "Run phase1_coarse_pilot first.")
        ids = set(pd.read_csv(prev)["combination_id"].astype(str).tolist())
        return [c for c in all_combos if c.combination_id in ids]

    if phase_name == "phase3_main_campaign":
        if phase3_manual_ids:
            unique_ids: List[str] = []
            seen = set()
            for cid in phase3_manual_ids:
                if cid not in seen:
                    unique_ids.append(cid)
                    seen.add(cid)

            combo_map = {c.combination_id: c for c in all_combos}
            missing = [cid for cid in unique_ids if cid not in combo_map]
            if missing:
                source = phase3_selection_path if phase3_selection_path else Path("<manual>")
                raise PipelineError(
                    f"Manual phase3 selection contains unknown combination IDs in {source}: {missing}"
                )
            if len(unique_ids) != 2:
                source = phase3_selection_path if phase3_selection_path else Path("<manual>")
                log(
                    f"[WARN] Manual phase3 selection has {len(unique_ids)} combinations in {source}; "
                    "the approved plan expects 2."
                )
            return [combo_map[cid] for cid in unique_ids]

        prev = root / "results/summaries/phase2_top2_combinations.csv"
        require_file(prev, "Run phase2_focused_pilot first.")
        ids = set(pd.read_csv(prev)["combination_id"].astype(str).tolist())
        return [c for c in all_combos if c.combination_id in ids]

    return []


def hard_pass(metrics: dict, filter_cfg: dict) -> bool:
    th = filter_cfg.get("hard_thresholds", {})
    pae_max = float(th.get("rf2_pae_max", 10.0))
    rmsd_max = float(th.get("design_rf2_rmsd_max", 2.0))
    return float(metrics.get("rf2_pae", 99.0)) < pae_max and float(metrics.get("design_rf2_rmsd", 99.0)) < rmsd_max


def load_or_empty_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def write_rows(path: Path, rows: List[dict], field_order: List[str]):
    atomic_write_csv(path, rows, field_order)


def cell_to_str(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def parse_int_maybe(value, default: int) -> int:
    token = cell_to_str(value)
    if not token:
        return int(default)
    try:
        return int(token)
    except ValueError:
        try:
            return int(float(token))
        except ValueError:
            return int(default)


def infer_candidate_structure_paths(root: Path, row: Dict[str, object]) -> Dict[str, object]:
    combo_id = cell_to_str(row.get("combination_id"))
    backbone_id = cell_to_str(row.get("backbone_id"))
    candidate_id = cell_to_str(row.get("candidate_id"))

    phases = ["phase3_main_campaign", "phase2_focused_pilot", "phase1_coarse_pilot"]

    if not cell_to_str(row.get("backbone_pdb")) and combo_id and backbone_id:
        for phase_name in phases:
            p = root / phase_name / "combinations" / combo_id / "backbones" / f"{backbone_id}.pdb"
            if p.exists():
                row["backbone_pdb"] = str(p)
                break

    seq_idx = None
    m = re.search(r"_s(\d+)$", candidate_id)
    if m:
        seq_idx = max(0, int(m.group(1)) - 1)

    if not cell_to_str(row.get("designed_pdb")) and combo_id and backbone_id and seq_idx is not None:
        for phase_name in phases:
            p = (
                root
                / phase_name
                / "combinations"
                / combo_id
                / "mpnn_aux"
                / backbone_id
                / f"{backbone_id}_outputs"
                / f"{backbone_id}_dldesign_{seq_idx}.pdb"
            )
            if p.exists():
                row["designed_pdb"] = str(p)
                break

    if not cell_to_str(row.get("rf2_best_pdb")) and combo_id and candidate_id:
        for phase_name in phases:
            best_pdb = (
                root
                / phase_name
                / "combinations"
                / combo_id
                / "rf2_metrics"
                / f"{candidate_id}_rf2_rf2_outputs"
                / f"{candidate_id}_best.pdb"
            )
            if best_pdb.exists():
                row["rf2_best_pdb"] = str(best_pdb)
                break

            rf2_json = (
                root
                / phase_name
                / "combinations"
                / combo_id
                / "rf2_metrics"
                / f"{candidate_id}_rf2.json"
            )
            if rf2_json.exists():
                try:
                    data = read_json(rf2_json)
                    rf2_best = cell_to_str(data.get("rf2_best_pdb"))
                    if rf2_best:
                        row["rf2_best_pdb"] = rf2_best
                        break
                except Exception:
                    pass

    if not cell_to_str(row.get("campaign_name")):
        campaign = ""
        if combo_id:
            m = re.match(r"^(.*)_H1\d+_H3\d+$", combo_id)
            if m:
                campaign = m.group(1)
        if campaign:
            row["campaign_name"] = campaign

    return row


def load_phase4_input_rows(context: dict, args: argparse.Namespace) -> Tuple[List[dict], Path]:
    root = context["root"]
    framework_parts = split_framework_and_cdr(context["nanobody_seq"], context["cdr"])
    phase4_token = cell_to_str(getattr(args, "phase4_input_csv", ""))
    auto_mode = phase4_token in {"", "__AUTO__", "AUTO"}
    if auto_mode:
        auto_candidates = [
            root / "phase3_selected.csv",
            root / "results/summaries/phase3_selected.csv",
            root / "results/summaries/phase3_top25_pre_h2.csv",
        ]
        phase4_input_path = next((p.resolve() for p in auto_candidates if p.exists()), None)
        if phase4_input_path is None:
            raise PipelineError(
                "No phase4 input CSV found in auto-detect paths: "
                "phase3_selected.csv, results/summaries/phase3_selected.csv, "
                "results/summaries/phase3_top25_pre_h2.csv. "
                "Please pass --phase4-input-csv explicitly."
            )
    else:
        phase4_input_path = resolve_path_like(root, phase4_token)
        if phase4_input_path is None:
            raise PipelineError("--phase4-input-csv is empty.")
        require_file(
            phase4_input_path,
            "Provide a valid phase4 input CSV path, or omit --phase4-input-csv for auto-detect.",
        )

    df = pd.read_csv(phase4_input_path)
    if df.empty:
        raise PipelineError(f"Phase4 input CSV is empty: {phase4_input_path}")
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    if "candidate_id" not in df.columns:
        raise PipelineError(
            f"Phase4 input CSV missing required column 'candidate_id': {phase4_input_path}"
        )

    phase3_by_candidate: Dict[str, dict] = {}
    for phase_name in ["phase3_main_campaign", "phase2_focused_pilot", "phase1_coarse_pilot"]:
        phase_rows = collect_phase_candidates(phase_name, root)
        for row in phase_rows:
            cid = cell_to_str(row.get("candidate_id"))
            if cid and cid not in phase3_by_candidate:
                phase3_by_candidate[cid] = row

    merged_rows: List[dict] = []
    seen: set = set()
    unresolved: List[str] = []
    for ridx, row in enumerate(df.to_dict(orient="records"), start=1):
        cid = cell_to_str(row.get("candidate_id"))
        if not cid:
            unresolved.append(f"row {ridx}: empty candidate_id")
            continue
        if cid in seen:
            log(f"[WARN] Phase4 input has duplicate candidate_id '{cid}'. Keeping first occurrence.")
            continue
        seen.add(cid)

        merged: Dict[str, object] = dict(phase3_by_candidate.get(cid, {}))
        for key, value in row.items():
            k = str(key).strip()
            v = cell_to_str(value)
            if v:
                merged[k] = v
            elif k not in merged:
                merged[k] = ""
        merged["candidate_id"] = cid
        merged = infer_candidate_structure_paths(root=root, row=merged)

        h1_seq = cell_to_str(merged.get("h1_sequence"))
        h3_seq = cell_to_str(merged.get("h3_sequence"))
        if not cell_to_str(merged.get("h1_length")) and h1_seq:
            merged["h1_length"] = len(h1_seq)
        if not cell_to_str(merged.get("h3_length")) and h3_seq:
            merged["h3_length"] = len(h3_seq)

        # If H2 (or H1/H3) is absent in a custom selection table, recover it from full_sequence.
        full_seq = cell_to_str(merged.get("full_sequence"))
        if full_seq:
            h1_len = parse_int_maybe(merged.get("h1_length"), len(h1_seq))
            h3_len = parse_int_maybe(merged.get("h3_length"), len(h3_seq))
            try:
                h1_guess, h2_guess, h3_guess = split_designed_sequence(
                    parts=framework_parts,
                    full_seq=full_seq,
                    h1_len=h1_len,
                    h3_len=h3_len,
                )
                if not cell_to_str(merged.get("h1_sequence")):
                    merged["h1_sequence"] = h1_guess
                if not cell_to_str(merged.get("h2_sequence")):
                    merged["h2_sequence"] = h2_guess
                if not cell_to_str(merged.get("h3_sequence")):
                    merged["h3_sequence"] = h3_guess
                if not cell_to_str(merged.get("h1_length")):
                    merged["h1_length"] = len(h1_guess)
                if not cell_to_str(merged.get("h3_length")):
                    merged["h3_length"] = len(h3_guess)
            except Exception:
                pass

        missing: List[str] = []
        for req in ["h1_sequence", "h2_sequence", "h3_sequence", "campaign_name"]:
            if not cell_to_str(merged.get(req)):
                missing.append(req)
        if not args.dry_run:
            if not (
                cell_to_str(merged.get("rf2_best_pdb"))
                or cell_to_str(merged.get("designed_pdb"))
                or cell_to_str(merged.get("backbone_pdb"))
            ):
                missing.append("rf2_best_pdb|designed_pdb|backbone_pdb")

        if missing:
            unresolved.append(f"{cid}: missing {', '.join(missing)}")
            continue
        merged_rows.append(merged)

    if unresolved:
        head = "; ".join(unresolved[:6])
        suffix = f" (+{len(unresolved) - 6} more)" if len(unresolved) > 6 else ""
        raise PipelineError(
            "Phase4 input rows unresolved. "
            f"{head}{suffix}. "
            "Provide candidate_id values that exist in phase3_main_campaign/combinations/*/candidates.csv, "
            "or include all required fields directly."
        )

    if len(merged_rows) != 25:
        log(
            f"[WARN] Phase4 input has {len(merged_rows)} candidates (approved plan target is 25). Continuing."
        )

    return merged_rows, phase4_input_path


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_h1_h3_from_combo_id(combination_id: str) -> Tuple[Optional[int], Optional[int]]:
    token = str(combination_id or "").strip()
    if not token:
        return None, None
    m = re.search(r"_H1(\d+)_H3(\d+)(?:_|$)", token)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _normalize_candidate_row(root: Path, row: Dict[str, object]) -> Dict[str, object]:
    merged = infer_candidate_structure_paths(root=root, row=dict(row))
    for key in (
        "rf2_best_pdb",
        "designed_pdb",
        "backbone_pdb",
        "parent_backbone_pdb",
    ):
        val = cell_to_str(merged.get(key))
        if val:
            resolved = resolve_path_like(root, val)
            if resolved is not None:
                merged[key] = str(resolved)
    return merged


def _candidate_index_for_rescue(root: Path) -> Dict[str, Dict[str, object]]:
    source_csvs = [
        root / "results/summaries/final25_h2_optimized_candidates.csv",
        root / "results/summaries/final25_h2_guardrailed.csv",
        root / "results/summaries/final_25_h2_optimized_candidates_table.csv",
        root / "results/rf2_passed/final25_rf2_passed_or_best.csv",
    ]
    index: Dict[str, Dict[str, object]] = {}
    for cpath in source_csvs:
        if not cpath.exists():
            continue
        try:
            rows = pd.read_csv(cpath).to_dict(orient="records")
        except Exception:
            continue
        for row in rows:
            cid = cell_to_str(row.get("candidate_id"))
            if not cid:
                continue
            if cid not in index:
                index[cid] = _normalize_candidate_row(root=root, row=row)
    return index


def _normalize_parent_sequence(seq: str, source: str, candidate_id: str) -> str:
    token = re.sub(r"\s+", "", str(seq or "").upper())
    if not token:
        return ""
    invalid = sorted(set(ch for ch in token if ch not in "ACDEFGHIKLMNPQRSTVWY"))
    if invalid:
        raise PipelineError(
            f"Invalid amino-acid character(s) in {source} for {candidate_id}: {''.join(invalid)}"
        )
    return token


def _manual_parent_sequence_index(context: dict) -> Dict[str, str]:
    rescue_cfg = context.get("cdr1_rescue_cfg", {}) or {}
    phase5_cfg = rescue_cfg.get("phase5", {}) if isinstance(rescue_cfg, dict) else {}
    raw = phase5_cfg.get("parent_full_sequences", {})
    out: Dict[str, str] = {}

    if isinstance(raw, dict):
        items = raw.items()
        for cid, seq in items:
            key = cell_to_str(cid)
            if not key:
                continue
            normalized = _normalize_parent_sequence(seq, "phase5.parent_full_sequences", key)
            if normalized:
                out[key] = normalized
        return out

    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            cid = cell_to_str(row.get("candidate_id"))
            if not cid:
                continue
            normalized = _normalize_parent_sequence(row.get("full_sequence", ""), "phase5.parent_full_sequences", cid)
            if normalized:
                out[cid] = normalized
    return out


def resolve_cdr1_rescue_parents(context: dict, parent_candidate_ids: Sequence[str]) -> List[dict]:
    root = context["root"]
    framework_parts = split_framework_and_cdr(context["nanobody_seq"], context["cdr"])
    index = _candidate_index_for_rescue(root)
    manual_seq_index = _manual_parent_sequence_index(context)
    fasta_seq_index: Dict[str, str] = {}
    af3_fasta = root / "af3_final25_nanobody.fasta"
    if af3_fasta.exists():
        try:
            for header, seq in read_sequence_file(af3_fasta):
                key = cell_to_str(header).split()[0]
                if key and seq:
                    fasta_seq_index[key] = _normalize_parent_sequence(seq, "af3_final25_nanobody.fasta", key)
        except Exception:
            fasta_seq_index = {}

    resolved: List[dict] = []
    for cid in parent_candidate_ids:
        if cid in index:
            row = dict(index[cid])
        else:
            # Robust fallback: if exact H2 variant is absent in final tables, use same backbone/stem variant
            # for structure path, while keeping the requested exact candidate ID and sequence (from FASTA).
            stem = cid.split("__H2v")[0] if "__H2v" in cid else cid
            candidates = [k for k in index.keys() if k.startswith(stem + "__H2v")]
            if not candidates:
                raise PipelineError(
                    "CDR1 rescue parent candidate not found in known result tables and no same-stem fallback exists: "
                    f"{cid}. Checked final25_h2_optimized_candidates.csv/final25_h2_guardrailed.csv."
                )
            row = dict(index[sorted(candidates)[0]])

        source_full_seq = _normalize_parent_sequence(
            cell_to_str(row.get("full_sequence")),
            "existing summary tables",
            cid,
        )
        exact_manual_seq = manual_seq_index.get(cid, "")
        exact_fasta_seq = fasta_seq_index.get(cid, "")
        full_seq = exact_manual_seq or exact_fasta_seq or source_full_seq
        if not full_seq:
            raise PipelineError(
                f"Parent candidate {cid} has empty full_sequence and no sequence in either "
                "phase5.parent_full_sequences or af3_final25_nanobody.fasta."
            )

        h1_len = parse_int_maybe(row.get("h1_length"), context["cdr"].h1_len)
        h3_len = parse_int_maybe(row.get("h3_length"), context["cdr"].h3_len)
        combo_id = cell_to_str(row.get("combination_id"))
        combo_h1, combo_h3 = _parse_h1_h3_from_combo_id(combo_id)
        if combo_h1 is not None:
            h1_len = combo_h1
        if combo_h3 is not None:
            h3_len = combo_h3

        h1_seq = cell_to_str(row.get("h1_sequence"))
        h2_seq = cell_to_str(row.get("h2_sequence"))
        h3_seq = cell_to_str(row.get("h3_sequence"))
        if not (h1_seq and h2_seq and h3_seq):
            try:
                h1_guess, h2_guess, h3_guess = split_designed_sequence(
                    parts=framework_parts,
                    full_seq=full_seq,
                    h1_len=h1_len,
                    h3_len=h3_len,
                )
                if not h1_seq:
                    h1_seq = h1_guess
                if not h2_seq:
                    h2_seq = h2_guess
                if not h3_seq:
                    h3_seq = h3_guess
            except Exception as exc:
                raise PipelineError(
                    f"Failed to recover H1/H2/H3 sequences for parent {cid}: {exc}"
                ) from exc

        parent_structure = (
            cell_to_str(row.get("rf2_best_pdb"))
            or cell_to_str(row.get("designed_pdb"))
            or cell_to_str(row.get("backbone_pdb"))
            or cell_to_str(row.get("parent_backbone_pdb"))
        )
        if not parent_structure:
            raise PipelineError(
                f"Parent candidate {cid} has no resolvable structure path (rf2_best_pdb/designed_pdb/backbone_pdb)."
            )

        resolved.append(
            {
                "candidate_id": cid,
                "combination_id": combo_id,
                "campaign_name": cell_to_str(row.get("campaign_name")),
                "h1_length": int(h1_len),
                "h2_length": len(h2_seq),
                "h3_length": int(h3_len),
                "h1_sequence": h1_seq,
                "h2_sequence": h2_seq,
                "h3_sequence": h3_seq,
                "full_sequence": full_seq,
                "structure_pdb": parent_structure,
            }
        )
    return resolved


def build_cdr1_rescue_hotspot_sets(
    hotspot_cfg: dict,
    antigen_chain_ids: Sequence[str],
    selected_set_names: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, object]]:
    sets = hotspot_cfg.get("cdr1_rescue_hotspot_sets", {})
    if not sets:
        raise PipelineError("Missing cdr1_rescue_hotspot_sets in data/configs/cdr1_rescue_hotspots.yaml")

    if selected_set_names:
        names = [n for n in selected_set_names if n]
    else:
        names = list(sets.keys())

    out: Dict[str, Dict[str, object]] = {}
    for set_name in names:
        if set_name not in sets:
            raise PipelineError(f"Unknown CDR1 rescue hotspot set: {set_name}")
        residues = [int(x) for x in sets[set_name].get("residues", [])]
        if not residues:
            raise PipelineError(f"Hotspot set {set_name} has empty residues list.")
        tokens: List[str] = []
        for chain_id in antigen_chain_ids:
            for resnum in residues:
                tokens.append(f"{chain_id}{resnum}")
        out[set_name] = {
            "residues": residues,
            "tokens": tokens,
            "description": cell_to_str(sets[set_name].get("description")),
        }
    return out


def enforce_cdr1_editable_positions(
    parent_full_seq: str,
    proposed_full_seq: str,
    editable_positions: Sequence[int],
) -> Tuple[str, List[int], str]:
    parent = str(parent_full_seq).strip().upper()
    proposed = str(proposed_full_seq).strip().upper()
    if not parent:
        return proposed, [], "Parent sequence missing; rescue mask not applied."
    if len(proposed) != len(parent):
        return parent, [], (
            f"Proposed sequence length mismatch (got {len(proposed)}, expected {len(parent)}); "
            "fallback to parent sequence."
        )

    out = list(parent)
    edited_positions: List[int] = []
    for p in editable_positions:
        idx = int(p) - 1
        if idx < 0 or idx >= len(out):
            continue
        aa = proposed[idx]
        if aa not in "ACDEFGHIKLMNPQRSTVWY":
            continue
        out[idx] = aa
        if aa != parent[idx]:
            edited_positions.append(int(p))
    return "".join(out), edited_positions, ""


def rescue_strict_relaxed_flags(
    rf2_pae: float,
    rf2_rmsd: float,
    strict_cfg: dict,
    relaxed_cfg: dict,
) -> Tuple[int, int]:
    strict = int(
        (rf2_pae < float(strict_cfg.get("rf2_pae_max", 10.0)))
        and (rf2_rmsd < float(strict_cfg.get("design_rf2_rmsd_max", 2.0)))
    )
    relaxed = int(
        (rf2_pae < float(relaxed_cfg.get("rf2_pae_max", 12.0)))
        and (rf2_rmsd < float(relaxed_cfg.get("design_rf2_rmsd_max", 2.5)))
    )
    return strict, relaxed


def rank_phase5_rescue_conditions(condition_rows: List[dict], top_k: int) -> Tuple[List[dict], List[dict]]:
    if not condition_rows:
        return [], []
    ranked = sorted(
        condition_rows,
        key=lambda x: (
            int(x.get("strict_pass_count", 0)),
            int(x.get("relaxed_pass_count", 0)),
            float(x.get("mean_ranking_score", 0.0)),
            -float(x.get("mean_design_rf2_rmsd", 999.0)),
            -float(x.get("mean_rf2_pae", 999.0)),
        ),
        reverse=True,
    )
    out: List[dict] = []
    for idx, row in enumerate(ranked, start=1):
        r = dict(row)
        r["rank"] = idx
        r["selected_for_phase6"] = int(idx <= max(1, top_k))
        out.append(r)
    selected = [r for r in out if int(r.get("selected_for_phase6", 0)) == 1]
    return out, selected


def export_af3_handoff_custom(context: dict, final_rows: List[dict], outdir: Path):
    root = context["root"]
    resolved_inputs = context["resolved_inputs"]
    resolved_targets = context["resolved_targets"]
    outdir.mkdir(parents=True, exist_ok=True)

    vp1_seq = read_sequence_file(Path(resolved_inputs["vp1_sequence_file"]))[0][1]
    pdom_seq = read_sequence_file(Path(resolved_inputs["p_domain_dimer_sequence_file"]))[0][1]

    submission_csv = outdir / "af3_web_submission_table.csv"
    rows = []
    for r in final_rows:
        rows.append(
            {
                "candidate_id": r.get("candidate_id", ""),
                "parent_candidate_id": r.get("parent_candidate_id", ""),
                "condition_id": r.get("condition_id", ""),
                "hotspot_set_name": r.get("hotspot_set_name", ""),
                "h1_sequence": r.get("h1_sequence", ""),
                "h2_sequence": r.get("h2_sequence", ""),
                "h3_sequence": r.get("h3_sequence", ""),
                "full_nanobody_sequence": r.get("full_sequence", ""),
                "rf2_pae": r.get("rf2_pae", ""),
                "design_rf2_rmsd": r.get("design_rf2_rmsd", ""),
                "strict_pass": r.get("strict_pass", ""),
                "relaxed_pass": r.get("relaxed_pass", ""),
                "ranking_score": r.get("ranking_score", ""),
                "rf2_best_pdb": r.get("rf2_best_pdb", ""),
                "notes": r.get("warning", ""),
            }
        )
    atomic_write_csv(
        submission_csv,
        rows,
        [
            "candidate_id",
            "parent_candidate_id",
            "condition_id",
            "hotspot_set_name",
            "h1_sequence",
            "h2_sequence",
            "h3_sequence",
            "full_nanobody_sequence",
            "rf2_pae",
            "design_rf2_rmsd",
            "strict_pass",
            "relaxed_pass",
            "ranking_score",
            "rf2_best_pdb",
            "notes",
        ],
    )

    fasta_out = outdir / "af3_final25_nanobody.fasta"
    with fasta_out.open("w", encoding="utf-8") as handle:
        for r in final_rows:
            handle.write(f">{r['candidate_id']}\n{r['full_sequence']}\n")

    context_txt = outdir / "af3_antigen_context.txt"
    context_txt.write_text(
        "\n".join(
            [
                "AF3 manual submission context (CDR1 rescue)",
                "==========================================",
                f"Full cleaned antigen target: {resolved_targets.get('full_cleaned_target', '')}",
                f"Cropped design target: {resolved_targets.get('cropped_design_target', '')}",
                f"Residue mapping table: {resolved_targets.get('mapping_table', '')}",
                f"VP1 sequence length: {len(vp1_seq)}",
                f"P-domain sequence length: {len(pdom_seq)}",
                "",
                "No local AF3 run was performed by this pipeline.",
                "Submit final candidates manually via AF3 web.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary_md = outdir / "af3_handoff_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                SAFETY_ETHICS_STATEMENT,
                "",
                "# AF3 Web Handoff (CDR1 Rescue)",
                "",
                f"- Submission table: `{submission_csv}`",
                f"- FASTA: `{fasta_out}`",
                f"- Antigen context: `{context_txt}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_cdr1_rescue_design(
    phase_name: str,
    conditions: List[RescueCondition],
    context: dict,
    args: argparse.Namespace,
    backbones_target: int,
    seqs_per_backbone: int,
    editable_positions: Sequence[int],
    strict_cfg: dict,
    relaxed_cfg: dict,
) -> List[dict]:
    root = context["root"]
    pipeline_cfg = context["pipeline_cfg"]
    tooling = context["tool_cfg"]
    rank_weights = pipeline_cfg.get("filters", {}).get("ranking_weights", {})
    seed_base = int(pipeline_cfg.get("project", {}).get("random_seed", 20260316))
    framework_parts = split_framework_and_cdr(context["nanobody_seq"], context["cdr"])

    phase_dir = root / phase_name
    cond_dir = phase_dir / "conditions"
    logs_dir = root / "logs" / phase_name
    ensure_dirs([phase_dir, cond_dir, logs_dir])

    if args.limit_per_combination is not None:
        backbones_target = min(backbones_target, int(args.limit_per_combination))
    if args.max_combinations is not None:
        conditions = conditions[: int(args.max_combinations)]

    status_path = phase_dir / "phase_status.json"
    manifest_path = phase_dir / "phase_manifest.csv"
    status = read_status(status_path)
    completed = set(status.get("completed_conditions", [])) if args.resume else set()
    manifest_rows: List[dict] = []
    condition_summaries: List[dict] = []

    target_pdb = Path(context["resolved_targets"]["cropped_design_target"])
    if not target_pdb.exists():
        raise PipelineError(f"Missing cropped design target for rescue phase: {target_pdb}")
    target_contig = str(context["rfdiffusion_target_contig"])

    for cond in conditions:
        parent_structure = Path(cond.parent_structure_pdb)
        if not args.dry_run:
            if not str(cond.parent_structure_pdb).strip():
                raise PipelineError(
                    f"{phase_name} missing parent structure path for condition {cond.condition_id}"
                )
            if not parent_structure.exists():
                raise PipelineError(
                    f"{phase_name} parent structure not found for {cond.condition_id}: {parent_structure}"
                )

        cdir = cond_dir / cond.phase_condition_id
        ensure_dirs([cdir, cdir / "backbones", cdir / "rf2_metrics", cdir / "mpnn_aux"])
        if args.resume and cond.phase_condition_id in completed:
            summary_path = cdir / "condition_summary.json"
            if summary_path.exists():
                condition_summaries.append(read_json(summary_path))
                manifest_rows.append(
                    {
                        "condition_id": cond.condition_id,
                        "phase_condition_id": cond.phase_condition_id,
                        "status": "skipped_resume",
                        "updated_at": now_str(),
                    }
                )
                continue

        backbones_csv = cdir / "backbones.csv"
        candidates_csv = cdir / "candidates.csv"
        backbones = load_or_empty_csv(backbones_csv)
        existing_backbone_ids = {cell_to_str(x.get("backbone_id")) for x in backbones}

        for i in range(1, backbones_target + 1):
            bb_id = f"{cond.phase_condition_id}_bb{i:03d}"
            if bb_id in existing_backbone_ids:
                continue
            bb_pdb = cdir / "backbones" / f"{bb_id}.pdb"
            run_rfdiffusion_backbone(
                cfg=tooling,
                combo={
                    "campaign_name": f"cdr1_rescue::{cond.hotspot_set_name}",
                    "h1_length": cond.parent_h1_length,
                    "h2_length": cond.parent_h2_length,
                    "h3_length": cond.parent_h3_length,
                },
                backbone_id=bb_id,
                target_pdb=target_pdb,
                framework_pdb=parent_structure,
                hotspots=cond.hotspot_tokens,
                target_contig=target_contig,
                binder_length=len(cond.parent_full_sequence),
                out_pdb=bb_pdb,
                seed=seed_base,
                log_file=logs_dir / f"{cond.phase_condition_id}_rfdiffusion.log",
                dry_run=args.dry_run,
                design_loops=f"H1:{cond.parent_h1_length}",
            )
            backbones.append(
                {
                    "condition_id": cond.condition_id,
                    "phase_condition_id": cond.phase_condition_id,
                    "parent_candidate_id": cond.parent_candidate_id,
                    "hotspot_set_name": cond.hotspot_set_name,
                    "backbone_id": bb_id,
                    "backbone_pdb": str(bb_pdb),
                    "backbone_signature": compute_backbone_signature(bb_pdb),
                }
            )
            existing_backbone_ids.add(bb_id)

        write_rows(
            backbones_csv,
            backbones,
            [
                "condition_id",
                "phase_condition_id",
                "parent_candidate_id",
                "hotspot_set_name",
                "backbone_id",
                "backbone_pdb",
                "backbone_signature",
            ],
        )

        candidates = load_or_empty_csv(candidates_csv)
        existing_candidate_ids = {cell_to_str(x.get("candidate_id")) for x in candidates}

        for bb in backbones:
            bb_id = cell_to_str(bb.get("backbone_id"))
            bb_pdb = Path(cell_to_str(bb.get("backbone_pdb")))
            expected = {f"{bb_id}_s{s:02d}" for s in range(1, seqs_per_backbone + 1)}
            if expected.issubset(existing_candidate_ids):
                continue

            mpnn_records = run_proteinmpnn_sequence_design(
                cfg=tooling,
                backbone_pdb=bb_pdb,
                out_dir=(cdir / "mpnn_aux" / bb_id),
                seed=seed_base,
                dry_run=args.dry_run,
                log_file=logs_dir / f"{cond.phase_condition_id}_mpnn.log",
                loops="H1",
                seqs_per_struct=seqs_per_backbone,
                temperature=0.1,
            )
            if not mpnn_records:
                raise PipelineError(f"ProteinMPNN produced no records for rescue backbone {bb_id}")

            for sidx in range(1, seqs_per_backbone + 1):
                cid = f"{bb_id}_s{sidx:02d}"
                if cid in existing_candidate_ids:
                    continue
                record = mpnn_records[min(sidx - 1, len(mpnn_records) - 1)]
                proposed_seq = cell_to_str(record.get("full_sequence"))
                constrained_seq, edited_pos, mask_warning = enforce_cdr1_editable_positions(
                    parent_full_seq=cond.parent_full_sequence,
                    proposed_full_seq=proposed_seq,
                    editable_positions=editable_positions,
                )
                threading_warning = ""

                threaded_pdb = cdir / "mpnn_aux" / bb_id / f"{cid}_threaded.pdb"
                designed_pdb_fallback = Path(cell_to_str(record.get("designed_pdb")) or str(bb_pdb))
                if args.dry_run or not tooling.execute_real_tools:
                    threaded_pdb = bb_pdb
                else:
                    try:
                        thread_sequence_on_backbone_pose(
                            backbone_pdb=bb_pdb,
                            binder_sequence=constrained_seq,
                            out_pdb=threaded_pdb,
                            binder_chain=context["cdr"].chain_id or "H",
                        )
                    except PipelineError as exc:
                        emsg = str(exc)
                        if "Failed to import rfantibody.util.pose" in emsg:
                            if designed_pdb_fallback.exists():
                                threaded_pdb = designed_pdb_fallback
                                if constrained_seq != proposed_seq:
                                    constrained_seq = proposed_seq
                                    edited_pos = []
                                    threading_warning = (
                                        "Pose-threading unavailable (rfantibody import failed); "
                                        "using ProteinMPNN designed_pdb directly, and rescue edit-mask "
                                        "was not strictly enforced for this candidate."
                                    )
                                else:
                                    threading_warning = (
                                        "Pose-threading unavailable (rfantibody import failed); "
                                        "using ProteinMPNN designed_pdb directly."
                                    )
                            else:
                                raise PipelineError(
                                    f"Threading failed and ProteinMPNN fallback PDB is missing for {cid}: "
                                    f"{designed_pdb_fallback}"
                                ) from exc
                        else:
                            raise

                rf2_json = cdir / "rf2_metrics" / f"{cid}_rf2.json"
                metrics = run_rf2_filter(
                    cfg=tooling,
                    input_pdb=threaded_pdb,
                    sequence=constrained_seq,
                    out_json=rf2_json,
                    dry_run=args.dry_run,
                    log_file=logs_dir / f"{cond.phase_condition_id}_rf2.log",
                    seed=seed_base,
                    context={
                        "candidate_id": cid,
                        "campaign_name": cond.parent_campaign_name,
                        "cdr3_contact_bias": 1,
                    },
                )

                heuristics = compute_interface_heuristics(
                    pdb_path=Path(str(metrics.get("rf2_best_pdb", threaded_pdb))),
                    parts=framework_parts,
                    h1_len=cond.parent_h1_length,
                    h3_len=cond.parent_h3_length,
                    hotspot_tokens=cond.hotspot_tokens,
                    cutoff=5.0,
                )
                metrics.update(heuristics)
                if "structural_plausibility" not in metrics:
                    metrics["structural_plausibility"] = max(0.0, min(1.0, float(metrics.get("rf2_pred_lddt", 0.0))))

                score = combine_weighted_score(metrics, rank_weights)
                strict_pass, relaxed_pass = rescue_strict_relaxed_flags(
                    rf2_pae=_safe_float(metrics.get("rf2_pae"), 99.0),
                    rf2_rmsd=_safe_float(metrics.get("design_rf2_rmsd"), 99.0),
                    strict_cfg=strict_cfg,
                    relaxed_cfg=relaxed_cfg,
                )
                warning_parts: List[str] = []
                if mask_warning:
                    warning_parts.append(mask_warning)
                if threading_warning:
                    warning_parts.append(threading_warning)
                if edited_pos:
                    warning_parts.append(f"CDR1 editable positions changed: {','.join(str(x) for x in edited_pos)}")
                warning = " | ".join(warning_parts)

                try:
                    h1_seq_c, h2_seq_c, h3_seq_c = split_designed_sequence(
                        parts=framework_parts,
                        full_seq=constrained_seq,
                        h1_len=cond.parent_h1_length,
                        h3_len=cond.parent_h3_length,
                    )
                except Exception:
                    h1_seq_c = cond.parent_h1_sequence
                    h2_seq_c = cond.parent_h2_sequence
                    h3_seq_c = cond.parent_h3_sequence

                candidates.append(
                    {
                        "phase": phase_name,
                        "condition_id": cond.condition_id,
                        "phase_condition_id": cond.phase_condition_id,
                        "parent_candidate_id": cond.parent_candidate_id,
                        "parent_combination_id": cond.parent_combination_id,
                        "parent_campaign_name": cond.parent_campaign_name,
                        "hotspot_set_name": cond.hotspot_set_name,
                        "hotspot_residues": ",".join(str(x) for x in cond.hotspot_residues),
                        "editable_cdr1_positions": ",".join(str(x) for x in editable_positions),
                        "backbone_id": bb_id,
                        "backbone_pdb": str(bb_pdb),
                        "backbone_signature": cell_to_str(bb.get("backbone_signature")),
                        "candidate_id": cid,
                        "threaded_pdb": str(threaded_pdb),
                        "rf2_best_pdb": str(metrics.get("rf2_best_pdb", "")),
                        "h1_sequence": h1_seq_c,
                        "h2_sequence": h2_seq_c,
                        "h3_sequence": h3_seq_c,
                        "full_sequence": constrained_seq,
                        "strict_pass": strict_pass,
                        "relaxed_pass": relaxed_pass,
                        "hard_filter_pass": strict_pass,
                        "rf2_pae": metrics.get("rf2_pae", ""),
                        "design_rf2_rmsd": metrics.get("design_rf2_rmsd", ""),
                        "hotspot_agreement": metrics.get("hotspot_agreement", ""),
                        "groove_localization": metrics.get("groove_localization", ""),
                        "h1_h3_role_consistency": metrics.get("h1_h3_role_consistency", ""),
                        "structural_plausibility": metrics.get("structural_plausibility", ""),
                        "target_contact_residue_count": metrics.get("target_contact_residue_count", ""),
                        "hotspot_overlap_count": metrics.get("hotspot_overlap_count", ""),
                        "ranking_score": round(float(score), 6),
                        "cdr1_edit_count": len(edited_pos),
                        "cdr1_edited_positions": ",".join(str(x) for x in edited_pos),
                        "cdr1_rescue_stable_interacting_positions_count": "",
                        "cdr1_rescue_wt_anchor_overlap_count": "",
                        "cdr1_rescue_hotspot_enrichment_score": "",
                        "warning": warning,
                    }
                )
                existing_candidate_ids.add(cid)

        candidate_fields = [
            "phase",
            "condition_id",
            "phase_condition_id",
            "parent_candidate_id",
            "parent_combination_id",
            "parent_campaign_name",
            "hotspot_set_name",
            "hotspot_residues",
            "editable_cdr1_positions",
            "backbone_id",
            "backbone_pdb",
            "backbone_signature",
            "candidate_id",
            "threaded_pdb",
            "rf2_best_pdb",
            "h1_sequence",
            "h2_sequence",
            "h3_sequence",
            "full_sequence",
            "strict_pass",
            "relaxed_pass",
            "hard_filter_pass",
            "rf2_pae",
            "design_rf2_rmsd",
            "hotspot_agreement",
            "groove_localization",
            "h1_h3_role_consistency",
            "structural_plausibility",
            "target_contact_residue_count",
            "hotspot_overlap_count",
            "ranking_score",
            "cdr1_edit_count",
            "cdr1_edited_positions",
            "cdr1_rescue_stable_interacting_positions_count",
            "cdr1_rescue_wt_anchor_overlap_count",
            "cdr1_rescue_hotspot_enrichment_score",
            "warning",
        ]
        write_rows(candidates_csv, candidates, candidate_fields)

        df = pd.DataFrame(candidates)
        if df.empty:
            summary = {
                "phase": phase_name,
                "condition_id": cond.condition_id,
                "phase_condition_id": cond.phase_condition_id,
                "parent_candidate_id": cond.parent_candidate_id,
                "hotspot_set_name": cond.hotspot_set_name,
                "editable_cdr1_positions": ",".join(str(x) for x in editable_positions),
                "total_designs": 0,
                "strict_pass_count": 0,
                "relaxed_pass_count": 0,
                "strict_pass_rate": 0.0,
                "relaxed_pass_rate": 0.0,
                "mean_ranking_score": 0.0,
                "mean_rf2_pae": 0.0,
                "mean_design_rf2_rmsd": 0.0,
                "unique_sequence_count": 0,
                "unique_backbone_count": 0,
                "cdr1_rescue_stable_interacting_positions_count": "",
                "cdr1_rescue_wt_anchor_overlap_count": "",
                "cdr1_rescue_hotspot_enrichment_score": "",
            }
        else:
            total = int(df.shape[0])
            strict_count = int(df["strict_pass"].astype(int).sum())
            relaxed_count = int(df["relaxed_pass"].astype(int).sum())
            summary = {
                "phase": phase_name,
                "condition_id": cond.condition_id,
                "phase_condition_id": cond.phase_condition_id,
                "parent_candidate_id": cond.parent_candidate_id,
                "hotspot_set_name": cond.hotspot_set_name,
                "editable_cdr1_positions": ",".join(str(x) for x in editable_positions),
                "total_designs": total,
                "strict_pass_count": strict_count,
                "relaxed_pass_count": relaxed_count,
                "strict_pass_rate": round(strict_count / max(1, total), 4),
                "relaxed_pass_rate": round(relaxed_count / max(1, total), 4),
                "mean_ranking_score": float(df["ranking_score"].astype(float).mean()),
                "mean_rf2_pae": float(df["rf2_pae"].astype(float).mean()),
                "mean_design_rf2_rmsd": float(df["design_rf2_rmsd"].astype(float).mean()),
                "unique_sequence_count": int(df["full_sequence"].nunique()),
                "unique_backbone_count": int(df["backbone_id"].nunique()),
                "cdr1_rescue_stable_interacting_positions_count": "",
                "cdr1_rescue_wt_anchor_overlap_count": "",
                "cdr1_rescue_hotspot_enrichment_score": "",
            }

        write_json(cdir / "condition_summary.json", summary)
        condition_summaries.append(summary)
        manifest_rows.append(
            {
                "condition_id": cond.condition_id,
                "phase_condition_id": cond.phase_condition_id,
                "parent_candidate_id": cond.parent_candidate_id,
                "hotspot_set_name": cond.hotspot_set_name,
                "status": "completed",
                "updated_at": now_str(),
            }
        )
        completed.add(cond.phase_condition_id)
        write_status(
            status_path,
            {
                "phase": phase_name,
                "updated_at": now_str(),
                "completed_conditions": sorted(completed),
            },
        )

    atomic_write_csv(
        manifest_path,
        manifest_rows,
        ["condition_id", "phase_condition_id", "parent_candidate_id", "hotspot_set_name", "status", "updated_at"],
    )
    return condition_summaries


def run_phase5_cdr1_rescue_pilot(context: dict, args: argparse.Namespace):
    root = context["root"]
    phase_cfg = context["phases_cfg"]["phases"]["phase5_cdr1_rescue_pilot"]
    rescue_cfg = context.get("cdr1_rescue_cfg", {})
    hotspot_cfg = context.get("cdr1_rescue_hotspots", {})
    if not rescue_cfg:
        raise PipelineError(
            "Missing CDR1 rescue phase config. Expected: data/configs/cdr1_rescue_phase.yaml"
        )
    if not hotspot_cfg:
        raise PipelineError(
            "Missing CDR1 rescue hotspot config. Expected: data/configs/cdr1_rescue_hotspots.yaml"
        )

    phase5_cfg = rescue_cfg.get("phase5", {})
    parent_ids = [cell_to_str(x) for x in phase5_cfg.get("parent_candidate_ids", []) if cell_to_str(x)]
    if not parent_ids:
        raise PipelineError("phase5.parent_candidate_ids is empty in cdr1 rescue config.")
    parents = resolve_cdr1_rescue_parents(context=context, parent_candidate_ids=parent_ids)

    editable_positions = [int(x) for x in rescue_cfg.get("cdr1_rescue", {}).get("editable_positions", [26, 27, 28, 30, 31, 32])]
    strict_cfg = rescue_cfg.get("cdr1_rescue", {}).get("strict_thresholds", {"rf2_pae_max": 10.0, "design_rf2_rmsd_max": 2.0})
    relaxed_cfg = rescue_cfg.get("cdr1_rescue", {}).get("relaxed_thresholds", {"rf2_pae_max": 12.0, "design_rf2_rmsd_max": 2.5})

    hotspot_names = [cell_to_str(x) for x in phase5_cfg.get("hotspot_set_names", []) if cell_to_str(x)]
    hotspot_sets = build_cdr1_rescue_hotspot_sets(
        hotspot_cfg=hotspot_cfg,
        antigen_chain_ids=context["pipeline_cfg"].get("target_prep", {}).get("antigen_chain_ids", ["A", "B"]),
        selected_set_names=hotspot_names,
    )

    conditions: List[RescueCondition] = []
    for p in parents:
        for hs_name, hs in hotspot_sets.items():
            cond_id = f"{p['candidate_id']}__{hs_name}"
            conditions.append(
                RescueCondition(
                    condition_id=cond_id,
                    phase_condition_id=slugify(cond_id),
                    parent_candidate_id=p["candidate_id"],
                    parent_combination_id=p["combination_id"],
                    parent_campaign_name=p["campaign_name"],
                    parent_h1_length=int(p["h1_length"]),
                    parent_h2_length=int(p["h2_length"]),
                    parent_h3_length=int(p["h3_length"]),
                    parent_full_sequence=p["full_sequence"],
                    parent_h1_sequence=p["h1_sequence"],
                    parent_h2_sequence=p["h2_sequence"],
                    parent_h3_sequence=p["h3_sequence"],
                    parent_structure_pdb=p["structure_pdb"],
                    hotspot_set_name=hs_name,
                    hotspot_residues=list(hs["residues"]),
                    hotspot_tokens=list(hs["tokens"]),
                )
            )

    summaries = run_cdr1_rescue_design(
        phase_name="phase5_cdr1_rescue_pilot",
        conditions=conditions,
        context=context,
        args=args,
        backbones_target=int(phase_cfg.get("backbones_per_combination", phase5_cfg.get("backbones_per_condition", 50))),
        seqs_per_backbone=int(phase_cfg.get("sequences_per_backbone", phase5_cfg.get("sequences_per_backbone", 1))),
        editable_positions=editable_positions,
        strict_cfg=strict_cfg,
        relaxed_cfg=relaxed_cfg,
    )

    top_k = int(rescue_cfg.get("phase6", {}).get("top_conditions", 1))
    ranked, selected = rank_phase5_rescue_conditions(summaries, top_k=top_k)

    out_summary = root / "results/summaries/phase5_cdr1_rescue_pilot_summary.csv"
    out_rank = root / "results/summaries/phase5_cdr1_rescue_condition_ranking.csv"
    out_sel = root / "results/summaries/phase5_selected_conditions.csv"
    if ranked:
        fields = list(ranked[0].keys())
        atomic_write_csv(out_summary, summaries, [k for k in fields if k not in {"rank", "selected_for_phase6"}])
        atomic_write_csv(out_rank, ranked, fields)
        atomic_write_csv(out_sel, selected, fields)
    else:
        atomic_write_csv(out_summary, [], [])
        atomic_write_csv(out_rank, [], [])
        atomic_write_csv(out_sel, [], [])


def run_phase6_cdr1_rescue_main(context: dict, args: argparse.Namespace):
    root = context["root"]
    phase_cfg = context["phases_cfg"]["phases"]["phase6_cdr1_rescue_main"]
    rescue_cfg = context.get("cdr1_rescue_cfg", {})
    hotspot_cfg = context.get("cdr1_rescue_hotspots", {})
    if not rescue_cfg or not hotspot_cfg:
        raise PipelineError("Missing cdr1 rescue config/hotspot definitions for phase6.")

    editable_positions = [int(x) for x in rescue_cfg.get("cdr1_rescue", {}).get("editable_positions", [26, 27, 28, 30, 31, 32])]
    strict_cfg = rescue_cfg.get("cdr1_rescue", {}).get("strict_thresholds", {"rf2_pae_max": 10.0, "design_rf2_rmsd_max": 2.0})
    relaxed_cfg = rescue_cfg.get("cdr1_rescue", {}).get("relaxed_thresholds", {"rf2_pae_max": 12.0, "design_rf2_rmsd_max": 2.5})
    phase6_cfg = rescue_cfg.get("phase6", {})
    phase5_cfg = rescue_cfg.get("phase5", {})

    parent_ids = [cell_to_str(x) for x in phase5_cfg.get("parent_candidate_ids", []) if cell_to_str(x)]
    parents = resolve_cdr1_rescue_parents(context=context, parent_candidate_ids=parent_ids)
    hotspot_names = [cell_to_str(x) for x in phase5_cfg.get("hotspot_set_names", []) if cell_to_str(x)]
    hotspot_sets = build_cdr1_rescue_hotspot_sets(
        hotspot_cfg=hotspot_cfg,
        antigen_chain_ids=context["pipeline_cfg"].get("target_prep", {}).get("antigen_chain_ids", ["A", "B"]),
        selected_set_names=hotspot_names,
    )

    all_cond_map: Dict[str, RescueCondition] = {}
    for p in parents:
        for hs_name, hs in hotspot_sets.items():
            cond_id = f"{p['candidate_id']}__{hs_name}"
            phase_cond_id = slugify(cond_id)
            all_cond_map[cond_id] = RescueCondition(
                condition_id=cond_id,
                phase_condition_id=phase_cond_id,
                parent_candidate_id=p["candidate_id"],
                parent_combination_id=p["combination_id"],
                parent_campaign_name=p["campaign_name"],
                parent_h1_length=int(p["h1_length"]),
                parent_h2_length=int(p["h2_length"]),
                parent_h3_length=int(p["h3_length"]),
                parent_full_sequence=p["full_sequence"],
                parent_h1_sequence=p["h1_sequence"],
                parent_h2_sequence=p["h2_sequence"],
                parent_h3_sequence=p["h3_sequence"],
                parent_structure_pdb=p["structure_pdb"],
                hotspot_set_name=hs_name,
                hotspot_residues=list(hs["residues"]),
                hotspot_tokens=list(hs["tokens"]),
            )

    manual_conditions = [cell_to_str(x) for x in phase6_cfg.get("manual_selected_conditions", []) if cell_to_str(x)]
    selected_condition_ids: List[str] = []
    if manual_conditions:
        selected_condition_ids = manual_conditions
    else:
        sel_csv = root / "results/summaries/phase5_selected_conditions.csv"
        require_file(sel_csv, "Run phase5_cdr1_rescue_pilot first, or set phase6.manual_selected_conditions.")
        sel_df = pd.read_csv(sel_csv)
        selected_condition_ids = [cell_to_str(x) for x in sel_df.get("condition_id", []).tolist() if cell_to_str(x)]

    if not selected_condition_ids:
        raise PipelineError("No selected rescue conditions available for phase6.")
    missing = [cid for cid in selected_condition_ids if cid not in all_cond_map]
    if missing:
        raise PipelineError(f"Phase6 selected condition(s) not found in phase5 parent/hotspot matrix: {missing}")

    selected_conditions = [all_cond_map[cid] for cid in selected_condition_ids]
    summaries = run_cdr1_rescue_design(
        phase_name="phase6_cdr1_rescue_main",
        conditions=selected_conditions,
        context=context,
        args=args,
        backbones_target=int(phase_cfg.get("backbones_per_combination", phase6_cfg.get("backbones_per_condition", 300))),
        seqs_per_backbone=int(phase_cfg.get("sequences_per_backbone", phase6_cfg.get("sequences_per_backbone", 2))),
        editable_positions=editable_positions,
        strict_cfg=strict_cfg,
        relaxed_cfg=relaxed_cfg,
    )
    atomic_write_csv(
        root / "results/summaries/phase6_cdr1_rescue_main_summary.csv",
        summaries,
        list(summaries[0].keys()) if summaries else [],
    )

    phase_dir = root / "phase6_cdr1_rescue_main" / "conditions"
    all_rows: List[dict] = []
    if phase_dir.exists():
        # Read only selected phase6 condition directories to avoid contamination from stale runs.
        for cond in selected_conditions:
            cdir = phase_dir / cond.phase_condition_id
            cfile = cdir / "candidates.csv"
            if cfile.exists():
                all_rows.extend(pd.read_csv(cfile).to_dict(orient="records"))
    if not all_rows:
        raise PipelineError("Phase6 generated no candidate rows.")

    df = pd.DataFrame(all_rows)
    df["strict_pass"] = df["strict_pass"].astype(int)
    df["relaxed_pass"] = df["relaxed_pass"].astype(int)
    df["ranking_score"] = df["ranking_score"].astype(float)
    df["rf2_pae"] = df["rf2_pae"].astype(float)
    df["design_rf2_rmsd"] = df["design_rf2_rmsd"].astype(float)
    df["rf2_sum_score"] = df["rf2_pae"] + df["design_rf2_rmsd"]
    # Phase6 RF2-only ranking: normalize first, then aggregate.
    # Requested normalization: RMSD/10, pAE/2; lower combined score is better.
    df["rf2_rmsd_norm"] = df["design_rf2_rmsd"] / 10.0
    df["rf2_pae_norm"] = df["rf2_pae"] / 2.0
    df["rf2_norm_score"] = df["rf2_rmsd_norm"] + df["rf2_pae_norm"]

    ranking_mode = cell_to_str(phase6_cfg.get("ranking_mode", "strict_then_ranking")).lower()
    use_rf2_sum_mode = ranking_mode in {"rf2_sum", "rf2_rmsd_pae", "rf2_pae_rmsd"}

    if use_rf2_sum_mode:
        ranked_df = df.sort_values(
            ["rf2_norm_score", "rf2_pae_norm", "rf2_rmsd_norm", "rf2_pae", "design_rf2_rmsd"],
            ascending=[True, True, True, True, True],
        )
        # greedy_sequence_dedup keeps highest score, so negate for min-objective mode.
        ranked_df["dedup_score"] = -ranked_df["rf2_norm_score"]
    else:
        ranked_df = df.sort_values(
            ["strict_pass", "relaxed_pass", "ranking_score", "design_rf2_rmsd", "rf2_pae"],
            ascending=[False, False, False, True, True],
        )
        ranked_df["dedup_score"] = ranked_df["ranking_score"]
    ranked_df = ranked_df.drop_duplicates(subset=["full_sequence"], keep="first")

    dedup_threshold = float(rescue_cfg.get("cdr1_rescue", {}).get("dedup_identity_threshold", 0.95))
    deduped = greedy_sequence_dedup(
        rows=ranked_df.to_dict(orient="records"),
        sequence_key="full_sequence",
        score_key="dedup_score",
        identity_threshold=dedup_threshold,
    )
    if use_rf2_sum_mode:
        final_ranked = sorted(
            deduped,
            key=lambda x: (
                float(x.get("rf2_norm_score", 999.0)),
                float(x.get("rf2_pae_norm", 999.0)),
                float(x.get("rf2_rmsd_norm", 999.0)),
                float(x.get("rf2_pae", 999.0)),
                float(x.get("design_rf2_rmsd", 999.0)),
            ),
        )
    else:
        final_ranked = sorted(
            deduped,
            key=lambda x: (
                int(x.get("strict_pass", 0)),
                int(x.get("relaxed_pass", 0)),
                float(x.get("ranking_score", 0.0)),
                -float(x.get("design_rf2_rmsd", 999.0)),
                -float(x.get("rf2_pae", 999.0)),
            ),
            reverse=True,
        )

    for idx, row in enumerate(final_ranked, 1):
        row["phase6_rank"] = idx
        row["phase6_ranking_mode"] = "rf2_norm_sum" if use_rf2_sum_mode else "strict_then_ranking"
    top_n = int(phase6_cfg.get("final_top_n", context["pipeline_cfg"].get("af3_handoff", {}).get("export_top_n", 25)))
    top_rows = final_ranked[:top_n]
    top_label = f"final{top_n}"

    out_rank = root / "results/summaries/phase6_cdr1_rescue_final_ranked_candidates.csv"
    out_top = root / f"results/summaries/{top_label}_cdr1_rescue_candidates.csv"
    out_table = root / f"results/summaries/final_{top_n}_cdr1_rescue_candidates_table.csv"
    fields = list(top_rows[0].keys()) if top_rows else list(final_ranked[0].keys())
    atomic_write_csv(out_rank, final_ranked, fields)
    atomic_write_csv(out_top, top_rows, fields)
    atomic_write_csv(out_table, top_rows, fields)

    # Keep legacy filenames for backward compatibility when top_n==25.
    if top_n == 25:
        atomic_write_csv(root / "results/summaries/final25_cdr1_rescue_candidates.csv", top_rows, fields)
        atomic_write_csv(root / "results/summaries/final_25_cdr1_rescue_candidates_table.csv", top_rows, fields)

    fasta_path = root / f"results/final_{top_n}/{top_label}_cdr1_rescue_sequences.fasta"
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w", encoding="utf-8") as handle:
        for row in top_rows:
            handle.write(f">{row['candidate_id']}\n{row['full_sequence']}\n")

    export_af3_handoff_custom(
        context=context,
        final_rows=top_rows,
        outdir=root / "results/af3_web_exports_cdr1_rescue",
    )


def parse_position_list(values: Sequence[object], field_name: str) -> List[int]:
    out: List[int] = []
    for item in values:
        token = cell_to_str(item)
        if not token:
            continue
        nums = re.findall(r"-?\d+", token)
        if not nums:
            raise PipelineError(f"Invalid position token '{token}' in {field_name}.")
        out.append(int(nums[-1]))
    uniq = sorted(set(x for x in out if x > 0))
    if not uniq:
        raise PipelineError(f"No valid residue positions found in {field_name}.")
    return uniq


def _build_named_hotspot_set(
    hotspot_cfg: dict,
    root_key: str,
    set_name: str,
    cfg_hint: str,
) -> Dict[str, object]:
    sets = hotspot_cfg.get(root_key, {})
    if not sets:
        raise PipelineError(
            f"Missing {root_key} in {cfg_hint}"
        )
    if set_name not in sets:
        raise PipelineError(f"Unknown hotspot set '{set_name}' in {cfg_hint}")
    node = sets[set_name] or {}
    tokens = [cell_to_str(x) for x in node.get("tokens", []) if cell_to_str(x)]
    if not tokens:
        residues = [int(x) for x in node.get("residues", [])]
        chains = [cell_to_str(x) for x in node.get("chains", []) if cell_to_str(x)]
        if residues and chains:
            for ch in chains:
                for resnum in residues:
                    tokens.append(f"{ch}{resnum}")
    if not tokens:
        raise PipelineError(f"Hotspot set {set_name} has no valid tokens/residues.")
    return {
        "set_name": set_name,
        "tokens": tokens,
        "residues": hotspot_numbers_from_tokens(tokens),
        "description": cell_to_str(node.get("description")),
    }


def build_test1_local_hotspot_set(hotspot_cfg: dict, set_name: str) -> Dict[str, object]:
    return _build_named_hotspot_set(
        hotspot_cfg=hotspot_cfg,
        root_key="test1_local_maturation_hotspot_sets",
        set_name=set_name,
        cfg_hint="data/configs/test1_local_maturation_hotspots.yaml",
    )


def build_champion_narrow50_hotspot_set(hotspot_cfg: dict, set_name: str) -> Dict[str, object]:
    return _build_named_hotspot_set(
        hotspot_cfg=hotspot_cfg,
        root_key="champion_narrow50_hotspot_sets",
        set_name=set_name,
        cfg_hint="data/configs/champion_narrow50_hotspots.yaml",
    )


def _candidate_index_for_local_maturation(root: Path) -> Dict[str, Dict[str, object]]:
    index = _candidate_index_for_rescue(root)
    source_csvs = [
        root / "results/summaries/phase6_cdr1_rescue_final_ranked_candidates.csv",
        root / "results/summaries/final_30_cdr1_rescue_candidates_table.csv",
        root / "results/summaries/final25_cdr1_rescue_candidates.csv",
        root / "results/summaries/final_25_cdr1_rescue_candidates_table.csv",
        root / "results/summaries/phase_next_test1_local_maturation_rf2_summary.csv",
        root / "results/summaries/phase_next_test1_local_maturation_strict_pass.csv",
    ]
    for cpath in source_csvs:
        if not cpath.exists():
            continue
        try:
            rows = pd.read_csv(cpath).to_dict(orient="records")
        except Exception:
            continue
        for row in rows:
            cid = cell_to_str(row.get("candidate_id"))
            if cid and cid not in index:
                index[cid] = _normalize_candidate_row(root=root, row=row)

    for phase_name in ("phase5_cdr1_rescue_pilot", "phase6_cdr1_rescue_main"):
        cond_root = root / phase_name / "conditions"
        if not cond_root.exists():
            continue
        for cfile in cond_root.glob("*/candidates.csv"):
            try:
                rows = pd.read_csv(cfile).to_dict(orient="records")
            except Exception:
                continue
            for row in rows:
                cid = cell_to_str(row.get("candidate_id"))
                if cid and cid not in index:
                    index[cid] = _normalize_candidate_row(root=root, row=row)

    next_root = root / "phase_next_test1_local_maturation" / "branches"
    if next_root.exists():
        for cfile in next_root.glob("*/candidates.csv"):
            try:
                rows = pd.read_csv(cfile).to_dict(orient="records")
            except Exception:
                continue
            for row in rows:
                cid = cell_to_str(row.get("candidate_id"))
                if cid and cid not in index:
                    index[cid] = _normalize_candidate_row(root=root, row=row)
    return index


def resolve_test1_parent_candidate(context: dict, parent_ref: str, local_cfg: dict) -> dict:
    root = context["root"]
    resolved_ref = cell_to_str(parent_ref)
    alias_source = "explicit_parent_ref"

    if resolved_ref.lower() == "test1":
        override = cell_to_str(local_cfg.get("test1_real_candidate_id"))
        if override:
            resolved_ref = override
            alias_source = "test1_real_candidate_id_override"
        else:
            summary_md = root / "results/summaries/test1_interface_maturation_summary.md"
            if summary_md.exists():
                text = summary_md.read_text(encoding="utf-8", errors="ignore")
                m = re.search(r"Resolved Test1 real candidate ID:\s*`([^`]+)`", text)
                if m:
                    resolved_ref = cell_to_str(m.group(1))
                    alias_source = "test1_interface_maturation_summary.md"
            if resolved_ref.lower() == "test1":
                resolved_ref = DEFAULT_TEST1_REAL_CANDIDATE_ID
                alias_source = "hardcoded_default_test1_id"

    index = _candidate_index_for_local_maturation(root)
    row: Optional[dict] = None
    sequence_override = ""
    def _preferred_structure_path(crow: dict) -> str:
        return (
            cell_to_str(crow.get("rf2_best_pdb"))
            or cell_to_str(crow.get("threaded_pdb"))
            or cell_to_str(crow.get("designed_pdb"))
            or cell_to_str(crow.get("backbone_pdb"))
            or cell_to_str(crow.get("parent_backbone_pdb"))
        )

    def _has_real_structure(crow: dict) -> bool:
        p = _preferred_structure_path(crow)
        if not p:
            return False
        if "MISSING_PARENT_PDB_PLACEHOLDER" in p:
            return False
        try:
            return Path(p).exists()
        except Exception:
            return False

    if resolved_ref in index:
        row = dict(index[resolved_ref])
    else:
        # Fallback: recover Test1 sequence from AF3 renamed summary, and recover structure from
        # the corresponding parent stem candidate (e.g., ..._H2v03_Set_* -> ...__H2v03).
        for seq_csv in [
            root / "results/summaries/af3_fold_test4_15_vs_wt_metrics_renamed.csv",
            root / "results/summaries/af3_fold_test16_21_vs_wt_metrics.csv",
        ]:
            if not seq_csv.exists():
                continue
            try:
                df_seq = pd.read_csv(seq_csv)
            except Exception:
                continue
            if "sequence" not in df_seq.columns:
                continue
            id_col = "renamed_candidate_id" if "renamed_candidate_id" in df_seq.columns else "candidate_id"
            if id_col not in df_seq.columns:
                continue
            q = df_seq[df_seq[id_col].astype(str) == resolved_ref]
            if not q.empty:
                sequence_override = _normalize_parent_sequence(
                    cell_to_str(q.iloc[0].get("sequence", "")),
                    f"{seq_csv.name}:{id_col}",
                    resolved_ref,
                )
                break

        stem = resolved_ref.split("_Set_")[0] if "_Set_" in resolved_ref else resolved_ref
        parent_stem = re.sub(r"_H2v(\d+)$", r"__H2v\1", stem)
        # Prefer any same-stem candidate row with an existing structure path.
        same_stem_rows: List[Tuple[str, dict]] = []
        for cid, crow in index.items():
            if cid.startswith(stem + "_Set_") or cid.startswith(stem + "__Set_"):
                same_stem_rows.append((cid, dict(crow)))
        same_stem_rows = sorted(
            same_stem_rows,
            key=lambda x: (_has_real_structure(x[1]), x[0]),
            reverse=True,
        )
        if same_stem_rows:
            row = same_stem_rows[0][1]
            alias_source += f" + same_stem_candidate_fallback({same_stem_rows[0][0]})"

        for probe in [stem, parent_stem]:
            if row is not None:
                break
            if probe in index:
                row = dict(index[probe])
                alias_source += f" + stem_fallback({probe})"
                break
            for cid, crow in index.items():
                if cid.startswith(probe + "__H2v"):
                    row = dict(crow)
                    alias_source += f" + stem_prefix_fallback({cid})"
                    break
            if row is not None:
                break

        if row is None:
            # Last-resort fallback: synthesize a parent row from explicit structure path + hardcoded/default sequence.
            explicit_structure = resolve_path_like(root, cell_to_str(local_cfg.get("test1_parent_structure_pdb", "")))
            derived_structure_candidates: List[Path] = []
            base = re.sub(r"_s\\d+$", "", resolved_ref)
            cond = re.sub(r"_bb\\d+$", "", base)
            derived_structure_candidates.extend(
                [
                    root
                    / "phase5_cdr1_rescue_pilot"
                    / "conditions"
                    / cond
                    / "rf2_metrics"
                    / f"{resolved_ref}_rf2_rf2_outputs"
                    / f"{resolved_ref}_best.pdb",
                    root / "phase5_cdr1_rescue_pilot" / "conditions" / cond / "backbones" / f"{base}.pdb",
                    root / "phase6_cdr1_rescue_main" / "conditions" / cond / "backbones" / f"{base}.pdb",
                    root / "phase6_cdr1_rescue_main" / "conditions" / cond / "backbones" / f"{cond}_bb001.pdb",
                ]
            )
            structure_fallback = None
            if explicit_structure is not None and explicit_structure.exists():
                structure_fallback = explicit_structure
                alias_source += " + explicit_parent_structure"
            else:
                for p in derived_structure_candidates:
                    if p.exists():
                        structure_fallback = p
                        alias_source += f" + derived_parent_structure({p.name})"
                        break

            if structure_fallback is None:
                raise PipelineError(
                    f"Resolved Test1 parent candidate '{resolved_ref}' was not found in known tables and "
                    "no fallback structure was found. Set test1_local_maturation.test1_parent_structure_pdb "
                    "in data/configs/test1_local_maturation_phase.yaml."
                )

            row = {
                "candidate_id": resolved_ref,
                "full_sequence": "",
                "rf2_best_pdb": str(structure_fallback),
                "campaign_name": "test1_local_maturation",
                "combination_id": "",
            }
    config_seq_override = _normalize_parent_sequence(
        cell_to_str(local_cfg.get("test1_full_sequence", "")),
        "test1_full_sequence",
        resolved_ref,
    ) if cell_to_str(local_cfg.get("test1_full_sequence", "")) else ""
    default_seq_override = (
        DEFAULT_TEST1_FULL_SEQUENCE
        if resolved_ref == DEFAULT_TEST1_REAL_CANDIDATE_ID
        else ""
    )
    full_seq = sequence_override or config_seq_override or default_seq_override or _normalize_parent_sequence(
        cell_to_str(row.get("full_sequence")),
        "test1 parent candidate row",
        resolved_ref,
    )
    if not full_seq:
        raise PipelineError(f"Resolved Test1 parent {resolved_ref} has empty full_sequence.")

    h1_len = parse_int_maybe(row.get("h1_length"), context["cdr"].h1_len)
    h3_len = parse_int_maybe(row.get("h3_length"), context["cdr"].h3_len)
    cid_h1, cid_h3 = _parse_h1_h3_from_combo_id(resolved_ref)
    if cid_h1 is not None:
        h1_len = int(cid_h1)
    if cid_h3 is not None:
        h3_len = int(cid_h3)
    h1_seq = cell_to_str(row.get("h1_sequence"))
    h2_seq = cell_to_str(row.get("h2_sequence"))
    h3_seq = cell_to_str(row.get("h3_sequence"))
    if not (h1_seq and h2_seq and h3_seq):
        parts = split_framework_and_cdr(context["nanobody_seq"], context["cdr"])
        try:
            h1_guess, h2_guess, h3_guess = split_designed_sequence(
                parts=parts,
                full_seq=full_seq,
                h1_len=h1_len,
                h3_len=h3_len,
            )
            h1_seq = h1_seq or h1_guess
            h2_seq = h2_seq or h2_guess
            h3_seq = h3_seq or h3_guess
        except Exception as exc:
            raise PipelineError(
                f"Failed to recover H1/H2/H3 sequences for resolved Test1 parent {resolved_ref}: {exc}"
            ) from exc

    parent_structure = _preferred_structure_path(row)
    if not parent_structure:
        raise PipelineError(
            f"Resolved Test1 parent {resolved_ref} has no usable structure path for fixed-backbone maturation."
        )

    out = {
        "candidate_id": resolved_ref,
        "alias_source": alias_source,
        "full_sequence": full_seq,
        "h1_length": int(h1_len),
        "h2_length": len(h2_seq),
        "h3_length": int(h3_len),
        "h1_sequence": h1_seq,
        "h2_sequence": h2_seq,
        "h3_sequence": h3_seq,
        "structure_pdb": parent_structure,
        "combination_id": cell_to_str(row.get("combination_id") or row.get("parent_combination_id")),
        "campaign_name": cell_to_str(row.get("campaign_name") or row.get("parent_campaign_name")),
        "rf2_pae": _safe_float(row.get("rf2_pae"), float("nan")),
        "design_vs_rf2_rmsd": _safe_float(
            row.get("design_vs_rf2_rmsd", row.get("design_rf2_rmsd")),
            float("nan"),
        ),
        "ranking_score": _safe_float(row.get("ranking_score"), float("nan")),
    }
    return out


def _looks_like_stage7_short_job_id(token: str) -> bool:
    return bool(re.fullmatch(r"spg\d+_\d+", str(token or "").strip().lower()))


def _read_af3_job_request_sequence(job_dir: Path) -> str:
    jfiles = sorted(job_dir.glob("*_job_request.json"))
    if not jfiles:
        return ""
    try:
        obj = read_json(jfiles[0], default={})
        if isinstance(obj, list):
            obj = obj[0] if obj else {}
        seq = (
            ((obj.get("sequences") or [])[0] or {})
            .get("proteinChain", {})
            .get("sequence", "")
        )
        return _normalize_parent_sequence(seq, f"{jfiles[0].name}:sequences[0]", job_dir.name)
    except Exception:
        return ""


def _iter_stage7_job_map_csvs(root: Path, local_cfg: dict) -> List[Path]:
    candidates: List[Path] = []
    explicit = resolve_path_like(root, cell_to_str(local_cfg.get("stage7_job_map_csv", "")))
    if explicit is not None and explicit.exists():
        candidates.append(explicit)

    glob_patterns = [
        "results/af3_web_exports_strict_pass*/af3_strict_pass_all_map.csv",
        "results/af3_web_exports_strict_pass*/af3_strict_pass_group*_map.csv",
    ]
    for pattern in glob_patterns:
        for p in root.glob(pattern):
            if p.exists():
                candidates.append(p.resolve())

    for p in root.rglob("af3_strict_pass_all_map.csv"):
        if p.exists():
            candidates.append(p.resolve())
    for p in root.rglob("af3_strict_pass_group*_map.csv"):
        if p.exists():
            candidates.append(p.resolve())

    uniq: Dict[str, Path] = {}
    for p in candidates:
        uniq[str(p)] = p
    return sorted(uniq.values(), key=lambda x: x.stat().st_mtime, reverse=True)


def _find_stage7_job_dir(stage7_dir: Path, job_name: str) -> Optional[Path]:
    direct = stage7_dir / job_name
    if direct.exists() and direct.is_dir():
        return direct
    for d in stage7_dir.iterdir():
        if not d.is_dir():
            continue
        jfiles = sorted(d.glob("*_job_request.json"))
        if not jfiles:
            continue
        try:
            obj = read_json(jfiles[0], default={})
            if isinstance(obj, list):
                obj = obj[0] if obj else {}
            if cell_to_str(obj.get("name")) == job_name:
                return d
        except Exception:
            continue
    return None


def _resolve_stage7_job_to_candidate_id(
    root: Path,
    local_cfg: dict,
    job_name: str,
    raw_candidate_id: str,
    stage7_job_dir: Optional[Path],
) -> Tuple[str, str]:
    raw = cell_to_str(raw_candidate_id)
    if raw and not _looks_like_stage7_short_job_id(raw):
        return raw, "stage7_ranked_summary.candidate_id"

    for mpath in _iter_stage7_job_map_csvs(root, local_cfg):
        try:
            df = pd.read_csv(mpath)
        except Exception:
            continue
        if "job_name" not in df.columns or "candidate_id" not in df.columns:
            continue
        q = df[df["job_name"].astype(str) == job_name]
        if q.empty:
            continue
        cid = cell_to_str(q.iloc[0].get("candidate_id"))
        if cid:
            return cid, str(mpath)

    # Sequence-based fallback mapping: AF3 job seq -> any known candidate with same full sequence.
    stage7_seq = _read_af3_job_request_sequence(stage7_job_dir) if stage7_job_dir is not None else ""
    if stage7_seq:
        index = _candidate_index_for_local_maturation(root)
        for cid, row in index.items():
            try:
                seq = _normalize_parent_sequence(
                    cell_to_str(row.get("full_sequence")),
                    f"candidate_index[{cid}]",
                    cid,
                )
            except Exception:
                continue
            if seq and seq == stage7_seq:
                return cid, "sequence_match_candidate_index"

    fallback = raw or job_name
    return fallback, "fallback_stage7_job_name"


def resolve_best_phase7_parent_candidate(context: dict, local_cfg: dict) -> dict:
    root = context["root"]
    ranked_path = resolve_path_like(
        root,
        cell_to_str(local_cfg.get("phase7_ranked_summary_csv", "results/summaries/af3_stage7_ranked_summary_with_wt_test1.csv")),
    )
    if ranked_path is None or not ranked_path.exists():
        raise PipelineError(
            "Missing Phase7 AF3 ranked summary for parent resolution. "
            "Set champion_narrow50.phase7_ranked_summary_csv to a valid file."
        )
    stage7_dir = resolve_path_like(
        root,
        cell_to_str(local_cfg.get("phase7_af3_results_dir", "AF3 Results/Stage7 AF3")),
    )
    if stage7_dir is None or not stage7_dir.exists():
        raise PipelineError(
            "Missing Stage7 AF3 directory for champion parent resolution. "
            "Set champion_narrow50.phase7_af3_results_dir to a valid directory."
        )

    df = pd.read_csv(ranked_path)
    if "source_group" in df.columns:
        df = df[df["source_group"].astype(str) == "Stage7"].copy()
    if "strict_pass_models" in df.columns and "n_models" in df.columns:
        strict = df[
            (df["n_models"].fillna(0).astype(int) > 0)
            & (df["strict_pass_models"].fillna(0).astype(int) == df["n_models"].fillna(0).astype(int))
        ].copy()
        if not strict.empty:
            df = strict
    if df.empty:
        raise PipelineError(
            f"No Stage7 rows available in ranked summary: {ranked_path}"
        )

    sort_cols = []
    sort_asc = []
    for col, asc in [
        ("interface_stability_score", False),
        ("best_pair_iptm_mean", False),
        ("best_pair_pae_min_mean", True),
    ]:
        if col in df.columns:
            sort_cols.append(col)
            sort_asc.append(asc)
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=sort_asc)
    best = df.iloc[0].to_dict()
    best_job = cell_to_str(best.get("job_name"))
    if not best_job:
        raise PipelineError(f"Top Stage7 row in {ranked_path} has empty job_name.")

    stage7_job_dir = _find_stage7_job_dir(stage7_dir, best_job)
    resolved_parent_ref, parent_ref_source = _resolve_stage7_job_to_candidate_id(
        root=root,
        local_cfg=local_cfg,
        job_name=best_job,
        raw_candidate_id=cell_to_str(best.get("candidate_id")),
        stage7_job_dir=stage7_job_dir,
    )

    parent = resolve_test1_parent_candidate(
        context=context,
        parent_ref=resolved_parent_ref,
        local_cfg=local_cfg,
    )
    parent["alias_source"] = (
        f"{parent.get('alias_source', '')} + phase7_best({best_job}) + {parent_ref_source}"
    ).strip(" +")
    parent["resolved_phase7_job_name"] = best_job
    parent["resolved_phase7_parent_ref"] = resolved_parent_ref
    parent["phase7_interface_stability_score"] = _safe_float(
        best.get("interface_stability_score"),
        float("nan"),
    )
    parent["phase7_best_pair_iptm_mean"] = _safe_float(
        best.get("best_pair_iptm_mean"),
        float("nan"),
    )
    parent["phase7_best_pair_pae_min_mean"] = _safe_float(
        best.get("best_pair_pae_min_mean"),
        float("nan"),
    )
    parent["phase7_strict_pass_models"] = parse_int_maybe(best.get("strict_pass_models"), 0)
    parent["phase7_n_models"] = parse_int_maybe(best.get("n_models"), 0)
    if stage7_job_dir is not None:
        parent["phase7_job_dir"] = str(stage7_job_dir)
    return parent


def mutate_local_positions(
    parent_seq: str,
    editable_positions: Sequence[int],
    rng,
    aa_alphabet: str,
    min_mutations: int,
    max_mutations: int,
) -> Tuple[str, List[int]]:
    seq = list(str(parent_seq).strip().upper())
    valid_positions = [p for p in sorted(set(int(x) for x in editable_positions)) if 1 <= p <= len(seq)]
    if not valid_positions:
        return "".join(seq), []

    max_m = max(1, min(int(max_mutations), len(valid_positions)))
    min_m = max(1, min(int(min_mutations), max_m))
    n_mut = rng.randint(min_m, max_m)
    selected = rng.sample(valid_positions, n_mut)

    edited: List[int] = []
    for pos in selected:
        idx = pos - 1
        native = seq[idx]
        options = [aa for aa in aa_alphabet if aa != native]
        if not options:
            continue
        seq[idx] = rng.choice(options)
        if seq[idx] != native:
            edited.append(pos)

    # Ensure at least one mutation if possible.
    if not edited:
        pos = valid_positions[0]
        idx = pos - 1
        native = seq[idx]
        options = [aa for aa in aa_alphabet if aa != native]
        if options:
            seq[idx] = options[0]
            edited.append(pos)
    return "".join(seq), sorted(set(edited))


def run_phase_next_test1_local_maturation(context: dict, args: argparse.Namespace):
    root = context["root"]
    phases = context["phases_cfg"].get("phases", {})
    phase_cfg = phases.get("phase_next_test1_local_maturation", {})
    local_cfg_root = context.get("test1_local_cfg", {}) or {}
    hotspot_cfg_root = context.get("test1_local_hotspots", {}) or {}
    if not local_cfg_root:
        raise PipelineError(
            "Missing local maturation config. Expected: data/configs/test1_local_maturation_phase.yaml"
        )
    if not hotspot_cfg_root:
        raise PipelineError(
            "Missing local maturation hotspot config. Expected: data/configs/test1_local_maturation_hotspots.yaml"
        )

    local_cfg = local_cfg_root.get("test1_local_maturation", {})
    if not local_cfg:
        raise PipelineError("Missing 'test1_local_maturation' block in local maturation config.")

    parent_ref = (
        cell_to_str(local_cfg.get("parent_candidate_id"))
        or cell_to_str(local_cfg.get("parent_candidate_ref"))
        or "Test1"
    )
    parent = resolve_test1_parent_candidate(context=context, parent_ref=parent_ref, local_cfg=local_cfg)

    hotspot_set_name = cell_to_str(local_cfg.get("hotspot_set_name", "Test1_defect_guided"))
    hotspot_set = build_test1_local_hotspot_set(hotspot_cfg_root, set_name=hotspot_set_name)

    strict_cfg = local_cfg.get("strict_thresholds", {"rf2_pae_max": 10.0, "design_rf2_rmsd_max": 2.0})
    relaxed_cfg = local_cfg.get("relaxed_thresholds", {"rf2_pae_max": 12.0, "design_rf2_rmsd_max": 2.5})
    fixed_core_positions = parse_position_list(
        local_cfg.get("fixed_core_positions", [27, 28, 29, 30, 33, 34]),
        "test1_local_maturation.fixed_core_positions",
    )

    raw_branches = local_cfg.get("branches", {})
    if not isinstance(raw_branches, dict) or not raw_branches:
        raise PipelineError("test1_local_maturation.branches is empty.")
    branches: List[LocalMaturationBranch] = []
    for bname, bcfg in raw_branches.items():
        editable_positions = parse_position_list(
            (bcfg or {}).get("editable_positions", []),
            f"test1_local_maturation.branches.{bname}.editable_positions",
        )
        branches.append(
            LocalMaturationBranch(
                branch_name=cell_to_str(bname),
                phase_branch_id=slugify(f"{parent['candidate_id']}__{bname}"),
                editable_positions=editable_positions,
            )
        )
    if args.max_combinations is not None:
        branches = branches[: int(args.max_combinations)]

    target_per_branch = int(
        phase_cfg.get(
            "candidates_per_branch",
            local_cfg.get("candidate_count_per_branch", 50),
        )
    )
    if args.limit_per_combination is not None:
        target_per_branch = min(target_per_branch, int(args.limit_per_combination))
    if target_per_branch <= 0:
        raise PipelineError("candidates_per_branch must be > 0 for phase_next_test1_local_maturation.")

    aa_alphabet = re.sub(r"[^A-Z]", "", cell_to_str(local_cfg.get("aa_alphabet", "ACDEFGHIKLMNPQRSTVWY")).upper())
    if not aa_alphabet:
        aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    min_mutations = int(local_cfg.get("min_mutations_per_candidate", 1))
    max_mutations_default = max(1, max(len(b.editable_positions) for b in branches))
    max_mutations = int(local_cfg.get("max_mutations_per_candidate", max_mutations_default))

    tooling = context["tool_cfg"]
    pipeline_cfg = context["pipeline_cfg"]
    rank_weights = pipeline_cfg.get("filters", {}).get("ranking_weights", {})
    seed_base = int(pipeline_cfg.get("project", {}).get("random_seed", 20260316))
    framework_parts = split_framework_and_cdr(context["nanobody_seq"], context["cdr"])

    phase_name = "phase_next_test1_local_maturation"
    phase_dir = root / phase_name
    branches_dir = phase_dir / "branches"
    logs_dir = root / "logs" / phase_name
    ensure_dirs([phase_dir, branches_dir, logs_dir])

    parent_structure = Path(parent["structure_pdb"])
    if not args.dry_run and not parent_structure.exists():
        raise PipelineError(
            f"Resolved Test1 parent structure does not exist: {parent_structure}"
        )

    status_path = phase_dir / "phase_status.json"
    manifest_path = phase_dir / "phase_manifest.csv"
    status = read_status(status_path)
    completed = set(status.get("completed_branches", [])) if args.resume else set()
    manifest_rows: List[dict] = []

    for branch in branches:
        bdir = branches_dir / branch.phase_branch_id
        ensure_dirs([bdir, bdir / "threaded", bdir / "rf2_metrics"])
        candidates_csv = bdir / "candidates.csv"
        candidates = load_or_empty_csv(candidates_csv)
        existing_ids = {cell_to_str(x.get("candidate_id")) for x in candidates}
        seen_sequences = {cell_to_str(x.get("full_sequence")) for x in candidates if cell_to_str(x.get("full_sequence"))}

        if args.resume and branch.phase_branch_id in completed and len(candidates) >= target_per_branch:
            manifest_rows.append(
                {
                    "branch_name": branch.branch_name,
                    "phase_branch_id": branch.phase_branch_id,
                    "status": "skipped_resume",
                    "updated_at": now_str(),
                }
            )
            continue

        for i in range(1, target_per_branch + 1):
            cid = f"{branch.phase_branch_id}_c{i:03d}"
            if cid in existing_ids:
                continue

            final_seq = parent["full_sequence"]
            edited_pos: List[int] = []
            warning_parts: List[str] = []
            duplicate_after_sampling = False
            for attempt in range(1, 97):
                rng = deterministic_rng(seed_base, f"{cid}::attempt{attempt}")
                proposed_seq, proposed_edited = mutate_local_positions(
                    parent_seq=parent["full_sequence"],
                    editable_positions=branch.editable_positions,
                    rng=rng,
                    aa_alphabet=aa_alphabet,
                    min_mutations=min_mutations,
                    max_mutations=max_mutations,
                )
                constrained_seq, constrained_edited, mask_warning = enforce_cdr1_editable_positions(
                    parent_full_seq=parent["full_sequence"],
                    proposed_full_seq=proposed_seq,
                    editable_positions=branch.editable_positions,
                )
                # Core Test1 primary-contact positions must remain unchanged.
                core_violations = [
                    p
                    for p in fixed_core_positions
                    if 1 <= p <= len(constrained_seq)
                    and constrained_seq[p - 1] != parent["full_sequence"][p - 1]
                ]
                if core_violations:
                    raise PipelineError(
                        f"{cid} violates fixed core positions: {core_violations}"
                    )
                if constrained_seq in seen_sequences and attempt < 96:
                    continue
                duplicate_after_sampling = constrained_seq in seen_sequences
                final_seq = constrained_seq
                edited_pos = constrained_edited or proposed_edited
                if mask_warning:
                    warning_parts.append(mask_warning)
                break

            if duplicate_after_sampling:
                warning_parts.append("Duplicate full sequence after resampling attempts.")
            if edited_pos:
                warning_parts.append(
                    "Edited positions: " + ",".join(str(x) for x in sorted(set(edited_pos)))
                )
            seen_sequences.add(final_seq)

            threaded_pdb = bdir / "threaded" / f"{cid}.pdb"
            threading_warning = ""
            if args.dry_run or not tooling.execute_real_tools:
                threaded_pdb = parent_structure
            else:
                try:
                    thread_sequence_on_backbone_pose(
                        backbone_pdb=parent_structure,
                        binder_sequence=final_seq,
                        out_pdb=threaded_pdb,
                        binder_chain=context["cdr"].chain_id or "H",
                    )
                except PipelineError as exc:
                    msg = str(exc)
                    if "Failed to import rfantibody.util.pose" in msg:
                        threaded_pdb = parent_structure
                        threading_warning = (
                            "Pose-threading unavailable (rfantibody import failed); "
                            "using parent backbone directly for RF2 input."
                        )
                    else:
                        raise

            rf2_json = bdir / "rf2_metrics" / f"{cid}_rf2.json"
            metrics = run_rf2_filter(
                cfg=tooling,
                input_pdb=threaded_pdb,
                sequence=final_seq,
                out_json=rf2_json,
                dry_run=args.dry_run,
                log_file=logs_dir / f"{branch.phase_branch_id}_rf2.log",
                seed=seed_base,
                context={
                    "candidate_id": cid,
                    "campaign_name": parent.get("campaign_name", "test1_local_maturation"),
                    "cdr3_contact_bias": 1,
                },
            )

            heuristics = compute_interface_heuristics(
                pdb_path=Path(str(metrics.get("rf2_best_pdb", threaded_pdb))),
                parts=framework_parts,
                h1_len=int(parent["h1_length"]),
                h3_len=int(parent["h3_length"]),
                hotspot_tokens=hotspot_set["tokens"],
                cutoff=5.0,
            )
            metrics.update(heuristics)
            if "structural_plausibility" not in metrics:
                metrics["structural_plausibility"] = max(0.0, min(1.0, float(metrics.get("rf2_pred_lddt", 0.0))))
            ranking_score = round(float(combine_weighted_score(metrics, rank_weights)), 6)

            strict_pass, relaxed_pass = rescue_strict_relaxed_flags(
                rf2_pae=_safe_float(metrics.get("rf2_pae"), 99.0),
                rf2_rmsd=_safe_float(metrics.get("design_rf2_rmsd"), 99.0),
                strict_cfg=strict_cfg,
                relaxed_cfg=relaxed_cfg,
            )

            if threading_warning:
                warning_parts.append(threading_warning)
            warning = " | ".join(warning_parts)

            candidates.append(
                {
                    "phase": phase_name,
                    "branch_name": branch.branch_name,
                    "phase_branch_id": branch.phase_branch_id,
                    "parent_candidate_id": parent["candidate_id"],
                    "parent_candidate_ref": parent_ref,
                    "parent_alias_source": parent["alias_source"],
                    "candidate_id": cid,
                    "editable_positions": ",".join(str(x) for x in branch.editable_positions),
                    "fixed_core_positions": ",".join(str(x) for x in fixed_core_positions),
                    "hotspot_set_name": hotspot_set["set_name"],
                    "hotspot_tokens": ",".join(hotspot_set["tokens"]),
                    "full_sequence": final_seq,
                    "strict_pass": int(strict_pass),
                    "relaxed_pass": int(relaxed_pass),
                    "rf2_pae": _safe_float(metrics.get("rf2_pae"), 99.0),
                    "design_vs_rf2_rmsd": _safe_float(metrics.get("design_rf2_rmsd"), 99.0),
                    "ranking_score": ranking_score,
                    "threaded_pdb": str(threaded_pdb),
                    "rf2_best_pdb": str(metrics.get("rf2_best_pdb", "")),
                    "cdr1_edit_count": len(edited_pos),
                    "cdr1_edited_positions": ",".join(str(x) for x in sorted(set(edited_pos))),
                    "warning": warning,
                }
            )
            existing_ids.add(cid)

        branch_fields = [
            "phase",
            "branch_name",
            "phase_branch_id",
            "parent_candidate_id",
            "parent_candidate_ref",
            "parent_alias_source",
            "candidate_id",
            "editable_positions",
            "fixed_core_positions",
            "hotspot_set_name",
            "hotspot_tokens",
            "full_sequence",
            "strict_pass",
            "relaxed_pass",
            "rf2_pae",
            "design_vs_rf2_rmsd",
            "ranking_score",
            "threaded_pdb",
            "rf2_best_pdb",
            "cdr1_edit_count",
            "cdr1_edited_positions",
            "warning",
        ]
        write_rows(candidates_csv, candidates, branch_fields)

        branch_complete = len(candidates) >= target_per_branch
        if branch_complete:
            completed.add(branch.phase_branch_id)
        manifest_rows.append(
            {
                "branch_name": branch.branch_name,
                "phase_branch_id": branch.phase_branch_id,
                "status": "completed" if branch_complete else "partial",
                "updated_at": now_str(),
            }
        )
        write_status(
            status_path,
            {
                "phase": phase_name,
                "updated_at": now_str(),
                "resolved_test1_parent_candidate_id": parent["candidate_id"],
                "completed_branches": sorted(completed),
            },
        )

    atomic_write_csv(
        manifest_path,
        manifest_rows,
        ["branch_name", "phase_branch_id", "status", "updated_at"],
    )

    all_rows: List[dict] = []
    for branch in branches:
        cfile = branches_dir / branch.phase_branch_id / "candidates.csv"
        if cfile.exists():
            all_rows.extend(pd.read_csv(cfile).to_dict(orient="records"))
    if not all_rows:
        raise PipelineError("phase_next_test1_local_maturation generated no candidate rows.")

    df = pd.DataFrame(all_rows)
    for col in ("strict_pass", "relaxed_pass"):
        df[col] = df[col].astype(int)
    for col in ("rf2_pae", "design_vs_rf2_rmsd", "ranking_score"):
        df[col] = df[col].astype(float)
    dup = df["full_sequence"].astype(str).duplicated(keep=False)
    df["unique_sequence_flag"] = (~dup).astype(int)
    df = df.sort_values(["branch_name", "rf2_pae", "design_vs_rf2_rmsd", "candidate_id"], ascending=[True, True, True, True])

    summary_csv = root / "results/summaries/phase_next_test1_local_maturation_rf2_summary.csv"
    strict_csv = root / "results/summaries/phase_next_test1_local_maturation_strict_pass.csv"
    strict_fasta = root / "results/summaries/phase_next_test1_local_maturation_strict_pass.fasta"
    summary_md = root / "results/summaries/phase_next_test1_local_maturation_summary.md"

    summary_fields = [
        "branch_name",
        "parent_candidate_id",
        "candidate_id",
        "editable_positions",
        "hotspot_set_name",
        "rf2_pae",
        "design_vs_rf2_rmsd",
        "ranking_score",
        "strict_pass",
        "relaxed_pass",
        "full_sequence",
        "unique_sequence_flag",
        "phase_branch_id",
        "hotspot_tokens",
        "cdr1_edit_count",
        "cdr1_edited_positions",
        "threaded_pdb",
        "rf2_best_pdb",
        "warning",
    ]
    atomic_write_csv(summary_csv, df.to_dict(orient="records"), summary_fields)

    strict_df = df[df["strict_pass"] == 1].copy()
    strict_df = strict_df.sort_values(["branch_name", "rf2_pae", "design_vs_rf2_rmsd", "candidate_id"], ascending=[True, True, True, True])
    strict_fields = [
        "branch_name",
        "parent_candidate_id",
        "candidate_id",
        "rf2_pae",
        "design_vs_rf2_rmsd",
        "ranking_score",
        "full_sequence",
        "unique_sequence_flag",
        "editable_positions",
        "hotspot_set_name",
    ]
    atomic_write_csv(strict_csv, strict_df.to_dict(orient="records"), strict_fields)

    strict_unique = strict_df.drop_duplicates(subset=["full_sequence"], keep="first")
    strict_fasta.parent.mkdir(parents=True, exist_ok=True)
    with strict_fasta.open("w", encoding="utf-8") as handle:
        for _, row in strict_unique.iterrows():
            handle.write(f">{row['candidate_id']}\n{row['full_sequence']}\n")

    branch_lines: List[str] = [SAFETY_ETHICS_STATEMENT, "", "# Test1 Local Maturation (RF2-Only)", ""]
    branch_lines.append(f"- Resolved Test1 parent candidate ID: `{parent['candidate_id']}`")
    branch_lines.append(f"- Test1 resolution source: `{parent['alias_source']}`")
    branch_lines.append(f"- Shared hotspot set: `{hotspot_set['set_name']}` ({', '.join(hotspot_set['tokens'])})")
    branch_lines.append("")

    branch_scores: Dict[str, dict] = {}
    top_n = int(local_cfg.get("top_n_per_branch_for_summary", 3))
    for branch in branches:
        bdf = df[df["branch_name"] == branch.branch_name].copy()
        total = int(bdf.shape[0])
        strict_count = int(bdf["strict_pass"].sum()) if total else 0
        relaxed_count = int(bdf["relaxed_pass"].sum()) if total else 0
        strict_rate = strict_count / max(1, total)
        relaxed_rate = relaxed_count / max(1, total)
        bdf["rf2_quality"] = bdf["rf2_pae"] + bdf["design_vs_rf2_rmsd"]
        bbest = bdf.sort_values(["rf2_quality", "rf2_pae", "design_vs_rf2_rmsd"]).head(1)
        btop = bdf.sort_values(["rf2_quality", "rf2_pae", "design_vs_rf2_rmsd"]).head(max(1, top_n))
        top3_quality = float(btop["rf2_quality"].mean()) if not btop.empty else 999.0
        top1_quality = float(bbest.iloc[0]["rf2_quality"]) if not bbest.empty else 999.0
        strict_unique_count = int(bdf[(bdf["strict_pass"] == 1) & (bdf["unique_sequence_flag"] == 1)].shape[0])
        promising = (strict_count >= 5) or (strict_rate >= 0.10 and top1_quality < 10.0)
        branch_scores[branch.branch_name] = {
            "strict_count": strict_count,
            "relaxed_count": relaxed_count,
            "top1_quality": top1_quality,
            "top3_quality": top3_quality,
            "strict_unique_count": strict_unique_count,
            "promising": promising,
        }

        branch_lines.append(f"## {branch.branch_name}")
        branch_lines.append(f"- Total candidates: {total}")
        branch_lines.append(f"- Strict pass: {strict_count}/{total} ({strict_rate:.1%})")
        branch_lines.append(f"- Relaxed pass: {relaxed_count}/{total} ({relaxed_rate:.1%})")
        if not bbest.empty:
            row = bbest.iloc[0]
            branch_lines.append(
                f"- Best RF2 candidate: `{row['candidate_id']}` "
                f"(pAE={float(row['rf2_pae']):.3f}, RMSD={float(row['design_vs_rf2_rmsd']):.3f})"
            )
        branch_lines.append("- Top 3 RF2 candidates:")
        for _, row in btop.iterrows():
            branch_lines.append(
                f"  - `{row['candidate_id']}` "
                f"(pAE={float(row['rf2_pae']):.3f}, RMSD={float(row['design_vs_rf2_rmsd']):.3f}, "
                f"quality={float(row['rf2_quality']):.3f})"
            )
        branch_lines.append(f"- Promising for expansion: {'yes' if promising else 'no'}")
        branch_lines.append("")

    branch_lines.append("## Branch Comparison")
    compare_rows = []
    for bname, info in branch_scores.items():
        compare_rows.append(
            (
                bname,
                int(info["strict_count"]),
                int(info["relaxed_count"]),
                float(info["top1_quality"]),
                float(info["top3_quality"]),
                int(info["strict_unique_count"]),
            )
        )
    compare_rows = sorted(compare_rows, key=lambda x: (x[1], x[2], -x[3], -x[4], x[5]), reverse=True)
    for bname, strict_count, relaxed_count, top1_q, top3_q, strict_unique_count in compare_rows:
        branch_lines.append(
            f"- {bname}: strict={strict_count}, relaxed={relaxed_count}, "
            f"top1_quality={top1_q:.3f}, top3_quality={top3_q:.3f}, "
            f"strict_unique_sequences={strict_unique_count}"
        )
    if compare_rows:
        branch_lines.append(
            f"- Most promising branch for next AF3/web evaluation and later expansion: **{compare_rows[0][0]}**"
        )

    summary_md.write_text("\n".join(branch_lines) + "\n", encoding="utf-8")


def run_phase_next_champion_narrow50(context: dict, args: argparse.Namespace):
    root = context["root"]
    phases = context["phases_cfg"].get("phases", {})
    phase_cfg = phases.get("phase_next_champion_narrow50", {})
    local_cfg_root = context.get("champion_narrow_cfg", {}) or {}
    hotspot_cfg_root = context.get("champion_narrow_hotspots", {}) or {}
    if not local_cfg_root:
        raise PipelineError(
            "Missing narrowed champion config. Expected: data/configs/champion_narrow50_phase.yaml"
        )
    if not hotspot_cfg_root:
        raise PipelineError(
            "Missing narrowed champion hotspot config. Expected: data/configs/champion_narrow50_hotspots.yaml"
        )

    local_cfg = local_cfg_root.get("champion_narrow50", {})
    if not local_cfg:
        raise PipelineError("Missing 'champion_narrow50' block in narrowed champion config.")

    parent = resolve_best_phase7_parent_candidate(context=context, local_cfg=local_cfg)
    hotspot_set_name = cell_to_str(local_cfg.get("hotspot_set_name", "Champion_consensus_narrow_patch"))
    hotspot_set = build_champion_narrow50_hotspot_set(hotspot_cfg_root, set_name=hotspot_set_name)

    strict_cfg = local_cfg.get("strict_thresholds", {"rf2_pae_max": 10.0, "design_rf2_rmsd_max": 2.0})
    relaxed_cfg = local_cfg.get("relaxed_thresholds", {"rf2_pae_max": 12.0, "design_rf2_rmsd_max": 2.5})
    fixed_core_positions = parse_position_list(
        local_cfg.get("fixed_core_positions", [27, 28, 29, 30, 33, 34]),
        "champion_narrow50.fixed_core_positions",
    )
    editable_positions = parse_position_list(
        local_cfg.get("editable_positions", [25, 26, 31, 37, 39]),
        "champion_narrow50.editable_positions",
    )
    branch_name = cell_to_str(local_cfg.get("line_name", "Champion_Narrow50"))

    target_total = int(
        phase_cfg.get(
            "candidates_total",
            local_cfg.get("candidates_total", 50),
        )
    )
    if args.limit_per_combination is not None:
        target_total = min(target_total, int(args.limit_per_combination))
    if target_total <= 0:
        raise PipelineError("phase_next_champion_narrow50 candidates_total must be > 0.")

    aa_alphabet = re.sub(
        r"[^A-Z]",
        "",
        cell_to_str(local_cfg.get("aa_alphabet", "ACDEFGHIKLMNPQRSTVWY")).upper(),
    )
    if not aa_alphabet:
        aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    min_mutations = int(local_cfg.get("min_mutations_per_candidate", 1))
    max_mutations = int(local_cfg.get("max_mutations_per_candidate", max(1, len(editable_positions))))

    tooling = context["tool_cfg"]
    pipeline_cfg = context["pipeline_cfg"]
    rank_weights = pipeline_cfg.get("filters", {}).get("ranking_weights", {})
    seed_base = int(pipeline_cfg.get("project", {}).get("random_seed", 20260316))
    framework_parts = split_framework_and_cdr(context["nanobody_seq"], context["cdr"])

    phase_name = "phase_next_champion_narrow50"
    phase_dir = root / phase_name
    line_id = slugify(f"{parent['candidate_id']}__champion_narrow50")
    line_dir = phase_dir / "line" / line_id
    logs_dir = root / "logs" / phase_name
    ensure_dirs([phase_dir, line_dir, line_dir / "threaded", line_dir / "rf2_metrics", logs_dir])

    status_path = phase_dir / "phase_status.json"
    candidates_csv = line_dir / "candidates.csv"
    candidates = load_or_empty_csv(candidates_csv)
    existing_ids = {cell_to_str(x.get("candidate_id")) for x in candidates}
    seen_sequences = {cell_to_str(x.get("full_sequence")) for x in candidates if cell_to_str(x.get("full_sequence"))}

    parent_structure = Path(parent["structure_pdb"])
    if not args.dry_run and not parent_structure.exists():
        raise PipelineError(
            f"Resolved champion parent structure does not exist: {parent_structure}"
        )

    for i in range(1, target_total + 1):
        cid = f"{line_id}_c{i:03d}"
        if cid in existing_ids:
            continue

        final_seq = parent["full_sequence"]
        edited_pos: List[int] = []
        warning_parts: List[str] = []
        duplicate_after_sampling = False
        for attempt in range(1, 97):
            rng = deterministic_rng(seed_base, f"{cid}::attempt{attempt}")
            proposed_seq, proposed_edited = mutate_local_positions(
                parent_seq=parent["full_sequence"],
                editable_positions=editable_positions,
                rng=rng,
                aa_alphabet=aa_alphabet,
                min_mutations=min_mutations,
                max_mutations=max_mutations,
            )
            constrained_seq, constrained_edited, mask_warning = enforce_cdr1_editable_positions(
                parent_full_seq=parent["full_sequence"],
                proposed_full_seq=proposed_seq,
                editable_positions=editable_positions,
            )
            core_violations = [
                p
                for p in fixed_core_positions
                if 1 <= p <= len(constrained_seq)
                and constrained_seq[p - 1] != parent["full_sequence"][p - 1]
            ]
            if core_violations:
                raise PipelineError(
                    f"{cid} violates fixed core positions: {core_violations}"
                )
            if constrained_seq in seen_sequences and attempt < 96:
                continue
            duplicate_after_sampling = constrained_seq in seen_sequences
            final_seq = constrained_seq
            edited_pos = constrained_edited or proposed_edited
            if mask_warning:
                warning_parts.append(mask_warning)
            break

        if duplicate_after_sampling:
            warning_parts.append("Duplicate full sequence after resampling attempts.")
        if edited_pos:
            warning_parts.append(
                "Edited positions: " + ",".join(str(x) for x in sorted(set(edited_pos)))
            )
        seen_sequences.add(final_seq)

        threaded_pdb = line_dir / "threaded" / f"{cid}.pdb"
        threading_warning = ""
        if args.dry_run or not tooling.execute_real_tools:
            threaded_pdb = parent_structure
        else:
            try:
                thread_sequence_on_backbone_pose(
                    backbone_pdb=parent_structure,
                    binder_sequence=final_seq,
                    out_pdb=threaded_pdb,
                    binder_chain=context["cdr"].chain_id or "H",
                )
            except PipelineError as exc:
                msg = str(exc)
                if "Failed to import rfantibody.util.pose" in msg:
                    threaded_pdb = parent_structure
                    threading_warning = (
                        "Pose-threading unavailable (rfantibody import failed); "
                        "using parent backbone directly for RF2 input."
                    )
                else:
                    raise

        rf2_json = line_dir / "rf2_metrics" / f"{cid}_rf2.json"
        metrics = run_rf2_filter(
            cfg=tooling,
            input_pdb=threaded_pdb,
            sequence=final_seq,
            out_json=rf2_json,
            dry_run=args.dry_run,
            log_file=logs_dir / f"{line_id}_rf2.log",
            seed=seed_base,
            context={
                "candidate_id": cid,
                "campaign_name": parent.get("campaign_name", "phase_next_champion_narrow50"),
                "cdr3_contact_bias": 1,
            },
        )

        heuristics = compute_interface_heuristics(
            pdb_path=Path(str(metrics.get("rf2_best_pdb", threaded_pdb))),
            parts=framework_parts,
            h1_len=int(parent["h1_length"]),
            h3_len=int(parent["h3_length"]),
            hotspot_tokens=hotspot_set["tokens"],
            cutoff=5.0,
        )
        metrics.update(heuristics)
        if "structural_plausibility" not in metrics:
            metrics["structural_plausibility"] = max(
                0.0, min(1.0, float(metrics.get("rf2_pred_lddt", 0.0)))
            )
        ranking_score = round(float(combine_weighted_score(metrics, rank_weights)), 6)

        strict_pass, relaxed_pass = rescue_strict_relaxed_flags(
            rf2_pae=_safe_float(metrics.get("rf2_pae"), 99.0),
            rf2_rmsd=_safe_float(metrics.get("design_rf2_rmsd"), 99.0),
            strict_cfg=strict_cfg,
            relaxed_cfg=relaxed_cfg,
        )

        if threading_warning:
            warning_parts.append(threading_warning)
        warning = " | ".join(warning_parts)
        candidates.append(
            {
                "phase": phase_name,
                "line_name": branch_name,
                "phase_line_id": line_id,
                "parent_candidate_id": parent["candidate_id"],
                "parent_ref_source": parent.get("alias_source", ""),
                "resolved_phase7_job_name": parent.get("resolved_phase7_job_name", ""),
                "candidate_id": cid,
                "editable_positions": ",".join(str(x) for x in editable_positions),
                "fixed_core_positions": ",".join(str(x) for x in fixed_core_positions),
                "hotspot_set_name": hotspot_set["set_name"],
                "hotspot_tokens": ",".join(hotspot_set["tokens"]),
                "full_sequence": final_seq,
                "strict_pass": int(strict_pass),
                "relaxed_pass": int(relaxed_pass),
                "rf2_pae": _safe_float(metrics.get("rf2_pae"), 99.0),
                "design_vs_rf2_rmsd": _safe_float(metrics.get("design_rf2_rmsd"), 99.0),
                "ranking_score": ranking_score,
                "threaded_pdb": str(threaded_pdb),
                "rf2_best_pdb": str(metrics.get("rf2_best_pdb", "")),
                "cdr1_edit_count": len(edited_pos),
                "cdr1_edited_positions": ",".join(str(x) for x in sorted(set(edited_pos))),
                "warning": warning,
            }
        )
        existing_ids.add(cid)

    field_order = [
        "phase",
        "line_name",
        "phase_line_id",
        "parent_candidate_id",
        "parent_ref_source",
        "resolved_phase7_job_name",
        "candidate_id",
        "editable_positions",
        "fixed_core_positions",
        "hotspot_set_name",
        "hotspot_tokens",
        "full_sequence",
        "strict_pass",
        "relaxed_pass",
        "rf2_pae",
        "design_vs_rf2_rmsd",
        "ranking_score",
        "threaded_pdb",
        "rf2_best_pdb",
        "cdr1_edit_count",
        "cdr1_edited_positions",
        "warning",
    ]
    write_rows(candidates_csv, candidates, field_order)

    write_status(
        status_path,
        {
            "phase": phase_name,
            "updated_at": now_str(),
            "resolved_parent_candidate_id": parent["candidate_id"],
            "resolved_parent_source": parent.get("alias_source", ""),
            "resolved_phase7_job_name": parent.get("resolved_phase7_job_name", ""),
            "completed": int(len(candidates) >= target_total),
            "candidate_rows": int(len(candidates)),
        },
    )

    if not candidates:
        raise PipelineError("phase_next_champion_narrow50 generated no candidate rows.")

    df = pd.DataFrame(candidates)
    for col in ("strict_pass", "relaxed_pass"):
        df[col] = df[col].astype(int)
    for col in ("rf2_pae", "design_vs_rf2_rmsd", "ranking_score"):
        df[col] = df[col].astype(float)
    dup = df["full_sequence"].astype(str).duplicated(keep=False)
    df["unique_sequence_flag"] = (~dup).astype(int)
    df = df.sort_values(["rf2_pae", "design_vs_rf2_rmsd", "candidate_id"], ascending=[True, True, True])

    summary_csv = root / "results/summaries/phase_next_champion_narrow50_rf2_summary.csv"
    strict_csv = root / "results/summaries/phase_next_champion_narrow50_strict_pass.csv"
    strict_fasta = root / "results/summaries/phase_next_champion_narrow50_strict_pass.fasta"
    summary_md = root / "results/summaries/phase_next_champion_narrow50_summary.md"

    summary_fields = [
        "parent_candidate_id",
        "candidate_id",
        "editable_positions",
        "hotspot_set_name",
        "rf2_pae",
        "design_vs_rf2_rmsd",
        "ranking_score",
        "strict_pass",
        "relaxed_pass",
        "full_sequence",
        "unique_sequence_flag",
        "line_name",
        "phase_line_id",
        "hotspot_tokens",
        "cdr1_edit_count",
        "cdr1_edited_positions",
        "threaded_pdb",
        "rf2_best_pdb",
        "warning",
    ]
    atomic_write_csv(summary_csv, df.to_dict(orient="records"), summary_fields)

    strict_df = df[df["strict_pass"] == 1].copy().sort_values(
        ["rf2_pae", "design_vs_rf2_rmsd", "candidate_id"],
        ascending=[True, True, True],
    )
    strict_fields = [
        "parent_candidate_id",
        "candidate_id",
        "rf2_pae",
        "design_vs_rf2_rmsd",
        "ranking_score",
        "full_sequence",
        "unique_sequence_flag",
        "editable_positions",
        "hotspot_set_name",
    ]
    atomic_write_csv(strict_csv, strict_df.to_dict(orient="records"), strict_fields)

    strict_unique = strict_df.drop_duplicates(subset=["full_sequence"], keep="first")
    strict_fasta.parent.mkdir(parents=True, exist_ok=True)
    with strict_fasta.open("w", encoding="utf-8") as handle:
        for _, row in strict_unique.iterrows():
            handle.write(f">{row['candidate_id']}\n{row['full_sequence']}\n")

    total = int(df.shape[0])
    strict_count = int(df["strict_pass"].sum())
    relaxed_count = int(df["relaxed_pass"].sum())
    strict_rate = strict_count / max(1, total)
    relaxed_rate = relaxed_count / max(1, total)
    df["rf2_quality"] = df["rf2_pae"] + df["design_vs_rf2_rmsd"]
    best_row = df.sort_values(["rf2_quality", "rf2_pae", "design_vs_rf2_rmsd"]).head(1)
    top3 = df.sort_values(["rf2_quality", "rf2_pae", "design_vs_rf2_rmsd"]).head(3)
    top1_quality = float(best_row.iloc[0]["rf2_quality"]) if not best_row.empty else 999.0
    promising = (strict_count >= 5) or (strict_rate >= 0.10 and top1_quality < 10.0)

    compare_lines: List[str] = []
    parent_pae = _safe_float(parent.get("rf2_pae"), float("nan"))
    parent_rmsd = _safe_float(parent.get("design_vs_rf2_rmsd"), float("nan"))
    if not math.isnan(parent_pae) and not math.isnan(parent_rmsd):
        parent_quality = parent_pae + parent_rmsd
        compare_lines.append(
            f"- Parent RF2 baseline (from existing summaries): pAE={parent_pae:.3f}, RMSD={parent_rmsd:.3f}, quality={parent_quality:.3f}"
        )
    prev_table = root / "results/summaries/phase_next_test1_local_maturation_rf2_summary.csv"
    if prev_table.exists():
        try:
            prev_df = pd.read_csv(prev_table)
            if not prev_df.empty and {"rf2_pae", "design_vs_rf2_rmsd"}.issubset(prev_df.columns):
                prev_df["rf2_quality"] = prev_df["rf2_pae"].astype(float) + prev_df["design_vs_rf2_rmsd"].astype(float)
                prow = prev_df.sort_values(["rf2_quality", "rf2_pae", "design_vs_rf2_rmsd"]).iloc[0]
                compare_lines.append(
                    f"- Recent benchmark (phase_next_test1_local_maturation best): "
                    f"candidate={cell_to_str(prow.get('candidate_id'))}, "
                    f"pAE={float(prow['rf2_pae']):.3f}, RMSD={float(prow['design_vs_rf2_rmsd']):.3f}, "
                    f"quality={float(prow['rf2_quality']):.3f}"
                )
        except Exception:
            pass
    phase6_table = root / "results/summaries/phase6_cdr1_rescue_final_ranked_candidates.csv"
    if phase6_table.exists():
        try:
            p6 = pd.read_csv(phase6_table)
            if not p6.empty and {"rf2_pae", "design_rf2_rmsd"}.issubset(p6.columns):
                p6["rf2_quality"] = p6["rf2_pae"].astype(float) + p6["design_rf2_rmsd"].astype(float)
                r6 = p6.sort_values(["rf2_quality", "rf2_pae", "design_rf2_rmsd"]).iloc[0]
                compare_lines.append(
                    f"- Recent benchmark (phase6 best): candidate={cell_to_str(r6.get('candidate_id'))}, "
                    f"pAE={float(r6['rf2_pae']):.3f}, RMSD={float(r6['design_rf2_rmsd']):.3f}, "
                    f"quality={float(r6['rf2_quality']):.3f}"
                )
        except Exception:
            pass

    lines = [SAFETY_ETHICS_STATEMENT, "", "# Champion Narrow50 Local Maturation (RF2-Only)", ""]
    lines.append(f"- Resolved parent candidate ID: `{parent['candidate_id']}`")
    lines.append(f"- Parent resolution source: `{parent.get('alias_source', '')}`")
    lines.append(f"- Stage7 best job used for parent resolution: `{parent.get('resolved_phase7_job_name', '')}`")
    lines.append(
        f"- Shared hotspot set: `{hotspot_set['set_name']}` ({', '.join(hotspot_set['tokens'])})"
    )
    lines.append("")
    lines.append(f"- Total candidates generated: {total}")
    lines.append(f"- Strict pass: {strict_count}/{total} ({strict_rate:.1%})")
    lines.append(f"- Relaxed pass: {relaxed_count}/{total} ({relaxed_rate:.1%})")
    if not best_row.empty:
        row = best_row.iloc[0]
        lines.append(
            f"- Best RF2 candidate: `{row['candidate_id']}` "
            f"(pAE={float(row['rf2_pae']):.3f}, RMSD={float(row['design_vs_rf2_rmsd']):.3f}, quality={float(row['rf2_quality']):.3f})"
        )
    lines.append("- Top 3 RF2 candidates:")
    for _, row in top3.iterrows():
        lines.append(
            f"  - `{row['candidate_id']}` "
            f"(pAE={float(row['rf2_pae']):.3f}, RMSD={float(row['design_vs_rf2_rmsd']):.3f}, quality={float(row['rf2_quality']):.3f})"
        )
    lines.append("")
    lines.append("## RF2 Benchmark Comparison")
    if compare_lines:
        lines.extend(compare_lines)
    else:
        lines.append("- No prior benchmark tables were available locally for quantitative comparison.")
    lines.append("")
    lines.append(
        f"- Promising for later AF3/manual evaluation and larger-scale expansion: {'yes' if promising else 'no'}"
    )
    lines.append("- This conclusion is RF2-only and does not claim AF3/interface improvement.")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_phase_design(
    phase_name: str,
    combinations: List[Combination],
    context: dict,
    args: argparse.Namespace,
) -> List[dict]:
    root = context["root"]
    phase_cfg = context["phases_cfg"]["phases"][phase_name]
    pipeline_cfg = context["pipeline_cfg"]
    tooling = context["tool_cfg"]
    filter_cfg = pipeline_cfg.get("filters", {})
    rank_weights = filter_cfg.get("ranking_weights", {})
    seed_base = int(pipeline_cfg.get("project", {}).get("random_seed", 20260316))

    framework_parts = split_framework_and_cdr(context["nanobody_seq"], context["cdr"])

    phase_dir = root / phase_name
    combos_dir = phase_dir / "combinations"
    logs_dir = root / "logs" / phase_name
    ensure_dirs([phase_dir, combos_dir, logs_dir])

    backbones_target = int(phase_cfg["backbones_per_combination"])
    seqs_per_backbone = int(phase_cfg["sequences_per_backbone"])
    if args.limit_per_combination is not None:
        backbones_target = min(backbones_target, int(args.limit_per_combination))

    if args.max_combinations is not None:
        combinations = combinations[: int(args.max_combinations)]

    status_path = phase_dir / "phase_status.json"
    manifest_path = phase_dir / "phase_manifest.csv"
    status = read_status(status_path)
    completed = set(status.get("completed_combinations", [])) if args.resume else set()
    manifest_rows: List[dict] = []

    combo_summaries: List[dict] = []

    target_pdb = Path(context["resolved_targets"]["cropped_design_target"])
    if not target_pdb.exists():
        raise PipelineError(f"Missing cropped design target: {target_pdb}")
    framework_pdb = Path(context.get("framework_pdb", "")) if context.get("framework_pdb", "") else None
    if not args.dry_run and framework_pdb is None:
        raise PipelineError(
            "Real execution requires inputs.nanobody_framework_pdb_file in data/configs/pipeline.yaml."
        )
    if not args.dry_run and framework_pdb and not framework_pdb.exists():
        raise PipelineError(f"Configured framework PDB does not exist: {framework_pdb}")
    target_contig = str(context["rfdiffusion_target_contig"])
    campaign_hotspot_tokens = context.get("campaign_hotspot_tokens", {})

    framework_fixed_len = (
        len(framework_parts["framework_prefix"])
        + len(framework_parts["framework_between_h1_h2"])
        + len(framework_parts["h2_native"])
        + len(framework_parts["framework_between_h2_h3"])
        + len(framework_parts["framework_suffix"])
    )

    for combo in combinations:
        combo_dir = combos_dir / combo.combination_id
        ensure_dirs([combo_dir])

        if args.resume and combo.combination_id in completed:
            summary_path = combo_dir / "combination_summary.json"
            if summary_path.exists():
                combo_summaries.append(read_json(summary_path))
                log(f"[{phase_name}] Skipping completed combination: {combo.combination_id}")
                manifest_rows.append(
                    {
                        "combination_id": combo.combination_id,
                        "campaign_name": combo.campaign_name,
                        "status": "skipped_resume",
                        "updated_at": now_str(),
                    }
                )
                continue

        backbones_csv = combo_dir / "backbones.csv"
        candidates_csv = combo_dir / "candidates.csv"
        metrics_dir = combo_dir / "rf2_metrics"
        seq_aux_dir = combo_dir / "mpnn_aux"
        backbone_dir = combo_dir / "backbones"
        ensure_dirs([metrics_dir, seq_aux_dir, backbone_dir])

        backbones = load_or_empty_csv(backbones_csv)
        backbones_changed = False
        for bb in backbones:
            sig = str(bb.get("backbone_signature", "")).strip()
            if sig:
                continue
            bb_pdb_existing = Path(str(bb.get("backbone_pdb", "")).strip())
            bb["backbone_signature"] = compute_backbone_signature(bb_pdb_existing)
            backbones_changed = True

        existing_backbone_ids = {x["backbone_id"] for x in backbones}

        # stage 1: backbone generation
        for i in range(1, backbones_target + 1):
            bb_id = f"{combo.combination_id}_bb{i:03d}"
            if bb_id in existing_backbone_ids:
                continue
            bb_pdb = backbone_dir / f"{bb_id}.pdb"
            binder_len = framework_fixed_len + int(combo.h1_length) + int(combo.h3_length)
            hotspot_tokens = campaign_hotspot_tokens.get(combo.campaign_name, [])
            run_rfdiffusion_backbone(
                cfg=tooling,
                combo={
                    **combo.__dict__,
                    "h2_length": context["cdr"].h2_len,
                },
                backbone_id=bb_id,
                target_pdb=target_pdb,
                framework_pdb=framework_pdb if framework_pdb else Path("MISSING_FRAMEWORK_PLACEHOLDER.pdb"),
                hotspots=hotspot_tokens,
                target_contig=target_contig,
                binder_length=binder_len,
                out_pdb=bb_pdb,
                seed=seed_base,
                log_file=logs_dir / f"{combo.combination_id}_rfdiffusion.log",
                dry_run=args.dry_run,
            )
            backbones.append(
                {
                    "combination_id": combo.combination_id,
                    "campaign_name": combo.campaign_name,
                    "backbone_id": bb_id,
                    "backbone_pdb": str(bb_pdb),
                    "backbone_signature": compute_backbone_signature(bb_pdb),
                }
            )
            existing_backbone_ids.add(bb_id)

        if backbones_changed:
            log(f"[{phase_name}] Refreshed missing backbone signatures for {combo.combination_id}.")
        write_rows(
            backbones_csv,
            backbones,
            ["combination_id", "campaign_name", "backbone_id", "backbone_pdb", "backbone_signature"],
        )

        # stage 2+3: sequence generation + RF2 filtering
        candidates = load_or_empty_csv(candidates_csv)
        existing_candidate_ids = {x["candidate_id"] for x in candidates}

        for bb in backbones:
            bb_id = bb["backbone_id"]
            bb_pdb = Path(bb["backbone_pdb"])
            expected_ids = {f"{bb_id}_s{sidx:02d}" for sidx in range(1, seqs_per_backbone + 1)}
            if expected_ids.issubset(existing_candidate_ids):
                continue

            mpnn_records = run_proteinmpnn_sequence_design(
                cfg=tooling,
                backbone_pdb=bb_pdb,
                out_dir=seq_aux_dir / bb_id,
                seed=seed_base,
                dry_run=args.dry_run,
                log_file=logs_dir / f"{combo.combination_id}_mpnn.log",
                loops="H1,H3",
                seqs_per_struct=seqs_per_backbone,
                temperature=0.1,
            )
            if not mpnn_records:
                raise PipelineError(f"ProteinMPNN produced no records for backbone {bb_id}")

            for sidx in range(1, seqs_per_backbone + 1):
                cid = f"{bb_id}_s{sidx:02d}"
                if cid in existing_candidate_ids:
                    continue

                record = mpnn_records[min(sidx - 1, len(mpnn_records) - 1)]
                designed_pdb = Path(record.get("designed_pdb", str(bb_pdb)))
                full_seq = str(record.get("full_sequence", "")).strip().upper()
                warning_msg = ""

                try:
                    h1_seq, h2_seq, h3_seq = split_designed_sequence(
                        parts=framework_parts,
                        full_seq=full_seq,
                        h1_len=combo.h1_length,
                        h3_len=combo.h3_length,
                    )
                except Exception as exc:
                    warning_msg = f"Designed sequence parsing fallback used: {exc}"
                    rng = deterministic_rng(seed_base, cid)
                    h1_seq = random_loop(combo.h1_length, rng)
                    h2_seq = framework_parts["h2_native"]
                    h3_seq = random_loop(combo.h3_length, rng)
                    full_seq = compose_nanobody_sequence(framework_parts, h1_seq=h1_seq, h2_seq=h2_seq, h3_seq=h3_seq)

                rf2_json = metrics_dir / f"{cid}_rf2.json"
                metrics = run_rf2_filter(
                    cfg=tooling,
                    input_pdb=designed_pdb,
                    sequence=full_seq,
                    out_json=rf2_json,
                    dry_run=args.dry_run,
                    log_file=logs_dir / f"{combo.combination_id}_rf2.log",
                    seed=seed_base,
                    context={
                        "candidate_id": cid,
                        "campaign_name": combo.campaign_name,
                        "cdr3_contact_bias": 1 if combo.h3_delta >= 0 else 0,
                    },
                )

                structure_for_contacts = Path(str(metrics.get("rf2_best_pdb", designed_pdb)))
                heuristics = compute_interface_heuristics(
                    pdb_path=structure_for_contacts,
                    parts=framework_parts,
                    h1_len=combo.h1_length,
                    h3_len=combo.h3_length,
                    hotspot_tokens=campaign_hotspot_tokens.get(combo.campaign_name, []),
                    cutoff=5.0,
                )
                metrics.update(heuristics)

                if "structural_plausibility" not in metrics:
                    metrics["structural_plausibility"] = max(0.0, min(1.0, float(metrics.get("rf2_pred_lddt", 0.0))))

                passed = hard_pass(metrics, filter_cfg)
                rank_score = combine_weighted_score(metrics, rank_weights)

                row = {
                    "phase": phase_name,
                    "combination_id": combo.combination_id,
                    "campaign_name": combo.campaign_name,
                    "hotspot_full_length_residues": ",".join(str(x) for x in combo.hotspot_full_length_residues),
                    "h1_delta": combo.h1_delta,
                    "h3_delta": combo.h3_delta,
                    "h1_length": combo.h1_length,
                    "h3_length": combo.h3_length,
                    "backbone_id": bb_id,
                    "backbone_pdb": str(bb_pdb),
                    "backbone_signature": str(bb.get("backbone_signature", "")),
                    "candidate_id": cid,
                    "designed_pdb": str(designed_pdb),
                    "rf2_best_pdb": str(metrics.get("rf2_best_pdb", "")),
                    "h1_sequence": h1_seq,
                    "h2_sequence": h2_seq,
                    "h3_sequence": h3_seq,
                    "full_sequence": full_seq,
                    "rf2_pae": metrics.get("rf2_pae", ""),
                    "design_rf2_rmsd": metrics.get("design_rf2_rmsd", ""),
                    "hotspot_agreement": metrics.get("hotspot_agreement", ""),
                    "groove_localization": metrics.get("groove_localization", ""),
                    "h1_h3_role_consistency": metrics.get("h1_h3_role_consistency", ""),
                    "structural_plausibility": metrics.get("structural_plausibility", ""),
                    "target_contact_residue_count": metrics.get("target_contact_residue_count", ""),
                    "hotspot_overlap_count": metrics.get("hotspot_overlap_count", ""),
                    "hard_filter_pass": int(passed),
                    "ranking_score": round(float(rank_score), 6),
                    "warning": warning_msg,
                }
                candidates.append(row)
                existing_candidate_ids.add(cid)

        candidate_fields = [
            "phase",
            "combination_id",
            "campaign_name",
            "hotspot_full_length_residues",
            "h1_delta",
            "h3_delta",
            "h1_length",
            "h3_length",
            "backbone_id",
            "backbone_pdb",
            "backbone_signature",
            "candidate_id",
            "designed_pdb",
            "rf2_best_pdb",
            "h1_sequence",
            "h2_sequence",
            "h3_sequence",
            "full_sequence",
            "rf2_pae",
            "design_rf2_rmsd",
            "hotspot_agreement",
            "groove_localization",
            "h1_h3_role_consistency",
            "structural_plausibility",
            "target_contact_residue_count",
            "hotspot_overlap_count",
            "hard_filter_pass",
            "ranking_score",
            "warning",
        ]
        write_rows(candidates_csv, candidates, candidate_fields)

        df = pd.DataFrame(candidates)
        if df.empty:
            summary = {
                "phase": phase_name,
                "combination_id": combo.combination_id,
                "campaign_name": combo.campaign_name,
                "h1_length": combo.h1_length,
                "h3_length": combo.h3_length,
                "total_candidates": 0,
                "hard_pass_candidates": 0,
                "unique_sequences": 0,
                "unique_ranking_scores": 0,
                "unique_backbone_signatures": 0,
                "best_ranking_score": 0.0,
                "mean_ranking_score": 0.0,
            }
        else:
            uniq_seq = int(df["full_sequence"].nunique()) if "full_sequence" in df.columns else 0
            uniq_rank = int(df["ranking_score"].nunique()) if "ranking_score" in df.columns else 0
            if "backbone_signature" in df.columns:
                sig_col = df["backbone_signature"].fillna("").astype(str).str.strip()
                known_backbone_sigs = int(sig_col.ne("").sum())
                uniq_backbone_sig = int(sig_col[sig_col.ne("")].nunique())
            else:
                known_backbone_sigs = 0
                uniq_backbone_sig = 0
            no_candidate_div = int(df.shape[0]) > 1 and uniq_seq <= 1 and uniq_rank <= 1
            no_backbone_div = known_backbone_sigs > 1 and uniq_backbone_sig <= 1
            if no_candidate_div or no_backbone_div:
                log(
                    f"[WARN] {combo.combination_id} diversity issue "
                    f"(unique_sequences={uniq_seq}, unique_ranking_scores={uniq_rank}, "
                    f"unique_backbone_signatures={uniq_backbone_sig}, known_backbone_signatures={known_backbone_sigs}). "
                    "Check deterministic settings and seed strategy for RFdiffusion/ProteinMPNN/RF2."
                )
                if phase_name in {"phase2_focused_pilot", "phase3_main_campaign"}:
                    raise PipelineError(
                        f"{phase_name} combination {combo.combination_id} failed diversity checks "
                        f"(unique_sequences={uniq_seq}, unique_ranking_scores={uniq_rank}, "
                        f"unique_backbone_signatures={uniq_backbone_sig})."
                    )
            summary = {
                "phase": phase_name,
                "combination_id": combo.combination_id,
                "campaign_name": combo.campaign_name,
                "h1_length": combo.h1_length,
                "h3_length": combo.h3_length,
                "total_candidates": int(df.shape[0]),
                "hard_pass_candidates": int(df["hard_filter_pass"].sum()),
                "unique_sequences": uniq_seq,
                "unique_ranking_scores": uniq_rank,
                "unique_backbone_signatures": uniq_backbone_sig,
                "best_ranking_score": float(df["ranking_score"].max()),
                "mean_ranking_score": float(df["ranking_score"].mean()),
                "mean_rf2_pae": float(df["rf2_pae"].mean()),
                "mean_design_rf2_rmsd": float(df["design_rf2_rmsd"].mean()),
            }

        write_json(combo_dir / "combination_summary.json", summary)
        combo_summaries.append(summary)
        manifest_rows.append(
            {
                "combination_id": combo.combination_id,
                "campaign_name": combo.campaign_name,
                "status": "completed",
                "updated_at": now_str(),
            }
        )

        completed.add(combo.combination_id)
        write_status(
            status_path,
            {
                "phase": phase_name,
                "updated_at": now_str(),
                "completed_combinations": sorted(completed),
            },
        )

    atomic_write_csv(
        manifest_path,
        manifest_rows,
        ["combination_id", "campaign_name", "status", "updated_at"],
    )

    # phase summary tables
    summary_fields = [
        "phase",
        "combination_id",
        "campaign_name",
        "h1_length",
        "h3_length",
        "total_candidates",
        "hard_pass_candidates",
        "unique_sequences",
        "unique_ranking_scores",
        "unique_backbone_signatures",
        "best_ranking_score",
        "mean_ranking_score",
        "mean_rf2_pae",
        "mean_design_rf2_rmsd",
    ]
    phase_summary_csv = root / "results/summaries" / f"{phase_name}_summary.csv"
    atomic_write_csv(phase_summary_csv, combo_summaries, summary_fields)

    # Per-combination pilot summary alias files for phase1/2
    if phase_name == "phase1_coarse_pilot":
        atomic_write_csv(root / "results/summaries/phase1_combination_summary.csv", combo_summaries, summary_fields)
        select_top_combinations(
            combo_summaries,
            top_n=8,
            out_csv=root / "results/summaries/phase1_top8_combinations.csv",
        )
    if phase_name == "phase2_focused_pilot":
        atomic_write_csv(root / "results/summaries/phase2_combination_summary.csv", combo_summaries, summary_fields)
        select_top_combinations(
            combo_summaries,
            top_n=2,
            out_csv=root / "results/summaries/phase2_top2_combinations.csv",
        )

    if phase_name == "phase3_main_campaign":
        select_top25_pre_h2(context=context, phase_name=phase_name)

    return combo_summaries


def select_top_combinations(summaries: List[dict], top_n: int, out_csv: Path):
    if not summaries:
        atomic_write_csv(out_csv, [], ["rank", "combination_id", "campaign_name", "h1_length", "h3_length"])
        return

    ranked = sorted(
        summaries,
        key=lambda x: (
            int(x.get("hard_pass_candidates", 0)),
            float(x.get("best_ranking_score", 0.0)),
            float(x.get("mean_ranking_score", 0.0)),
        ),
        reverse=True,
    )
    rows = []
    for i, row in enumerate(ranked[:top_n], start=1):
        rows.append(
            {
                "rank": i,
                "combination_id": row["combination_id"],
                "campaign_name": row["campaign_name"],
                "h1_length": row["h1_length"],
                "h3_length": row["h3_length"],
                "hard_pass_candidates": row.get("hard_pass_candidates", 0),
                "best_ranking_score": row.get("best_ranking_score", 0),
            }
        )

    atomic_write_csv(
        out_csv,
        rows,
        [
            "rank",
            "combination_id",
            "campaign_name",
            "h1_length",
            "h3_length",
            "hard_pass_candidates",
            "best_ranking_score",
        ],
    )


def collect_phase_candidates(phase_name: str, root: Path) -> List[dict]:
    phase_dir = root / phase_name / "combinations"
    if not phase_dir.exists():
        return []
    rows = []
    for combo_dir in sorted(phase_dir.iterdir()):
        cfile = combo_dir / "candidates.csv"
        if cfile.exists():
            rows.extend(pd.read_csv(cfile).to_dict(orient="records"))
    return rows


def select_top25_pre_h2(context: dict, phase_name: str):
    root = context["root"]
    pipeline_cfg = context["pipeline_cfg"]

    rows = collect_phase_candidates(phase_name, root)
    if not rows:
        raise PipelineError("No phase3 candidates found for pre-H2 selection.")

    passed = [r for r in rows if int(r.get("hard_filter_pass", 0)) == 1]
    if not passed:
        # Fail gracefully but preserve top scoring if nothing passes.
        passed = sorted(rows, key=lambda x: float(x.get("ranking_score", 0.0)), reverse=True)

    dedup_threshold = float(pipeline_cfg.get("postprocess", {}).get("sequence_dedup_identity_threshold", 0.95))
    deduped = greedy_sequence_dedup(
        rows=passed,
        sequence_key="full_sequence",
        score_key="ranking_score",
        identity_threshold=dedup_threshold,
    )
    ranked = sorted(deduped, key=lambda x: float(x.get("ranking_score", 0.0)), reverse=True)
    top25 = ranked[:25]

    out = root / "results/summaries/phase3_top25_pre_h2.csv"
    fields = [
        "phase",
        "combination_id",
        "campaign_name",
        "h1_length",
        "h3_length",
        "backbone_id",
        "backbone_pdb",
        "candidate_id",
        "designed_pdb",
        "rf2_best_pdb",
        "h1_sequence",
        "h2_sequence",
        "h3_sequence",
        "full_sequence",
        "rf2_pae",
        "design_rf2_rmsd",
        "hotspot_agreement",
        "groove_localization",
        "h1_h3_role_consistency",
        "structural_plausibility",
        "target_contact_residue_count",
        "hotspot_overlap_count",
        "hard_filter_pass",
        "ranking_score",
        "warning",
    ]
    atomic_write_csv(out, top25, fields)
    atomic_write_csv(root / "results/summaries/top25_pre_h2_table.csv", top25, fields)


def run_phase4_h2_refine(context: dict, args: argparse.Namespace):
    root = context["root"]
    pipeline_cfg = context["pipeline_cfg"]
    phase_cfg = context["phases_cfg"]["phases"]["phase4_h2_refine"]
    tooling = context["tool_cfg"]

    rows, pre_h2_path = load_phase4_input_rows(context=context, args=args)
    log(f"[phase4_h2_refine] Loaded {len(rows)} parent candidates from: {pre_h2_path}")

    variants_n = int(phase_cfg.get("h2_variants_per_candidate", 4))
    seed_base = int(pipeline_cfg.get("project", {}).get("random_seed", 20260316)) + 404

    framework_parts = split_framework_and_cdr(context["nanobody_seq"], context["cdr"])
    filter_cfg = pipeline_cfg.get("filters", {})
    rank_weights = filter_cfg.get("ranking_weights", {})

    phase_dir = root / "phase4_h2_refine"
    logs_dir = root / "logs/phase4_h2_refine"
    out_metrics_dir = phase_dir / "rf2_metrics"
    out_aux_dir = phase_dir / "mpnn_aux"
    ensure_dirs([phase_dir, logs_dir, out_metrics_dir, out_aux_dir])

    final_rows = []
    manifest_rows = []
    for row in rows:
        parent_id = cell_to_str(row.get("candidate_id"))
        h1_seq_parent = cell_to_str(row.get("h1_sequence"))
        h2_parent = cell_to_str(row.get("h2_sequence"))
        h3_seq_parent = cell_to_str(row.get("h3_sequence"))
        parent_backbone = cell_to_str(row.get("backbone_id"))
        parent_backbone_pdb = cell_to_str(row.get("backbone_pdb"))
        campaign_name = cell_to_str(row.get("campaign_name"))
        h1_len = parse_int_maybe(row.get("h1_length"), len(h1_seq_parent))
        h3_len = parse_int_maybe(row.get("h3_length"), len(h3_seq_parent))

        parent_input_pdb = (
            resolve_path_like(root, cell_to_str(row.get("rf2_best_pdb")))
            or resolve_path_like(root, cell_to_str(row.get("designed_pdb")))
            or resolve_path_like(root, parent_backbone_pdb)
        )
        if parent_input_pdb is None:
            if args.dry_run:
                parent_input_pdb = root / "MISSING_PARENT_PDB_PLACEHOLDER.pdb"
            else:
                raise PipelineError(f"Phase4 requires a parent structure path for {parent_id}, but none was provided.")
        if not args.dry_run and not parent_input_pdb.exists():
            raise PipelineError(
                f"Phase4 requires a real parent structure PDB. Missing for {parent_id}: {parent_input_pdb}"
            )

        mpnn_records = run_proteinmpnn_sequence_design(
            cfg=tooling,
            backbone_pdb=parent_input_pdb,
            out_dir=out_aux_dir / parent_id,
            seed=seed_base,
            dry_run=args.dry_run,
            log_file=logs_dir / f"{parent_id}_phase4_mpnn.log",
            loops="H2",
            seqs_per_struct=variants_n,
            temperature=0.1,
        )
        if not mpnn_records:
            raise PipelineError(f"Phase4 ProteinMPNN produced no H2 variants for {parent_id}")

        variant_rows = []
        for i in range(1, variants_n + 1):
            vid = f"{parent_id}__H2v{i:02d}"
            record = mpnn_records[min(i - 1, len(mpnn_records) - 1)]
            designed_pdb = Path(record.get("designed_pdb", str(parent_input_pdb)))
            full_seq = str(record.get("full_sequence", "")).strip().upper()
            warning_msg = ""

            try:
                h1_seq, h2_seq, h3_seq = split_designed_sequence(
                    parts=framework_parts,
                    full_seq=full_seq,
                    h1_len=h1_len,
                    h3_len=h3_len,
                )
            except Exception as exc:
                warning_msg = f"H2 parse fallback used: {exc}"
                rng = deterministic_rng(seed_base, vid)
                h2_seq = mutate_h2_only(h2_parent, rng=rng, n_mut=2)
                h1_seq = h1_seq_parent
                h3_seq = h3_seq_parent
                full_seq = compose_nanobody_sequence(framework_parts, h1_seq=h1_seq, h2_seq=h2_seq, h3_seq=h3_seq)

            # Enforce phase4 policy: H1/H3 fixed; only H2 may vary.
            if h1_seq != h1_seq_parent or h3_seq != h3_seq_parent:
                warning_msg = (
                    warning_msg + " | " if warning_msg else ""
                ) + "Phase4 enforces fixed H1/H3; replaced designed H1/H3 with parent sequences."
                h1_seq = h1_seq_parent
                h3_seq = h3_seq_parent
                full_seq = compose_nanobody_sequence(framework_parts, h1_seq=h1_seq, h2_seq=h2_seq, h3_seq=h3_seq)

            rf2_json = out_metrics_dir / f"{vid}_rf2.json"
            metrics = run_rf2_filter(
                cfg=tooling,
                input_pdb=designed_pdb,
                sequence=full_seq,
                out_json=rf2_json,
                dry_run=args.dry_run,
                log_file=logs_dir / f"{parent_id}_phase4_rf2.log",
                seed=seed_base,
                context={
                    "candidate_id": vid,
                    "campaign_name": campaign_name,
                    "cdr3_contact_bias": 1,
                },
            )

            structure_for_contacts = Path(str(metrics.get("rf2_best_pdb", designed_pdb)))
            heuristics = compute_interface_heuristics(
                pdb_path=structure_for_contacts,
                parts=framework_parts,
                h1_len=h1_len,
                h3_len=h3_len,
                hotspot_tokens=context.get("campaign_hotspot_tokens", {}).get(campaign_name, []),
                cutoff=5.0,
            )
            metrics.update(heuristics)
            if "structural_plausibility" not in metrics:
                metrics["structural_plausibility"] = max(0.0, min(1.0, float(metrics.get("rf2_pred_lddt", 0.0))))

            passed = hard_pass(metrics, filter_cfg)
            score = combine_weighted_score(metrics, rank_weights)

            variant_rows.append(
                {
                    "parent_candidate_id": parent_id,
                    "candidate_id": vid,
                    "phase": "phase4_h2_refine",
                    "combination_id": row.get("combination_id", ""),
                    "campaign_name": campaign_name,
                    "h1_length": h1_len,
                    "h3_length": h3_len,
                    "parent_backbone_id": parent_backbone,
                    "parent_backbone_pdb": parent_backbone_pdb,
                    "designed_pdb": str(designed_pdb),
                    "rf2_best_pdb": str(metrics.get("rf2_best_pdb", "")),
                    "h1_sequence": h1_seq,
                    "h2_sequence": h2_seq,
                    "h3_sequence": h3_seq,
                    "full_sequence": full_seq,
                    "rf2_pae": metrics.get("rf2_pae", ""),
                    "design_rf2_rmsd": metrics.get("design_rf2_rmsd", ""),
                    "hotspot_agreement": metrics.get("hotspot_agreement", ""),
                    "groove_localization": metrics.get("groove_localization", ""),
                    "h1_h3_role_consistency": metrics.get("h1_h3_role_consistency", ""),
                    "structural_plausibility": metrics.get("structural_plausibility", ""),
                    "target_contact_residue_count": metrics.get("target_contact_residue_count", ""),
                    "hotspot_overlap_count": metrics.get("hotspot_overlap_count", ""),
                    "hard_filter_pass": int(passed),
                    "ranking_score": round(float(score), 6),
                    "warning": warning_msg,
                }
            )

        passed_rows = [x for x in variant_rows if int(x["hard_filter_pass"]) == 1]
        if passed_rows:
            best = sorted(passed_rows, key=lambda x: float(x["ranking_score"]), reverse=True)[0]
        else:
            best = sorted(variant_rows, key=lambda x: float(x["ranking_score"]), reverse=True)[0]
            prior = str(best.get("warning", "")).strip()
            msg = "No H2 variants passed hard RF2 filters; selected best-scoring fallback variant."
            best["warning"] = f"{prior} | {msg}" if prior else msg
        final_rows.append(best)
        manifest_rows.append(
            {
                "parent_candidate_id": parent_id,
                "selected_candidate_id": best["candidate_id"],
                "status": "completed",
                "updated_at": now_str(),
            }
        )

    dedup_threshold = float(pipeline_cfg.get("postprocess", {}).get("sequence_dedup_identity_threshold", 0.95))
    deduped = greedy_sequence_dedup(
        rows=final_rows,
        sequence_key="full_sequence",
        score_key="ranking_score",
        identity_threshold=dedup_threshold,
    )
    ranked = sorted(deduped, key=lambda x: float(x["ranking_score"]), reverse=True)
    top25 = ranked[:25]

    final_csv = root / "results/summaries/final25_h2_optimized_candidates.csv"
    fields = [
        "phase",
        "candidate_id",
        "parent_candidate_id",
        "parent_backbone_id",
        "parent_backbone_pdb",
        "combination_id",
        "campaign_name",
        "h1_length",
        "h3_length",
        "designed_pdb",
        "rf2_best_pdb",
        "h1_sequence",
        "h2_sequence",
        "h3_sequence",
        "full_sequence",
        "rf2_pae",
        "design_rf2_rmsd",
        "hotspot_agreement",
        "groove_localization",
        "h1_h3_role_consistency",
        "structural_plausibility",
        "target_contact_residue_count",
        "hotspot_overlap_count",
        "hard_filter_pass",
        "ranking_score",
        "warning",
    ]
    atomic_write_csv(final_csv, top25, fields)
    atomic_write_csv(root / "results/summaries/final_25_h2_optimized_candidates_table.csv", top25, fields)
    atomic_write_csv(
        root / "results/summaries/phase4_h2_refine_summary.csv",
        top25,
        fields,
    )
    atomic_write_csv(
        phase_dir / "phase_manifest.csv",
        manifest_rows,
        ["parent_candidate_id", "selected_candidate_id", "status", "updated_at"],
    )

    # Required final metadata
    metadata = {
        "generated_at": now_str(),
        "phase": "phase4_h2_refine",
        "total_selected": len(top25),
        "source_pre_h2_table": str(pre_h2_path),
        "final_table": str(final_csv),
        "notes": [
            "Backbone fixed during H2 optimization.",
            "H1 and H3 fixed during H2 optimization.",
            "No local AlphaFold 3 execution performed by design.",
        ],
    }
    write_json(root / "results/summaries/final_metadata.json", metadata)

    # Final FASTA output
    fasta_path = root / "results/final_25/final25_nanobody_sequences.fasta"
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w", encoding="utf-8") as handle:
        for row in top25:
            handle.write(f">{row['candidate_id']}\n{row['full_sequence']}\n")

    # RF2 passed set copy
    rf2_pass_csv = root / "results/rf2_passed/final25_rf2_passed_or_best.csv"
    rf2_pass_csv.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(rf2_pass_csv, top25, fields)

    # AF3 export package
    export_af3_handoff(context=context, final_rows=top25)


def export_af3_handoff(context: dict, final_rows: List[dict]):
    root = context["root"]
    resolved_inputs = context["resolved_inputs"]
    resolved_targets = context["resolved_targets"]

    outdir = root / "results/af3_web_exports"
    outdir.mkdir(parents=True, exist_ok=True)

    vp1_seq = read_sequence_file(Path(resolved_inputs["vp1_sequence_file"]))[0][1]
    pdom_seq = read_sequence_file(Path(resolved_inputs["p_domain_dimer_sequence_file"]))[0][1]

    submission_csv = outdir / "af3_web_submission_table.csv"
    fields = [
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
    ]
    rows = []
    for r in final_rows:
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
                "parent_backbone_pdb": r.get("parent_backbone_pdb", ""),
                "rf2_best_pdb": r.get("rf2_best_pdb", ""),
                "notes": r.get("warning", ""),
            }
        )
    atomic_write_csv(submission_csv, rows, fields)

    fasta_out = outdir / "af3_final25_nanobody.fasta"
    with fasta_out.open("w", encoding="utf-8") as handle:
        for r in final_rows:
            handle.write(f">{r['candidate_id']}\n{r['full_sequence']}\n")

    context_txt = outdir / "af3_antigen_context.txt"
    context_txt.write_text(
        "\n".join(
            [
                "AF3 manual submission context",
                "============================",
                f"Full cleaned antigen target: {resolved_targets.get('full_cleaned_target', '')}",
                f"Cropped design target: {resolved_targets.get('cropped_design_target', '')}",
                f"Residue mapping table: {resolved_targets.get('mapping_table', '')}",
                f"VP1 sequence length: {len(vp1_seq)}",
                f"P-domain sequence length: {len(pdom_seq)}",
                "",
                "No local AF3 run was performed by this pipeline.",
                "Submit final candidates manually via AF3 web.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary_md = outdir / "af3_handoff_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                SAFETY_ETHICS_STATEMENT,
                "",
                "# AF3 Web Handoff",
                "",
                "This package contains the final 25 H2-optimized nanobody candidates for manual AF3 web submission.",
                "",
                f"- Submission table: `{submission_csv}`",
                f"- FASTA: `{fasta_out}`",
                f"- Antigen context: `{context_txt}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_project_summary(context: dict):
    root = context["root"]
    summary_md = root / "results/summaries/project_summary.md"

    phase_files = [
        ("phase0_smoke", root / "results/summaries/phase0_smoke_summary.csv"),
        ("phase1_coarse_pilot", root / "results/summaries/phase1_coarse_pilot_summary.csv"),
        ("phase2_focused_pilot", root / "results/summaries/phase2_focused_pilot_summary.csv"),
        ("phase3_main_campaign", root / "results/summaries/phase3_main_campaign_summary.csv"),
        ("phase4_h2_refine", root / "results/summaries/phase4_h2_refine_summary.csv"),
        ("phase5_cdr1_rescue_pilot", root / "results/summaries/phase5_cdr1_rescue_pilot_summary.csv"),
        ("phase6_cdr1_rescue_main", root / "results/summaries/phase6_cdr1_rescue_main_summary.csv"),
        (
            "phase_next_test1_local_maturation",
            root / "results/summaries/phase_next_test1_local_maturation_rf2_summary.csv",
        ),
        (
            "phase_next_champion_narrow50",
            root / "results/summaries/phase_next_champion_narrow50_rf2_summary.csv",
        ),
    ]

    lines = [SAFETY_ETHICS_STATEMENT, "", "# Pipeline Summary", ""]
    for phase_name, path in phase_files:
        if path.exists():
            try:
                df = pd.read_csv(path)
                lines.append(f"- {phase_name}: {df.shape[0]} rows in `{path.name}`")
            except Exception:
                lines.append(f"- {phase_name}: generated `{path.name}`")
        else:
            lines.append(f"- {phase_name}: not run yet")

    lines.extend(
        [
            "",
            "Notes:",
            "- RFantibody core workflow was used (RFdiffusion -> ProteinMPNN -> RF2).",
            "- No local AlphaFold 3 execution was performed.",
        ]
    )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_single_phase(phase_name: str, context: dict, args: argparse.Namespace):
    phase_cfg = context["phases_cfg"]["phases"].get(phase_name)
    if phase_cfg is None:
        raise PipelineError(f"Unknown phase in config: {phase_name}")

    cdr = context["cdr"]
    all_combos = generate_all_combinations(
        campaign_cfg=context["campaign_cfg"],
        design_cfg=context["design_cfg"],
        native_h1=cdr.h1_len,
        native_h3=cdr.h3_len,
    )

    if phase_name in {"phase0_smoke", "phase1_coarse_pilot", "phase2_focused_pilot", "phase3_main_campaign"}:
        combos = combos_for_phase(
            phase_name,
            phase_cfg,
            all_combos,
            context["root"],
            phase2_manual_ids=context.get("phase2_manual_selection_ids", []),
            phase2_selection_path=Path(context["phase2_selection_path"])
            if context.get("phase2_selection_path")
            else None,
            phase3_manual_ids=context.get("phase3_manual_selection_ids", []),
            phase3_selection_path=Path(context["phase3_selection_path"])
            if context.get("phase3_selection_path")
            else None,
        )
        if not combos:
            raise PipelineError(f"No combinations selected for {phase_name}.")
        run_phase_design(phase_name=phase_name, combinations=combos, context=context, args=args)
    elif phase_name == "phase4_h2_refine":
        run_phase4_h2_refine(context=context, args=args)
    elif phase_name == "phase5_cdr1_rescue_pilot":
        run_phase5_cdr1_rescue_pilot(context=context, args=args)
    elif phase_name == "phase6_cdr1_rescue_main":
        run_phase6_cdr1_rescue_main(context=context, args=args)
    elif phase_name == "phase_next_test1_local_maturation":
        run_phase_next_test1_local_maturation(context=context, args=args)
    elif phase_name == "phase_next_champion_narrow50":
        run_phase_next_champion_narrow50(context=context, args=args)
    else:
        raise PipelineError(f"Unsupported phase: {phase_name}")


def main() -> int:
    args = parse_args()
    context = load_base_context(args)

    # default flags from config unless overridden explicitly
    dry_default = bool(context["pipeline_cfg"].get("execution", {}).get("dry_run_default", True))
    resume_default = bool(context["pipeline_cfg"].get("execution", {}).get("resume_default", True))
    if args.execute:
        args.dry_run = False
    elif not args.dry_run and dry_default:
        # keep explicit dry-run default unless caller passes --dry-run false equivalent (not implemented)
        args.dry_run = True
    if not args.resume and resume_default:
        args.resume = True
    if args.no_resume:
        args.resume = False

    if not args.dry_run:
        tool_cfg = context["tool_cfg"]
        missing = []
        if not tool_cfg.execute_real_tools:
            missing.append("tooling.execute_real_tools=true (or tooling.detected.yaml with execute_real_tools=true)")
        if not tool_cfg.rfdiffusion_prefix:
            missing.append("tools.rfdiffusion_prefix")
        if not tool_cfg.proteinmpnn_prefix:
            missing.append("tools.proteinmpnn_prefix")
        if not tool_cfg.rf2_prefix:
            missing.append("tools.rf2_prefix")
        if not tool_cfg.rfdiffusion_weights:
            missing.append("checkpoints.rfdiffusion_weights")
        if not tool_cfg.proteinmpnn_weights:
            missing.append("checkpoints.proteinmpnn_weights")
        if not tool_cfg.rf2_weights:
            missing.append("checkpoints.rf2_weights")
        framework_pdb = str(context.get("framework_pdb", "")).strip()
        if not framework_pdb:
            missing.append("inputs.nanobody_framework_pdb_file")
        elif not Path(framework_pdb).exists():
            missing.append("inputs.nanobody_framework_pdb_file (path not found)")
        if missing:
            raise PipelineError(
                "Cannot run with --execute because real tooling is unresolved. Missing: "
                + ", ".join(missing)
                + ". Run scripts/autodetect_runtime_and_tooling.py and/or edit data/configs/tooling.yaml."
            )

    phases_order = [
        "phase0_smoke",
        "phase1_coarse_pilot",
        "phase2_focused_pilot",
        "phase3_main_campaign",
        "phase4_h2_refine",
        "phase5_cdr1_rescue_pilot",
        "phase6_cdr1_rescue_main",
        "phase_next_test1_local_maturation",
        "phase_next_champion_narrow50",
    ]

    if args.phase == "all":
        for p in phases_order:
            log(f"Running {p} ...")
            run_single_phase(p, context=context, args=args)
    else:
        run_single_phase(args.phase, context=context, args=args)

    write_project_summary(context)
    log("Pipeline run complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
