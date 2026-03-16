#!/usr/bin/env python3
"""Master orchestrator for Norovirus CHDC2094 nanobody redesign pipeline."""

from __future__ import annotations

import argparse
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
    return parser.parse_args()


def require_file(path: Path, hint: str):
    if not path.exists():
        raise PipelineError(f"Missing required file: {path}. {hint}")


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

    cdr = load_cdr_boundaries(root / args.cdr_config)

    nanobody_path = Path(resolved_inputs.get("nanobody_sequence_file", ""))
    if not nanobody_path.exists():
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
        framework_pdb = (root / framework_cfg).resolve()
        if not framework_pdb.exists():
            raise PipelineError(
                f"Configured nanobody framework PDB does not exist: {framework_pdb}. "
                "Please set inputs.nanobody_framework_pdb_file to a valid structure."
            )

    cropped_target = Path(resolved_targets.get("cropped_design_target", ""))
    mapping_table = Path(resolved_targets.get("mapping_table", ""))
    if not cropped_target.exists():
        raise PipelineError(
            f"Resolved cropped target missing: {cropped_target}. Run scripts/prepare_targets.py first."
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
    model = next(parser.get_structure("candidate", str(pdb_path)).get_models())

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
        prev = root / "results/summaries/phase1_top8_combinations.csv"
        require_file(prev, "Run phase1_coarse_pilot first.")
        ids = set(pd.read_csv(prev)["combination_id"].astype(str).tolist())
        return [c for c in all_combos if c.combination_id in ids]

    if phase_name == "phase3_main_campaign":
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
                }
            )
            existing_backbone_ids.add(bb_id)

        write_rows(backbones_csv, backbones, ["combination_id", "campaign_name", "backbone_id", "backbone_pdb"])

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
                "best_ranking_score": 0.0,
                "mean_ranking_score": 0.0,
            }
        else:
            summary = {
                "phase": phase_name,
                "combination_id": combo.combination_id,
                "campaign_name": combo.campaign_name,
                "h1_length": combo.h1_length,
                "h3_length": combo.h3_length,
                "total_candidates": int(df.shape[0]),
                "hard_pass_candidates": int(df["hard_filter_pass"].sum()),
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

    pre_h2_path = root / "results/summaries/phase3_top25_pre_h2.csv"
    require_file(pre_h2_path, "Run phase3_main_campaign first.")

    rows = pd.read_csv(pre_h2_path).to_dict(orient="records")
    if len(rows) == 0:
        raise PipelineError("phase3_top25_pre_h2.csv is empty.")

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
        parent_id = str(row["candidate_id"])
        h1_seq_parent = str(row["h1_sequence"])
        h2_parent = str(row["h2_sequence"])
        h3_seq_parent = str(row["h3_sequence"])
        parent_backbone = str(row.get("backbone_id", ""))
        parent_backbone_pdb = str(row.get("backbone_pdb", ""))
        campaign_name = str(row.get("campaign_name", ""))
        h1_len = int(row.get("h1_length", len(h1_seq_parent)))
        h3_len = int(row.get("h3_length", len(h3_seq_parent)))

        parent_input_pdb = Path(str(row.get("rf2_best_pdb", "")).strip() or str(row.get("designed_pdb", "")).strip() or parent_backbone_pdb)
        if not parent_input_pdb.exists():
            parent_input_pdb = Path(parent_backbone_pdb)
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
                    "h1_length": row.get("h1_length", ""),
                    "h3_length": row.get("h3_length", ""),
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
        combos = combos_for_phase(phase_name, phase_cfg, all_combos, context["root"])
        if not combos:
            raise PipelineError(f"No combinations selected for {phase_name}.")
        run_phase_design(phase_name=phase_name, combinations=combos, context=context, args=args)
    elif phase_name == "phase4_h2_refine":
        run_phase4_h2_refine(context=context, args=args)
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
