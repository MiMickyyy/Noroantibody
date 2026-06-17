#!/usr/bin/env python3
"""Memory-safe Stage1 5O04 hotspot-transfer campaign runner.

Runs one 5O04-informed Stage1 condition at a time:
RFdiffusion -> ProteinMPNN -> RF2 -> AF3Score for RF2-relaxed candidates only.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_common import (  # noqa: E402
    PipelineError,
    SAFETY_ETHICS_STATEMENT,
    atomic_write_csv,
    deterministic_rng,
    ensure_dirs,
    load_cdr_boundaries,
    log,
    read_sequence_file,
    read_yaml,
    slugify,
    write_json,
)
from run_pipeline import (  # noqa: E402
    AF3SCORE_FIELDS,
    blank_af3score_fields,
    build_target_contig,
    compute_backbone_signature,
    ensure_combined_score_column,
    hard_pass,
    maybe_run_af3score_validation,
    relaxed_surrogate_pass,
    split_designed_sequence,
    split_framework_and_cdr,
    target_chain_segments,
)
from tool_wrappers import (  # noqa: E402
    combine_weighted_score,
    load_tool_config,
    run_proteinmpnn_sequence_design,
    run_rfdiffusion_backbone,
    run_rf2_filter,
)


CORE_CROP_TO_FULL = {49: 273, 53: 277, 238: 462, 241: 465, 243: 467}
MONITOR_CROP_TO_FULL = {48: 272, 49: 273, 53: 277, 238: 462, 239: 463, 240: 464, 241: 465, 243: 467}
CDR1_RANGE = (23, 34)
CDR2_RANGE = (50, 58)
CDR3_RANGE = (97, 106)
NEXT_HOTSPOT_GROUP1 = ("A271", "A466")
NEXT_HOTSPOT_GROUP2 = ("A464", "B479")
NEXT_HOTSPOT_GROUP3 = ("A224", "A272", "B482", "A225")
NEXT_MONITORING_HOTSPOTS = tuple(dict.fromkeys(NEXT_HOTSPOT_GROUP1 + NEXT_HOTSPOT_GROUP2 + NEXT_HOTSPOT_GROUP3))


@dataclass(frozen=True)
class Stage1Condition:
    condition_index: int
    condition_name: str
    design_group: str
    cdr1_length: int
    cdr3_length: int
    hotspot_tokens: Tuple[str, ...]
    open_cdr1: bool
    open_cdr2: bool
    open_cdr3: bool
    flexible_backbone_regions: str
    length_variable_regions: str
    sequence_design_regions: str
    fixed_regions: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default="outputs/stage1_5O04_hotspot_transfer")
    p.add_argument("--pipeline-config", default="data/configs/stage1_5o04/pipeline.stage1_5o04.yaml")
    p.add_argument("--tooling-config", default="data/configs/stage1_5o04/tooling.hpcc.yaml")
    p.add_argument("--resolved-inputs", default="data/configs/stage1_5o04/resolved_inputs.hpcc.yaml")
    p.add_argument("--resolved-targets", default="data/configs/stage1_5o04/resolved_targets.full_target.yaml")
    p.add_argument("--cdr-config", default="data/configs/cdr_boundaries.yaml")
    p.add_argument("--condition-index", type=int, default=None)
    p.add_argument("--max-conditions", type=int, default=None)
    p.add_argument("--backbones-per-condition", type=int, default=20)
    p.add_argument("--seqs-per-backbone", type=int, default=1)
    p.add_argument("--limit-backbones", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    return p.parse_args()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


def memory_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def resolve_path(root: Path, value: str) -> Path:
    p = Path(str(value).strip()).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def read_resolved_inputs(root: Path, path: Path) -> dict:
    data = read_yaml(path)
    return data.get("resolved_inputs", data)


def build_conditions() -> List[Stage1Condition]:
    rows: List[Stage1Condition] = []
    idx = 0
    for n1 in (1, 2):
        for g1 in combinations(NEXT_HOTSPOT_GROUP1, n1):
            for n2 in (1, 2):
                for g2 in combinations(NEXT_HOTSPOT_GROUP2, n2):
                    for n3 in range(0, len(NEXT_HOTSPOT_GROUP3) + 1):
                        for g3 in combinations(NEXT_HOTSPOT_GROUP3, n3):
                            tokens = tuple(g1 + g2 + g3)
                            if len(tokens) > 6:
                                continue
                            open_cdr1 = bool({"A271", "A272", "A464", "B464"} & set(tokens))
                            open_cdr3 = bool({"A466", "A224", "A464", "B464"} & set(tokens))
                            open_cdr2 = "B479" in tokens and "B482" in tokens
                            open_regions = []
                            if open_cdr1:
                                open_regions.append("CDR1:23-34")
                            if open_cdr2:
                                open_regions.append("CDR2:50-58")
                            if open_cdr3:
                                open_regions.append("CDR3:97-106")
                            fixed = []
                            if not open_cdr1:
                                fixed.append("CDR1:23-34")
                            if not open_cdr2:
                                fixed.append("CDR2:50-58")
                            if not open_cdr3:
                                fixed.append("CDR3:97-106")
                            fixed.append("framework_sequence_outside_open_CDRs")
                            fixed.append("antigen")
                            hotspot_label = "_".join(tokens)
                            cdr_label = "".join(
                                [
                                    "H1" if open_cdr1 else "",
                                    "H2" if open_cdr2 else "",
                                    "H3" if open_cdr3 else "",
                                ]
                            ) or "none"
                            condition_name = f"baker_hs{idx:03d}_{cdr_label}_{hotspot_label}"
                            rows.append(
                                Stage1Condition(
                                    condition_index=idx,
                                    condition_name=condition_name,
                                    design_group="baker_avg5_hotspot_combo",
                                    cdr1_length=12,
                                    cdr3_length=10,
                                    hotspot_tokens=tokens,
                                    open_cdr1=open_cdr1,
                                    open_cdr2=open_cdr2,
                                    open_cdr3=open_cdr3,
                                    flexible_backbone_regions=";".join(open_regions),
                                    length_variable_regions=";".join(
                                        r for r in ["CDR1:10-14" if open_cdr1 else "", "CDR3:10-14" if open_cdr3 else ""] if r
                                    ),
                                    sequence_design_regions=";".join(open_regions),
                                    fixed_regions=";".join(fixed),
                                )
                            )
                            idx += 1
    if len(rows) != 135:
        raise PipelineError(f"Expected 135 hotspot combinations, got {len(rows)}")
    return rows


def canonical_hotspot_token(token: str) -> str:
    token = str(token).strip().upper()
    # User shorthand/typo guard: all A479 references in this experiment mean B479.
    if token == "A479":
        return "B479"
    return token


def hotspot_key(token: str) -> Tuple[str, int]:
    token = canonical_hotspot_token(token)
    if len(token) < 2 or not token[0].isalpha() or not token[1:].isdigit():
        raise PipelineError(f"Invalid hotspot token: {token}")
    return token[0], int(token[1:])


def hotspot_tokens_to_keys(tokens: Sequence[str]) -> Tuple[Tuple[str, int], ...]:
    out = []
    seen = set()
    for token in tokens:
        key = hotspot_key(token)
        if key not in seen:
            out.append(key)
            seen.add(key)
    return tuple(out)


def hotspot_keys_to_tokens(keys: Iterable[Tuple[str, int]]) -> List[str]:
    return [f"{chain}{resnum}" for chain, resnum in sorted(set(keys), key=lambda x: (x[0], x[1]))]


def weighted_length_schedule(total: int, weights: Dict[int, float], seed_base: int, key: str) -> List[int]:
    if total <= 0:
        return []
    normalized = {int(k): float(v) for k, v in weights.items() if float(v) > 0}
    if not normalized:
        raise PipelineError("Length schedule weights are empty.")
    weight_sum = sum(normalized.values())
    raw = {length: (weight / weight_sum) * total for length, weight in normalized.items()}
    counts = {length: int(math.floor(value)) for length, value in raw.items()}
    remainder = total - sum(counts.values())
    if remainder > 0:
        ranked = sorted(raw, key=lambda length: (raw[length] - counts[length], normalized[length], -length), reverse=True)
        for length in ranked[:remainder]:
            counts[length] += 1
    schedule: List[int] = []
    for length in sorted(counts):
        schedule.extend([length] * counts[length])
    rng = deterministic_rng(seed_base, key)
    rng.shuffle(schedule)
    return schedule


def candidate_length_plan(condition: Stage1Condition, cdr, total: int, seed_base: int) -> List[Tuple[int, int]]:
    if condition.open_cdr1:
        h1 = weighted_length_schedule(
            total,
            {10: 0.10, 11: 0.10, 12: 0.60, 13: 0.10, 14: 0.10},
            seed_base,
            f"{condition.condition_name}:h1",
        )
    else:
        h1 = [int(cdr.h1_len)] * total

    if condition.open_cdr3:
        h3 = weighted_length_schedule(
            total,
            {10: 0.20, 11: 0.20, 12: 0.20, 13: 0.20, 14: 0.20},
            seed_base,
            f"{condition.condition_name}:h3",
        )
    else:
        h3 = [int(cdr.h3_len)] * total
    return list(zip(h1, h3))


def design_loop_specs(condition: Stage1Condition, h1_len: int, h2_len: int, h3_len: int) -> Tuple[str, str]:
    rf_loops = []
    mpnn_loops = []
    if condition.open_cdr1:
        rf_loops.append(f"H1:{h1_len}")
        mpnn_loops.append("H1")
    if condition.open_cdr2:
        rf_loops.append(f"H2:{h2_len}")
        mpnn_loops.append("H2")
    if condition.open_cdr3:
        rf_loops.append(f"H3:{h3_len}")
        mpnn_loops.append("H3")
    if not rf_loops:
        raise PipelineError(f"No open CDR region for condition {condition.condition_name}")
    return ",".join(rf_loops), ",".join(mpnn_loops)


def write_run_configs(root: Path, out_root: Path, conditions: Sequence[Stage1Condition], args: argparse.Namespace):
    cfg_dir = out_root / "configs"
    ensure_dirs([cfg_dir])
    manifest_rows = []
    for c in conditions:
        row = dict(c.__dict__)
        row["hotspot_tokens"] = ";".join(c.hotspot_tokens)
        row["backbones_per_condition"] = args.backbones_per_condition
        row["sequences_per_backbone"] = args.seqs_per_backbone
        manifest_rows.append(row)
    atomic_write_csv(
        out_root / "run_manifest.csv",
        manifest_rows,
        [
            "condition_index",
            "condition_name",
            "design_group",
            "cdr1_length",
            "cdr3_length",
            "hotspot_tokens",
            "open_cdr1",
            "open_cdr2",
            "open_cdr3",
            "flexible_backbone_regions",
            "length_variable_regions",
            "sequence_design_regions",
            "fixed_regions",
            "backbones_per_condition",
            "sequences_per_backbone",
        ],
    )
    for src in [args.pipeline_config, args.tooling_config, args.resolved_inputs, args.resolved_targets, args.cdr_config]:
        p = resolve_path(root, src)
        if p.exists():
            shutil.copyfile(p, cfg_dir / p.name)


def existing_condition_rows(summary_csv: Path, expected_rows: int) -> Tuple[List[dict], set]:
    if not summary_csv.exists():
        return [], set()
    try:
        df = pd.read_csv(summary_csv)
    except Exception:
        return [], set()
    if df.empty:
        return [], set()
    rows = df.to_dict(orient="records")
    completed = {str(row.get("candidate_id", "")) for row in rows if str(row.get("candidate_id", "")).strip()}
    if len(rows) < expected_rows:
        log(f"[resume] partial condition rows found: {summary_csv} rows={len(rows)}")
    return rows, completed


def _append_af3_async_manifest(out_root: Path, record: dict) -> None:
    manifest = out_root / "af3score_async_jobs.csv"
    fields = [
        "submitted_at",
        "condition_name",
        "candidate_id",
        "af3score_job_id",
        "af3score_input_pdb",
        "af3score_output_dir",
        "af3score_metric_csv",
        "af3score_parent_stdout",
        "af3score_parent_stderr",
    ]
    write_header = not manifest.exists()
    with manifest.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in fields})


def submit_af3score_async(
    *,
    root: Path,
    out_root: Path,
    condition: Stage1Condition,
    pipeline_cfg: dict,
    candidate_id: str,
    af3_input_pdb: Path,
    ranking_score: float,
    scope_dir: Path,
    logs_dir: Path,
    dry_run: bool,
) -> dict:
    af3_cfg = pipeline_cfg.get("af3score", {}) or {}
    if dry_run:
        return blank_af3score_fields(1, ranking_score, "submitted_async_dry_run")

    script = root / str(af3_cfg.get("async_submit_script", "scripts/slurm/stage1_5o04_af3score_async.sbatch"))
    if not script.exists():
        raise PipelineError(f"AF3Score async submit script missing: {script}")
    if not af3_input_pdb.exists():
        raise PipelineError(f"AF3Score async input PDB missing: {af3_input_pdb}")

    safe_stem = slugify(candidate_id)
    out_dir = scope_dir / "af3score" / safe_stem
    input_dir = out_dir / "input_pdb"
    async_logs = logs_dir / "af3score_async"
    ensure_dirs([input_dir, async_logs])
    input_copy = input_dir / f"{safe_stem}.pdb"
    shutil.copyfile(af3_input_pdb, input_copy)

    parent_stdout = async_logs / f"{safe_stem}_parent_%j.out"
    parent_stderr = async_logs / f"{safe_stem}_parent_%j.err"
    export_values = {
        "AF3_ASYNC_INPUT_DIR": str(input_dir),
        "AF3_ASYNC_OUTPUT_DIR": str(out_dir),
        "AF3_ASYNC_NUM_JOBS": "1",
        "AF3_ASYNC_CANDIDATE_ID": candidate_id,
        "AF3SCORE_PARTITION": str(af3_cfg.get("async_partition", "short_gpu")),
        "AF3SCORE_QOS": str(af3_cfg.get("async_qos", "short_gpu")),
        "AF3SCORE_TIME": str(af3_cfg.get("async_time", "02:00:00")),
        "AF3SCORE_GRES": str(af3_cfg.get("async_gres", "gpu:a100:1")),
        "AF3SCORE_CUDA_MODULE": str(af3_cfg.get("async_cuda_module", "cuda/12.8")),
        "AF3SCORE_FLASH_ATTENTION": str(af3_cfg.get("async_flash_attention", "xla")),
    }
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in export_values.items())
    cmd = [
        "sbatch",
        "--parsable",
        "--partition=batch",
        "--cpus-per-task=4",
        "--mem=16G",
        "--time=2-00:00:00",
        "--output",
        str(parent_stdout),
        "--error",
        str(parent_stderr),
        "--export",
        export_arg,
        str(script),
    ]
    try:
        job_id = subprocess.check_output(cmd, cwd=root, text=True).strip().splitlines()[-1]
    except subprocess.CalledProcessError as exc:
        raise PipelineError(f"AF3Score async sbatch failed for {candidate_id}: {exc}") from exc

    fields = blank_af3score_fields(1, ranking_score, "submitted_async")
    fields.update(
        {
            "af3score_metric_csv": str(out_dir / "af3score_metrics.csv"),
            "af3score_input_pdb": str(input_copy),
            "af3score_output_dir": str(out_dir),
            "af3score_job_id": job_id,
        }
    )
    _append_af3_async_manifest(
        out_root,
        {
            "submitted_at": now_iso(),
            "condition_name": condition.condition_name,
            "candidate_id": candidate_id,
            "af3score_job_id": job_id,
            "af3score_input_pdb": str(input_copy),
            "af3score_output_dir": str(out_dir),
            "af3score_metric_csv": str(out_dir / "af3score_metrics.csv"),
            "af3score_parent_stdout": str(parent_stdout),
            "af3score_parent_stderr": str(parent_stderr),
        },
    )
    with (logs_dir / "condition.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{now_iso()} af3score_async_submitted candidate_id={candidate_id} job_id={job_id}\n")
    return fields


def residue_contact(res_a, res_b, cutoff: float) -> bool:
    for atom_a in res_a.get_atoms():
        if atom_a.element == "H":
            continue
        for atom_b in res_b.get_atoms():
            if atom_b.element == "H":
                continue
            if atom_a - atom_b <= cutoff:
                return True
    return False


def cdr_index_sets(parts: dict, h1_len: int, h3_len: int) -> Tuple[set, set, set]:
    f0 = len(parts["framework_prefix"])
    f1 = len(parts["framework_between_h1_h2"])
    h2 = len(parts["h2_native"])
    f2 = len(parts["framework_between_h2_h3"])
    h1_start = f0
    h1_end = h1_start + h1_len
    h2_start = h1_end + f1
    h2_end = h2_start + h2
    h3_start = h2_end + f2
    h3_end = h3_start + h3_len
    return set(range(h1_start, h1_end)), set(range(h2_start, h2_end)), set(range(h3_start, h3_end))


def compute_5o04_contacts(
    pdb_path: Path,
    parts: dict,
    h1_len: int,
    h3_len: int,
    selected_hotspot_tokens: Sequence[str],
    monitoring_hotspot_tokens: Sequence[str] = NEXT_MONITORING_HOTSPOTS,
    cutoff: float = 5.0,
) -> dict:
    selected_keys = set(hotspot_tokens_to_keys(selected_hotspot_tokens))
    monitoring_keys = set(hotspot_tokens_to_keys(monitoring_hotspot_tokens))
    base = {
        "contact_count_to_core_hotspots": 0,
        "contact_count_to_monitoring_epitope": 0,
        "selected_hotspot_total": len(selected_keys),
        "selected_hotspot_contacts": "",
        "selected_hotspot_exact_contact_count": 0,
        "selected_hotspot_residue_contact_count": 0,
        "monitoring_hotspot_total": len(monitoring_keys),
        "monitoring_hotspot_contacts": "",
        "monitoring_hotspot_exact_contact_count": 0,
        "monitoring_hotspot_residue_contact_count": 0,
        "target_chain_ids": "",
        "cdr1_contact_count": 0,
        "cdr2_contact_count": 0,
        "cdr3_contact_count": 0,
        "cdr1_contact_fraction": 0.0,
        "cdr2_contact_fraction": 0.0,
        "cdr3_contact_fraction": 0.0,
        "cdr1_dominant_flag": 0,
        "cdr3_support_flag": 0,
        "cdr2_low_contact_flag": 1,
        "wt_like_interface_recovery_score": 0.0,
    }
    for crop in CORE_CROP_TO_FULL:
        base[f"contacts_to_crop_{crop}"] = 0
    for crop in MONITOR_CROP_TO_FULL:
        base.setdefault(f"contacts_to_crop_{crop}", 0)
    if not pdb_path.exists():
        return base

    parser = PDBParser(QUIET=True)
    try:
        model = next(parser.get_structure("stage1_candidate", str(pdb_path)).get_models())
    except Exception:
        return base

    chains = {}
    for chain in model.get_chains():
        residues = [r for r in chain.get_residues() if r.id[0] == " " and is_aa(r, standard=False)]
        if residues:
            chains[str(chain.id)] = residues
    if not chains:
        return base

    binder_chain = "H" if "H" in chains else min(chains, key=lambda c: len(chains[c]))
    binder_res = chains[binder_chain]
    target_res = [(cid, r) for cid, residues in chains.items() if cid != binder_chain for r in residues]
    if not target_res:
        return base
    target_chain_ids = {cid for cid, _ in target_res}
    base["target_chain_ids"] = ";".join(sorted(target_chain_ids))

    h1_idx, h2_idx, h3_idx = cdr_index_sets(parts, h1_len, h3_len)
    cdr_contact_targets = {"cdr1": set(), "cdr2": set(), "cdr3": set()}
    target_contacted_by_cdr = set()

    for cid, tres in target_res:
        tkey = (cid, int(tres.id[1]))
        for bidx, bres in enumerate(binder_res):
            cdr_name = None
            if bidx in h1_idx:
                cdr_name = "cdr1"
            elif bidx in h2_idx:
                cdr_name = "cdr2"
            elif bidx in h3_idx:
                cdr_name = "cdr3"
            if cdr_name is None:
                continue
            if residue_contact(bres, tres, cutoff=cutoff):
                cdr_contact_targets[cdr_name].add(tkey)
                target_contacted_by_cdr.add(tkey)

    core_full = set(CORE_CROP_TO_FULL.values())
    monitor_full = set(MONITOR_CROP_TO_FULL.values())
    contacted_full_nums = {resnum for _, resnum in target_contacted_by_cdr}

    selected_exact = selected_keys & target_contacted_by_cdr
    monitoring_exact = monitoring_keys & target_contacted_by_cdr
    selected_by_residue = {key for key in selected_keys if key[1] in contacted_full_nums}
    monitoring_by_residue = {key for key in monitoring_keys if key[1] in contacted_full_nums}
    selected_chain_available = bool({chain for chain, _ in selected_keys} & target_chain_ids)
    monitoring_chain_available = bool({chain for chain, _ in monitoring_keys} & target_chain_ids)

    selected_for_score = selected_exact if selected_chain_available else selected_by_residue
    monitoring_for_score = monitoring_exact if monitoring_chain_available else monitoring_by_residue
    base["selected_hotspot_contacts"] = ";".join(hotspot_keys_to_tokens(selected_for_score))
    base["selected_hotspot_exact_contact_count"] = len(selected_exact)
    base["selected_hotspot_residue_contact_count"] = len(selected_by_residue)
    base["monitoring_hotspot_contacts"] = ";".join(hotspot_keys_to_tokens(monitoring_for_score))
    base["monitoring_hotspot_exact_contact_count"] = len(monitoring_exact)
    base["monitoring_hotspot_residue_contact_count"] = len(monitoring_by_residue)
    base["contact_count_to_core_hotspots"] = len(selected_for_score)
    base["contact_count_to_monitoring_epitope"] = len(monitoring_for_score)
    for crop, full in {**MONITOR_CROP_TO_FULL, **CORE_CROP_TO_FULL}.items():
        base[f"contacts_to_crop_{crop}"] = len({x for x in target_contacted_by_cdr if x[1] == full})

    c1 = len(cdr_contact_targets["cdr1"])
    c2 = len(cdr_contact_targets["cdr2"])
    c3 = len(cdr_contact_targets["cdr3"])
    total_cdr_contacts = max(1, c1 + c2 + c3)
    base["cdr1_contact_count"] = c1
    base["cdr2_contact_count"] = c2
    base["cdr3_contact_count"] = c3
    base["cdr1_contact_fraction"] = round(c1 / total_cdr_contacts, 4)
    base["cdr2_contact_fraction"] = round(c2 / total_cdr_contacts, 4)
    base["cdr3_contact_fraction"] = round(c3 / total_cdr_contacts, 4)
    base["cdr1_dominant_flag"] = int(c1 >= max(c2, c3) and c1 > 0)
    base["cdr3_support_flag"] = int(c3 > 0)
    base["cdr2_low_contact_flag"] = int(c2 <= max(1, int(0.25 * total_cdr_contacts)))

    core_frac = len(selected_for_score) / max(1, len(selected_keys))
    monitor_frac = len(monitoring_for_score) / max(1, len(monitoring_keys))
    cdr_mode = (
        0.45 * base["cdr1_contact_fraction"]
        + 0.30 * min(1.0, base["cdr3_contact_fraction"] / 0.35)
        + 0.25 * (1.0 - base["cdr2_contact_fraction"])
    )
    base["wt_like_interface_recovery_score"] = round(0.45 * core_frac + 0.25 * monitor_frac + 0.30 * cdr_mode, 6)
    return base


def compact_af3score_dir(path_value: str) -> int:
    if not path_value:
        return 0
    out_dir = Path(path_value)
    if not out_dir.exists() or not out_dir.is_dir():
        return 0
    removed = 0
    keep_names = {"af3score_metrics.csv"}
    for child in out_dir.iterdir():
        if child.name in keep_names or child.name.endswith("_af3score.json"):
            continue
        if child.name in {"af3_input_batch", "single_chain_cif", "json", "logs", "input_pdb", "single_seq.csv"}:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            removed += 1
    return removed


def cleanup_condition(condition_dir: Path, rows: Sequence[dict]) -> dict:
    removed = {"failed_rf2_dirs": 0, "failed_mpnn_dirs": 0, "af3_temp_items": 0}
    relaxed_pass_ids = {str(r.get("candidate_id")) for r in rows if int(float(r.get("rf2_relaxed_pass") or 0)) == 1}
    for r in rows:
        cid = str(r.get("candidate_id", ""))
        if cid not in relaxed_pass_ids:
            rf2_dir = condition_dir / "rf2_metrics" / f"{cid}_rf2_rf2_outputs"
            if rf2_dir.exists():
                shutil.rmtree(rf2_dir, ignore_errors=True)
                removed["failed_rf2_dirs"] += 1
    for r in rows:
        if str(r.get("af3score_status", "")) in {"completed", "dry_run"}:
            removed["af3_temp_items"] += compact_af3score_dir(str(r.get("af3score_output_dir", "")))
    return removed


def is_condition_complete(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    return int(df.shape[0]) >= int(expected_rows) and "combined_ranking_score" in df.columns


def run_condition(root: Path, args: argparse.Namespace, condition: Stage1Condition) -> List[dict]:
    out_root = resolve_path(root, args.output_root)
    condition_dir = out_root / "conditions" / condition.condition_name
    logs_dir = out_root / "logs" / condition.condition_name
    ensure_dirs([condition_dir, logs_dir])

    expected = int(args.backbones_per_condition) * int(args.seqs_per_backbone)
    summary_csv = condition_dir / "condition_summary_compact.csv"
    if not args.no_resume and is_condition_complete(summary_csv, expected):
        log(f"[resume] 完整 condition 已存在，跳过：{condition.condition_name}")
        return pd.read_csv(summary_csv).to_dict(orient="records")
    rows, completed_ids = existing_condition_rows(summary_csv, expected) if not args.no_resume else ([], set())

    pipeline_cfg = read_yaml(resolve_path(root, args.pipeline_config))
    tooling = load_tool_config(resolve_path(root, args.tooling_config))
    resolved_inputs = read_resolved_inputs(root, resolve_path(root, args.resolved_inputs))
    resolved_targets = read_yaml(resolve_path(root, args.resolved_targets))
    cdr = load_cdr_boundaries(resolve_path(root, args.cdr_config))
    if (cdr.h1, cdr.h2, cdr.h3) != (CDR1_RANGE, CDR2_RANGE, CDR3_RANGE):
        raise PipelineError(f"CDR definitions changed unexpectedly: H1={cdr.h1}, H2={cdr.h2}, H3={cdr.h3}")

    nanobody_seq = read_sequence_file(resolve_path(root, resolved_inputs["nanobody_sequence_file"]))[0][1]
    parts = split_framework_and_cdr(nanobody_seq, cdr)
    target_pdb = resolve_path(root, resolved_targets["cropped_design_target"])
    framework_pdb = resolve_path(root, resolved_inputs["nanobody_framework_pdb_file"])
    target_contig = build_target_contig(target_chain_segments(target_pdb), ["A", "B"])

    if args.execute and not tooling.execute_real_tools:
        raise PipelineError("tooling.execute_real_tools is false; refusing real Stage1 launch.")
    dry_run = bool(args.dry_run or not args.execute)
    seed_base = int(pipeline_cfg.get("project", {}).get("random_seed", 20260316))
    filter_cfg = pipeline_cfg.get("filters", {})
    rank_weights = filter_cfg.get("ranking_weights", {})

    mem_log = logs_dir / "memory.log"
    with mem_log.open("a", encoding="utf-8") as handle:
        handle.write(f"{now_iso()} condition_start memory_mb={memory_mb():.2f}\n")

    backbone_count = int(args.limit_backbones or args.backbones_per_condition)
    seqs_per_backbone = int(args.seqs_per_backbone)
    if seqs_per_backbone != 1:
        raise PipelineError("This Baker hotspot matrix expects --seqs-per-backbone 1 so each design has its own loop lengths.")
    length_plan = candidate_length_plan(condition, cdr, backbone_count, seed_base)
    hotspot_tokens = [canonical_hotspot_token(token) for token in condition.hotspot_tokens]
    backbones_dir = condition_dir / "backbones"
    mpnn_dir = condition_dir / "mpnn_aux"
    rf2_dir = condition_dir / "rf2_metrics"
    ensure_dirs([backbones_dir, mpnn_dir, rf2_dir])

    for i in range(1, backbone_count + 1):
        h1_len, h3_len = length_plan[i - 1]
        rf_design_loops, mpnn_loops = design_loop_specs(condition, h1_len, cdr.h2_len, h3_len)
        bb_id = f"{condition.condition_name}_H1L{h1_len:02d}_H3L{h3_len:02d}_bb{i:03d}"
        expected_cids = {f"{bb_id}_s{sidx:02d}" for sidx in range(1, seqs_per_backbone + 1)}
        if expected_cids and expected_cids.issubset(completed_ids):
            continue
        bb_pdb = backbones_dir / f"{bb_id}.pdb"
        run_rfdiffusion_backbone(
            cfg=tooling,
            combo={
                "condition_name": condition.condition_name,
                "campaign_name": "Baker_avg5_hotspot_combo_2700",
                "h1_length": h1_len,
                "h2_length": cdr.h2_len,
                "h3_length": h3_len,
            },
            backbone_id=bb_id,
            target_pdb=target_pdb,
            framework_pdb=framework_pdb,
            hotspots=hotspot_tokens,
            target_contig=target_contig,
            binder_length=len(parts["framework_prefix"])
            + len(parts["framework_between_h1_h2"])
            + cdr.h2_len
            + len(parts["framework_between_h2_h3"])
            + len(parts["framework_suffix"])
            + h1_len
            + h3_len,
            out_pdb=bb_pdb,
            seed=seed_base,
            log_file=logs_dir / "rfdiffusion.log",
            dry_run=dry_run,
            design_loops=rf_design_loops,
        )
        mpnn_records = run_proteinmpnn_sequence_design(
            cfg=tooling,
            backbone_pdb=bb_pdb,
            out_dir=mpnn_dir / bb_id,
            seed=seed_base,
            dry_run=dry_run,
            log_file=logs_dir / "proteinmpnn.log",
            loops=mpnn_loops,
            seqs_per_struct=seqs_per_backbone,
            temperature=0.1,
        )
        if not mpnn_records:
            raise PipelineError(f"ProteinMPNN produced no records for {bb_id}")

        for sidx in range(1, seqs_per_backbone + 1):
            cid = f"{bb_id}_s{sidx:02d}"
            if cid in completed_ids:
                continue
            record = mpnn_records[min(sidx - 1, len(mpnn_records) - 1)]
            designed_pdb = Path(record.get("designed_pdb", str(bb_pdb)))
            full_seq = str(record.get("full_sequence", "")).strip().upper()
            try:
                h1_seq, h2_seq, h3_seq = split_designed_sequence(parts, full_seq, h1_len, h3_len)
            except Exception:
                rng = deterministic_rng(seed_base, cid)
                h1_seq = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(h1_len))
                h2_seq = parts["h2_native"]
                h3_seq = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(h3_len))

            rf2_json = rf2_dir / f"{cid}_rf2.json"
            metrics = run_rf2_filter(
                cfg=tooling,
                input_pdb=designed_pdb,
                sequence=full_seq,
                out_json=rf2_json,
                dry_run=dry_run,
                log_file=logs_dir / "rf2.log",
                seed=seed_base,
                context={"candidate_id": cid, "campaign_name": "Baker_avg5_hotspot_combo_2700", "cdr3_contact_bias": int(condition.open_cdr3)},
            )
            structure_for_contacts = Path(str(metrics.get("rf2_best_pdb") or designed_pdb))
            contacts = compute_5o04_contacts(
                structure_for_contacts,
                parts,
                h1_len,
                h3_len,
                selected_hotspot_tokens=hotspot_tokens,
                monitoring_hotspot_tokens=NEXT_MONITORING_HOTSPOTS,
            )
            metrics.update(contacts)
            strict_pass = hard_pass(metrics, filter_cfg)
            relaxed_pass = relaxed_surrogate_pass(metrics, filter_cfg)
            selected_total = max(1, int(contacts.get("selected_hotspot_total") or 0))
            monitoring_total = max(1, int(contacts.get("monitoring_hotspot_total") or 0))
            rf2_rank = combine_weighted_score(
                {
                    "rf2_pae": metrics.get("rf2_pae", 99.0),
                    "design_rf2_rmsd": metrics.get("design_rf2_rmsd", 99.0),
                    "hotspot_agreement": min(1.0, contacts["contact_count_to_core_hotspots"] / selected_total),
                    "groove_localization": min(1.0, contacts["contact_count_to_monitoring_epitope"] / monitoring_total),
                    "h1_h3_role_consistency": 1.0 if contacts["cdr1_dominant_flag"] and contacts["cdr3_support_flag"] else 0.0,
                    "structural_plausibility": metrics.get("structural_plausibility", 0.0),
                },
                rank_weights,
            )
            af3_cfg = pipeline_cfg.get("af3score", {}) or {}
            if bool(af3_cfg.get("async_submit", False)) and bool(af3_cfg.get("enabled", False)):
                if bool(af3_cfg.get("score_relaxed_only", True)) and not relaxed_pass:
                    af3_fields = blank_af3score_fields(int(relaxed_pass), rf2_rank, "skipped_rf2_relaxed_gate")
                elif relaxed_pass:
                    af3_fields = submit_af3score_async(
                        root=root,
                        out_root=out_root,
                        condition=condition,
                        pipeline_cfg=pipeline_cfg,
                        candidate_id=cid,
                        af3_input_pdb=Path(str(metrics.get("rf2_best_pdb") or designed_pdb)),
                        ranking_score=rf2_rank,
                        scope_dir=condition_dir,
                        logs_dir=logs_dir,
                        dry_run=dry_run,
                    )
                else:
                    af3_fields = blank_af3score_fields(int(relaxed_pass), rf2_rank, "submitted_async_not_relaxed")
            else:
                af3_fields = maybe_run_af3score_validation(
                    context={"pipeline_cfg": pipeline_cfg, "tool_cfg": tooling},
                    args=argparse.Namespace(dry_run=dry_run),
                    phase_name="stage1_5O04_hotspot_transfer",
                    candidate_id=cid,
                    rf2_input_pdb=designed_pdb,
                    metrics=metrics,
                    ranking_score=rf2_rank,
                    rf2_relaxed_pass=relaxed_pass,
                    scope_dir=condition_dir,
                    logs_dir=logs_dir,
                    seed_base=seed_base,
                )
            row = {
                "condition_name": condition.condition_name,
                "design_group": condition.design_group,
                "hotspot_tokens": ";".join(hotspot_tokens),
                "open_cdr1": int(condition.open_cdr1),
                "open_cdr2": int(condition.open_cdr2),
                "open_cdr3": int(condition.open_cdr3),
                "cdr1_length": h1_len,
                "cdr3_length": h3_len,
                "backbone_id": bb_id,
                "sequence_id": f"s{sidx:02d}",
                "candidate_id": cid,
                "backbone_pdb": str(bb_pdb) if relaxed_pass else "",
                "designed_pdb": str(designed_pdb) if relaxed_pass else "",
                "rf2_best_pdb": str(metrics.get("rf2_best_pdb", "")) if relaxed_pass else "",
                "rf2_pae": metrics.get("rf2_pae", ""),
                "design_rf2_rmsd": metrics.get("design_rf2_rmsd", ""),
                "rf2_strict_pass": int(strict_pass),
                "rf2_relaxed_pass": int(relaxed_pass),
                "rf2_rank_score": round(float(rf2_rank), 6),
                "ranking_score": round(float(rf2_rank), 6),
                "h1_sequence": h1_seq,
                "h2_sequence": h2_seq,
                "h3_sequence": h3_seq,
                "full_sequence": full_seq,
                "backbone_signature": compute_backbone_signature(bb_pdb),
                "retained_file_policy": "required_inputs_and_metrics" if relaxed_pass else "metadata_only",
            }
            row.update(af3_fields)
            row.update(contacts)
            rows.append(row)
            completed_ids.add(cid)
            atomic_write_csv(summary_csv, rows, master_fields())

    cleanup = cleanup_condition(condition_dir, rows)
    with (logs_dir / "condition.log").open("a", encoding="utf-8") as handle:
        handle.write(
            f"{now_iso()} completed candidates={len(rows)} strict={sum(int(r['rf2_strict_pass']) for r in rows)} "
            f"relaxed={sum(int(r['rf2_relaxed_pass']) for r in rows)} "
            f"af3_attempted={sum(str(r.get('af3score_status')) in {'completed','dry_run','submitted_async'} for r in rows)} "
            f"af3_skipped={sum(str(r.get('af3score_status')) == 'skipped_rf2_relaxed_gate' for r in rows)} "
            f"cleanup={json.dumps(cleanup, sort_keys=True)}\n"
        )
    with mem_log.open("a", encoding="utf-8") as handle:
        handle.write(f"{now_iso()} condition_end memory_mb={memory_mb():.2f}\n")
    write_json(condition_dir / "condition_status.json", {"completed": True, "rows": len(rows), "cleanup": cleanup, "updated_at": now_iso()})
    return rows


def master_fields() -> List[str]:
    return [
        "condition_name",
        "design_group",
        "hotspot_tokens",
        "open_cdr1",
        "open_cdr2",
        "open_cdr3",
        "cdr1_length",
        "cdr3_length",
        "backbone_id",
        "sequence_id",
        "candidate_id",
        "rf2_pae",
        "design_rf2_rmsd",
        "rf2_strict_pass",
        "rf2_relaxed_pass",
        "rf2_rank_score",
        "af3score_status",
        "af3score_ptm",
        "af3score_iptm",
        "af3score_plddt",
        "af3score_pae",
        "af3score_ipsae",
        "af3score_rank_score",
        "combined_ranking_score",
        "contact_count_to_core_hotspots",
        "contact_count_to_monitoring_epitope",
        "selected_hotspot_total",
        "selected_hotspot_contacts",
        "selected_hotspot_exact_contact_count",
        "selected_hotspot_residue_contact_count",
        "monitoring_hotspot_total",
        "monitoring_hotspot_contacts",
        "monitoring_hotspot_exact_contact_count",
        "monitoring_hotspot_residue_contact_count",
        "target_chain_ids",
        "contacts_to_crop_49",
        "contacts_to_crop_53",
        "contacts_to_crop_238",
        "contacts_to_crop_241",
        "contacts_to_crop_243",
        "contacts_to_crop_48",
        "contacts_to_crop_239",
        "contacts_to_crop_240",
        "cdr1_contact_count",
        "cdr2_contact_count",
        "cdr3_contact_count",
        "cdr1_contact_fraction",
        "cdr2_contact_fraction",
        "cdr3_contact_fraction",
        "cdr1_dominant_flag",
        "cdr3_support_flag",
        "cdr2_low_contact_flag",
        "wt_like_interface_recovery_score",
        "backbone_pdb",
        "designed_pdb",
        "rf2_best_pdb",
        "af3score_metric_csv",
        "af3score_input_pdb",
        "af3score_output_dir",
        "af3score_job_id",
        "retained_file_policy",
        "h1_sequence",
        "h2_sequence",
        "h3_sequence",
        "full_sequence",
        "backbone_signature",
    ]


def read_all_condition_rows(out_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((out_root / "conditions").glob("*/condition_summary_compact.csv")):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=master_fields())
    df = pd.concat(frames, ignore_index=True)
    return ensure_combined_score_column(df)


def intended_cdr_mode(df: pd.DataFrame) -> pd.Series:
    return (
        (pd.to_numeric(df.get("cdr1_dominant_flag", 0), errors="coerce").fillna(0) == 1)
        & (pd.to_numeric(df.get("cdr3_support_flag", 0), errors="coerce").fillna(0) == 1)
        & (pd.to_numeric(df.get("cdr2_low_contact_flag", 0), errors="coerce").fillna(0) == 1)
    )


def merge_outputs(root: Path, args: argparse.Namespace, conditions: Sequence[Stage1Condition]):
    out_root = resolve_path(root, args.output_root)
    ensure_dirs([out_root])
    df = read_all_condition_rows(out_root)
    if df.empty:
        atomic_write_csv(out_root / "stage1_master_results.csv", [], master_fields())
        return
    for col in ["rf2_relaxed_pass", "rf2_strict_pass"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)
    numeric_cols = [
        "rf2_pae",
        "design_rf2_rmsd",
        "af3score_iptm",
        "af3score_plddt",
        "af3score_pae",
        "af3score_ipsae",
        "combined_ranking_score",
        "wt_like_interface_recovery_score",
        "cdr1_contact_fraction",
        "cdr2_contact_fraction",
        "cdr3_contact_fraction",
        "contact_count_to_core_hotspots",
        "contact_count_to_monitoring_epitope",
        "selected_hotspot_total",
        "selected_hotspot_exact_contact_count",
        "selected_hotspot_residue_contact_count",
        "monitoring_hotspot_total",
        "monitoring_hotspot_exact_contact_count",
        "monitoring_hotspot_residue_contact_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["intended_cdr_mode_flag"] = intended_cdr_mode(df).astype(int)
    selected_total = pd.to_numeric(df.get("selected_hotspot_total", 1), errors="coerce").fillna(1).clip(lower=1)
    core_norm = pd.to_numeric(df.get("contact_count_to_core_hotspots", 0), errors="coerce").fillna(0) / selected_total
    wt = pd.to_numeric(df.get("wt_like_interface_recovery_score", 0), errors="coerce").fillna(0)
    combined = pd.to_numeric(df.get("combined_ranking_score", 0), errors="coerce").fillna(0)
    mode = df["intended_cdr_mode_flag"].fillna(0)
    df["balanced_selection_score"] = (0.40 * combined + 0.35 * wt + 0.15 * core_norm + 0.10 * mode).round(6)

    df.to_csv(out_root / "stage1_master_results.csv", index=False)
    df.to_csv(out_root / "rf2_results.csv", index=False)
    af3_cols = [c for c in master_fields() if c.startswith("af3score_") or c in {"condition_name", "candidate_id", "rf2_relaxed_pass", "combined_ranking_score"}]
    df[af3_cols].to_csv(out_root / "af3score_metrics.csv", index=False)

    scored = df[(df["rf2_relaxed_pass"] == 1) & (df["af3score_status"].isin(["completed", "dry_run"]))].copy()
    scored.sort_values("combined_ranking_score", ascending=False).head(100).to_csv(out_root / "stage1_top100_combined.csv", index=False)
    df.sort_values("wt_like_interface_recovery_score", ascending=False).head(50).to_csv(out_root / "stage1_top50_wt_like_interface.csv", index=False)
    df.sort_values("balanced_selection_score", ascending=False).head(50).to_csv(out_root / "stage1_top50_balanced.csv", index=False)
    df.sort_values(["condition_name", "balanced_selection_score", "combined_ranking_score"], ascending=[True, False, False]).groupby("condition_name", as_index=False).head(1).to_csv(out_root / "stage1_per_condition_best.csv", index=False)

    summary = (
        df.groupby(
            [
                "condition_name",
                "design_group",
                "hotspot_tokens",
                "open_cdr1",
                "open_cdr2",
                "open_cdr3",
                "cdr1_length",
                "cdr3_length",
            ],
            as_index=False,
        )
        .agg(
            total_generated=("candidate_id", "count"),
            rf2_strict_count=("rf2_strict_pass", "sum"),
            rf2_relaxed_count=("rf2_relaxed_pass", "sum"),
            af3score_attempted_count=("af3score_status", lambda s: int(s.isin(["completed", "dry_run", "submitted_async"]).sum())),
            af3score_skipped_count=("af3score_status", lambda s: int((s == "skipped_rf2_relaxed_gate").sum())),
            mean_rf2_pae=("rf2_pae", "mean"),
            mean_rf2_rmsd=("design_rf2_rmsd", "mean"),
            mean_af3score_iptm=("af3score_iptm", "mean"),
            mean_af3score_plddt=("af3score_plddt", "mean"),
            mean_af3score_pae=("af3score_pae", "mean"),
            mean_af3score_ipsae=("af3score_ipsae", "mean"),
            mean_combined_ranking_score=("combined_ranking_score", "mean"),
            mean_wt_like_interface_recovery_score=("wt_like_interface_recovery_score", "mean"),
            mean_cdr1_contact_fraction=("cdr1_contact_fraction", "mean"),
            mean_cdr2_contact_fraction=("cdr2_contact_fraction", "mean"),
            mean_cdr3_contact_fraction=("cdr3_contact_fraction", "mean"),
            mean_selected_hotspot_contacts=("contact_count_to_core_hotspots", "mean"),
            mean_monitoring_hotspot_contacts=("contact_count_to_monitoring_epitope", "mean"),
            intended_cdr_mode_count=("intended_cdr_mode_flag", "sum"),
        )
        .sort_values(["mean_wt_like_interface_recovery_score", "mean_combined_ranking_score"], ascending=[False, False])
    )
    summary.to_csv(out_root / "stage1_condition_summary.csv", index=False)
    write_readme(root, out_root, args, conditions, df, summary)


def write_readme(root: Path, out_root: Path, args: argparse.Namespace, conditions: Sequence[Stage1Condition], df: pd.DataFrame, summary: pd.DataFrame):
    af3_prefix = read_yaml(resolve_path(root, args.tooling_config)).get("af3score", {}).get("command_prefix", "")
    job_ids = out_root / "slurm_job_ids.txt"
    lines = [
        SAFETY_ETHICS_STATEMENT,
        "",
        "# Stage1 Baker Avg5 Hotspot Combo Run Summary",
        "",
        f"- Date/time: {now_iso()}",
        f"- Git commit hash: `{git_commit(root)}`",
        "- Exact command template: `python scripts/run_stage1_5o04_campaign.py --execute --output-root outputs/stage1_baker_hotspot_2700_full_af3 --pipeline-config data/configs/stage1_5o04/pipeline.stage1_baker_hotspot_2700.yaml --backbones-per-condition 20 --seqs-per-backbone 1`",
        f"- AF3Score command_prefix: `{af3_prefix}`",
        "- HPCC environment notes: persistent one-GPU worker; do not pass `--condition-index`; the job should keep the same allocation and run all 135 hotspot combinations sequentially.",
        f"- Number of conditions: {len(conditions)}",
        f"- Planned designs: {len(conditions) * int(args.backbones_per_condition) * int(args.seqs_per_backbone)}",
        f"- Completed candidate rows currently merged: {int(df.shape[0])}",
        f"- RF2 strict: {int(pd.to_numeric(df.get('rf2_strict_pass', 0), errors='coerce').fillna(0).sum())}",
        f"- RF2 relaxed: {int(pd.to_numeric(df.get('rf2_relaxed_pass', 0), errors='coerce').fillna(0).sum())}",
        f"- AF3Score attempted: {int(df.get('af3score_status', pd.Series(dtype=str)).isin(['completed', 'dry_run']).sum())}",
        f"- AF3Score skipped: {int((df.get('af3score_status', pd.Series(dtype=str)) == 'skipped_rf2_relaxed_gate').sum())}",
        f"- Slurm job IDs: `{job_ids}`" if job_ids.exists() else "- Slurm job IDs: not recorded yet.",
        "",
        "## Main output files",
        f"- `{out_root / 'run_manifest.csv'}`",
        f"- `{out_root / 'stage1_master_results.csv'}`",
        f"- `{out_root / 'stage1_condition_summary.csv'}`",
        f"- `{out_root / 'stage1_top100_combined.csv'}`",
        f"- `{out_root / 'stage1_top50_wt_like_interface.csv'}`",
        f"- `{out_root / 'stage1_top50_balanced.csv'}`",
        f"- `{out_root / 'stage1_per_condition_best.csv'}`",
        "",
        "## Numbering sanity check",
        "- This run uses full VP1 residue numbering in the full cleaned P-domain dimer target.",
        "- Hotspot group 1: A271/A466, choose 1-2.",
        "- Hotspot group 2: A464/B479, choose 1-2. Any A479 shorthand is canonicalized to B479.",
        "- Hotspot group 3: A224/A272/B482/A225, choose 0-4.",
        "- Total hotspot count per condition is capped at 6, yielding 135 combinations.",
        "- CDR opening: A271 or A272 opens CDR1; A466 or A224 opens CDR3; A464 opens CDR1 and CDR3; B479+B482 additionally opens CDR2 without length change.",
        "- CDR1 open length distribution: 10/11/13/14 each 10%, length 12 at 60%; CDR3 open length distribution: 10-14 uniform.",
        "",
        "## Cleanup policy",
        "- cleanup_mode: after_each_condition",
        "- retain_failed_relaxed: metadata_only",
        "- retain_rf2_relaxed_pass: required_inputs_and_metrics",
        "- retain_top_structures: true",
        "- retain_all_raw_intermediates: false",
        "",
        "## Failed or incomplete jobs",
        "- Check per-condition `condition_status.json` and `logs/<condition>/condition.log`.",
    ]
    (out_root / "README_stage1_run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(".").resolve()
    out_root = resolve_path(root, args.output_root)
    ensure_dirs([out_root, out_root / "logs", out_root / "conditions"])
    conditions = build_conditions()
    if args.max_conditions is not None:
        conditions = conditions[: int(args.max_conditions)]
    write_run_configs(root, out_root, conditions, args)

    if args.prepare_only:
        merge_outputs(root, args, conditions)
        return 0
    if args.merge_only:
        merge_outputs(root, args, conditions)
        return 0

    selected = conditions
    if args.condition_index is not None:
        selected = [c for c in conditions if c.condition_index == int(args.condition_index)]
        if not selected:
            raise PipelineError(f"Unknown condition index: {args.condition_index}")

    for condition in selected:
        log(f"运行 condition {condition.condition_index}: {condition.condition_name}")
        run_condition(root, args, condition)
        merge_outputs(root, args, conditions)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
