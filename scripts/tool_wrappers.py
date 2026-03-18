#!/usr/bin/env python3
"""Wrappers for RFantibody (RFdiffusion / ProteinMPNN / RF2) execution.

Dry-run mode writes deterministic mock outputs for end-to-end validation.
Execute mode runs real commands discovered from tooling config.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1

from pipeline_common import (
    PipelineError,
    deterministic_rng,
    now_str,
    read_yaml,
    run_command,
    sanitize_pdb_for_rfantibody,
)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class ToolConfig:
    execute_real_tools: bool
    rfdiffusion_prefix: List[str]
    proteinmpnn_prefix: List[str]
    rf2_prefix: List[str]
    rfdiffusion_cwd: Optional[Path]
    proteinmpnn_cwd: Optional[Path]
    rf2_cwd: Optional[Path]
    rfdiffusion_weights: Optional[str]
    proteinmpnn_weights: Optional[str]
    rf2_weights: Optional[str]


def _parse_prefix(value) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        token = value.strip()
        if token.upper() == "AUTO_DETECT_FROM_REPO_AND_ENV" or not token:
            return []
        return [token]
    return []


def _parse_optional_path(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip()
        if not token or token.upper() == "AUTO_DETECT_FROM_REPO_AND_ENV":
            return None
        return token
    return None


def _parse_optional_cwd(value) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        p = Path(token).expanduser().resolve()
        return p
    return None


def load_tool_config(path: Path) -> ToolConfig:
    cfg = read_yaml(path)
    detected_cfg = {}
    detected_path = path.with_name("tooling.detected.yaml")
    if detected_path.exists():
        detected_cfg = read_yaml(detected_path)

    def merged_tool(tool_name: str) -> dict:
        manual = cfg.get(tool_name, {}) or {}
        detected = detected_cfg.get(tool_name, {}) or {}
        merged = dict(detected)
        merged.update({k: v for k, v in manual.items() if v not in (None, "", "AUTO_DETECT_FROM_REPO_AND_ENV")})
        return merged

    rfd_tool = merged_tool("rfdiffusion")
    mpnn_tool = merged_tool("proteinmpnn")
    rf2_tool = merged_tool("rf2")

    rfd = _parse_prefix(rfd_tool.get("command_prefix", cfg.get("rfdiffusion", {}).get("command_prefix", [])))
    mpnn = _parse_prefix(mpnn_tool.get("command_prefix", cfg.get("proteinmpnn", {}).get("command_prefix", [])))
    rf2 = _parse_prefix(rf2_tool.get("command_prefix", cfg.get("rf2", {}).get("command_prefix", [])))

    execute_flag = bool(cfg.get("execute_real_tools", False))
    if not execute_flag:
        execute_flag = bool(detected_cfg.get("execute_real_tools", False))

    ckpt_cfg = cfg.get("checkpoints", {}) or {}
    ckpt_detected = detected_cfg.get("checkpoints", {}) or {}

    rfdiff_ckpt = _parse_optional_path(ckpt_cfg.get("rfdiffusion_weights"))
    mpnn_ckpt = _parse_optional_path(ckpt_cfg.get("proteinmpnn_weights"))
    rf2_ckpt = _parse_optional_path(ckpt_cfg.get("rf2_weights"))
    if not rfdiff_ckpt:
        rfdiff_ckpt = _parse_optional_path(ckpt_detected.get("rfdiffusion_weights"))
    if not mpnn_ckpt:
        mpnn_ckpt = _parse_optional_path(ckpt_detected.get("proteinmpnn_weights"))
    if not rf2_ckpt:
        rf2_ckpt = _parse_optional_path(ckpt_detected.get("rf2_weights"))

    return ToolConfig(
        execute_real_tools=execute_flag,
        rfdiffusion_prefix=rfd,
        proteinmpnn_prefix=mpnn,
        rf2_prefix=rf2,
        rfdiffusion_cwd=_parse_optional_cwd(rfd_tool.get("run_cwd")),
        proteinmpnn_cwd=_parse_optional_cwd(mpnn_tool.get("run_cwd")),
        rf2_cwd=_parse_optional_cwd(rf2_tool.get("run_cwd")),
        rfdiffusion_weights=rfdiff_ckpt,
        proteinmpnn_weights=mpnn_ckpt,
        rf2_weights=rf2_ckpt,
    )


def random_loop(length: int, rng) -> str:
    if length <= 0:
        raise PipelineError(f"Loop length must be >0, got {length}")
    return "".join(rng.choice(AA_ALPHABET) for _ in range(length))


def mutate_h2_only(h2_seq: str, rng, n_mut: int = 2) -> str:
    arr = list(h2_seq)
    n_mut = max(1, min(n_mut, len(arr)))
    positions = sorted(rng.sample(range(len(arr)), n_mut))
    for idx in positions:
        arr[idx] = rng.choice(AA_ALPHABET)
    return "".join(arr)


def _prefix_last_token(prefix: Sequence[str]) -> str:
    if not prefix:
        return ""
    return Path(prefix[-1]).name.lower()


def _is_cli_prefix(prefix: Sequence[str], tool_name: str) -> bool:
    if not prefix:
        return False
    if len(prefix) >= 3 and prefix[0] == "uv" and prefix[1] == "run" and prefix[2] == tool_name:
        return True
    return _prefix_last_token(prefix) == tool_name


def _is_script_prefix(prefix: Sequence[str], script_basename: str) -> bool:
    if not prefix:
        return False
    return _prefix_last_token(prefix) == script_basename.lower()


def _chain_sequence_from_pdb(
    pdb_path: Path,
    preferred_chain_ids: Sequence[str] = ("H",),
    ignore_chain_ids: Sequence[str] = ("T",),
) -> Tuple[str, str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", str(pdb_path))
    model = next(structure.get_models())

    chain_seqs: Dict[str, str] = {}
    for chain in model.get_chains():
        residues = [r for r in chain.get_residues() if r.id[0] == " " and is_aa(r, standard=False)]
        if not residues:
            continue
        seq = "".join(protein_letters_3to1.get(r.get_resname().upper(), "X") for r in residues)
        chain_seqs[str(chain.id)] = seq

    if not chain_seqs:
        raise PipelineError(f"No protein chains parsed from PDB: {pdb_path}")

    for cid in preferred_chain_ids:
        if cid in chain_seqs:
            return cid, chain_seqs[cid]

    filtered = [(cid, seq) for cid, seq in chain_seqs.items() if cid not in set(ignore_chain_ids)]
    if filtered:
        cid, seq = sorted(filtered, key=lambda x: len(x[1]), reverse=True)[0]
        return cid, seq

    cid, seq = sorted(chain_seqs.items(), key=lambda x: len(x[1]), reverse=True)[0]
    return cid, seq


def _collect_mpnn_outputs(out_dir: Path, input_tag: str) -> List[Path]:
    if not out_dir.exists():
        return []
    regex = re.compile(rf"^{re.escape(input_tag)}_dldesign_(\d+)\.pdb$")

    tagged: List[Tuple[int, Path]] = []
    fallback: List[Path] = []
    for path in out_dir.glob("*.pdb"):
        m = regex.match(path.name)
        if m:
            tagged.append((int(m.group(1)), path))
        else:
            fallback.append(path)

    if tagged:
        tagged.sort(key=lambda x: x[0])
        return [p for _, p in tagged]

    fallback.sort()
    return fallback


def _parse_rf2_scores(best_pdb: Path) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for line in best_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("SCORE"):
            continue
        # Expected format: SCORE key: value
        m = re.match(r"^SCORE\s+([^:]+):\s+([-+0-9.eE]+)\s*$", line.strip())
        if not m:
            continue
        key = m.group(1).strip()
        val = float(m.group(2))
        scores[key] = val
    return scores


def _best_rf2_pdb(output_dir: Path, input_stem: str) -> Optional[Path]:
    direct = output_dir / f"{input_stem}_best.pdb"
    if direct.exists():
        return direct
    cands = sorted(output_dir.glob("*_best.pdb"))
    return cands[0] if cands else None


def _append_log_line(log_path: Path, message: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_str()}] {message}\n")


def _sanitize_input_pdb_for_rfdiffusion(src: Path, role: str, log_file: Path) -> Path:
    if src.suffix.lower() != ".pdb":
        _append_log_line(
            log_file,
            f"[WARN] {role} input is not .pdb ({src}); skipping sanitizer and passing original path.",
        )
        return src

    clean_path = src.with_name(f"{src.stem}.rfab_clean.pdb")
    if clean_path.exists() and clean_path.stat().st_mtime >= src.stat().st_mtime:
        return clean_path

    stats = sanitize_pdb_for_rfantibody(src, clean_path)
    _append_log_line(
        log_file,
        (
            f"Sanitized {role} input for RFantibody: src={src} clean={clean_path} "
            f"atoms_in={stats['atoms_in']} atoms_kept={stats['atoms_kept']} "
            f"dropped_altloc={stats['dropped_altloc']} "
            f"dropped_duplicates={stats['dropped_duplicate_atom_records']} "
            f"residues_missing_backbone={stats['residues_missing_backbone']}"
        ),
    )
    if stats["residues_missing_backbone"] > 0:
        raise PipelineError(
            f"{role} input has {stats['residues_missing_backbone']} residues missing N/CA/C after sanitization: "
            f"{clean_path}"
        )
    return clean_path


def _log_contains_any(log_file: Path, patterns: Sequence[str]) -> bool:
    if not log_file.exists():
        return False
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    return any(p.lower() in lowered for p in patterns)


def run_rfdiffusion_backbone(
    cfg: ToolConfig,
    combo: dict,
    backbone_id: str,
    target_pdb: Path,
    framework_pdb: Path,
    hotspots: Sequence[str],
    target_contig: str,
    binder_length: int,
    out_pdb: Path,
    seed: int,
    log_file: Path,
    dry_run: bool,
):
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    if dry_run or not cfg.execute_real_tools:
        rng = deterministic_rng(seed, backbone_id)
        content = (
            f"REMARK MOCK_BACKBONE {backbone_id}\n"
            f"REMARK CAMPAIGN {combo['campaign_name']}\n"
            f"REMARK HOTSPOTS {','.join(str(x) for x in hotspots)}\n"
            f"REMARK TARGET_CONTIG {target_contig}\n"
            f"REMARK BINDER_LENGTH {binder_length}\n"
            f"REMARK H1_LENGTH {combo['h1_length']}\n"
            f"REMARK H2_LENGTH {combo.get('h2_length', '')}\n"
            f"REMARK H3_LENGTH {combo['h3_length']}\n"
            f"REMARK RNG {rng.random():.6f}\n"
            "END\n"
        )
        out_pdb.write_text(content, encoding="utf-8")
        return

    if not cfg.rfdiffusion_prefix:
        raise PipelineError("execute_real_tools=true but rfdiffusion.command_prefix is empty")
    if not framework_pdb.exists():
        raise PipelineError(f"Missing framework PDB for RFdiffusion: {framework_pdb}")

    safe_target_pdb = _sanitize_input_pdb_for_rfdiffusion(target_pdb, role="target", log_file=log_file)
    safe_framework_pdb = _sanitize_input_pdb_for_rfdiffusion(framework_pdb, role="framework", log_file=log_file)

    h1_len = int(combo["h1_length"])
    h2_len = int(combo.get("h2_length", 0))
    h3_len = int(combo["h3_length"])
    if h2_len <= 0:
        raise PipelineError("H2 length must be available in combo for RFantibody RFdiffusion")

    out_prefix = str(out_pdb.with_suffix(""))

    def build_cmd(*, deterministic: bool, diffuser_t: int) -> List[str]:
        cmd = list(cfg.rfdiffusion_prefix)
        if _is_cli_prefix(cmd, "rfdiffusion"):
            cmd += [
                "--target",
                str(safe_target_pdb),
                "--framework",
                str(safe_framework_pdb),
                "--output",
                out_prefix,
                "--num-designs",
                "1",
                "--design-loops",
                f"H1:{h1_len},H2:{h2_len},H3:{h3_len}",
                "--diffuser-t",
                str(diffuser_t),
                "--final-step",
                "1",
                "--no-trajectory",
            ]
            if deterministic:
                cmd.append("--deterministic")
            if hotspots:
                cmd += ["--hotspots", ",".join(hotspots)]
            if cfg.rfdiffusion_weights:
                cmd += ["--weights", str(cfg.rfdiffusion_weights)]
            return cmd

        if _is_script_prefix(cmd, "rfdiffusion_inference.py"):
            cmd += [
                "--config-name",
                "antibody",
                f"antibody.target_pdb={str(safe_target_pdb)}",
                f"antibody.framework_pdb={str(safe_framework_pdb)}",
                f"inference.output_prefix={out_prefix}",
                "inference.num_designs=1",
                f"antibody.design_loops=[H1:{h1_len},H2:{h2_len},H3:{h3_len}]",
                f"diffuser.T={diffuser_t}",
                "inference.final_step=1",
                "inference.write_trajectory=False",
            ]
            if deterministic:
                cmd.append("inference.deterministic=True")
            if hotspots:
                cmd.append(f"ppi.hotspot_res=[{','.join(hotspots)}]")
            if cfg.rfdiffusion_weights:
                cmd.append(f"inference.ckpt_override_path={str(cfg.rfdiffusion_weights)}")
            return cmd

        raise PipelineError(
            "Unsupported rfdiffusion command prefix. Expected CLI 'rfdiffusion' or script 'rfdiffusion_inference.py'."
        )

    # Do not force deterministic sampling in production phases; diversity across
    # backbones is required for meaningful per-combination ranking.
    cmd = build_cmd(deterministic=False, diffuser_t=50)
    code = run_command(cmd, log_path=log_file, dry_run=False, cwd=cfg.rfdiffusion_cwd)
    if code != 0 and _log_contains_any(
        log_file,
        patterns=[
            "non-positive determinant",
            "left-handed or null coordinate frame",
            "rotation matrix",
        ],
    ):
        _append_log_line(
            log_file,
            (
                "Detected RFdiffusion rotation-frame instability. "
                "Retrying once with deterministic=False and diffuser_t=200 for stability."
            ),
        )
        retry_cmd = build_cmd(deterministic=False, diffuser_t=200)
        code = run_command(retry_cmd, log_path=log_file, dry_run=False, cwd=cfg.rfdiffusion_cwd)

    if code != 0:
        raise PipelineError(f"RFdiffusion command failed for {backbone_id}; see {log_file}")

    generated_prefix = out_pdb.with_suffix("")
    candidates = [
        generated_prefix.parent / f"{generated_prefix.name}_0.pdb",
        generated_prefix.parent / f"{generated_prefix.name}.pdb",
    ]
    if out_pdb.exists():
        return
    for cand in candidates:
        if cand.exists():
            shutil.copyfile(cand, out_pdb)
            return

    raise PipelineError(
        f"RFdiffusion finished but expected output not found for {backbone_id}. Checked: {', '.join(str(x) for x in candidates)}"
    )


def run_proteinmpnn_sequence_design(
    cfg: ToolConfig,
    backbone_pdb: Path,
    out_dir: Path,
    seed: int,
    dry_run: bool,
    log_file: Path,
    loops: str,
    seqs_per_struct: int,
    temperature: float = 0.1,
) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run or not cfg.execute_real_tools:
        rng = deterministic_rng(seed, backbone_pdb.stem)
        seq_len = 120
        try:
            _, seq = _chain_sequence_from_pdb(backbone_pdb)
            if seq:
                seq_len = len(seq)
        except Exception:
            pass

        out = []
        for i in range(seqs_per_struct):
            seq = "".join(rng.choice(AA_ALPHABET) for _ in range(seq_len))
            out.append(
                {
                    "design_index": i,
                    "designed_pdb": str(backbone_pdb),
                    "chain_id": "H",
                    "full_sequence": seq,
                }
            )
        return out

    if not cfg.proteinmpnn_prefix:
        raise PipelineError("execute_real_tools=true but proteinmpnn.command_prefix is empty")
    if not backbone_pdb.exists():
        raise PipelineError(f"ProteinMPNN input backbone missing: {backbone_pdb}")

    input_dir = out_dir / f"{backbone_pdb.stem}_input"
    output_dir = out_dir / f"{backbone_pdb.stem}_outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_copy = input_dir / f"{backbone_pdb.stem}.pdb"
    if not input_copy.exists() or input_copy.stat().st_mtime < backbone_pdb.stat().st_mtime:
        shutil.copyfile(backbone_pdb, input_copy)

    cmd = list(cfg.proteinmpnn_prefix)

    if _is_cli_prefix(cmd, "proteinmpnn"):
        cmd += [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--loops",
            loops,
            "--seqs-per-struct",
            str(seqs_per_struct),
            "--temperature",
            str(temperature),
        ]
        if cfg.proteinmpnn_weights:
            cmd += ["--weights", str(cfg.proteinmpnn_weights)]
    elif _is_script_prefix(cmd, "proteinmpnn_interface_design.py"):
        cmd += [
            "-pdbdir",
            str(input_dir),
            "-outpdbdir",
            str(output_dir),
            "-loop_string",
            loops,
            "-seqs_per_struct",
            str(seqs_per_struct),
            "-temperature",
            str(temperature),
        ]
        if cfg.proteinmpnn_weights:
            cmd += ["-checkpoint_path", str(cfg.proteinmpnn_weights)]
    else:
        raise PipelineError(
            "Unsupported proteinmpnn command prefix. Expected CLI 'proteinmpnn' or script 'proteinmpnn_interface_design.py'."
        )

    code = run_command(cmd, log_path=log_file, dry_run=False, cwd=cfg.proteinmpnn_cwd)
    if code != 0:
        raise PipelineError(f"ProteinMPNN command failed for {backbone_pdb}; see {log_file}")

    output_pdbs = _collect_mpnn_outputs(output_dir, input_tag=backbone_pdb.stem)
    if not output_pdbs:
        raise PipelineError(f"ProteinMPNN produced no designed PDBs in {output_dir} for {backbone_pdb.stem}")

    records: List[dict] = []
    for i, pdb_path in enumerate(output_pdbs):
        chain_id, seq = _chain_sequence_from_pdb(pdb_path)
        records.append(
            {
                "design_index": i,
                "designed_pdb": str(pdb_path),
                "chain_id": chain_id,
                "full_sequence": seq,
            }
        )
    return records


def run_rf2_filter(
    cfg: ToolConfig,
    input_pdb: Path,
    sequence: str,
    out_json: Path,
    dry_run: bool,
    log_file: Path,
    seed: int,
    context: dict,
):
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if dry_run or not cfg.execute_real_tools:
        key = context.get("candidate_id", out_json.stem)
        rng = deterministic_rng(seed, key)

        base = 10.0
        if "campaign_A_core" in context.get("campaign_name", ""):
            base -= 0.8
        if context.get("cdr3_contact_bias", 0) > 0:
            base -= 0.3

        pae = max(3.0, min(16.0, rng.uniform(base - 3.0, base + 3.0)))
        rmsd = max(0.4, min(4.0, rng.uniform(1.0, 2.8)))
        hotspot_agreement = min(1.0, max(0.0, rng.uniform(0.35, 0.98)))
        groove_localization = min(1.0, max(0.0, rng.uniform(0.30, 0.95)))
        h1h3_consistency = min(1.0, max(0.0, rng.uniform(0.25, 0.95)))
        structural_plausibility = min(1.0, max(0.0, rng.uniform(0.3, 0.95)))

        metrics = {
            "rf2_pae": round(float(pae), 4),
            "design_rf2_rmsd": round(float(rmsd), 4),
            "hotspot_agreement": round(float(hotspot_agreement), 4),
            "groove_localization": round(float(groove_localization), 4),
            "h1_h3_role_consistency": round(float(h1h3_consistency), 4),
            "structural_plausibility": round(float(structural_plausibility), 4),
        }
        out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    if not cfg.rf2_prefix:
        raise PipelineError("execute_real_tools=true but rf2.command_prefix is empty")
    if not input_pdb.exists():
        raise PipelineError(f"RF2 input PDB missing: {input_pdb}")

    rf2_out_dir = out_json.parent / f"{out_json.stem}_rf2_outputs"
    rf2_out_dir.mkdir(parents=True, exist_ok=True)

    # Keep reproducibility while avoiding identical RF2 seeds for every candidate.
    seed_key = str(context.get("candidate_id", input_pdb.stem))
    rf2_seed = deterministic_rng(seed, f"rf2::{seed_key}").randint(1, 2_000_000_000)

    cmd = list(cfg.rf2_prefix)

    if _is_cli_prefix(cmd, "rf2"):
        cmd += [
            "--input-pdb",
            str(input_pdb),
            "--output-dir",
            str(rf2_out_dir),
            "--num-recycles",
            "10",
            "--hotspot-show-prop",
            "0.1",
            "--seed",
            str(rf2_seed),
            "--cautious",
        ]
        if cfg.rf2_weights:
            cmd += ["--weights", str(cfg.rf2_weights)]
    elif _is_script_prefix(cmd, "rf2_predict.py"):
        cmd += [
            f"input.pdb={str(input_pdb)}",
            f"output.pdb_dir={str(rf2_out_dir)}",
            "inference.num_recycles=10",
            "inference.cautious=True",
            "inference.hotspot_show_proportion=0.1",
            f"+inference.seed={rf2_seed}",
        ]
        if cfg.rf2_weights:
            cmd += [f"model.model_weights={str(cfg.rf2_weights)}"]
    else:
        raise PipelineError("Unsupported rf2 command prefix. Expected CLI 'rf2' or script 'rf2_predict.py'.")

    code = run_command(cmd, log_path=log_file, dry_run=False, cwd=cfg.rf2_cwd)
    if code != 0:
        raise PipelineError(f"RF2 command failed for {out_json.stem}; see {log_file}")

    best_pdb = _best_rf2_pdb(rf2_out_dir, input_stem=input_pdb.stem)
    if best_pdb is None:
        raise PipelineError(f"RF2 finished but no *_best.pdb was found in {rf2_out_dir}")

    score_dict = _parse_rf2_scores(best_pdb)

    interaction_pae = float(score_dict.get("interaction_pae", score_dict.get("pae", 99.0)))
    rmsd = float(
        score_dict.get(
            "framework_aligned_cdr_rmsd",
            score_dict.get(
                "framework_aligned_antibody_rmsd",
                score_dict.get("target_aligned_cdr_rmsd", score_dict.get("target_aligned_antibody_rmsd", 99.0)),
            ),
        )
    )
    pred_lddt = float(score_dict.get("pred_lddt", 0.0))

    metrics = {
        "rf2_pae": round(interaction_pae, 4),
        "design_rf2_rmsd": round(rmsd, 4),
        "rf2_interaction_pae": round(float(score_dict.get("interaction_pae", interaction_pae)), 4),
        "rf2_mean_pae": round(float(score_dict.get("pae", interaction_pae)), 4),
        "rf2_pred_lddt": round(pred_lddt, 4),
        "structural_plausibility": round(max(0.0, min(1.0, pred_lddt)), 4),
        "rf2_best_pdb": str(best_pdb),
    }

    # Attach raw SCORE dictionary for auditability.
    for key, value in score_dict.items():
        metrics[f"rf2_score_{key}"] = round(float(value), 4)

    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def combine_weighted_score(metrics: dict, weights: dict) -> float:
    # Normalize pAE and RMSD so higher is better after inversion.
    pae_score = max(0.0, 1.0 - (float(metrics.get("rf2_pae", 99.0)) / 20.0))
    rmsd_score = max(0.0, 1.0 - (float(metrics.get("design_rf2_rmsd", 99.0)) / 4.0))
    rf2_self_consistency = 0.5 * (pae_score + rmsd_score)

    score = 0.0
    score += float(weights.get("rf2_self_consistency", 0.4)) * rf2_self_consistency
    score += float(weights.get("hotspot_agreement", 0.25)) * float(metrics.get("hotspot_agreement", 0.0))
    score += float(weights.get("docking_localization", 0.2)) * float(metrics.get("groove_localization", 0.0))
    score += float(weights.get("structural_plausibility", 0.1)) * float(metrics.get("structural_plausibility", 0.0))
    score += float(weights.get("h1_h3_role_consistency", 0.05)) * float(
        metrics.get("h1_h3_role_consistency", 0.0)
    )
    return float(score)
