#!/usr/bin/env python3
"""Re-score AF3 result models with RF2 and summarize per-job metrics.

Workflow:
1) Scan AF3 result folders (fold_*)
2) Convert selected AF3 model CIFs to sanitized PDBs
3) Run RF2 filtering wrapper on each converted model
4) Export model-level and job-level summary CSVs

Notes:
- This is a structural consistency re-score only (RF2 proxy metrics).
- It does not compute true binding free energy or experimental affinity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from Bio.PDB import MMCIFParser, PDBIO

from pipeline_common import (
    PipelineError,
    load_cdr_boundaries,
    log,
    now_str,
    read_yaml,
    sanitize_pdb_for_rfantibody,
)
from tool_wrappers import load_tool_config, run_rf2_filter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-score AF3 outputs using RF2")
    p.add_argument("--af3-results-dir", default="AF3 Results")
    p.add_argument("--tooling-config", default="data/configs/tooling.yaml")
    p.add_argument("--pipeline-config", default="data/configs/pipeline.yaml")
    p.add_argument("--cdr-config", default="data/configs/cdr_boundaries.yaml")
    p.add_argument("--job-map-csv", default="results/af3_web_exports/af3_26_jobs_map.csv")
    p.add_argument("--outdir", default="results/summaries")
    p.add_argument("--workdir", default="results/af3_rf2_rescore")
    p.add_argument("--logdir", default="logs/af3_rf2_rescore")
    p.add_argument(
        "--model-indices",
        default="0",
        help="Comma list like 0,1,2 or 'all' for all AF3 models found per job",
    )
    p.add_argument("--max-jobs", type=int, default=None, help="Debug limiter")
    p.add_argument("--seed", type=int, default=20260316)
    p.add_argument("--execute", action="store_true", help="Force real RF2 execution mode")
    p.add_argument("--dry-run", action="store_true", help="Force dry-run mode")
    p.add_argument("--no-resume", action="store_true", help="Do not reuse existing RF2 json results")
    p.add_argument("--strict-pae-max", type=float, default=10.0)
    p.add_argument("--strict-rmsd-max", type=float, default=2.0)
    p.add_argument("--soft-pae-max", type=float, default=12.0)
    p.add_argument("--soft-rmsd-max", type=float, default=2.5)
    return p.parse_args()


def parse_model_indices(value: str) -> Optional[List[int]]:
    token = str(value).strip().lower()
    if token == "all":
        return None
    if not token:
        raise PipelineError("--model-indices is empty")
    out: List[int] = []
    for x in token.split(","):
        x = x.strip()
        if not x:
            continue
        idx = int(x)
        if idx < 0:
            raise PipelineError(f"Model index must be >= 0, got {idx}")
        out.append(idx)
    if not out:
        raise PipelineError("No valid model indices parsed from --model-indices")
    return sorted(set(out))


def infer_candidate_id(job_name: str) -> str:
    # Example:
    # fold_14_campaign_b_a_plus_d_rim_bridge_h112_h310_bb046_s01_h2v02
    # fold_26_wildtype_nanobody_121_aa
    import re

    m = re.match(r"^fold_\d+_(.+)$", job_name)
    return (m.group(1) if m else job_name).strip()


def load_job_map(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        j = str(row.get("job_name", "")).strip()
        if not j:
            continue
        out[j] = {
            "candidate_id": str(row.get("candidate_id", "")).strip(),
            "source": str(row.get("source", "")).strip(),
        }
    return out


def list_job_dirs(root: Path, max_jobs: Optional[int]) -> List[Path]:
    jobs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    if max_jobs is not None:
        jobs = jobs[: max(0, int(max_jobs))]
    return jobs


def list_model_cifs(job_dir: Path, wanted_indices: Optional[Sequence[int]]) -> List[Tuple[int, Path]]:
    found: List[Tuple[int, Path]] = []
    prefix = f"{job_dir.name}_model_"
    for cif in sorted(job_dir.glob(f"{prefix}*.cif")):
        stem = cif.stem  # ..._model_0
        tail = stem.split("_model_")[-1]
        if not tail.isdigit():
            continue
        idx = int(tail)
        found.append((idx, cif))
    if wanted_indices is None:
        return found
    keep = set(int(x) for x in wanted_indices)
    return [(i, p) for (i, p) in found if i in keep]


def _parse_atom_residue_key(line: str) -> Tuple[str, str, str]:
    chain = line[21] if len(line) > 21 else " "
    resseq = line[22:26] if len(line) > 25 else "    "
    icode = line[26] if len(line) > 26 else " "
    return chain, resseq, icode


def _residue_counts_from_atom_lines(lines: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    seen = set()
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        chain, resseq, icode = _parse_atom_residue_key(line)
        key = (chain, resseq, icode)
        if key in seen:
            continue
        seen.add(key)
        counts[chain] = counts.get(chain, 0) + 1
    return counts


def convert_clean_pdb_to_hlt_remarked(
    clean_pdb: Path,
    out_hlt_pdb: Path,
    cdr_ranges: Dict[str, Tuple[int, int]],
) -> Dict[str, object]:
    """Convert a sanitized PDB into minimal HLT-remarked format for RF2.

    Strategy:
    - Identify nanobody chain as shortest chain by residue count -> map to H.
    - Merge all remaining chains as target -> map to T.
    - Reorder residues in output: all H residues first, then T residues.
    - Renumber residues globally 1..N to satisfy RF2 remark indexing logic.
    - Add H1/H2/H3 REMARK lines from configured CDR ranges.
    """
    src_lines = clean_pdb.read_text(encoding="utf-8", errors="ignore").splitlines()
    atom_lines = [ln for ln in src_lines if ln.startswith("ATOM")]
    if not atom_lines:
        raise PipelineError(f"No ATOM lines found in {clean_pdb}")

    chain_counts = _residue_counts_from_atom_lines(atom_lines)
    if not chain_counts:
        raise PipelineError(f"Could not infer chain residue counts from {clean_pdb}")

    # Shortest chain is treated as nanobody chain.
    nanobody_chain = sorted(chain_counts.items(), key=lambda kv: (kv[1], kv[0]))[0][0]

    residue_order: List[Tuple[str, str, str]] = []
    residue_atoms: Dict[Tuple[str, str, str], List[str]] = {}
    for ln in atom_lines:
        key = _parse_atom_residue_key(ln)
        if key not in residue_atoms:
            residue_atoms[key] = []
            residue_order.append(key)
        residue_atoms[key].append(ln.rstrip("\n"))

    h_keys = [k for k in residue_order if k[0] == nanobody_chain]
    t_keys = [k for k in residue_order if k[0] != nanobody_chain]
    if not h_keys:
        raise PipelineError(f"Failed to identify nanobody residues in {clean_pdb}")
    if not t_keys:
        raise PipelineError(f"Failed to identify target residues in {clean_pdb}")

    new_order = h_keys + t_keys
    key_to_new: Dict[Tuple[str, str, str], Tuple[str, int]] = {}
    for i, key in enumerate(new_order, start=1):
        new_chain = "H" if key[0] == nanobody_chain else "T"
        key_to_new[key] = (new_chain, i)

    out_lines: List[str] = []
    serial = 1
    for key in new_order:
        new_chain, new_resnum = key_to_new[key]
        for raw in residue_atoms[key]:
            line = raw if len(raw) >= 80 else raw.ljust(80)
            # Atom serial (cols 7-11), chain (col 22), resseq (cols 23-26), icode (col 27)
            line = f"{line[:6]}{serial:5d}{line[11:]}"
            line = line[:21] + new_chain + f"{new_resnum:4d}" + " " + line[27:]
            out_lines.append(line.rstrip())
            serial += 1

    out_lines.append("TER")

    h_len = len(h_keys)
    # Add CDR remarks only if they fall within heavy-chain length after remap.
    for loop_name in ("H1", "H2", "H3"):
        start, end = cdr_ranges[loop_name]
        for resi in range(start, end + 1):
            if 1 <= resi <= h_len:
                out_lines.append(f"REMARK PDBinfo-LABEL:{resi:5d} {loop_name}")
    out_lines.append("REMARK AF3_TO_HLT_REMARKED")

    out_hlt_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_hlt_pdb.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return {
        "nanobody_chain_original": nanobody_chain,
        "chain_counts": chain_counts,
        "h_len": h_len,
        "t_len": len(t_keys),
    }


def convert_cif_to_clean_pdb(cif_path: Path, out_pdb: Path) -> Dict[str, int]:
    if out_pdb.exists() and out_pdb.stat().st_mtime >= cif_path.stat().st_mtime:
        return {}

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    raw_pdb = out_pdb.with_suffix(".raw.pdb")
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("af3", str(cif_path))

    io_obj = PDBIO()
    io_obj.set_structure(structure)
    io_obj.save(str(raw_pdb))

    stats = sanitize_pdb_for_rfantibody(raw_pdb, out_pdb)
    try:
        raw_pdb.unlink(missing_ok=True)
    except Exception:
        pass
    return stats


def is_pass(pae: float, rmsd: float, pae_max: float, rmsd_max: float) -> int:
    return int((float(pae) < float(pae_max)) and (float(rmsd) < float(rmsd_max)))


def safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def build_job_summary(model_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: List[dict] = []
    for job_name, g in model_df.groupby("job_name", sort=True):
        ok = g[g["status"] == "ok"].copy()
        rec = {
            "job_name": job_name,
            "candidate_id": g["candidate_id"].iloc[0],
            "source": g["source"].iloc[0],
            "is_wildtype": int(g["is_wildtype"].iloc[0]),
            "models_attempted": int(len(g)),
            "models_ok": int(len(ok)),
            "strict_pass_count": 0,
            "soft_pass_count": 0,
            "best_model_index": "",
            "best_rf2_pae": math.nan,
            "best_design_rf2_rmsd": math.nan,
            "mean_rf2_pae": math.nan,
            "mean_design_rf2_rmsd": math.nan,
        }
        if not ok.empty:
            ok["strict_pass"] = ok.apply(
                lambda r: is_pass(
                    r["rf2_pae"],
                    r["design_rf2_rmsd"],
                    args.strict_pae_max,
                    args.strict_rmsd_max,
                ),
                axis=1,
            )
            ok["soft_pass"] = ok.apply(
                lambda r: is_pass(
                    r["rf2_pae"],
                    r["design_rf2_rmsd"],
                    args.soft_pae_max,
                    args.soft_rmsd_max,
                ),
                axis=1,
            )
            rec["strict_pass_count"] = int(ok["strict_pass"].sum())
            rec["soft_pass_count"] = int(ok["soft_pass"].sum())
            rec["mean_rf2_pae"] = float(ok["rf2_pae"].astype(float).mean())
            rec["mean_design_rf2_rmsd"] = float(ok["design_rf2_rmsd"].astype(float).mean())
            best = ok.sort_values(["rf2_pae", "design_rf2_rmsd"], ascending=[True, True]).iloc[0]
            rec["best_model_index"] = int(best["model_index"])
            rec["best_rf2_pae"] = float(best["rf2_pae"])
            rec["best_design_rf2_rmsd"] = float(best["design_rf2_rmsd"])
        rows.append(rec)
    return pd.DataFrame(rows)


def build_vs_wt(job_df: pd.DataFrame) -> pd.DataFrame:
    wt = job_df[job_df["is_wildtype"] == 1]
    if wt.empty:
        return pd.DataFrame()

    wt_best = wt.sort_values(["best_rf2_pae", "best_design_rf2_rmsd"], ascending=[True, True]).iloc[0]
    wt_pae = safe_float(wt_best.get("best_rf2_pae"))
    wt_rmsd = safe_float(wt_best.get("best_design_rf2_rmsd"))

    rows: List[dict] = []
    for _, r in job_df.iterrows():
        if int(r.get("is_wildtype", 0)) == 1:
            continue
        pae = safe_float(r.get("best_rf2_pae"))
        rmsd = safe_float(r.get("best_design_rf2_rmsd"))
        rows.append(
            {
                "job_name": r.get("job_name", ""),
                "candidate_id": r.get("candidate_id", ""),
                "source": r.get("source", ""),
                "strict_pass_count": r.get("strict_pass_count", 0),
                "soft_pass_count": r.get("soft_pass_count", 0),
                "best_rf2_pae": pae,
                "best_design_rf2_rmsd": rmsd,
                "delta_vs_wt_pae": pae - wt_pae if not math.isnan(pae) and not math.isnan(wt_pae) else math.nan,
                "delta_vs_wt_rmsd": (
                    rmsd - wt_rmsd if not math.isnan(rmsd) and not math.isnan(wt_rmsd) else math.nan
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["strict_pass_count", "soft_pass_count", "best_rf2_pae", "best_design_rf2_rmsd"],
        ascending=[False, False, True, True],
    )


def main() -> int:
    args = parse_args()
    root = Path(".").resolve()
    af3_dir = (root / args.af3_results_dir).resolve()
    outdir = (root / args.outdir).resolve()
    workdir = (root / args.workdir).resolve()
    logdir = (root / args.logdir).resolve()
    job_map_csv = (root / args.job_map_csv).resolve()
    pipeline_cfg = read_yaml((root / args.pipeline_config).resolve())
    cdr = load_cdr_boundaries((root / args.cdr_config).resolve())

    if not af3_dir.exists():
        raise PipelineError(f"AF3 results dir not found: {af3_dir}")
    outdir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    tool_cfg = load_tool_config((root / args.tooling_config).resolve())
    if args.execute:
        tool_cfg.execute_real_tools = True
    if args.dry_run:
        tool_cfg.execute_real_tools = False

    wanted_indices = parse_model_indices(args.model_indices)
    job_map = load_job_map(job_map_csv)
    jobs = list_job_dirs(af3_dir, max_jobs=args.max_jobs)
    if not jobs:
        raise PipelineError(f"No fold_* jobs found under {af3_dir}")

    rank_weights = (
        pipeline_cfg.get("filters", {})
        .get("ranking_weights", {})
    )

    model_rows: List[dict] = []
    for jidx, job_dir in enumerate(jobs, start=1):
        job_name = job_dir.name
        mapped = job_map.get(job_name, {})
        candidate_id = mapped.get("candidate_id") or infer_candidate_id(job_name)
        source = mapped.get("source", "")
        is_wildtype = int("wildtype" in job_name.lower() or candidate_id.lower().startswith("wildtype"))

        models = list_model_cifs(job_dir, wanted_indices=wanted_indices)
        if not models:
            model_rows.append(
                {
                    "job_name": job_name,
                    "candidate_id": candidate_id,
                    "source": source,
                    "is_wildtype": is_wildtype,
                    "model_index": -1,
                    "af3_model_cif": "",
                    "converted_pdb": "",
                    "rf2_json": "",
                    "rf2_best_pdb": "",
                    "rf2_pae": math.nan,
                    "design_rf2_rmsd": math.nan,
                    "rf2_pred_lddt": math.nan,
                    "ranking_score": math.nan,
                    "status": "error",
                    "error": "No model CIFs matched --model-indices",
                    "updated_at": now_str(),
                }
            )
            continue

        log(f"[{jidx}/{len(jobs)}] Re-scoring {job_name} ({len(models)} model(s))")
        for midx, cif_path in models:
            per_model_dir = workdir / job_name / f"model_{midx}"
            per_model_dir.mkdir(parents=True, exist_ok=True)
            pdb_path = per_model_dir / f"{job_name}_model_{midx}.rfab_clean.pdb"
            hlt_pdb_path = per_model_dir / f"{job_name}_model_{midx}.rfab_hlt_remarked.pdb"
            rf2_json = per_model_dir / f"{job_name}_model_{midx}_rf2.json"
            rf2_log = logdir / f"{job_name}_rf2.log"

            row = {
                "job_name": job_name,
                "candidate_id": candidate_id,
                "source": source,
                "is_wildtype": is_wildtype,
                "model_index": midx,
                "af3_model_cif": str(cif_path),
                "converted_pdb": str(pdb_path),
                "rf2_input_pdb": str(hlt_pdb_path),
                "rf2_json": str(rf2_json),
                "rf2_best_pdb": "",
                "rf2_pae": math.nan,
                "design_rf2_rmsd": math.nan,
                "rf2_pred_lddt": math.nan,
                "ranking_score": math.nan,
                "status": "error",
                "error": "",
                "updated_at": now_str(),
            }
            try:
                convert_cif_to_clean_pdb(cif_path, pdb_path)
                convert_clean_pdb_to_hlt_remarked(
                    clean_pdb=pdb_path,
                    out_hlt_pdb=hlt_pdb_path,
                    cdr_ranges={"H1": cdr.h1, "H2": cdr.h2, "H3": cdr.h3},
                )
                if (not args.no_resume) and rf2_json.exists():
                    # Reuse prior RF2 result JSON
                    metrics = json.loads(rf2_json.read_text(encoding="utf-8"))
                else:
                    metrics = run_rf2_filter(
                        cfg=tool_cfg,
                        input_pdb=hlt_pdb_path,
                        sequence="",
                        out_json=rf2_json,
                        dry_run=(not tool_cfg.execute_real_tools),
                        log_file=rf2_log,
                        seed=args.seed,
                        context={
                            "candidate_id": f"{job_name}_m{midx}",
                            "campaign_name": candidate_id,
                            "cdr3_contact_bias": 0,
                        },
                    )

                row["rf2_best_pdb"] = str(metrics.get("rf2_best_pdb", ""))
                row["rf2_pae"] = safe_float(metrics.get("rf2_pae"))
                row["design_rf2_rmsd"] = safe_float(metrics.get("design_rf2_rmsd"))
                row["rf2_pred_lddt"] = safe_float(metrics.get("rf2_pred_lddt"))
                # Same weighted score entrypoint as pipeline, with available fields.
                row["ranking_score"] = (
                    0.4 * max(0.0, 1.0 - row["rf2_pae"] / 20.0)
                    + 0.4 * max(0.0, 1.0 - row["design_rf2_rmsd"] / 4.0)
                    + 0.2 * max(0.0, min(1.0, row["rf2_pred_lddt"] if not math.isnan(row["rf2_pred_lddt"]) else 0.0))
                )
                # If pipeline ranking weights are available, keep this info for audit.
                if isinstance(rank_weights, dict) and rank_weights:
                    row["ranking_weights_source"] = "pipeline.filters.ranking_weights (partial metrics available)"
                else:
                    row["ranking_weights_source"] = "fallback_rf2_pae_rmsd_plddt"
                row["status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                row["error"] = str(exc)
                row["status"] = "error"
            row["updated_at"] = now_str()
            model_rows.append(row)

    model_df = pd.DataFrame(model_rows)
    if model_df.empty:
        raise PipelineError("No model rows generated")

    # Pass flags on model-level rows
    model_df["strict_pass"] = model_df.apply(
        lambda r: is_pass(
            safe_float(r.get("rf2_pae")),
            safe_float(r.get("design_rf2_rmsd")),
            args.strict_pae_max,
            args.strict_rmsd_max,
        )
        if r.get("status") == "ok"
        else 0,
        axis=1,
    )
    model_df["soft_pass"] = model_df.apply(
        lambda r: is_pass(
            safe_float(r.get("rf2_pae")),
            safe_float(r.get("design_rf2_rmsd")),
            args.soft_pae_max,
            args.soft_rmsd_max,
        )
        if r.get("status") == "ok"
        else 0,
        axis=1,
    )

    job_df = build_job_summary(model_df, args=args)
    vs_wt_df = build_vs_wt(job_df)

    model_csv = outdir / "af3_rf2_model_level.csv"
    job_csv = outdir / "af3_rf2_job_summary.csv"
    vs_wt_csv = outdir / "af3_rf2_ranked_designs_vs_wt.csv"
    model_df.sort_values(["job_name", "model_index"], ascending=[True, True]).to_csv(model_csv, index=False)
    job_df.sort_values(["strict_pass_count", "soft_pass_count", "best_rf2_pae"], ascending=[False, False, True]).to_csv(
        job_csv, index=False
    )
    if not vs_wt_df.empty:
        vs_wt_df.to_csv(vs_wt_csv, index=False)

    strict_models = int(model_df["strict_pass"].sum())
    soft_models = int(model_df["soft_pass"].sum())
    ok_models = int((model_df["status"] == "ok").sum())
    total_models = int(len(model_df))

    print(f"Wrote: {model_csv}")
    print(f"Wrote: {job_csv}")
    if vs_wt_csv.exists():
        print(f"Wrote: {vs_wt_csv}")
    print(
        (
            "AF3->RF2 rescore complete: "
            f"ok_models={ok_models}/{total_models}, "
            f"strict_pass={strict_models}, soft_pass={soft_models}"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
