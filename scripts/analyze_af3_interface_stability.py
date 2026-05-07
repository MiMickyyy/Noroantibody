#!/usr/bin/env python3
"""Analyze AF3 result folders for nanobody-antigen interface stability.

This script computes model-level and job-level interface consistency metrics using:
- AF3 summary confidence fields (iptm, chain_pair_iptm, chain_pair_pae_min, ranking_score)
- geometric interface contacts from model CIF coordinates

Outputs:
- results/summaries/af3_interface_model_level.csv
- results/summaries/af3_interface_job_summary.csv
- results/summaries/af3_interface_ranked_designs_vs_wt.csv
- results/summaries/af3_interface_analysis_report.txt
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AF3 interface stability")
    parser.add_argument("--af3-results-dir", default="AF3 Results")
    parser.add_argument("--job-map-csv", default="results/af3_web_exports/af3_26_jobs_map.csv")
    parser.add_argument("--contact-cutoff", type=float, default=4.5)
    parser.add_argument("--outdir", default="results/summaries")
    return parser.parse_args()


def load_job_map(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: Dict[str, dict] = {}
    for _, row in df.iterrows():
        job_name = str(row.get("job_name", "")).strip()
        if not job_name:
            continue
        out[job_name] = {
            "candidate_id": str(row.get("candidate_id", "")).strip(),
            "source": str(row.get("source", "")).strip(),
        }
    return out


def load_job_request(job_request_json: Path) -> dict:
    obj = json.loads(job_request_json.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        if not obj:
            raise ValueError(f"Empty job request: {job_request_json}")
        return obj[0]
    return obj


def detect_chain_roles(full_data_json: Path) -> Tuple[List[str], str, List[str], Dict[str, int]]:
    obj = json.loads(full_data_json.read_text(encoding="utf-8"))
    token_chain_ids = obj.get("token_chain_ids", [])
    if not token_chain_ids:
        raise ValueError(f"No token_chain_ids in {full_data_json}")

    order: List[str] = []
    for c in token_chain_ids:
        if c not in order:
            order.append(c)
    counts = Counter(token_chain_ids)
    chain_counts = {c: int(counts[c]) for c in order}

    # Nanobody chain is expected to be shortest chain.
    nanobody_chain = min(order, key=lambda c: chain_counts.get(c, 10**9))
    antigen_chains = [c for c in order if c != nanobody_chain]
    return order, nanobody_chain, antigen_chains, chain_counts


def matrix_pair_value(matrix, chain_order: Sequence[str], chain_i: str, chain_j: str) -> float:
    if not isinstance(matrix, list) or not matrix:
        return float("nan")
    try:
        i = chain_order.index(chain_i)
        j = chain_order.index(chain_j)
        return float(matrix[i][j])
    except Exception:
        return float("nan")


def is_heavy_atom(atom) -> bool:
    element = (atom.element or "").strip().upper()
    name = atom.get_name().strip().upper()
    if element:
        return element != "H"
    return not name.startswith("H")


def residue_label(chain_id: str, residue) -> str:
    het, resseq, icode = residue.id
    if het != " ":
        return ""
    i = str(icode).strip()
    return f"{chain_id}:{int(resseq)}{i}"


def extract_chain_atoms_and_residues(chain) -> Tuple[np.ndarray, List[str]]:
    coords = []
    atom_res_labels: List[str] = []
    chain_id = str(chain.id)
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue
        rlabel = residue_label(chain_id, residue)
        if not rlabel:
            continue
        for atom in residue.get_atoms():
            if not is_heavy_atom(atom):
                continue
            coords.append(atom.coord.astype(float))
            atom_res_labels.append(rlabel)
    if not coords:
        return np.empty((0, 3), dtype=float), []
    return np.asarray(coords, dtype=float), atom_res_labels


def compute_contacts_for_model(
    cif_path: Path,
    nanobody_chain: str,
    antigen_chains: Sequence[str],
    cutoff: float,
) -> Dict[str, dict]:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("af3", str(cif_path))
    model = next(structure.get_models())
    chains = {str(c.id): c for c in model.get_chains()}

    if nanobody_chain not in chains:
        raise ValueError(f"Missing nanobody chain {nanobody_chain} in {cif_path.name}")

    nb_coords, nb_atom_res = extract_chain_atoms_and_residues(chains[nanobody_chain])
    if nb_coords.shape[0] == 0:
        raise ValueError(f"No heavy atoms found for nanobody chain {nanobody_chain} in {cif_path.name}")

    out: Dict[str, dict] = {}
    for achain in antigen_chains:
        if achain not in chains:
            out[achain] = {
                "contact_pairs": 0,
                "nanobody_contact_residues": set(),
                "antigen_contact_residues": set(),
            }
            continue
        ag_coords, ag_atom_res = extract_chain_atoms_and_residues(chains[achain])
        if ag_coords.shape[0] == 0:
            out[achain] = {
                "contact_pairs": 0,
                "nanobody_contact_residues": set(),
                "antigen_contact_residues": set(),
            }
            continue

        tree = cKDTree(ag_coords)
        neighbors = tree.query_ball_point(nb_coords, r=cutoff)

        contact_pairs = 0
        nb_res_set: Set[str] = set()
        ag_res_set: Set[str] = set()
        for i, js in enumerate(neighbors):
            if not js:
                continue
            contact_pairs += len(js)
            nb_res_set.add(nb_atom_res[i])
            for j in js:
                ag_res_set.add(ag_atom_res[j])

        out[achain] = {
            "contact_pairs": int(contact_pairs),
            "nanobody_contact_residues": nb_res_set,
            "antigen_contact_residues": ag_res_set,
        }
    return out


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mean_pairwise_jaccard(sets: List[Set[str]]) -> float:
    if not sets:
        return float("nan")
    if len(sets) == 1:
        return 1.0
    vals = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
    return float(np.mean(vals)) if vals else float("nan")


def compute_stability_score(row: dict) -> float:
    # Convert to bounded factors.
    iptm = float(row.get("iptm_mean", float("nan")))
    pair_iptm = float(row.get("best_pair_iptm_mean", float("nan")))
    pair_pae = float(row.get("best_pair_pae_min_mean", float("nan")))
    ag_j = float(row.get("dominant_antigen_residue_jaccard", float("nan")))
    dom_cons = float(row.get("dominant_chain_consistency", float("nan")))
    rank_std = float(row.get("ranking_score_std", float("nan")))
    clash_frac = float(row.get("has_clash_fraction", float("nan")))

    def nz(v, default=0.0):
        return default if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)

    iptm = nz(iptm)
    pair_iptm = nz(pair_iptm)
    pair_pae = nz(pair_pae, 20.0)
    ag_j = nz(ag_j)
    dom_cons = nz(dom_cons)
    rank_std = nz(rank_std, 0.1)
    clash_frac = nz(clash_frac, 1.0)

    pae_term = max(0.0, min(1.0, 1.0 - pair_pae / 20.0))
    std_term = max(0.0, min(1.0, 1.0 - rank_std / 0.1))

    score = (
        0.25 * iptm
        + 0.20 * pair_iptm
        + 0.20 * pae_term
        + 0.20 * ag_j
        + 0.10 * dom_cons
        + 0.05 * std_term
        - 0.10 * clash_frac
    )
    return float(max(0.0, min(1.0, score)))


def main() -> int:
    args = parse_args()
    root = Path(".").resolve()
    af3_dir = (root / args.af3_results_dir).resolve()
    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not af3_dir.exists():
        raise FileNotFoundError(f"AF3 results folder not found: {af3_dir}")

    job_map = load_job_map((root / args.job_map_csv).resolve())
    model_rows: List[dict] = []
    job_rows: List[dict] = []

    run_dirs = sorted([d for d in af3_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    if not run_dirs:
        raise RuntimeError(f"No fold_* directories found under {af3_dir}")

    for run_dir in run_dirs:
        job_request_files = sorted(run_dir.glob("*_job_request.json"))
        if not job_request_files:
            continue
        job_obj = load_job_request(job_request_files[0])
        job_name = str(job_obj.get("name", run_dir.name)).strip()
        map_row = job_map.get(job_name, {})
        candidate_id = str(map_row.get("candidate_id", "")).strip() or re.sub(r"^\d+_", "", job_name)
        is_wt = candidate_id.lower() == "wildtype" or "wildtype" in job_name.lower()

        full_data0 = sorted(run_dir.glob("*_full_data_0.json"))
        if not full_data0:
            continue

        chain_order, nanobody_chain, antigen_chains, chain_counts = detect_chain_roles(full_data0[0])
        if not antigen_chains:
            continue

        summary_files = sorted(run_dir.glob("*_summary_confidences_*.json"))
        if not summary_files:
            continue

        per_model_rows = []
        for sfile in summary_files:
            m = re.search(r"_summary_confidences_(\d+)\.json$", sfile.name)
            model_idx = int(m.group(1)) if m else -1
            s = json.loads(sfile.read_text(encoding="utf-8"))
            cif = run_dir / sfile.name.replace("_summary_confidences_", "_model_").replace(".json", ".cif")
            full_data = run_dir / sfile.name.replace("_summary_confidences_", "_full_data_")

            pair_iptm = s.get("chain_pair_iptm", [])
            pair_pae_min = s.get("chain_pair_pae_min", [])

            ag_pair_iptm = [
                matrix_pair_value(pair_iptm, chain_order, nanobody_chain, ach) for ach in antigen_chains
            ]
            ag_pair_pae = [
                matrix_pair_value(pair_pae_min, chain_order, nanobody_chain, ach) for ach in antigen_chains
            ]
            best_pair_iptm = float(np.nanmax(ag_pair_iptm)) if ag_pair_iptm else float("nan")
            best_pair_pae_min = float(np.nanmin(ag_pair_pae)) if ag_pair_pae else float("nan")

            contacts = compute_contacts_for_model(
                cif_path=cif,
                nanobody_chain=nanobody_chain,
                antigen_chains=antigen_chains,
                cutoff=args.contact_cutoff,
            )
            dominant_chain = max(
                antigen_chains,
                key=lambda c: contacts.get(c, {}).get("contact_pairs", 0),
            )
            dominant_contact_pairs = int(contacts.get(dominant_chain, {}).get("contact_pairs", 0))
            nb_res = contacts.get(dominant_chain, {}).get("nanobody_contact_residues", set())
            ag_res = contacts.get(dominant_chain, {}).get("antigen_contact_residues", set())

            row = {
                "run_dir": run_dir.name,
                "job_name": job_name,
                "candidate_id": candidate_id,
                "is_wildtype": int(is_wt),
                "model_index": model_idx,
                "chain_order": ",".join(chain_order),
                "nanobody_chain": nanobody_chain,
                "antigen_chains": ",".join(antigen_chains),
                "chain_counts": ";".join(f"{k}:{v}" for k, v in chain_counts.items()),
                "ranking_score": float(s.get("ranking_score", float("nan"))),
                "iptm": float(s.get("iptm", float("nan"))),
                "ptm": float(s.get("ptm", float("nan"))),
                "has_clash": float(s.get("has_clash", float("nan"))),
                "fraction_disordered": float(s.get("fraction_disordered", float("nan"))),
                "pair_iptm_nb_ag1": ag_pair_iptm[0] if len(ag_pair_iptm) > 0 else float("nan"),
                "pair_iptm_nb_ag2": ag_pair_iptm[1] if len(ag_pair_iptm) > 1 else float("nan"),
                "pair_pae_min_nb_ag1": ag_pair_pae[0] if len(ag_pair_pae) > 0 else float("nan"),
                "pair_pae_min_nb_ag2": ag_pair_pae[1] if len(ag_pair_pae) > 1 else float("nan"),
                "best_pair_iptm": best_pair_iptm,
                "best_pair_pae_min": best_pair_pae_min,
                "dominant_antigen_chain": dominant_chain,
                "dominant_contact_pairs": dominant_contact_pairs,
                "dominant_nanobody_contact_res_count": len(nb_res),
                "dominant_antigen_contact_res_count": len(ag_res),
                "dominant_nanobody_contact_residues": ",".join(sorted(nb_res)),
                "dominant_antigen_contact_residues": ",".join(sorted(ag_res)),
            }
            per_model_rows.append(row)
            model_rows.append(row)

        dfm = pd.DataFrame(per_model_rows).sort_values("model_index")
        if dfm.empty:
            continue

        dominant_chain_consistency = (
            dfm["dominant_antigen_chain"].value_counts(normalize=True).iloc[0]
            if not dfm["dominant_antigen_chain"].empty
            else float("nan")
        )
        ag_sets = [set(str(x).split(",")) if str(x).strip() else set() for x in dfm["dominant_antigen_contact_residues"]]
        nb_sets = [set(str(x).split(",")) if str(x).strip() else set() for x in dfm["dominant_nanobody_contact_residues"]]
        ag_sets = [s - {""} for s in ag_sets]
        nb_sets = [s - {""} for s in nb_sets]

        summary = {
            "run_dir": run_dir.name,
            "job_name": job_name,
            "candidate_id": candidate_id,
            "is_wildtype": int(is_wt),
            "n_models": int(len(dfm)),
            "nanobody_chain": nanobody_chain,
            "antigen_chains": ",".join(antigen_chains),
            "ranking_score_mean": float(dfm["ranking_score"].mean()),
            "ranking_score_std": float(dfm["ranking_score"].std(ddof=0)),
            "iptm_mean": float(dfm["iptm"].mean()),
            "iptm_std": float(dfm["iptm"].std(ddof=0)),
            "ptm_mean": float(dfm["ptm"].mean()),
            "best_pair_iptm_mean": float(dfm["best_pair_iptm"].mean()),
            "best_pair_iptm_std": float(dfm["best_pair_iptm"].std(ddof=0)),
            "best_pair_pae_min_mean": float(dfm["best_pair_pae_min"].mean()),
            "best_pair_pae_min_std": float(dfm["best_pair_pae_min"].std(ddof=0)),
            "has_clash_fraction": float(dfm["has_clash"].mean()),
            "fraction_disordered_mean": float(dfm["fraction_disordered"].mean()),
            "dominant_contact_pairs_mean": float(dfm["dominant_contact_pairs"].mean()),
            "dominant_contact_pairs_std": float(dfm["dominant_contact_pairs"].std(ddof=0)),
            "dominant_chain_consistency": float(dominant_chain_consistency),
            "dominant_antigen_residue_jaccard": float(mean_pairwise_jaccard(ag_sets)),
            "dominant_nanobody_residue_jaccard": float(mean_pairwise_jaccard(nb_sets)),
        }
        summary["interface_stability_score"] = compute_stability_score(summary)
        job_rows.append(summary)

    model_df = pd.DataFrame(model_rows)
    job_df = pd.DataFrame(job_rows)
    if model_df.empty or job_df.empty:
        raise RuntimeError("No AF3 jobs parsed successfully.")

    model_csv = outdir / "af3_interface_model_level.csv"
    job_csv = outdir / "af3_interface_job_summary.csv"
    model_pass_csv = outdir / "af3_interface_model_pass_summary.csv"
    rank_csv = outdir / "af3_interface_ranked_designs_vs_wt.csv"
    report_txt = outdir / "af3_interface_analysis_report.txt"

    # Model-level hard-pass heuristic for stable interface geometry.
    # These are proxy rules, not physical binding-energy calculations.
    model_df["model_interface_pass_strict"] = (
        (model_df["best_pair_iptm"].astype(float) >= 0.50)
        & (model_df["best_pair_pae_min"].astype(float) <= 10.0)
        & (model_df["iptm"].astype(float) >= 0.75)
        & (model_df["has_clash"].astype(float) <= 0.0)
    ).astype(int)

    pass_summary = (
        model_df.groupby(["job_name", "candidate_id", "is_wildtype"], as_index=False)
        .agg(
            n_models=("model_index", "count"),
            strict_pass_models=("model_interface_pass_strict", "sum"),
            iptm_mean=("iptm", "mean"),
            best_pair_iptm_mean=("best_pair_iptm", "mean"),
            best_pair_pae_min_mean=("best_pair_pae_min", "mean"),
        )
    )
    pass_summary["strict_model_pass_rate"] = (
        pass_summary["strict_pass_models"] / pass_summary["n_models"]
    )

    job_df = job_df.merge(
        pass_summary[
            [
                "job_name",
                "strict_pass_models",
                "strict_model_pass_rate",
            ]
        ],
        on="job_name",
        how="left",
    )
    model_df.sort_values(["job_name", "model_index"]).to_csv(model_csv, index=False)
    job_df.sort_values("interface_stability_score", ascending=False).to_csv(job_csv, index=False)
    pass_summary.sort_values(
        ["strict_model_pass_rate", "best_pair_iptm_mean", "best_pair_pae_min_mean"],
        ascending=[False, False, True],
    ).to_csv(model_pass_csv, index=False)

    wt = job_df[job_df["is_wildtype"] == 1]
    wt_row = wt.iloc[0] if not wt.empty else None
    designs = job_df[job_df["is_wildtype"] == 0].copy()
    if wt_row is not None:
        for col in ["interface_stability_score", "iptm_mean", "best_pair_iptm_mean", "best_pair_pae_min_mean"]:
            designs[f"delta_{col}_vs_wt"] = designs[col] - float(wt_row[col])
    designs = designs.sort_values(
        ["interface_stability_score", "best_pair_iptm_mean", "best_pair_pae_min_mean"],
        ascending=[False, False, True],
    )
    designs.to_csv(rank_csv, index=False)

    top10 = designs.head(10)
    with report_txt.open("w", encoding="utf-8") as handle:
        handle.write("AF3 Interface Stability Analysis\n")
        handle.write("================================\n\n")
        handle.write(f"AF3 result folders parsed: {job_df.shape[0]}\n")
        handle.write(f"Designs parsed: {designs.shape[0]}\n")
        handle.write(f"Wildtype detected: {0 if wt_row is None else 1}\n\n")
        if wt_row is not None:
            handle.write("Wildtype summary\n")
            handle.write("----------------\n")
            handle.write(
                f"job={wt_row['job_name']}, stability={wt_row['interface_stability_score']:.4f}, "
                f"iptm_mean={wt_row['iptm_mean']:.4f}, pair_iptm_mean={wt_row['best_pair_iptm_mean']:.4f}, "
                f"pair_pae_min_mean={wt_row['best_pair_pae_min_mean']:.4f}, "
                f"strict_pass={int(wt_row.get('strict_pass_models', 0))}/{int(wt_row.get('n_models', 0))}\n\n"
            )
        handle.write("Top 10 designs by interface stability score\n")
        handle.write("------------------------------------------\n")
        for _, r in top10.iterrows():
            handle.write(
                f"{r['job_name']}: stability={r['interface_stability_score']:.4f}, "
                f"iptm={r['iptm_mean']:.4f}, pair_iptm={r['best_pair_iptm_mean']:.4f}, "
                f"pair_pae_min={r['best_pair_pae_min_mean']:.4f}, "
                f"ag_jaccard={r['dominant_antigen_residue_jaccard']:.4f}, "
                f"chain_cons={r['dominant_chain_consistency']:.4f}, "
                f"strict_pass={int(r.get('strict_pass_models', 0))}/{int(r.get('n_models', 0))}\n"
            )

    print(f"Wrote: {model_csv}")
    print(f"Wrote: {job_csv}")
    print(f"Wrote: {model_pass_csv}")
    print(f"Wrote: {rank_csv}")
    print(f"Wrote: {report_txt}")
    print(f"Parsed jobs: {job_df.shape[0]}, designs: {designs.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
