#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from Bio.PDB import MMCIFParser
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Head-to-head AF3 analysis: WT vs p9c_052")
    p.add_argument(
        "--wt-dir",
        default="AF3 Results/Stage6 AF3/fold_089_wt",
        help="WT AF3 directory",
    )
    p.add_argument(
        "--candidate-dir",
        default="AF3 Results/Stage 9 AF3/p9c_052",
        help="p9c_052 AF3 directory",
    )
    p.add_argument("--cdr-config", default="data/configs/cdr_boundaries.yaml")
    p.add_argument(
        "--outdir",
        default="results/summaries/wt_vs_p9c052_af3_head2head",
        help="Output analysis directory",
    )
    p.add_argument("--contact-cutoff", type=float, default=4.5)
    p.add_argument("--hb-cutoff", type=float, default=3.5)
    p.add_argument("--salt-cutoff", type=float, default=4.0)
    p.add_argument("--clash-cutoff", type=float, default=2.0)
    p.add_argument("--close-cutoff", type=float, default=2.4)
    p.add_argument("--sidechain-overclose-cutoff", type=float, default=2.2)
    return p.parse_args()


@dataclass
class ModelRecord:
    system: str
    run_dir: Path
    run_name: str
    model_index: int
    summary_path: Path
    full_data_path: Path
    cif_path: Path
    job_name: str


def load_json(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj[0]
    return obj


def unique_in_order(seq: Sequence[str]) -> List[str]:
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def parse_cdr_config(path: Path) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    nb_chain = str(cfg.get("nanobody_chain_id", "C"))
    bounds = cfg.get("cdr_boundaries", {})
    cdr = {}
    for k in ["H1", "H2", "H3"]:
        v = bounds.get(k)
        if not v or len(v) != 2:
            raise ValueError(f"Missing CDR boundary for {k} in {path}")
        cdr[k] = (int(v[0]), int(v[1]))
    return nb_chain, cdr


def label_from_residue(chain_id: str, residue) -> str:
    het, resseq, icode = residue.id
    if het != " ":
        return ""
    ins = str(icode).strip()
    return f"{chain_id}:{int(resseq)}{ins}"


def parse_label(label: str) -> Tuple[str, int]:
    m = re.match(r"^([^:]+):(\d+)", str(label))
    if not m:
        return "", -1
    return m.group(1), int(m.group(2))


def cdr_region(resnum: int, cdr_bounds: Dict[str, Tuple[int, int]]) -> str:
    if cdr_bounds["H1"][0] <= resnum <= cdr_bounds["H1"][1]:
        return "CDR1"
    if cdr_bounds["H2"][0] <= resnum <= cdr_bounds["H2"][1]:
        return "CDR2"
    if cdr_bounds["H3"][0] <= resnum <= cdr_bounds["H3"][1]:
        return "CDR3"
    return "framework"


def is_heavy_atom(atom) -> bool:
    element = (atom.element or "").strip().upper()
    name = atom.get_name().strip().upper()
    if element:
        return element != "H"
    return not name.startswith("H")


def is_backbone_atom(name: str) -> bool:
    return name.strip().upper() in {"N", "CA", "C", "O", "OXT"}


def residue_plddt_from_structure(structure, cdr_bounds: Dict[str, Tuple[int, int]]) -> pd.DataFrame:
    model = next(structure.get_models())
    rows = []
    for chain in model.get_chains():
        cid = str(chain.id)
        for residue in chain.get_residues():
            if residue.id[0] != " ":
                continue
            label = label_from_residue(cid, residue)
            if not label:
                continue
            vals = [float(a.bfactor) for a in residue.get_atoms() if is_heavy_atom(a)]
            if not vals:
                continue
            _, rnum = parse_label(label)
            rows.append(
                {
                    "chain_id": cid,
                    "residue_label": label,
                    "resnum": rnum,
                    "resname": residue.get_resname().strip(),
                    "mean_plddt": float(np.mean(vals)),
                    "cdr_region": cdr_region(rnum, cdr_bounds),
                }
            )
    return pd.DataFrame(rows)


def extract_ca_dict(chain) -> Dict[Tuple[str, int, str], np.ndarray]:
    cid = str(chain.id)
    out = {}
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue
        if "CA" not in residue:
            continue
        _, resseq, icode = residue.id
        out[(cid, int(resseq), str(icode).strip())] = residue["CA"].coord.astype(float)
    return out


def extract_chain_atoms_detailed(chain) -> Tuple[np.ndarray, List[dict]]:
    coords = []
    meta = []
    cid = str(chain.id)
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue
        label = label_from_residue(cid, residue)
        if not label:
            continue
        _, rnum = parse_label(label)
        resname = residue.get_resname().strip().upper()
        for atom in residue.get_atoms():
            if not is_heavy_atom(atom):
                continue
            aname = atom.get_name().strip().upper()
            coords.append(atom.coord.astype(float))
            meta.append(
                {
                    "chain_id": cid,
                    "residue_label": label,
                    "resnum": rnum,
                    "resname": resname,
                    "atom_name": aname,
                    "element": (atom.element or "").strip().upper(),
                    "is_backbone": is_backbone_atom(aname),
                }
            )
    if not coords:
        return np.empty((0, 3), dtype=float), []
    return np.asarray(coords, dtype=float), meta


def parse_model_records(system: str, run_dir: Path) -> List[ModelRecord]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run dir: {run_dir}")

    job_request = sorted(run_dir.glob("*_job_request.json"))
    if not job_request:
        raise RuntimeError(f"No *_job_request.json in {run_dir}")
    job_obj = load_json(job_request[0])
    job_name = str(job_obj.get("name", run_dir.name)).strip()

    out: List[ModelRecord] = []
    for s in sorted(run_dir.glob("*_summary_confidences_*.json")):
        m = re.search(r"_summary_confidences_(\d+)\.json$", s.name)
        if not m:
            continue
        idx = int(m.group(1))
        full_data = run_dir / s.name.replace("_summary_confidences_", "_full_data_")
        cif = run_dir / s.name.replace("_summary_confidences_", "_model_").replace(".json", ".cif")
        if not full_data.exists() or not cif.exists():
            continue
        out.append(
            ModelRecord(
                system=system,
                run_dir=run_dir,
                run_name=run_dir.name,
                model_index=idx,
                summary_path=s,
                full_data_path=full_data,
                cif_path=cif,
                job_name=job_name,
            )
        )
    if not out:
        raise RuntimeError(f"No complete models in {run_dir}")
    return sorted(out, key=lambda x: x.model_index)


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    X = P - cP
    Y = Q - cQ
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    t = cQ - cP @ R
    return R, t


def rotation_angle_deg(R: np.ndarray) -> float:
    val = (np.trace(R) - 1.0) / 2.0
    val = max(-1.0, min(1.0, float(val)))
    return float(np.degrees(np.arccos(val)))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def hydrophobic_res(resname: str) -> bool:
    return resname.upper() in {
        "ALA",
        "VAL",
        "LEU",
        "ILE",
        "MET",
        "PHE",
        "TRP",
        "TYR",
        "PRO",
    }


def acidic_atom(resname: str, atom_name: str) -> bool:
    rn = resname.upper()
    an = atom_name.upper()
    return (rn == "ASP" and an.startswith("OD")) or (rn == "GLU" and an.startswith("OE"))


def basic_atom(resname: str, atom_name: str) -> bool:
    rn = resname.upper()
    an = atom_name.upper()
    if rn == "LYS":
        return an == "NZ"
    if rn == "ARG":
        return an.startswith("NH") or an == "NE"
    if rn == "HIS":
        return an.startswith("ND") or an.startswith("NE")
    return False


def build_token_index_map(full_data_obj: dict) -> Dict[Tuple[str, int], List[int]]:
    chain_ids = full_data_obj.get("token_chain_ids", [])
    res_ids = full_data_obj.get("token_res_ids", [])
    out: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for i, (ch, rr) in enumerate(zip(chain_ids, res_ids)):
        try:
            resnum = int(rr)
        except Exception:
            try:
                resnum = int(float(rr))
            except Exception:
                continue
        out[(str(ch), resnum)].append(i)
    return out


def dominant_contact_type(row: dict) -> str:
    flags = []
    if row.get("salt_bridge", 0) > 0:
        flags.append("salt_bridge")
    if row.get("hydrogen_bond", 0) > 0:
        flags.append("hydrogen_bond")
    if row.get("hydrophobic_packing", 0) > 0:
        flags.append("hydrophobic_packing")
    if not flags:
        return "heavy_atom_contact"
    if len(flags) == 1:
        return flags[0]
    return "mixed"


def main() -> int:
    args = parse_args()
    root = Path(".").resolve()
    wt_dir = (root / args.wt_dir).resolve()
    cand_dir = (root / args.candidate_dir).resolve()
    outdir = (root / args.outdir).resolve()

    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    subdirs = {
        "residue_plddt_tables": outdir / "residue_plddt_tables",
        "pae_matrices": outdir / "pae_matrices",
        "summary_plots": outdir / "summary_plots",
        "pose_comparison_figures": outdir / "pose_comparison_figures",
        "contact_heatmaps": outdir / "contact_heatmaps",
        "interface_confidence_figures": outdir / "interface_confidence_figures",
        "representative_annotated_structures": outdir / "representative_annotated_structures",
    }
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)

    nb_chain, cdr_bounds = parse_cdr_config((root / args.cdr_config).resolve())

    wt_records = parse_model_records("WT", wt_dir)
    cand_records = parse_model_records("p9c_052", cand_dir)
    records = wt_records + cand_records

    parser = MMCIFParser(QUIET=True)

    raw_rows = []
    contact_rows = []
    per_cdr_rows = []
    local_conf_rows = []
    strain_rows = []
    pose_rows = []

    model_cache = {}

    # For local hotspot PAE around known target patch.
    antigen_hotspot_res = [121, 122, 124, 217, 218, 219]
    core_nb_res = [27, 28, 29, 30, 33, 34]

    for rec in records:
        summary = load_json(rec.summary_path)
        full_data = load_json(rec.full_data_path)
        structure = parser.get_structure(f"{rec.system}_{rec.model_index}", str(rec.cif_path))
        model = next(structure.get_models())
        chains = {str(c.id): c for c in model.get_chains()}

        chain_order = unique_in_order([str(x) for x in full_data.get("token_chain_ids", [])])
        if nb_chain not in chains:
            # fallback to shortest chain if config mismatched
            chain_counts = Counter(full_data.get("token_chain_ids", []))
            nb_chain_eff = min(chain_order, key=lambda c: chain_counts.get(c, 10**9)) if chain_order else nb_chain
        else:
            nb_chain_eff = nb_chain

        antigen_chains = [c for c in chain_order if c != nb_chain_eff]
        if not antigen_chains:
            antigen_chains = [c for c in chains if c != nb_chain_eff]

        # chain-pair metrics from raw AF3 summary
        pair_iptm = summary.get("chain_pair_iptm", [])
        pair_pae = summary.get("chain_pair_pae_min", [])

        def pair_val(matrix, ci, cj):
            if not isinstance(matrix, list) or not matrix:
                return float("nan")
            try:
                i = chain_order.index(ci)
                j = chain_order.index(cj)
                return float(matrix[i][j])
            except Exception:
                return float("nan")

        ag_pair_iptm = [pair_val(pair_iptm, nb_chain_eff, c) for c in antigen_chains]
        ag_pair_pae = [pair_val(pair_pae, nb_chain_eff, c) for c in antigen_chains]

        atom_plddts = np.asarray(full_data.get("atom_plddts", []), dtype=float)
        mean_plddt = float(np.mean(atom_plddts)) if atom_plddts.size else float("nan")

        raw_rows.append(
            {
                "system": rec.system,
                "run_name": rec.run_name,
                "job_name": rec.job_name,
                "model_index": rec.model_index,
                "ranking_score": float(summary.get("ranking_score", float("nan"))),
                "ipTM": float(summary.get("iptm", float("nan"))),
                "pTM": float(summary.get("ptm", float("nan"))),
                "mean_pLDDT": mean_plddt,
                "has_clash": float(summary.get("has_clash", float("nan"))),
                "fraction_disordered": float(summary.get("fraction_disordered", float("nan"))),
                "chain_pair_iptm_nb_ag_max": float(np.nanmax(ag_pair_iptm)) if ag_pair_iptm else float("nan"),
                "chain_pair_iptm_nb_ag_mean": float(np.nanmean(ag_pair_iptm)) if ag_pair_iptm else float("nan"),
                "chain_pair_pae_min_nb_ag_min": float(np.nanmin(ag_pair_pae)) if ag_pair_pae else float("nan"),
                "chain_pair_pae_min_nb_ag_mean": float(np.nanmean(ag_pair_pae)) if ag_pair_pae else float("nan"),
                "chain_order": ",".join(chain_order),
                "nanobody_chain": nb_chain_eff,
                "antigen_chains": ",".join(antigen_chains),
                "summary_json": str(rec.summary_path),
                "full_data_json": str(rec.full_data_path),
                "cif_path": str(rec.cif_path),
            }
        )

        # PAE matrix export
        pae = np.asarray(full_data.get("pae", []), dtype=float)
        np.save(subdirs["pae_matrices"] / f"{rec.system}_model{rec.model_index}_pae.npy", pae)

        # Residue pLDDT export
        res_plddt_df = residue_plddt_from_structure(structure, cdr_bounds)
        res_plddt_df.insert(0, "model_index", rec.model_index)
        res_plddt_df.insert(0, "job_name", rec.job_name)
        res_plddt_df.insert(0, "system", rec.system)
        res_plddt_df.to_csv(
            subdirs["residue_plddt_tables"] / f"{rec.system}_model{rec.model_index}_residue_plddt.csv",
            index=False,
        )

        # Build coordinate caches
        nb_coords, nb_meta = extract_chain_atoms_detailed(chains[nb_chain_eff])
        ag_coords_all = []
        ag_meta_all = []
        ag_ca_dict = {}
        for ac in antigen_chains:
            if ac not in chains:
                continue
            ccoords, cmeta = extract_chain_atoms_detailed(chains[ac])
            if ccoords.size:
                ag_coords_all.append(ccoords)
                ag_meta_all.extend(cmeta)
            ag_ca_dict.update(extract_ca_dict(chains[ac]))
        if ag_coords_all:
            ag_coords = np.vstack(ag_coords_all)
        else:
            ag_coords = np.empty((0, 3), dtype=float)

        nb_ca_dict = extract_ca_dict(chains[nb_chain_eff])

        # Interface contacts
        per_pair = {}
        if nb_coords.size and ag_coords.size:
            tree = cKDTree(ag_coords)
            neighbors = tree.query_ball_point(nb_coords, r=args.contact_cutoff)
            for i, js in enumerate(neighbors):
                if not js:
                    continue
                nmeta = nb_meta[i]
                for j in js:
                    ameta = ag_meta_all[j]
                    d = float(np.linalg.norm(nb_coords[i] - ag_coords[j]))
                    key = (nmeta["residue_label"], ameta["residue_label"])
                    row = per_pair.setdefault(
                        key,
                        {
                            "min_distance": 999.0,
                            "heavy_atom_contacts": 0,
                            "hydrogen_bond": 0,
                            "salt_bridge": 0,
                            "hydrophobic_packing": 0,
                            "nb_resname": nmeta["resname"],
                            "ag_resname": ameta["resname"],
                        },
                    )
                    row["min_distance"] = min(row["min_distance"], d)
                    row["heavy_atom_contacts"] += 1

                    # Hydrogen bond heuristic
                    if (
                        nmeta["element"] in {"N", "O", "S"}
                        and ameta["element"] in {"N", "O", "S"}
                        and d <= args.hb_cutoff
                    ):
                        row["hydrogen_bond"] = 1

                    # Salt bridge heuristic
                    if d <= args.salt_cutoff:
                        c1 = acidic_atom(nmeta["resname"], nmeta["atom_name"]) and basic_atom(
                            ameta["resname"], ameta["atom_name"]
                        )
                        c2 = acidic_atom(ameta["resname"], ameta["atom_name"]) and basic_atom(
                            nmeta["resname"], nmeta["atom_name"]
                        )
                        if c1 or c2:
                            row["salt_bridge"] = 1

                    # Hydrophobic packing heuristic
                    if (
                        hydrophobic_res(nmeta["resname"])
                        and hydrophobic_res(ameta["resname"])
                        and nmeta["element"] == "C"
                        and ameta["element"] == "C"
                        and d <= args.contact_cutoff
                    ):
                        row["hydrophobic_packing"] = 1

        interface_nb = set()
        interface_ag = set()
        interface_pairs = []
        for (nb_res, ag_res), v in sorted(per_pair.items()):
            _, nb_rnum = parse_label(nb_res)
            ct = dominant_contact_type(v)
            contact_rows.append(
                {
                    "system": rec.system,
                    "job_name": rec.job_name,
                    "model_index": rec.model_index,
                    "nanobody_residue": nb_res,
                    "antigen_residue": ag_res,
                    "residue_pair": f"{nb_res}|{ag_res}",
                    "cdr_region": cdr_region(nb_rnum, cdr_bounds),
                    "contact_type": ct,
                    "min_distance": float(v["min_distance"]),
                    "heavy_atom_contacts": int(v["heavy_atom_contacts"]),
                    "hydrogen_bond": int(v["hydrogen_bond"]),
                    "salt_bridge": int(v["salt_bridge"]),
                    "hydrophobic_packing": int(v["hydrophobic_packing"]),
                }
            )
            interface_nb.add(nb_res)
            interface_ag.add(ag_res)
            interface_pairs.append((nb_res, ag_res))

        # per-CDR summary
        if interface_pairs:
            tmp = pd.DataFrame(
                [
                    {
                        "cdr_region": cdr_region(parse_label(nb)[1], cdr_bounds),
                        "nanobody_residue": nb,
                        "antigen_residue": ag,
                    }
                    for nb, ag in interface_pairs
                ]
            )
            for reg, g in tmp.groupby("cdr_region"):
                per_cdr_rows.append(
                    {
                        "system": rec.system,
                        "job_name": rec.job_name,
                        "model_index": rec.model_index,
                        "cdr_region": reg,
                        "residue_pair_count": int(g.shape[0]),
                        "nanobody_residue_count": int(g["nanobody_residue"].nunique()),
                        "antigen_residue_count": int(g["antigen_residue"].nunique()),
                    }
                )

        # Local interface confidence
        token_map = build_token_index_map(full_data)
        pae_mat = np.asarray(full_data.get("pae", []), dtype=float)
        res_plddt_map = dict(zip(res_plddt_df["residue_label"], res_plddt_df["mean_plddt"]))

        nb_plddt_vals = [res_plddt_map[x] for x in interface_nb if x in res_plddt_map]
        ag_plddt_vals = [res_plddt_map[x] for x in interface_ag if x in res_plddt_map]

        pair_pae_vals = []
        for nb_res, ag_res in interface_pairs:
            nb_c, nb_n = parse_label(nb_res)
            ag_c, ag_n = parse_label(ag_res)
            nidx = token_map.get((nb_c, nb_n), [])
            aidx = token_map.get((ag_c, ag_n), [])
            for i in nidx:
                for j in aidx:
                    if i < pae_mat.shape[0] and j < pae_mat.shape[1]:
                        pair_pae_vals.append(float((pae_mat[i, j] + pae_mat[j, i]) / 2.0))

        hotspot_vals = []
        for nb_r in core_nb_res:
            nidx = token_map.get((nb_chain_eff, nb_r), [])
            for ac in antigen_chains:
                for ar in antigen_hotspot_res:
                    aidx = token_map.get((ac, ar), [])
                    for i in nidx:
                        for j in aidx:
                            if i < pae_mat.shape[0] and j < pae_mat.shape[1]:
                                hotspot_vals.append(float((pae_mat[i, j] + pae_mat[j, i]) / 2.0))

        local_conf_rows.append(
            {
                "system": rec.system,
                "job_name": rec.job_name,
                "model_index": rec.model_index,
                "interface_nb_residue_count": int(len(interface_nb)),
                "interface_ag_residue_count": int(len(interface_ag)),
                "interface_nb_mean_pLDDT": float(np.mean(nb_plddt_vals)) if nb_plddt_vals else float("nan"),
                "interface_ag_mean_pLDDT": float(np.mean(ag_plddt_vals)) if ag_plddt_vals else float("nan"),
                "interface_pair_pae_mean": float(np.mean(pair_pae_vals)) if pair_pae_vals else float("nan"),
                "hotspot_local_pae_mean": float(np.mean(hotspot_vals)) if hotspot_vals else float("nan"),
                "hotspot_local_pae_min": float(np.min(hotspot_vals)) if hotspot_vals else float("nan"),
            }
        )

        # Strain checks
        clash_count = 0
        close_count = 0
        sidechain_overclose = 0
        min_dist = float("nan")
        if nb_coords.size and ag_coords.size:
            tree = cKDTree(ag_coords)
            dists, idxs = tree.query(nb_coords, k=1)
            if np.size(dists):
                min_dist = float(np.min(dists))

            neigh = tree.query_ball_point(nb_coords, r=args.close_cutoff)
            for i, js in enumerate(neigh):
                nmeta = nb_meta[i]
                for j in js:
                    ameta = ag_meta_all[j]
                    d = float(np.linalg.norm(nb_coords[i] - ag_coords[j]))
                    if d < args.clash_cutoff:
                        clash_count += 1
                    if d < args.close_cutoff:
                        close_count += 1
                    if (
                        d < args.sidechain_overclose_cutoff
                        and (not nmeta["is_backbone"])
                        and (not ameta["is_backbone"])
                    ):
                        sidechain_overclose += 1

        # CDR backbone geometry outliers via CA distances
        cdr_outliers = 0
        cdr_mean_abs_dev = []
        if nb_chain_eff in chains:
            ca_dict = extract_ca_dict(chains[nb_chain_eff])
            by_num = {k[1]: v for k, v in ca_dict.items()}
            for h in ["H1", "H2", "H3"]:
                s, e = cdr_bounds[h]
                for r in range(s, e):
                    if r in by_num and (r + 1) in by_num:
                        d = float(np.linalg.norm(by_num[r + 1] - by_num[r]))
                        cdr_mean_abs_dev.append(abs(d - 3.8))
                        if d < 3.4 or d > 4.2:
                            cdr_outliers += 1

        strain_rows.append(
            {
                "system": rec.system,
                "job_name": rec.job_name,
                "model_index": rec.model_index,
                "steric_clash_count_lt2p0": int(clash_count),
                "abnormally_close_count_lt2p4": int(close_count),
                "sidechain_overclose_count_lt2p2": int(sidechain_overclose),
                "min_interchain_heavy_distance": min_dist,
                "cdr_backbone_outlier_count": int(cdr_outliers),
                "cdr_backbone_mean_abs_dev_from_3p8": float(np.mean(cdr_mean_abs_dev)) if cdr_mean_abs_dev else float("nan"),
            }
        )

        model_cache[(rec.system, rec.model_index)] = {
            "record": rec,
            "structure": structure,
            "nb_chain": nb_chain_eff,
            "antigen_chains": antigen_chains,
            "nb_ca": nb_ca_dict,
            "ag_ca": ag_ca_dict,
            "interface_ag_res": set(interface_ag),
            "token_map": token_map,
            "pae": pae_mat,
        }

    raw_df = pd.DataFrame(raw_rows).sort_values(["system", "model_index"])
    raw_df.to_csv(outdir / "raw_confidence_per_model.csv", index=False)

    # Summary plots for raw confidence
    sns.set_theme(style="whitegrid")
    plot_metrics = [
        "ranking_score",
        "ipTM",
        "pTM",
        "mean_pLDDT",
        "chain_pair_iptm_nb_ag_mean",
        "chain_pair_pae_min_nb_ag_min",
    ]
    for m in plot_metrics:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=raw_df, x="system", y=m)
        sns.stripplot(data=raw_df, x="system", y=m, color="black", alpha=0.6, size=4)
        plt.title(f"WT vs p9c_052: {m}")
        plt.tight_layout()
        plt.savefig(subdirs["summary_plots"] / f"{m}_wt_vs_p9c052.png", dpi=180)
        plt.close()

    # Contact outputs
    contact_df = pd.DataFrame(contact_rows)
    contact_df.to_csv(outdir / "interface_contacts_long.csv", index=False)

    # Consensus contact maps
    cons_rows = []
    for sys, g in contact_df.groupby("system"):
        n_models = raw_df[raw_df["system"] == sys]["model_index"].nunique()
        for (nb_res, ag_res), gg in g.groupby(["nanobody_residue", "antigen_residue"]):
            pair = f"{nb_res}|{ag_res}"
            occ = gg["model_index"].nunique()
            typ = gg["contact_type"].mode().iloc[0]
            _, rn = parse_label(nb_res)
            cons_rows.append(
                {
                    "system": sys,
                    "residue_pair": pair,
                    "nanobody_residue": nb_res,
                    "antigen_residue": ag_res,
                    "cdr_region": cdr_region(rn, cdr_bounds),
                    "models_with_contact": int(occ),
                    "n_models": int(n_models),
                    "occupancy": float(occ / max(1, n_models)),
                    "dominant_contact_type": typ,
                }
            )
    cons_df = pd.DataFrame(cons_rows)
    cons_wt = cons_df[cons_df["system"] == "WT"].sort_values(["occupancy", "residue_pair"], ascending=[False, True])
    cons_p9 = cons_df[cons_df["system"] == "p9c_052"].sort_values(["occupancy", "residue_pair"], ascending=[False, True])
    cons_wt.to_csv(outdir / "interface_contact_consensus_WT.csv", index=False)
    cons_p9.to_csv(outdir / "interface_contact_consensus_p9c052.csv", index=False)

    per_cdr_df = pd.DataFrame(per_cdr_rows)
    per_cdr_df.to_csv(outdir / "per_CDR_contact_summary.csv", index=False)

    # contact heatmaps: occupancy matrices
    def plot_contact_heatmap(cons_sub: pd.DataFrame, tag: str):
        if cons_sub.empty:
            return
        m = cons_sub.copy()
        m["nb"] = m["nanobody_residue"].str.replace(r"^[^:]+:", "", regex=True)
        m["ag"] = m["antigen_residue"].str.replace(r"^[^:]+:", "", regex=True)
        pivot = m.pivot_table(index="nb", columns="ag", values="occupancy", aggfunc="max", fill_value=0.0)
        if pivot.shape[0] > 40:
            pivot = pivot.iloc[:40, :]
        if pivot.shape[1] > 50:
            pivot = pivot.iloc[:, :50]
        plt.figure(figsize=(max(8, 0.25 * pivot.shape[1]), max(5, 0.25 * pivot.shape[0])))
        sns.heatmap(pivot, cmap="viridis", vmin=0, vmax=1)
        plt.title(f"Contact occupancy heatmap: {tag}")
        plt.tight_layout()
        plt.savefig(subdirs["contact_heatmaps"] / f"contact_heatmap_{tag}.png", dpi=180)
        plt.close()

    plot_contact_heatmap(cons_wt, "WT")
    plot_contact_heatmap(cons_p9, "p9c052")

    # Pose metrics
    ref_key = ("WT", min(k[1] for k in model_cache if k[0] == "WT"))
    ref = model_cache[ref_key]
    ref_nb = ref["nb_ca"]
    ref_ag = ref["ag_ca"]

    # Reference epitope center: WT reference interface residues on antigen
    ref_epi_pts = []
    for lab in ref["interface_ag_res"]:
        ch, rn = parse_label(lab)
        key = (ch, rn, "")
        cands = [k for k in ref_ag if k[0] == ch and k[1] == rn]
        if cands:
            ref_epi_pts.append(ref_ag[cands[0]])
    if not ref_epi_pts:
        ref_epi_pts = list(ref_ag.values())
    ref_epi_center = np.mean(np.asarray(ref_epi_pts), axis=0)

    ref_nb_keys = set(ref_nb.keys())
    ref_nb_pts = np.asarray([ref_nb[k] for k in sorted(ref_nb_keys)], dtype=float)
    ref_nb_com = ref_nb_pts.mean(axis=0)

    def principal_axis(points: np.ndarray) -> np.ndarray:
        if points.shape[0] < 3:
            return np.array([1.0, 0.0, 0.0])
        X = points - points.mean(axis=0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        v = vt[0]
        return v / max(1e-8, np.linalg.norm(v))

    ref_axis = principal_axis(ref_nb_pts)

    # Paratope residue set from CDRs
    paratope_res = set(range(cdr_bounds["H1"][0], cdr_bounds["H1"][1] + 1))
    paratope_res |= set(range(cdr_bounds["H2"][0], cdr_bounds["H2"][1] + 1))
    paratope_res |= set(range(cdr_bounds["H3"][0], cdr_bounds["H3"][1] + 1))

    for key, item in sorted(model_cache.items(), key=lambda x: (x[0][0], x[0][1])):
        sys, midx = key
        nb = item["nb_ca"]
        ag = item["ag_ca"]

        common_ag = sorted(set(ag.keys()) & set(ref_ag.keys()))
        if len(common_ag) < 20:
            continue
        P = np.asarray([ag[k] for k in common_ag], dtype=float)
        Q = np.asarray([ref_ag[k] for k in common_ag], dtype=float)
        R, t = kabsch_align(P, Q)

        # align nb coordinates after antigen anchoring
        common_nb = sorted(set(nb.keys()) & ref_nb_keys)
        if len(common_nb) < 20:
            continue
        nb_aligned = np.asarray([nb[k] @ R + t for k in common_nb], dtype=float)
        nb_ref = np.asarray([ref_nb[k] for k in common_nb], dtype=float)
        pose_rmsd = float(np.sqrt(np.mean(np.sum((nb_aligned - nb_ref) ** 2, axis=1))))

        # residual rigid-body transform on nanobody
        R_nb, t_nb = kabsch_align(nb_aligned, nb_ref)
        rot_deg = rotation_angle_deg(R_nb)

        nb_all = np.asarray([v @ R + t for v in nb.values()], dtype=float)
        nb_com = nb_all.mean(axis=0)
        trans_vec = nb_com - ref_nb_com
        trans_norm = float(np.linalg.norm(trans_vec))

        # paratope COM
        par_pts = []
        cdr3_pts = []
        for k_nb, v in nb.items():
            ch, rn, _ = k_nb
            if ch != item["nb_chain"]:
                continue
            vv = v @ R + t
            if rn in paratope_res:
                par_pts.append(vv)
            if cdr_bounds["H3"][0] <= rn <= cdr_bounds["H3"][1]:
                cdr3_pts.append(vv)
        par_com = np.mean(np.asarray(par_pts), axis=0) if par_pts else nb_com
        cdr3_com = np.mean(np.asarray(cdr3_pts), axis=0) if cdr3_pts else nb_com

        paratope_dist = float(np.linalg.norm(par_com - ref_epi_center))
        cdr3_angle = angle_between(cdr3_com - nb_com, ref_epi_center - nb_com)

        ax = principal_axis(nb_all)
        axis_angle = min(angle_between(ax, ref_axis), angle_between(-ax, ref_axis))

        pose_rows.append(
            {
                "system": sys,
                "job_name": item["record"].job_name,
                "model_index": midx,
                "pose_rmsd_nb_ca": pose_rmsd,
                "translation_x": float(trans_vec[0]),
                "translation_y": float(trans_vec[1]),
                "translation_z": float(trans_vec[2]),
                "translation_norm": trans_norm,
                "rotation_diff_deg": rot_deg,
                "paratope_epitope_distance": paratope_dist,
                "cdr3_to_epitope_angle_deg": cdr3_angle,
                "principal_axis_angle_deg": axis_angle,
            }
        )

    pose_df = pd.DataFrame(pose_rows).sort_values(["system", "model_index"])

    # clustering
    feat_cols = [
        "translation_x",
        "translation_y",
        "translation_z",
        "rotation_diff_deg",
        "paratope_epitope_distance",
        "cdr3_to_epitope_angle_deg",
    ]
    X = pose_df[feat_cols].to_numpy(dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd < 1e-8] = 1.0
    Xz = (X - mu) / sd
    if Xz.shape[0] >= 3:
        Z = linkage(Xz, method="ward")
        labels = fcluster(Z, t=2, criterion="maxclust")
    else:
        labels = np.ones(Xz.shape[0], dtype=int)
    pose_df["pose_cluster"] = labels

    pose_df.to_csv(outdir / "pose_metrics.csv", index=False)
    pose_df[["system", "job_name", "model_index", "pose_cluster"]].to_csv(
        outdir / "pose_cluster_assignments.csv", index=False
    )

    # pose figures
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=pose_df,
        x="translation_x",
        y="translation_y",
        hue="system",
        style="pose_cluster",
        s=80,
    )
    plt.title("Antigen-aligned nanobody translation (x/y)")
    plt.tight_layout()
    plt.savefig(subdirs["pose_comparison_figures"] / "translation_xy_scatter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=pose_df, x="system", y="pose_rmsd_nb_ca")
    sns.stripplot(data=pose_df, x="system", y="pose_rmsd_nb_ca", color="black", alpha=0.6)
    plt.title("Pose RMSD to WT reference (antigen-aligned)")
    plt.tight_layout()
    plt.savefig(subdirs["pose_comparison_figures"] / "pose_rmsd_boxplot.png", dpi=180)
    plt.close()

    occ = pose_df.groupby(["system", "pose_cluster"]).size().reset_index(name="count")
    plt.figure(figsize=(7, 4))
    sns.barplot(data=occ, x="pose_cluster", y="count", hue="system")
    plt.title("Pose cluster occupancy")
    plt.tight_layout()
    plt.savefig(subdirs["pose_comparison_figures"] / "pose_cluster_occupancy.png", dpi=180)
    plt.close()

    # local interface confidence outputs
    local_df = pd.DataFrame(local_conf_rows).sort_values(["system", "model_index"])
    local_df.to_csv(outdir / "local_interface_confidence.csv", index=False)

    for m in [
        "interface_nb_mean_pLDDT",
        "interface_ag_mean_pLDDT",
        "interface_pair_pae_mean",
        "hotspot_local_pae_mean",
    ]:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=local_df, x="system", y=m)
        sns.stripplot(data=local_df, x="system", y=m, color="black", alpha=0.6)
        plt.title(f"Local interface confidence: {m}")
        plt.tight_layout()
        plt.savefig(subdirs["interface_confidence_figures"] / f"{m}_wt_vs_p9c052.png", dpi=180)
        plt.close()

    # strain checks
    strain_df = pd.DataFrame(strain_rows).sort_values(["system", "model_index"])
    strain_df.to_csv(outdir / "strain_checks.csv", index=False)

    # representative structures (closest to system centroid in pose features)
    rep_dir = subdirs["representative_annotated_structures"]
    for sys, g in pose_df.groupby("system"):
        G = g[feat_cols].to_numpy(float)
        center = np.nanmean(G, axis=0)
        dist = np.linalg.norm(G - center, axis=1)
        idx = int(np.argmin(dist))
        row = g.iloc[idx]
        model_idx = int(row["model_index"])
        rec = model_cache[(sys, model_idx)]["record"]

        dst_cif = rep_dir / f"{sys}_representative_model{model_idx}.cif"
        shutil.copy2(rec.cif_path, dst_cif)
        shutil.copy2(rec.summary_path, rep_dir / f"{sys}_representative_model{model_idx}_summary.json")

        note = rep_dir / f"{sys}_representative_model{model_idx}_notes.txt"
        note.write_text(
            "\n".join(
                [
                    f"system: {sys}",
                    f"job_name: {rec.job_name}",
                    f"model_index: {model_idx}",
                    f"pose_cluster: {int(row['pose_cluster'])}",
                    f"pose_rmsd_nb_ca: {row['pose_rmsd_nb_ca']:.4f}",
                    f"rotation_diff_deg: {row['rotation_diff_deg']:.4f}",
                    f"paratope_epitope_distance: {row['paratope_epitope_distance']:.4f}",
                    f"cdr3_to_epitope_angle_deg: {row['cdr3_to_epitope_angle_deg']:.4f}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    # Evidence-based conclusion (no custom aggregate score decision)
    def med(df: pd.DataFrame, col: str, sys: str) -> float:
        x = df[df["system"] == sys][col].astype(float)
        return float(np.nanmedian(x)) if len(x) else float("nan")

    raw_eval = {
        "ipTM": (med(raw_df, "ipTM", "WT"), med(raw_df, "ipTM", "p9c_052")),
        "pTM": (med(raw_df, "pTM", "WT"), med(raw_df, "pTM", "p9c_052")),
        "ranking_score": (med(raw_df, "ranking_score", "WT"), med(raw_df, "ranking_score", "p9c_052")),
        "mean_pLDDT": (med(raw_df, "mean_pLDDT", "WT"), med(raw_df, "mean_pLDDT", "p9c_052")),
        "pair_iptm": (med(raw_df, "chain_pair_iptm_nb_ag_mean", "WT"), med(raw_df, "chain_pair_iptm_nb_ag_mean", "p9c_052")),
        "pair_pae_min": (med(raw_df, "chain_pair_pae_min_nb_ag_min", "WT"), med(raw_df, "chain_pair_pae_min_nb_ag_min", "p9c_052")),
    }

    wt_dom_cluster = int(pose_df[pose_df["system"] == "WT"]["pose_cluster"].mode().iloc[0])
    p9_in_wt_cluster = int((pose_df[(pose_df["system"] == "p9c_052")]["pose_cluster"] == wt_dom_cluster).sum())
    p9_total = int((pose_df["system"] == "p9c_052").sum())

    wt_pose_spread = float(np.nanmedian(pose_df[pose_df["system"] == "WT"]["pose_rmsd_nb_ca"]))
    p9_pose_spread = float(np.nanmedian(pose_df[pose_df["system"] == "p9c_052"]["pose_rmsd_nb_ca"]))

    wt_clash_med = med(strain_df, "steric_clash_count_lt2p0", "WT")
    p9_clash_med = med(strain_df, "steric_clash_count_lt2p0", "p9c_052")
    wt_overclose_med = med(strain_df, "sidechain_overclose_count_lt2p2", "WT")
    p9_overclose_med = med(strain_df, "sidechain_overclose_count_lt2p2", "p9c_052")

    cons_pair = cons_df.pivot_table(
        index="residue_pair", columns="system", values="occupancy", aggfunc="max", fill_value=0.0
    ).reset_index()
    preserved = int(((cons_pair.get("WT", 0) >= 0.6) & (cons_pair.get("p9c_052", 0) >= 0.6)).sum())
    gained = int(((cons_pair.get("WT", 0) < 0.4) & (cons_pair.get("p9c_052", 0) >= 0.6)).sum())
    lost = int(((cons_pair.get("WT", 0) >= 0.6) & (cons_pair.get("p9c_052", 0) < 0.4)).sum())

    # Raw-metric qualitative call
    better_count = 0
    tie_count = 0
    for k, (w, p) in raw_eval.items():
        if np.isnan(w) or np.isnan(p):
            continue
        if k == "pair_pae_min":
            diff = w - p
        else:
            diff = p - w
        if abs(diff) < 0.01:
            tie_count += 1
        elif diff > 0:
            better_count += 1

    if better_count >= 4:
        raw_call = "consistently outperform"
    elif better_count >= 2:
        raw_call = "partially outperform"
    elif better_count == 1 and tie_count >= 2:
        raw_call = "roughly match"
    else:
        raw_call = "not outperform"

    if p9_pose_spread < wt_pose_spread - 0.2:
        repro_call = "more consistent"
    elif p9_pose_spread <= wt_pose_spread + 0.2:
        repro_call = "comparable"
    else:
        repro_call = "less consistent"

    same_logic = p9_in_wt_cluster / max(1, p9_total) >= 0.6
    logic_call = "same pose family" if same_logic else "alternative pose family"

    strain_flag = (p9_clash_med > wt_clash_med) or (p9_overclose_med > wt_overclose_med)

    if raw_call == "consistently outperform" and repro_call in {"more consistent", "comparable"} and not strain_flag:
        final_label = "credible WT challenger"
    elif raw_call in {"partially outperform", "roughly match"} and repro_call == "comparable" and not strain_flag:
        final_label = "roughly tied with WT"
    elif raw_call in {"partially outperform", "roughly match"}:
        final_label = "interesting but not yet convincing"
    else:
        final_label = "likely false positive"

    summary_md = outdir / "head2head_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# WT vs p9c_052 AF3 Head-to-Head (Evidence-First)",
                "",
                "## Inputs",
                f"- WT dir: `{wt_dir}`",
                f"- p9c_052 dir: `{cand_dir}`",
                f"- Models used: WT={len(wt_records)}, p9c_052={len(cand_records)}",
                "",
                "## A. Raw AF3-native confidence",
                f"- Call: **{raw_call}**",
                "- Median metrics (WT -> p9c_052):",
                f"  - ipTM: {raw_eval['ipTM'][0]:.4f} -> {raw_eval['ipTM'][1]:.4f}",
                f"  - pTM: {raw_eval['pTM'][0]:.4f} -> {raw_eval['pTM'][1]:.4f}",
                f"  - ranking_score: {raw_eval['ranking_score'][0]:.4f} -> {raw_eval['ranking_score'][1]:.4f}",
                f"  - mean pLDDT: {raw_eval['mean_pLDDT'][0]:.2f} -> {raw_eval['mean_pLDDT'][1]:.2f}",
                f"  - chain-pair iPTM (nb-antigen mean): {raw_eval['pair_iptm'][0]:.4f} -> {raw_eval['pair_iptm'][1]:.4f}",
                f"  - chain-pair min PAE (best): {raw_eval['pair_pae_min'][0]:.4f} -> {raw_eval['pair_pae_min'][1]:.4f} (lower is better)",
                "",
                "## B. Full-interface reproducibility",
                f"- Call: **{repro_call}**",
                f"- Pose RMSD median (antigen-aligned, to WT reference): WT={wt_pose_spread:.3f}, p9c_052={p9_pose_spread:.3f}",
                f"- p9c_052 models in WT-dominant cluster: {p9_in_wt_cluster}/{p9_total}",
                "",
                "## C. Binding logic relation",
                f"- Call: **{logic_call}**",
                f"- Conserved interface pairs (occupancy >=0.6 in both): {preserved}",
                f"- p9c_052 gained high-occupancy pairs vs WT: {gained}",
                f"- p9c_052 lost WT high-occupancy pairs: {lost}",
                "",
                "## D. Strain / hard-fit artifacts",
                f"- WT median steric clashes (<2.0A): {wt_clash_med:.1f}; p9c_052: {p9_clash_med:.1f}",
                f"- WT median sidechain overclose (<2.2A): {wt_overclose_med:.1f}; p9c_052: {p9_overclose_med:.1f}",
                f"- Hard-fit warning flag: {'YES' if strain_flag else 'NO'}",
                "",
                "## E. Overall evidence-based classification",
                f"- **{final_label}**",
                "- Interpretation is conservative and based on raw AF3 metrics + pose/interface reproducibility + strain checks.",
                "",
                "## Notes",
                "- No custom aggregate stability score used for final decision.",
                "- Local interface evidence emphasized over global-only ranking.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # concise strain interpretation
    (outdir / "strain_interpretation.txt").write_text(
        "\n".join(
            [
                "Strain / forced-fit brief interpretation",
                "======================================",
                f"WT median clashes(<2.0A): {wt_clash_med:.2f}",
                f"p9c_052 median clashes(<2.0A): {p9_clash_med:.2f}",
                f"WT median sidechain overclose(<2.2A): {wt_overclose_med:.2f}",
                f"p9c_052 median sidechain overclose(<2.2A): {p9_overclose_med:.2f}",
                f"Flagged potential hard-fit artifact: {'YES' if strain_flag else 'NO'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Zip package
    zip_path = outdir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(outdir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(outdir.parent))

    print(f"Wrote: {outdir / 'raw_confidence_per_model.csv'}")
    print(f"Wrote: {outdir / 'pose_metrics.csv'}")
    print(f"Wrote: {outdir / 'pose_cluster_assignments.csv'}")
    print(f"Wrote: {outdir / 'interface_contacts_long.csv'}")
    print(f"Wrote: {outdir / 'interface_contact_consensus_WT.csv'}")
    print(f"Wrote: {outdir / 'interface_contact_consensus_p9c052.csv'}")
    print(f"Wrote: {outdir / 'per_CDR_contact_summary.csv'}")
    print(f"Wrote: {outdir / 'local_interface_confidence.csv'}")
    print(f"Wrote: {outdir / 'strain_checks.csv'}")
    print(f"Wrote: {outdir / 'head2head_summary.md'}")
    print(f"Packaged: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
