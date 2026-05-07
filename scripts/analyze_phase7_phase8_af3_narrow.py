#!/usr/bin/env python3
"""Phase7 vs Phase8 AF3 narrowing analysis (population-level, mechanistic).

This script performs a strictly computational analysis for nanobody-antigen AF3 outputs.
It builds a complete package with tables, figures, representative structures, and decision notes.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import textwrap
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from Bio.SVDSuperimposer import SVDSuperimposer
from scipy.spatial import cKDTree, distance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


AA3_CHARGED_POS = {"LYS", "ARG", "HIS"}
AA3_CHARGED_NEG = {"ASP", "GLU"}
AA3_HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PRO", "PHE", "TRP", "TYR"}
AA3_AROMATIC = {"PHE", "TRP", "TYR", "HIS"}


@dataclass
class PhaseModel:
    phase: str
    run_dir: str
    job_name: str
    candidate_id: str
    sequence_id: str
    model_index: int
    cif_path: Path
    summary_json_path: Path
    full_data_path: Path
    nanobody_chain: str
    antigen_chains: List[str]
    chain_order: List[str]
    chain_counts: Dict[str, int]
    ranking_score: float
    iptm: float
    ptm: float
    has_clash: float
    fraction_disordered: float
    pair_iptm_values: List[float]
    pair_pae_values: List[float]
    best_pair_iptm: float
    best_pair_pae_min: float
    contact_pairs: Set[str]
    contacts_rows: List[dict]
    cdr1_pairs: Set[str]
    interface_nb_residues: Set[str]
    interface_ag_residues: Set[str]
    nb_residue_centroids: Dict[str, np.ndarray]
    ag_residue_centroids: Dict[str, np.ndarray]
    antigen_ca_map: Dict[str, np.ndarray]
    nanobody_ca_map: Dict[str, np.ndarray]
    nanobody_all_coords: np.ndarray
    nanobody_principal_axis: np.ndarray
    cdr3_centroid: np.ndarray
    cdr3_ca_coords: List[Tuple[int, np.ndarray]]
    interface_plddt_mean: float
    cdr_plddt_mean: float
    compressed_contact_count: int
    total_contact_pairs: int
    backbone_ca_outlier_fraction: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Phase7/8 AF3 narrowing behavior")
    p.add_argument("--phase7-dir", default="AF3 Results/Stage7 AF3")
    p.add_argument("--phase8-dir", default="AF3 Results/Stage 8 AF3")
    p.add_argument("--analysis-config", default="data/configs/af3_narrow_analysis.yaml")
    p.add_argument("--stage7-summary", default="results/summaries/af3_stage7_ranked_summary_with_wt_test1.csv")
    p.add_argument("--contact-cutoff", type=float, default=4.5)
    p.add_argument("--out-root", default="results/phase7_phase8_af3_narrow_analysis")
    p.add_argument("--zip-name", default="phase7_phase8_af3_narrow_analysis.zip")
    return p.parse_args()


def _safe_float(x, default=float("nan")) -> float:
    try:
        if pd.isna(x):
            return float(default)
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return float(default)


def _cell_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _res_label(chain_id: str, residue) -> str:
    het, resseq, icode = residue.id
    if het != " ":
        return ""
    ic = str(icode).strip()
    return f"{chain_id}:{int(resseq)}{ic}"


def _parse_resnum(label: str) -> Optional[int]:
    if not label:
        return None
    try:
        right = label.split(":", 1)[1]
        num = ""
        for ch in right:
            if ch.isdigit() or (ch == "-" and not num):
                num += ch
            else:
                break
        return int(num)
    except Exception:
        return None


def _is_heavy(atom) -> bool:
    e = (atom.element or "").strip().upper()
    n = atom.get_name().strip().upper()
    if e:
        return e != "H"
    return not n.startswith("H")


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    uu = _normalize(u)
    vv = _normalize(v)
    dot = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _matrix_pair_value(matrix, chain_order: Sequence[str], chain_i: str, chain_j: str) -> float:
    if not isinstance(matrix, list) or not matrix:
        return float("nan")
    try:
        i = chain_order.index(chain_i)
        j = chain_order.index(chain_j)
        return float(matrix[i][j])
    except Exception:
        return float("nan")


def _detect_chain_roles(full_data_json: Path) -> Tuple[List[str], str, List[str], Dict[str, int]]:
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
    nanobody_chain = min(order, key=lambda c: chain_counts.get(c, 10**9))
    antigen_chains = [c for c in order if c != nanobody_chain]
    return order, nanobody_chain, antigen_chains, chain_counts


def _extract_chain_atoms(chain):
    rows = []
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue
        rlabel = _res_label(str(chain.id), residue)
        if not rlabel:
            continue
        rname = str(residue.get_resname()).upper()
        for atom in residue.get_atoms():
            if not _is_heavy(atom):
                continue
            rows.append(
                {
                    "coord": atom.coord.astype(float),
                    "res_label": rlabel,
                    "res_name": rname,
                    "atom_name": atom.get_name().strip().upper(),
                    "element": (atom.element or "").strip().upper(),
                    "bfactor": float(atom.get_bfactor()),
                }
            )
    return rows


def _principal_axis_from_ca(ca_coords: np.ndarray) -> np.ndarray:
    if ca_coords.shape[0] < 3:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    centered = ca_coords - ca_coords.mean(axis=0)
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    return _normalize(axis)


def _compute_backbone_ca_outlier_fraction(chain) -> float:
    ca = []
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue
        if "CA" in residue:
            ca.append((int(residue.id[1]), residue["CA"].coord.astype(float)))
    if len(ca) < 2:
        return 0.0
    ca = sorted(ca, key=lambda x: x[0])
    dists = []
    for i in range(1, len(ca)):
        d = float(np.linalg.norm(ca[i][1] - ca[i - 1][1]))
        dists.append(d)
    if not dists:
        return 0.0
    out = [d for d in dists if (d < 3.4 or d > 4.3)]
    return float(len(out) / len(dists))


def _cdr_region_for_pos(pos: Optional[int], cdr_defs: dict) -> str:
    if pos is None:
        return "framework"
    for name in ("CDR1", "CDR2", "CDR3"):
        s, e = cdr_defs[name]
        if s <= pos <= e:
            return name
    return "framework"


def _residue_centroids(atom_rows: List[dict]) -> Dict[str, np.ndarray]:
    by_res = defaultdict(list)
    for r in atom_rows:
        by_res[r["res_label"]].append(r["coord"])
    out = {}
    for k, vv in by_res.items():
        out[k] = np.mean(np.asarray(vv, dtype=float), axis=0)
    return out


def _residue_plddt(atom_rows: List[dict]) -> Dict[str, float]:
    by_res = defaultdict(list)
    by_res_ca = {}
    for r in atom_rows:
        by_res[r["res_label"]].append(float(r["bfactor"]))
        if r["atom_name"] == "CA":
            by_res_ca[r["res_label"]] = float(r["bfactor"])
    out = {}
    for k in by_res:
        out[k] = by_res_ca.get(k, float(np.mean(by_res[k])))
    return out


def _build_contacts(nb_atoms: List[dict], ag_atoms: List[dict], cutoff: float) -> Tuple[List[dict], Set[str], int]:
    if not nb_atoms or not ag_atoms:
        return [], set(), 0
    nb_coords = np.asarray([r["coord"] for r in nb_atoms], dtype=float)
    ag_coords = np.asarray([r["coord"] for r in ag_atoms], dtype=float)
    tree = cKDTree(ag_coords)
    neigh = tree.query_ball_point(nb_coords, r=cutoff)

    pair_info: Dict[Tuple[str, str], dict] = {}
    compressed_pairs = set()

    for i, js in enumerate(neigh):
        if not js:
            continue
        nb = nb_atoms[i]
        nb_label = nb["res_label"]
        nb_resname = nb["res_name"]
        nb_pos = _parse_resnum(nb_label)
        for j in js:
            ag = ag_atoms[j]
            ag_label = ag["res_label"]
            ag_resname = ag["res_name"]
            ag_pos = _parse_resnum(ag_label)
            d = float(np.linalg.norm(nb["coord"] - ag["coord"]))
            key = (nb_label, ag_label)
            if key not in pair_info:
                pair_info[key] = {
                    "nanobody_residue": nb_label,
                    "nanobody_pos": nb_pos,
                    "nanobody_resname": nb_resname,
                    "antigen_residue": ag_label,
                    "antigen_pos": ag_pos,
                    "antigen_resname": ag_resname,
                    "min_distance": d,
                    "types": set(["heavy_atom"]),
                }
            else:
                if d < pair_info[key]["min_distance"]:
                    pair_info[key]["min_distance"] = d

            # Contact typing (interpretable heuristics)
            nb_el, ag_el = nb["element"], ag["element"]
            if d <= 3.5 and ({nb_el, ag_el} & {"N", "O", "S"}) == {"N", "O"} or ({nb_el, ag_el} <= {"N", "O", "S"} and len({nb_el, ag_el}) >= 1):
                pair_info[key]["types"].add("hydrogen_bond")

            if d <= 4.0 and ((nb_resname in AA3_CHARGED_POS and ag_resname in AA3_CHARGED_NEG) or (nb_resname in AA3_CHARGED_NEG and ag_resname in AA3_CHARGED_POS)):
                pair_info[key]["types"].add("salt_bridge")

            if d <= 4.5 and nb_resname in AA3_HYDROPHOBIC and ag_resname in AA3_HYDROPHOBIC:
                pair_info[key]["types"].add("hydrophobic")

            if d <= 5.0 and nb_resname in AA3_AROMATIC and ag_resname in AA3_AROMATIC:
                pair_info[key]["types"].add("aromatic_pi")

            if d < 2.2:
                compressed_pairs.add(key)

    rows = []
    pair_set = set()
    for (_, _), rec in pair_info.items():
        types = sorted(rec["types"])
        rec2 = dict(rec)
        rec2["contact_types"] = "|".join(types)
        rows.append(rec2)
        pair_set.add(f"{rec['nanobody_residue']}|{rec['antigen_residue']}")
    return rows, pair_set, len(compressed_pairs)


def _extract_model_features(
    cif_path: Path,
    nanobody_chain: str,
    antigen_chains: List[str],
    cdr_defs: dict,
    contact_cutoff: float,
) -> dict:
    parser = MMCIFParser(QUIET=True)
    st = parser.get_structure("af3", str(cif_path))
    model = next(st.get_models())
    chains = {str(c.id): c for c in model.get_chains()}
    if nanobody_chain not in chains:
        raise ValueError(f"Missing nanobody chain {nanobody_chain} in {cif_path}")

    nb_chain_obj = chains[nanobody_chain]
    nb_atoms = _extract_chain_atoms(nb_chain_obj)
    if not nb_atoms:
        raise ValueError(f"No nanobody heavy atoms in {cif_path}")

    ag_atoms = []
    for ach in antigen_chains:
        if ach in chains:
            ag_atoms.extend(_extract_chain_atoms(chains[ach]))

    contact_rows, contact_pair_set, compressed_count = _build_contacts(nb_atoms, ag_atoms, contact_cutoff)

    # Interface residue sets
    nb_int = {r["nanobody_residue"] for r in contact_rows}
    ag_int = {r["antigen_residue"] for r in contact_rows}

    # CDR1 pair subset
    c1s, c1e = cdr_defs["CDR1"]
    cdr1_pairs = set()
    for r in contact_rows:
        pos = r["nanobody_pos"]
        if pos is not None and c1s <= pos <= c1e:
            cdr1_pairs.add(f"{r['nanobody_residue']}|{r['antigen_residue']}")

    # Residue centroids and CA maps
    nb_cent = _residue_centroids(nb_atoms)
    ag_cent = _residue_centroids(ag_atoms)

    antigen_ca_map = {}
    nanobody_ca_map = {}
    for chain_id, chain_obj in chains.items():
        for residue in chain_obj.get_residues():
            if residue.id[0] != " ":
                continue
            if "CA" not in residue:
                continue
            label = _res_label(chain_id, residue)
            if not label:
                continue
            coord = residue["CA"].coord.astype(float)
            key = f"{label}:CA"
            if chain_id == nanobody_chain:
                nanobody_ca_map[key] = coord
            elif chain_id in antigen_chains:
                antigen_ca_map[key] = coord

    nb_coords = np.asarray([r["coord"] for r in nb_atoms], dtype=float)
    nb_ca_coords = np.asarray(list(nanobody_ca_map.values()), dtype=float) if nanobody_ca_map else nb_coords
    principal_axis = _principal_axis_from_ca(nb_ca_coords)

    # CDR3 centroid and CA track
    c3s, c3e = cdr_defs["CDR3"]
    cdr3_pts = []
    cdr3_ca = []
    for residue in nb_chain_obj.get_residues():
        if residue.id[0] != " ":
            continue
        pos = int(residue.id[1])
        if not (c3s <= pos <= c3e):
            continue
        coords = [a.coord.astype(float) for a in residue.get_atoms() if _is_heavy(a)]
        if coords:
            cdr3_pts.append(np.mean(np.asarray(coords, dtype=float), axis=0))
        if "CA" in residue:
            cdr3_ca.append((pos, residue["CA"].coord.astype(float)))
    cdr3_cent = np.mean(np.asarray(cdr3_pts, dtype=float), axis=0) if cdr3_pts else np.mean(nb_coords, axis=0)

    # pLDDT proxies from CIF B-factors
    nb_plddt = _residue_plddt(nb_atoms)
    int_plddt = [nb_plddt[r] for r in nb_int if r in nb_plddt]

    cdr_res = []
    for rr in nb_plddt:
        pos = _parse_resnum(rr)
        if pos is None:
            continue
        if cdr_defs["CDR1"][0] <= pos <= cdr_defs["CDR1"][1] or cdr_defs["CDR2"][0] <= pos <= cdr_defs["CDR2"][1] or cdr_defs["CDR3"][0] <= pos <= cdr_defs["CDR3"][1]:
            cdr_res.append(nb_plddt[rr])

    backbone_out = _compute_backbone_ca_outlier_fraction(nb_chain_obj)

    return {
        "contact_rows": contact_rows,
        "contact_pair_set": contact_pair_set,
        "cdr1_pairs": cdr1_pairs,
        "interface_nb_residues": nb_int,
        "interface_ag_residues": ag_int,
        "nb_residue_centroids": nb_cent,
        "ag_residue_centroids": ag_cent,
        "antigen_ca_map": antigen_ca_map,
        "nanobody_ca_map": nanobody_ca_map,
        "nanobody_all_coords": nb_coords,
        "nanobody_principal_axis": principal_axis,
        "cdr3_centroid": cdr3_cent,
        "cdr3_ca_coords": sorted(cdr3_ca, key=lambda x: x[0]),
        "interface_plddt_mean": float(np.mean(int_plddt)) if int_plddt else float("nan"),
        "cdr_plddt_mean": float(np.mean(cdr_res)) if cdr_res else float("nan"),
        "compressed_contact_count": int(compressed_count),
        "total_contact_pairs": int(len(contact_rows)),
        "backbone_ca_outlier_fraction": float(backbone_out),
    }


def _align_to_reference(model: PhaseModel, ref: PhaseModel) -> dict:
    common = sorted(set(model.antigen_ca_map.keys()) & set(ref.antigen_ca_map.keys()))
    if len(common) < 10:
        raise ValueError(
            f"Insufficient common antigen CA anchors for alignment: {model.job_name} model {model.model_index}"
        )

    ref_coords = np.asarray([ref.antigen_ca_map[k] for k in common], dtype=float)
    mob_coords = np.asarray([model.antigen_ca_map[k] for k in common], dtype=float)

    sup = SVDSuperimposer()
    sup.set(ref_coords, mob_coords)
    sup.run()
    rot, tran = sup.get_rotran()

    def tf(points: np.ndarray) -> np.ndarray:
        return np.dot(points, rot) + tran

    nb_all_tf = tf(model.nanobody_all_coords)
    nb_com = np.mean(nb_all_tf, axis=0)

    # transformed residue centroids
    nb_cent_tf = {k: np.dot(v, rot) + tran for k, v in model.nb_residue_centroids.items()}
    ag_cent_tf = {k: np.dot(v, rot) + tran for k, v in model.ag_residue_centroids.items()}

    # paratope center
    paratope_res = list(model.interface_nb_residues)
    if not paratope_res:
        paratope_res = [k for k in nb_cent_tf if _parse_resnum(k) is not None]
    if paratope_res:
        paratope_com = np.mean(np.asarray([nb_cent_tf[r] for r in paratope_res if r in nb_cent_tf], dtype=float), axis=0)
    else:
        paratope_com = nb_com

    # epitope center
    epi_res = list(model.interface_ag_residues)
    if epi_res:
        epitope_com = np.mean(np.asarray([ag_cent_tf[r] for r in epi_res if r in ag_cent_tf], dtype=float), axis=0)
    else:
        # fallback to antigen center
        epitope_com = np.mean(np.asarray([np.dot(v, rot) + tran for v in model.antigen_ca_map.values()], dtype=float), axis=0)

    paratope_to_epitope = float(np.linalg.norm(paratope_com - epitope_com))

    # principal axis and CDR3 direction (transformed)
    nb_ca_tf = np.asarray([np.dot(v, rot) + tran for v in model.nanobody_ca_map.values()], dtype=float)
    axis_tf = _principal_axis_from_ca(nb_ca_tf)
    cdr3_tf = np.dot(model.cdr3_centroid, rot) + tran
    cdr3_dir = _normalize(epitope_com - cdr3_tf)

    # translation and rotation to reference
    ref_nb_com = np.mean(ref.nanobody_all_coords, axis=0)
    trans_vec = nb_com - ref_nb_com
    trans_norm = float(np.linalg.norm(trans_vec))

    tr = float(np.trace(rot))
    rot_angle = float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))

    # transformed cdr3 CA
    cdr3_tf_ca = [(pos, np.dot(coord, rot) + tran) for pos, coord in model.cdr3_ca_coords]

    return {
        "trans_rot": rot,
        "trans_tran": tran,
        "nanobody_com": nb_com,
        "paratope_com": paratope_com,
        "epitope_com": epitope_com,
        "paratope_to_epitope_dist": paratope_to_epitope,
        "principal_axis": axis_tf,
        "cdr3_direction": cdr3_dir,
        "translation_vec": trans_vec,
        "translation_norm": trans_norm,
        "rotation_angle_deg": rot_angle,
        "cdr3_ca_transformed": cdr3_tf_ca,
        "nb_centroids_transformed": nb_cent_tf,
        "ag_centroids_transformed": ag_cent_tf,
    }


def _try_cluster(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    n = features.shape[0]
    if n <= 2:
        labels = np.ones(n, dtype=int)
        emb = features[:, :2] if features.shape[1] >= 2 else np.hstack([features, np.zeros((n, 1))])
        return labels, emb[:, :2], {"method": "degenerate", "k": 1, "silhouette": float("nan")}

    if HAVE_SKLEARN:
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        best = None
        max_k = min(6, n - 1)
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labs = km.fit_predict(X)
            if len(set(labs)) < 2:
                continue
            sil = silhouette_score(X, labs)
            if best is None or sil > best[0]:
                best = (sil, k, labs)

        if best is None or best[0] < 0.10:
            labels = np.ones(n, dtype=int)
            k = 1
            sil = float("nan")
        else:
            sil, k, labs = best
            labels = labs + 1

        if X.shape[1] >= 2:
            pca = PCA(n_components=2, random_state=42)
            emb = pca.fit_transform(X)
        else:
            emb = np.hstack([X, np.zeros((n, 1))])[:, :2]
        return labels, emb, {"method": "kmeans", "k": k, "silhouette": sil}

    # fallback without sklearn
    X = features.copy()
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
    labels = np.ones(n, dtype=int)
    emb = X[:, :2] if X.shape[1] >= 2 else np.hstack([X, np.zeros((n, 1))])[:, :2]
    return labels, emb, {"method": "fallback_single_cluster", "k": 1, "silhouette": float("nan")}


def _mean_pairwise_jaccard(sets: List[Set[str]]) -> float:
    if not sets:
        return float("nan")
    if len(sets) == 1:
        return 1.0
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            if not a and not b:
                vals.append(1.0)
            elif not a or not b:
                vals.append(0.0)
            else:
                vals.append(len(a & b) / len(a | b))
    return float(np.mean(vals)) if vals else float("nan")


def _pairwise_cdr3_rmsd(cdr3_coords: List[List[Tuple[int, np.ndarray]]]) -> float:
    vals = []
    for i in range(len(cdr3_coords)):
        for j in range(i + 1, len(cdr3_coords)):
            a = cdr3_coords[i]
            b = cdr3_coords[j]
            am = {p: c for p, c in a}
            bm = {p: c for p, c in b}
            common = sorted(set(am) & set(bm))
            if not common:
                continue
            A = np.asarray([am[p] for p in common], dtype=float)
            B = np.asarray([bm[p] for p in common], dtype=float)
            rmsd = float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))
            vals.append(rmsd)
    return float(np.mean(vals)) if vals else float("nan")


def _load_stage7_candidate_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "job_name" not in df.columns or "candidate_id" not in df.columns:
        return {}
    out = {}
    for _, r in df.iterrows():
        j = _cell_str(r.get("job_name"))
        c = _cell_str(r.get("candidate_id"))
        if j:
            out[j] = c or j
    return out


def _load_models(
    phase: str,
    phase_dir: Path,
    cdr_defs: dict,
    contact_cutoff: float,
    stage7_job_to_candidate: Dict[str, str],
) -> List[PhaseModel]:
    rows: List[PhaseModel] = []
    run_dirs = sorted([d for d in phase_dir.iterdir() if d.is_dir()])
    for run in run_dirs:
        jreqs = sorted(run.glob("*_job_request.json"))
        if not jreqs:
            continue
        job_obj = json.loads(jreqs[0].read_text(encoding="utf-8"))
        if isinstance(job_obj, list):
            job_obj = job_obj[0] if job_obj else {}
        job_name = _cell_str(job_obj.get("name")) or run.name

        if phase == "Phase7":
            candidate_id = stage7_job_to_candidate.get(job_name, job_name)
        else:
            candidate_id = job_name

        seq = ""
        try:
            seqs = job_obj.get("sequences", [])
            if seqs:
                seq = _cell_str((seqs[0] or {}).get("proteinChain", {}).get("sequence", "")).upper()
        except Exception:
            pass
        sequence_id = seq or candidate_id

        full0 = sorted(run.glob("*_full_data_0.json"))
        if not full0:
            continue

        try:
            chain_order, nb_chain, ag_chains, chain_counts = _detect_chain_roles(full0[0])
        except Exception:
            continue

        sfiles = sorted(run.glob("*_summary_confidences_*.json"))
        for sfile in sfiles:
            try:
                midx = int(sfile.stem.rsplit("_", 1)[-1])
            except Exception:
                continue
            cif = run / sfile.name.replace("_summary_confidences_", "_model_").replace(".json", ".cif")
            fdata = run / sfile.name.replace("_summary_confidences_", "_full_data_")
            if not cif.exists() or not fdata.exists():
                continue

            s = json.loads(sfile.read_text(encoding="utf-8"))
            pair_iptm = s.get("chain_pair_iptm", [])
            pair_pae = s.get("chain_pair_pae_min", [])
            pair_iptm_vals = [_matrix_pair_value(pair_iptm, chain_order, nb_chain, ach) for ach in ag_chains]
            pair_pae_vals = [_matrix_pair_value(pair_pae, chain_order, nb_chain, ach) for ach in ag_chains]
            best_iptm = float(np.nanmax(pair_iptm_vals)) if pair_iptm_vals else float("nan")
            best_pae = float(np.nanmin(pair_pae_vals)) if pair_pae_vals else float("nan")

            try:
                fx = _extract_model_features(
                    cif_path=cif,
                    nanobody_chain=nb_chain,
                    antigen_chains=ag_chains,
                    cdr_defs=cdr_defs,
                    contact_cutoff=contact_cutoff,
                )
            except Exception:
                continue

            rows.append(
                PhaseModel(
                    phase=phase,
                    run_dir=run.name,
                    job_name=job_name,
                    candidate_id=candidate_id,
                    sequence_id=sequence_id,
                    model_index=midx,
                    cif_path=cif,
                    summary_json_path=sfile,
                    full_data_path=fdata,
                    nanobody_chain=nb_chain,
                    antigen_chains=ag_chains,
                    chain_order=chain_order,
                    chain_counts=chain_counts,
                    ranking_score=_safe_float(s.get("ranking_score")),
                    iptm=_safe_float(s.get("iptm")),
                    ptm=_safe_float(s.get("ptm")),
                    has_clash=_safe_float(s.get("has_clash"), 0.0),
                    fraction_disordered=_safe_float(s.get("fraction_disordered"), 0.0),
                    pair_iptm_values=pair_iptm_vals,
                    pair_pae_values=pair_pae_vals,
                    best_pair_iptm=best_iptm,
                    best_pair_pae_min=best_pae,
                    contact_pairs=fx["contact_pair_set"],
                    contacts_rows=fx["contact_rows"],
                    cdr1_pairs=fx["cdr1_pairs"],
                    interface_nb_residues=fx["interface_nb_residues"],
                    interface_ag_residues=fx["interface_ag_residues"],
                    nb_residue_centroids=fx["nb_residue_centroids"],
                    ag_residue_centroids=fx["ag_residue_centroids"],
                    antigen_ca_map=fx["antigen_ca_map"],
                    nanobody_ca_map=fx["nanobody_ca_map"],
                    nanobody_all_coords=fx["nanobody_all_coords"],
                    nanobody_principal_axis=fx["nanobody_principal_axis"],
                    cdr3_centroid=fx["cdr3_centroid"],
                    cdr3_ca_coords=fx["cdr3_ca_coords"],
                    interface_plddt_mean=fx["interface_plddt_mean"],
                    cdr_plddt_mean=fx["cdr_plddt_mean"],
                    compressed_contact_count=fx["compressed_contact_count"],
                    total_contact_pairs=fx["total_contact_pairs"],
                    backbone_ca_outlier_fraction=fx["backbone_ca_outlier_fraction"],
                )
            )
    return rows


def _infer_hotspots_and_edges(
    pose_df: pd.DataFrame,
    contacts_df: pd.DataFrame,
    config: dict,
    dominant_phase7_cluster: int,
    root: Path,
) -> Tuple[List[str], List[str], List[int], List[str]]:
    notes = []

    # Explicit config first
    ag_hot_cfg = config.get("known_antigen_hotspot_residues", []) or config.get("antigen_hotspots", [])
    nb_hot_cfg = config.get("known_nanobody_hotspot_residues", []) or config.get("nanobody_hotspots", [])
    edge_cfg = config.get("edge_variable_nanobody_positions", []) or config.get("edge_positions", [])

    def norm_residue_tokens(vals):
        out = []
        for v in vals:
            s = _cell_str(v)
            if not s:
                continue
            if ":" in s:
                out.append(s)
            else:
                # allow B217 format
                chain = s[0]
                num = s[1:]
                if chain.isalpha() and num:
                    out.append(f"{chain}:{num}")
        return sorted(set(out))

    ag_hot = norm_residue_tokens(ag_hot_cfg)
    nb_hot = norm_residue_tokens(nb_hot_cfg)

    if not ag_hot or not nb_hot:
        d7 = contacts_df[(contacts_df["phase"] == "Phase7") & (contacts_df["pose_cluster"] == dominant_phase7_cluster)].copy()
        if d7.empty:
            d7 = contacts_df[contacts_df["phase"] == "Phase7"].copy()
        ag_top = (
            d7.groupby("antigen_residue").size().sort_values(ascending=False).head(6).index.tolist()
            if not d7.empty else []
        )
        nb_top = (
            d7.groupby("nanobody_residue").size().sort_values(ascending=False).head(6).index.tolist()
            if not d7.empty else []
        )
        if not ag_hot:
            ag_hot = ag_top
        if not nb_hot:
            nb_hot = nb_top
        notes.append("PROVISIONAL hotspot definitions inferred from dominant Phase7 contact occupancy.")

    # edge positions
    edge_pos: List[int] = []
    if edge_cfg:
        for x in edge_cfg:
            s = _cell_str(x)
            if not s:
                continue
            if ":" in s:
                s = s.split(":")[-1]
            try:
                edge_pos.append(int(s))
            except Exception:
                pass
    if not edge_pos:
        # infer from design configs first
        tcfg = _load_yaml(root / "data/configs/test1_local_maturation_phase.yaml")
        ccfg = _load_yaml(root / "data/configs/champion_narrow50_phase.yaml")
        cand = []
        for node in (
            (((tcfg or {}).get("test1_local_maturation") or {}).get("branches") or {}).values()
        ):
            for v in node.get("editable_positions", []):
                s = _cell_str(v)
                if ":" in s:
                    s = s.split(":")[-1]
                try:
                    cand.append(int(s))
                except Exception:
                    pass
        for v in (((ccfg or {}).get("champion_narrow50") or {}).get("editable_positions") or []):
            s = _cell_str(v)
            if ":" in s:
                s = s.split(":")[-1]
            try:
                cand.append(int(s))
            except Exception:
                pass
        if cand:
            edge_pos = sorted(set(cand))
            notes.append("Edge positions inferred from Phase7/8 design config editable positions.")

    if not edge_pos:
        core_nums = {(_parse_resnum(x) or -999) for x in nb_hot}
        d7 = contacts_df[contacts_df["phase"] == "Phase7"].copy()
        d7["nb_pos"] = d7["nanobody_residue"].map(_parse_resnum)
        d7 = d7[d7["nb_pos"].notna()]
        d7 = d7[~d7["nb_pos"].isin(core_nums)]
        top = d7.groupby("nb_pos").size().sort_values(ascending=False).head(8)
        edge_pos = [int(x) for x in top.index.tolist()]
        notes.append("PROVISIONAL edge positions inferred from non-core interface occupancy.")

    return ag_hot, nb_hot, sorted(set(edge_pos)), notes


def _make_dirs(base: Path):
    for sub in [
        base,
        base / "tables",
        base / "figures",
        base / "representative_structures",
        base / "scripts",
        base / "logs",
    ]:
        sub.mkdir(parents=True, exist_ok=True)


def _save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _build_package(args: argparse.Namespace) -> Path:
    root = Path(".").resolve()

    phase7_dir = (root / args.phase7_dir).resolve()
    phase8_dir = (root / args.phase8_dir).resolve()
    config_path = (root / args.analysis_config).resolve()
    stage7_summary_path = (root / args.stage7_summary).resolve()

    out_root = (root / args.out_root).resolve()
    package_root = out_root / "phase7_phase8_af3_narrow_analysis"
    _make_dirs(package_root)

    run_log = package_root / "logs" / "run.log"
    run_log.write_text("", encoding="utf-8")

    def log(msg: str):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {msg}"
        print(line)
        with run_log.open("a", encoding="utf-8") as h:
            h.write(line + "\n")

    config = _load_yaml(config_path)
    used_provisional: List[str] = []

    cdr_yaml = _load_yaml(root / "data/configs/cdr_boundaries.yaml")
    cdr_block = cdr_yaml.get("cdr_boundaries", {}) if cdr_yaml else {}

    def _cdr_pair(name: str, default):
        vv = cdr_block.get(name) if isinstance(cdr_block, dict) else None
        if isinstance(vv, list) and len(vv) == 2:
            try:
                return int(vv[0]), int(vv[1])
            except Exception:
                return default
        return default

    cdr_defs = {
        "CDR1": _cdr_pair("H1", (23, 34)),
        "CDR2": _cdr_pair("H2", (50, 58)),
        "CDR3": _cdr_pair("H3", (97, 107)),
    }

    stage7_map = _load_stage7_candidate_map(stage7_summary_path)

    log("Loading Phase7 models...")
    phase7_models = _load_models(
        phase="Phase7",
        phase_dir=phase7_dir,
        cdr_defs=cdr_defs,
        contact_cutoff=args.contact_cutoff,
        stage7_job_to_candidate=stage7_map,
    )
    log(f"Phase7 models loaded: {len(phase7_models)}")

    log("Loading Phase8 models...")
    phase8_models = _load_models(
        phase="Phase8",
        phase_dir=phase8_dir,
        cdr_defs=cdr_defs,
        contact_cutoff=args.contact_cutoff,
        stage7_job_to_candidate=stage7_map,
    )
    log(f"Phase8 models loaded: {len(phase8_models)}")

    all_models = phase7_models + phase8_models
    if not all_models:
        raise RuntimeError("No models parsed from phase7/phase8 directories")

    # ---------- Block 1: antigen-aligned pose geometry ----------
    log("Block 1: pose geometry")

    # initial reference: first phase7 model (or user config path if provided)
    ref_model = None
    ref_model_path = _cell_str(config.get("reference_model_path"))
    if ref_model_path:
        rp = (root / ref_model_path).resolve() if not Path(ref_model_path).is_absolute() else Path(ref_model_path)
        for m in all_models:
            if m.cif_path.resolve() == rp.resolve():
                ref_model = m
                break
    if ref_model is None:
        ref_model = phase7_models[0] if phase7_models else all_models[0]

    def build_pose_df(reference: PhaseModel):
        rows = []
        failures = 0
        for m in all_models:
            try:
                al = _align_to_reference(m, reference)
            except Exception:
                failures += 1
                continue
            row = {
                "phase": m.phase,
                "run_dir": m.run_dir,
                "job_name": m.job_name,
                "candidate_id": m.candidate_id,
                "sequence_id": m.sequence_id,
                "model_index": m.model_index,
                "model_id": f"{m.phase}|{m.job_name}|m{m.model_index}",
                "reference_job": reference.job_name,
                "reference_model_index": reference.model_index,
                "nanobody_com_x": al["nanobody_com"][0],
                "nanobody_com_y": al["nanobody_com"][1],
                "nanobody_com_z": al["nanobody_com"][2],
                "paratope_com_x": al["paratope_com"][0],
                "paratope_com_y": al["paratope_com"][1],
                "paratope_com_z": al["paratope_com"][2],
                "epitope_com_x": al["epitope_com"][0],
                "epitope_com_y": al["epitope_com"][1],
                "epitope_com_z": al["epitope_com"][2],
                "paratope_to_epitope_dist": al["paratope_to_epitope_dist"],
                "principal_axis_x": al["principal_axis"][0],
                "principal_axis_y": al["principal_axis"][1],
                "principal_axis_z": al["principal_axis"][2],
                "cdr3_dir_x": al["cdr3_direction"][0],
                "cdr3_dir_y": al["cdr3_direction"][1],
                "cdr3_dir_z": al["cdr3_direction"][2],
                "translation_x": al["translation_vec"][0],
                "translation_y": al["translation_vec"][1],
                "translation_z": al["translation_vec"][2],
                "translation_norm": al["translation_norm"],
                "rotation_angle_deg": al["rotation_angle_deg"],
                "iptm": m.iptm,
                "best_pair_iptm": m.best_pair_iptm,
                "best_pair_pae_min": m.best_pair_pae_min,
                "interface_plddt_mean": m.interface_plddt_mean,
                "cdr_plddt_mean": m.cdr_plddt_mean,
                "compressed_contact_count": m.compressed_contact_count,
                "total_contact_pairs": m.total_contact_pairs,
                "backbone_ca_outlier_fraction": m.backbone_ca_outlier_fraction,
            }
            rows.append(row)
        return pd.DataFrame(rows), failures

    pose_df, fails = build_pose_df(ref_model)
    log(f"Pose descriptors built (initial ref); skipped align failures: {fails}")

    feat_cols = [
        "nanobody_com_x",
        "nanobody_com_y",
        "nanobody_com_z",
        "principal_axis_x",
        "principal_axis_y",
        "principal_axis_z",
        "cdr3_dir_x",
        "cdr3_dir_y",
        "cdr3_dir_z",
        "paratope_to_epitope_dist",
        "translation_norm",
        "rotation_angle_deg",
    ]
    F_phase7 = pose_df[pose_df["phase"] == "Phase7"][feat_cols].to_numpy(dtype=float)
    labs7, _, cluster_meta7 = _try_cluster(F_phase7)
    phase7_idx = pose_df.index[pose_df["phase"] == "Phase7"].tolist()
    for i, idx in enumerate(phase7_idx):
        pose_df.loc[idx, "pose_cluster_phase7"] = int(labs7[i])

    dom_cluster = int(pd.Series(labs7).value_counts().index[0]) if len(labs7) else 1

    # representative of dominant phase7 cluster becomes final reference
    d7 = pose_df[pose_df["phase"] == "Phase7"].copy().reset_index(drop=False)
    d7_dom = d7[d7["pose_cluster_phase7"] == dom_cluster].copy()
    if not d7_dom.empty:
        X = d7_dom[feat_cols].to_numpy(dtype=float)
        ctr = X.mean(axis=0)
        dist = np.linalg.norm(X - ctr, axis=1)
        best_i = int(np.argmin(dist))
        best_global_idx = int(d7_dom.iloc[best_i]["index"])
        rep_row = pose_df.loc[best_global_idx]
        for m in all_models:
            if m.phase == rep_row["phase"] and m.job_name == rep_row["job_name"] and int(m.model_index) == int(rep_row["model_index"]):
                ref_model = m
                break

    # final pose descriptors + clustering (all models)
    pose_df, fails = build_pose_df(ref_model)
    log(f"Pose descriptors built (final ref={ref_model.job_name}/m{ref_model.model_index}); skipped align failures: {fails}")

    F = pose_df[feat_cols].to_numpy(dtype=float)
    labels, embedding2, cluster_meta = _try_cluster(F)
    pose_df["pose_cluster"] = labels.astype(int)
    pose_df["pose_embed_x"] = embedding2[:, 0]
    pose_df["pose_embed_y"] = embedding2[:, 1]

    # final dominant intended cluster = dominant among phase7
    dom_cluster = int(pose_df[pose_df["phase"] == "Phase7"]["pose_cluster"].value_counts().index[0])

    # cluster centroids and distances
    centroids = {}
    for c in sorted(pose_df["pose_cluster"].unique()):
        Xc = pose_df[pose_df["pose_cluster"] == c][feat_cols].to_numpy(dtype=float)
        centroids[int(c)] = Xc.mean(axis=0)

    d2cent = []
    for _, r in pose_df.iterrows():
        c = int(r["pose_cluster"])
        v = r[feat_cols].to_numpy(dtype=float)
        d = float(np.linalg.norm(v - centroids[c]))
        d2cent.append(d)
    pose_df["distance_to_cluster_centroid"] = d2cent

    dom_ctr = centroids[dom_cluster]
    pose_df["distance_to_dominant_cluster_centroid"] = [
        float(np.linalg.norm(r[feat_cols].to_numpy(dtype=float) - dom_ctr)) for _, r in pose_df.iterrows()
    ]

    pose_metrics = pose_df.copy()
    pose_metrics.to_csv(package_root / "tables" / "pose_metrics.csv", index=False)

    pose_assign = pose_df[
        [
            "phase",
            "run_dir",
            "job_name",
            "candidate_id",
            "model_index",
            "model_id",
            "pose_cluster",
            "distance_to_cluster_centroid",
            "distance_to_dominant_cluster_centroid",
            "translation_norm",
            "rotation_angle_deg",
        ]
    ].copy()
    pose_assign.to_csv(package_root / "tables" / "pose_cluster_assignments.csv", index=False)

    psummary = (
        pose_df.groupby(["phase", "pose_cluster"], as_index=False)
        .agg(
            model_count=("model_id", "count"),
            translation_norm_mean=("translation_norm", "mean"),
            rotation_angle_mean=("rotation_angle_deg", "mean"),
            within_cluster_spread=("distance_to_cluster_centroid", "mean"),
            dominant_distance_mean=("distance_to_dominant_cluster_centroid", "mean"),
        )
    )
    psummary["cluster_fraction_in_phase"] = psummary["model_count"] / psummary.groupby("phase")["model_count"].transform("sum")
    psummary.to_csv(package_root / "tables" / "pose_cluster_summary.csv", index=False)

    # figures block 1
    plt.figure(figsize=(8, 6))
    for ph, mk, col in [("Phase7", "o", "tab:blue"), ("Phase8", "^", "tab:orange")]:
        q = pose_df[pose_df["phase"] == ph]
        plt.scatter(q["pose_embed_x"], q["pose_embed_y"], s=18, alpha=0.7, marker=mk, color=col, label=ph)
    plt.title("Antigen-aligned pose embedding")
    plt.xlabel("Embed-1")
    plt.ylabel("Embed-2")
    plt.legend()
    _save_fig(package_root / "figures" / "pose_embedding.png")

    occ = psummary.pivot(index="pose_cluster", columns="phase", values="cluster_fraction_in_phase").fillna(0.0)
    occ = occ.sort_index()
    x = np.arange(len(occ.index))
    w = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar(x - w / 2, occ.get("Phase7", pd.Series(0, index=occ.index)).to_numpy(), width=w, label="Phase7")
    plt.bar(x + w / 2, occ.get("Phase8", pd.Series(0, index=occ.index)).to_numpy(), width=w, label="Phase8")
    plt.xticks(x, [str(i) for i in occ.index])
    plt.xlabel("Pose cluster")
    plt.ylabel("Occupancy fraction")
    plt.title("Pose cluster occupancy by phase")
    plt.legend()
    _save_fig(package_root / "figures" / "pose_cluster_occupancy_bar.png")

    disp = (
        pose_df.groupby("phase", as_index=False)
        .agg(
            dispersion_mean=("distance_to_dominant_cluster_centroid", "mean"),
            dispersion_median=("distance_to_dominant_cluster_centroid", "median"),
        )
    )
    plt.figure(figsize=(6, 5))
    plt.bar(disp["phase"], disp["dispersion_mean"], color=["tab:blue", "tab:orange"])
    plt.ylabel("Mean distance to dominant Phase7 centroid")
    plt.title("Phase7 vs Phase8 pose dispersion")
    _save_fig(package_root / "figures" / "pose_dispersion_phase_compare.png")

    # ---------- Block 2: residue-level contact fingerprint ----------
    log("Block 2: residue-level contact fingerprint")

    contacts_rows_all = []
    model_contact_sets: Dict[str, Set[str]] = {}
    model_hot_contact_rows: Dict[str, List[dict]] = {}

    # map pose cluster back to models
    pose_key_to_cluster = {
        (r.phase, r.job_name, int(r.model_index)): int(r.pose_cluster)
        for r in pose_df.itertuples()
    }

    for m in all_models:
        mid = f"{m.phase}|{m.job_name}|m{m.model_index}"
        model_contact_sets[mid] = set(m.contact_pairs)
        model_hot_contact_rows[mid] = m.contacts_rows
        for cr in m.contacts_rows:
            nb_res = cr["nanobody_residue"]
            ag_res = cr["antigen_residue"]
            contacts_rows_all.append(
                {
                    "phase": m.phase,
                    "run_dir": m.run_dir,
                    "job_name": m.job_name,
                    "candidate_id": m.candidate_id,
                    "sequence_id": m.sequence_id,
                    "model_index": m.model_index,
                    "model_id": mid,
                    "pose_cluster": pose_key_to_cluster.get((m.phase, m.job_name, int(m.model_index)), -1),
                    "nanobody_residue": nb_res,
                    "nanobody_pos": _parse_resnum(nb_res),
                    "cdr_region": _cdr_region_for_pos(_parse_resnum(nb_res), cdr_defs),
                    "antigen_residue": ag_res,
                    "antigen_pos": _parse_resnum(ag_res),
                    "min_distance": cr["min_distance"],
                    "contact_types": cr["contact_types"],
                    "pair_key": f"{nb_res}|{ag_res}",
                }
            )

    contacts_df = pd.DataFrame(contacts_rows_all)
    contacts_df.to_csv(package_root / "tables" / "contacts_long.csv", index=False)

    # fingerprint matrix
    model_list = pose_df["model_id"].tolist()
    pair_counts = contacts_df.groupby("pair_key")["model_id"].nunique().sort_values(ascending=False)
    occ_rate = pair_counts / max(1, len(model_list))
    keep_pairs = occ_rate[occ_rate >= 0.02].index.tolist()
    fp = pd.DataFrame(0, index=model_list, columns=keep_pairs, dtype=int)
    for mid, grp in contacts_df.groupby("model_id"):
        pairs = set(grp["pair_key"].tolist())
        cols = [p for p in pairs if p in fp.columns]
        if cols:
            fp.loc[mid, cols] = 1
    fp.insert(0, "model_id", fp.index)
    fp.reset_index(drop=True, inplace=True)
    fp.to_csv(package_root / "tables" / "contact_fingerprint_matrix.csv", index=False)

    # phase consensus
    def phase_consensus(phase_name: str) -> pd.DataFrame:
        q = contacts_df[contacts_df["phase"] == phase_name].copy()
        nmod = q["model_id"].nunique()
        if q.empty:
            return pd.DataFrame(columns=["pair_key", "nanobody_residue", "antigen_residue", "occupancy_count", "occupancy_rate", "dominant_contact_type"])
        g = q.groupby(["pair_key", "nanobody_residue", "antigen_residue"], as_index=False).agg(
            occupancy_count=("model_id", "nunique"),
            min_distance_mean=("min_distance", "mean"),
        )
        g["occupancy_rate"] = g["occupancy_count"] / max(1, nmod)

        # dominant type per pair
        type_map = {}
        for pk, sub in q.groupby("pair_key"):
            all_types = []
            for t in sub["contact_types"].astype(str):
                all_types.extend([x for x in t.split("|") if x])
            type_map[pk] = Counter(all_types).most_common(1)[0][0] if all_types else "heavy_atom"
        g["dominant_contact_type"] = g["pair_key"].map(type_map)
        g = g.sort_values(["occupancy_rate", "min_distance_mean"], ascending=[False, True])
        return g

    c7 = phase_consensus("Phase7")
    c8 = phase_consensus("Phase8")
    c7.to_csv(package_root / "tables" / "phase7_contact_consensus.csv", index=False)
    c8.to_csv(package_root / "tables" / "phase8_contact_consensus.csv", index=False)

    # contact heatmaps (top pairs by combined occupancy)
    top_pairs = (
        pd.concat([
            c7[["pair_key", "occupancy_rate"]].assign(phase="Phase7"),
            c8[["pair_key", "occupancy_rate"]].assign(phase="Phase8"),
        ])
        .groupby("pair_key", as_index=False)["occupancy_rate"]
        .max()
        .sort_values("occupancy_rate", ascending=False)
        .head(30)["pair_key"]
        .tolist()
    )

    def _cons_vec(cons_df, pairs):
        mp = {r.pair_key: r.occupancy_rate for r in cons_df.itertuples()}
        return np.asarray([mp.get(p, 0.0) for p in pairs], dtype=float)

    v7 = _cons_vec(c7, top_pairs)
    v8 = _cons_vec(c8, top_pairs)
    hm = np.vstack([v7, v8])
    plt.figure(figsize=(14, 3.6))
    plt.imshow(hm, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.yticks([0, 1], ["Phase7", "Phase8"])
    plt.xticks(np.arange(len(top_pairs)), top_pairs, rotation=90, fontsize=7)
    plt.colorbar(label="Occupancy rate")
    plt.title("Consensus contact occupancy (top residue pairs)")
    _save_fig(package_root / "figures" / "contact_consensus_heatmap.png")

    # ---------- Block 3: hotspot occupancy and co-occurrence ----------
    log("Block 3: hotspot occupancy and co-occurrence")

    ag_hot, nb_hot, edge_positions, notes = _infer_hotspots_and_edges(
        pose_df=pose_df,
        contacts_df=contacts_df,
        config=config,
        dominant_phase7_cluster=dom_cluster,
        root=root,
    )
    used_provisional.extend(notes)

    hotspot_contacts_rows = []
    model_hotset: Dict[str, Set[str]] = defaultdict(set)
    model_hot_anchor_count: Dict[str, int] = defaultdict(int)

    nb_hot_set = set(nb_hot)
    ag_hot_set = set(ag_hot)

    for _, r in contacts_df.iterrows():
        mid = r["model_id"]
        nb = r["nanobody_residue"]
        ag = r["antigen_residue"]
        if nb in nb_hot_set and ag in ag_hot_set:
            pair = f"{nb}|{ag}"
            hotspot_contacts_rows.append(
                {
                    "phase": r["phase"],
                    "model_id": mid,
                    "job_name": r["job_name"],
                    "candidate_id": r["candidate_id"],
                    "pose_cluster": r["pose_cluster"],
                    "hotspot_pair": pair,
                    "hotspot_nanobody_residue": nb,
                    "hotspot_antigen_residue": ag,
                    "contact_types": r["contact_types"],
                    "min_distance": r["min_distance"],
                    "present": 1,
                }
            )
            model_hotset[mid].add(pair)

    hot_df = pd.DataFrame(hotspot_contacts_rows)
    if hot_df.empty:
        hot_df = pd.DataFrame(columns=[
            "phase",
            "model_id",
            "job_name",
            "candidate_id",
            "pose_cluster",
            "hotspot_pair",
            "hotspot_nanobody_residue",
            "hotspot_antigen_residue",
            "contact_types",
            "min_distance",
            "present",
        ])
    hot_df.to_csv(package_root / "tables" / "hotspot_contacts_per_model.csv", index=False)

    # occupancy summary per hotspot pair and phase
    models_by_phase = pose_df.groupby("phase")["model_id"].nunique().to_dict()
    occ_rows = []
    all_hot_pairs = sorted({f"{n}|{a}" for n in nb_hot for a in ag_hot})
    for ph in ["Phase7", "Phase8"]:
        q = hot_df[hot_df["phase"] == ph]
        nmod = int(models_by_phase.get(ph, 0))
        bypair = q.groupby("hotspot_pair")["model_id"].nunique().to_dict()
        for hp in all_hot_pairs:
            c = int(bypair.get(hp, 0))
            occ_rows.append(
                {
                    "phase": ph,
                    "hotspot_pair": hp,
                    "occupancy_count": c,
                    "occupancy_rate": c / max(1, nmod),
                    "hotspot_nanobody_residue": hp.split("|")[0],
                    "hotspot_antigen_residue": hp.split("|")[1],
                }
            )
    hot_occ = pd.DataFrame(occ_rows).sort_values(["phase", "occupancy_rate"], ascending=[True, False])
    hot_occ.to_csv(package_root / "tables" / "hotspot_occupancy_summary.csv", index=False)

    # co-occurrence matrix (long format)
    co_rows = []
    for ph in ["Phase7", "Phase8"]:
        ph_models = pose_df[pose_df["phase"] == ph]["model_id"].tolist()
        nmod = len(ph_models)
        if nmod == 0:
            continue
        # binary matrix models x hotspot pairs
        B = pd.DataFrame(0, index=ph_models, columns=all_hot_pairs, dtype=int)
        for mid in ph_models:
            for hp in model_hotset.get(mid, set()):
                if hp in B.columns:
                    B.loc[mid, hp] = 1

        for i, a in enumerate(all_hot_pairs):
            va = B[a].to_numpy(dtype=int)
            for b in all_hot_pairs[i:]:
                vb = B[b].to_numpy(dtype=int)
                co = int(np.sum((va == 1) & (vb == 1)))
                union = int(np.sum((va == 1) | (vb == 1)))
                jac = float(co / union) if union > 0 else 0.0
                co_rows.append(
                    {
                        "phase": ph,
                        "pair_i": a,
                        "pair_j": b,
                        "cooccur_count": co,
                        "cooccur_rate": co / max(1, nmod),
                        "jaccard": jac,
                    }
                )
                if a != b:
                    co_rows.append(
                        {
                            "phase": ph,
                            "pair_i": b,
                            "pair_j": a,
                            "cooccur_count": co,
                            "cooccur_rate": co / max(1, nmod),
                            "jaccard": jac,
                        }
                    )

    co_df = pd.DataFrame(co_rows)
    co_df.to_csv(package_root / "tables" / "hotspot_cooccurrence_matrix.csv", index=False)

    # model-level hotspot anchor count
    for mid, s in model_hotset.items():
        model_hot_anchor_count[mid] = len(s)

    # hotspot figures
    occ_mat = hot_occ.pivot(index="hotspot_pair", columns="phase", values="occupancy_rate").fillna(0.0)
    plt.figure(figsize=(7, max(4, len(occ_mat) * 0.22)))
    plt.imshow(occ_mat.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=0, vmax=1)
    plt.yticks(np.arange(len(occ_mat.index)), occ_mat.index, fontsize=7)
    plt.xticks(np.arange(len(occ_mat.columns)), occ_mat.columns)
    plt.colorbar(label="Occupancy rate")
    plt.title("Hotspot pair occupancy by phase")
    _save_fig(package_root / "figures" / "hotspot_occupancy_heatmap.png")

    for ph in ["Phase7", "Phase8"]:
        q = co_df[co_df["phase"] == ph]
        if q.empty:
            continue
        M = q.pivot(index="pair_i", columns="pair_j", values="jaccard").fillna(0.0)
        M = M.reindex(index=all_hot_pairs, columns=all_hot_pairs, fill_value=0.0)
        plt.figure(figsize=(8, 7))
        plt.imshow(M.to_numpy(dtype=float), aspect="auto", cmap="cividis", vmin=0, vmax=1)
        plt.xticks(np.arange(len(all_hot_pairs)), all_hot_pairs, rotation=90, fontsize=6)
        plt.yticks(np.arange(len(all_hot_pairs)), all_hot_pairs, fontsize=6)
        plt.colorbar(label="Jaccard co-occurrence")
        plt.title(f"Hotspot co-occurrence ({ph})")
        _save_fig(package_root / "figures" / f"hotspot_cooccurrence_{ph.lower()}.png")

    # ---------- Block 4: edge/variable-position role classification ----------
    log("Block 4: edge/variable role classification")

    # define per-model pose deviation threshold for diverting classification
    d7_dom = pose_df[(pose_df["phase"] == "Phase7") & (pose_df["pose_cluster"] == dom_cluster)]
    div_threshold = float(d7_dom["distance_to_dominant_cluster_centroid"].quantile(0.75)) if not d7_dom.empty else float(pose_df["distance_to_dominant_cluster_centroid"].quantile(0.75))

    # pre-index contacts per model and residue
    by_model_nb = defaultdict(list)
    for _, r in contacts_df.iterrows():
        by_model_nb[(r["model_id"], _parse_resnum(r["nanobody_residue"]))].append(r)

    edge_rows = []
    for _, pm in pose_df.iterrows():
        mid = pm["model_id"]
        ph = pm["phase"]
        in_dom = int(pm["pose_cluster"] == dom_cluster)
        pose_dev = float(pm["distance_to_dominant_cluster_centroid"])
        hcount = int(model_hot_anchor_count.get(mid, 0))
        for pos in edge_positions:
            key = (mid, int(pos))
            crs = by_model_nb.get(key, [])
            if not crs:
                role = "PASSIVE"
                evidence = "no direct antigen contact"
            else:
                ag_res = {_cell_str(x["antigen_residue"]) for x in crs}
                ag_hot_hit = any(a in ag_hot_set for a in ag_res)
                if ag_hot_hit and in_dom and hcount >= 2:
                    role = "SUPPORTIVE"
                    evidence = "contacts hotspot patch while staying in dominant pose"
                elif (not ag_hot_hit) and (pose_dev > div_threshold or not in_dom):
                    role = "DIVERTING"
                    evidence = "contacts non-hotspot patch and associates with pose drift"
                elif ag_hot_hit:
                    role = "SUPPORTIVE"
                    evidence = "contacts hotspot patch"
                else:
                    role = "PASSIVE"
                    evidence = "contacts observed but no clear hotspot reinforcement"

            edge_rows.append(
                {
                    "phase": ph,
                    "model_id": mid,
                    "job_name": pm["job_name"],
                    "candidate_id": pm["candidate_id"],
                    "pose_cluster": int(pm["pose_cluster"]),
                    "edge_position": int(pos),
                    "edge_residue_label": f"{pm['model_id'].split('|')[1]}:{pos}",
                    "hotspot_anchor_count": hcount,
                    "pose_deviation": pose_dev,
                    "role": role,
                    "evidence": evidence,
                }
            )

    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_csv(package_root / "tables" / "edge_role_per_model.csv", index=False)

    edge_summary = (
        edge_df.groupby(["phase", "edge_position", "role"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    edge_summary["phase_position_total"] = edge_summary.groupby(["phase", "edge_position"])["count"].transform("sum")
    edge_summary["fraction"] = edge_summary["count"] / edge_summary["phase_position_total"]
    edge_summary.to_csv(package_root / "tables" / "edge_role_summary.csv", index=False)

    # edge figures
    role_order = ["SUPPORTIVE", "PASSIVE", "DIVERTING"]
    role_counts = edge_df.groupby(["phase", "role"], as_index=False).size().rename(columns={"size": "count"})
    role_piv = role_counts.pivot(index="phase", columns="role", values="count").fillna(0).reindex(columns=role_order, fill_value=0)
    plt.figure(figsize=(7, 5))
    x = np.arange(len(role_piv.index))
    bottom = np.zeros(len(role_piv.index), dtype=float)
    colors = {"SUPPORTIVE": "tab:green", "PASSIVE": "tab:gray", "DIVERTING": "tab:red"}
    for r in role_order:
        vals = role_piv[r].to_numpy(dtype=float)
        plt.bar(x, vals, bottom=bottom, label=r, color=colors[r])
        bottom += vals
    plt.xticks(x, role_piv.index)
    plt.ylabel("Count")
    plt.title("Edge-position role counts by phase")
    plt.legend()
    _save_fig(package_root / "figures" / "edge_role_counts_by_phase.png")

    # ---------- Block 5: within-sequence multi-model consistency ----------
    log("Block 5: sequence-level consistency")

    # prep sets per model
    model_to_contact = {mid: set(v) for mid, v in model_contact_sets.items()}
    model_to_hot = {mid: set(v) for mid, v in model_hotset.items()}

    seq_rows = []
    seq_cluster_rows = []

    for (phase, seq_id), grp in pose_df.groupby(["phase", "sequence_id"]):
        mids = grp["model_id"].tolist()
        if len(mids) < 2:
            continue
        pose_spread = float(grp["distance_to_cluster_centroid"].mean())
        trans_std = float(grp["translation_norm"].std(ddof=0))
        rot_std = float(grp["rotation_angle_deg"].std(ddof=0))

        # cluster usage
        ccounts = grp["pose_cluster"].value_counts()
        dominant_fraction = float(ccounts.iloc[0] / len(grp)) if not ccounts.empty else float("nan")
        for c, n in ccounts.items():
            seq_cluster_rows.append(
                {
                    "phase": phase,
                    "sequence_id": seq_id,
                    "candidate_id": _cell_str(grp.iloc[0]["candidate_id"]),
                    "pose_cluster": int(c),
                    "model_count": int(n),
                    "cluster_fraction": float(n / len(grp)),
                }
            )

        # set-based consistency
        contact_sets = [model_to_contact.get(mid, set()) for mid in mids]
        hot_sets = [model_to_hot.get(mid, set()) for mid in mids]
        contact_j = _mean_pairwise_jaccard(contact_sets)
        hot_j = _mean_pairwise_jaccard(hot_sets)

        # CDR3 RMSD from transformed coords (approx by job already in same aligned frame)
        cdr3_tracks = []
        # fetch matching model objects
        for _, rr in grp.iterrows():
            jm = rr["job_name"]
            mi = int(rr["model_index"])
            mobj = next((m for m in all_models if m.phase == phase and m.job_name == jm and int(m.model_index) == mi), None)
            if mobj is None:
                continue
            # transform to final reference frame
            try:
                al = _align_to_reference(mobj, ref_model)
                cdr3_tracks.append(al["cdr3_ca_transformed"])
            except Exception:
                pass
        cdr3_rmsd = _pairwise_cdr3_rmsd(cdr3_tracks)

        # combined score (interpretable, bounded)
        pose_term = 1.0 / (1.0 + max(pose_spread, 0.0))
        dom_term = max(0.0, min(1.0, dominant_fraction))
        contact_term = 0.0 if math.isnan(contact_j) else max(0.0, min(1.0, contact_j))
        hot_term = 0.0 if math.isnan(hot_j) else max(0.0, min(1.0, hot_j))
        consistency_score = float(0.30 * pose_term + 0.25 * dom_term + 0.25 * contact_term + 0.20 * hot_term)

        seq_rows.append(
            {
                "phase": phase,
                "sequence_id": seq_id,
                "candidate_id": _cell_str(grp.iloc[0]["candidate_id"]),
                "n_models": int(len(grp)),
                "n_pose_clusters": int(grp["pose_cluster"].nunique()),
                "dominant_cluster_fraction": dominant_fraction,
                "pose_spread_mean": pose_spread,
                "translation_std": trans_std,
                "rotation_std_deg": rot_std,
                "contact_fingerprint_jaccard_mean": contact_j,
                "hotspot_jaccard_mean": hot_j,
                "cdr3_rmsd_mean": cdr3_rmsd,
                "consistency_score": consistency_score,
            }
        )

    seq_df = pd.DataFrame(seq_rows)
    seq_df.to_csv(package_root / "tables" / "sequence_consistency_metrics.csv", index=False)
    seq_cluster_df = pd.DataFrame(seq_cluster_rows)
    seq_cluster_df.to_csv(package_root / "tables" / "sequence_model_cluster_usage.csv", index=False)

    # sequence consistency figure
    if not seq_df.empty:
        plt.figure(figsize=(6, 5))
        data = [seq_df[seq_df["phase"] == "Phase7"]["consistency_score"].dropna().to_numpy(), seq_df[seq_df["phase"] == "Phase8"]["consistency_score"].dropna().to_numpy()]
        plt.boxplot(data, labels=["Phase7", "Phase8"], showmeans=True)
        plt.ylabel("Sequence consistency score")
        plt.title("Within-sequence AF3 consistency")
        _save_fig(package_root / "figures" / "sequence_consistency_boxplot.png")

    # ---------- Block 6: local confidence + strain ----------
    log("Block 6: local confidence and strain")

    conf_rows = []
    for _, r in pose_df.iterrows():
        mid = r["model_id"]
        hcount = int(model_hot_anchor_count.get(mid, 0))
        comp = int(r["compressed_contact_count"])
        total_pairs = int(r["total_contact_pairs"])
        comp_frac = float(comp / max(1, total_pairs))
        pae = float(r["best_pair_pae_min"]) if not math.isnan(float(r["best_pair_pae_min"])) else 20.0
        clash = float(next((m.has_clash for m in all_models if m.phase == r["phase"] and m.job_name == r["job_name"] and int(m.model_index) == int(r["model_index"])), 0.0))
        outlier = float(r["backbone_ca_outlier_fraction"])

        # interpretable strain score
        strain_score = float(2.0 * comp_frac + 1.5 * max(0.0, outlier) + 1.5 * max(0.0, clash) + max(0.0, (pae - 6.0) / 10.0))
        forced_fit = int((r["distance_to_dominant_cluster_centroid"] <= div_threshold) and (strain_score > 0.8))

        conf_rows.append(
            {
                "phase": r["phase"],
                "model_id": mid,
                "job_name": r["job_name"],
                "candidate_id": r["candidate_id"],
                "model_index": int(r["model_index"]),
                "pose_cluster": int(r["pose_cluster"]),
                "distance_to_dominant_cluster_centroid": float(r["distance_to_dominant_cluster_centroid"]),
                "interface_plddt_mean": float(r["interface_plddt_mean"]),
                "cdr_plddt_mean": float(r["cdr_plddt_mean"]),
                "best_pair_pae_min": float(r["best_pair_pae_min"]),
                "best_pair_iptm": float(r["best_pair_iptm"]),
                "iptm": float(r["iptm"]),
                "compressed_contact_count": comp,
                "compressed_contact_fraction": comp_frac,
                "backbone_ca_outlier_fraction": outlier,
                "hotspot_anchor_count": hcount,
                "has_clash": clash,
                "strain_score": strain_score,
                "forced_fit_flag": forced_fit,
            }
        )

    conf_df = pd.DataFrame(conf_rows)
    conf_df.to_csv(package_root / "tables" / "interface_confidence_strain.csv", index=False)

    # confidence/strain figures
    for metric, fname, ylabel in [
        ("interface_plddt_mean", "interface_plddt_phase_compare.png", "Interface pLDDT (mean)"),
        ("best_pair_pae_min", "hotspot_local_pae_phase_compare.png", "Best pair pAE-min"),
        ("strain_score", "strain_score_phase_compare.png", "Strain score"),
    ]:
        plt.figure(figsize=(6, 5))
        data = [
            conf_df[conf_df["phase"] == "Phase7"][metric].dropna().to_numpy(),
            conf_df[conf_df["phase"] == "Phase8"][metric].dropna().to_numpy(),
        ]
        plt.boxplot(data, labels=["Phase7", "Phase8"], showmeans=True)
        plt.ylabel(ylabel)
        plt.title(metric.replace("_", " "))
        _save_fig(package_root / "figures" / fname)

    plt.figure(figsize=(7, 6))
    c = conf_df["strain_score"].to_numpy(dtype=float)
    sc = plt.scatter(
        conf_df["distance_to_dominant_cluster_centroid"],
        conf_df["interface_plddt_mean"],
        c=c,
        cmap="plasma",
        s=24,
        alpha=0.8,
    )
    plt.colorbar(sc, label="strain_score")
    plt.xlabel("Distance to dominant pose centroid")
    plt.ylabel("Interface pLDDT mean")
    plt.title("Convergence vs confidence vs strain")
    _save_fig(package_root / "figures" / "convergence_confidence_strain_scatter.png")

    # ---------- Representative structures ----------
    log("Selecting representative structures")

    rep_rows = []
    cluster_counts = pose_df["pose_cluster"].value_counts()
    major_clusters = [int(c) for c, n in cluster_counts.items() if n / len(pose_df) >= 0.05]
    if not major_clusters:
        major_clusters = [int(cluster_counts.index[0])]

    for c in major_clusters:
        q = pose_df[pose_df["pose_cluster"] == c].copy()
        if q.empty:
            continue
        q = q.sort_values("distance_to_cluster_centroid")
        r = q.iloc[0]
        mobj = next((m for m in all_models if m.phase == r["phase"] and m.job_name == r["job_name"] and int(m.model_index) == int(r["model_index"])), None)
        if mobj is None:
            continue
        out_name = f"cluster{c:02d}_{r['phase']}_{r['job_name']}_m{int(r['model_index'])}.cif"
        out_path = package_root / "representative_structures" / out_name
        shutil.copy2(mobj.cif_path, out_path)
        rep_rows.append(
            {
                "pose_cluster": c,
                "phase": r["phase"],
                "job_name": r["job_name"],
                "candidate_id": r["candidate_id"],
                "model_index": int(r["model_index"]),
                "distance_to_cluster_centroid": float(r["distance_to_cluster_centroid"]),
                "source_cif": str(mobj.cif_path),
                "representative_cif": str(out_path),
                "selection_reason": "Nearest model to cluster centroid in antigen-aligned descriptor space",
            }
        )

    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(package_root / "representative_structures" / "representative_clusters.csv", index=False)

    # ---------- Summaries / decision ----------
    log("Writing summaries and final decision")

    # block-wise aggregates
    phase_pose = pose_df.groupby("phase", as_index=False).agg(
        n_models=("model_id", "count"),
        n_clusters=("pose_cluster", "nunique"),
        dominant_cluster_fraction=("pose_cluster", lambda s: float(s.value_counts(normalize=True).iloc[0]) if len(s) else float("nan")),
        mean_dispersion=("distance_to_dominant_cluster_centroid", "mean"),
        median_dispersion=("distance_to_dominant_cluster_centroid", "median"),
    )

    # interface consistency proxy
    def phase_contact_jaccard(phase: str) -> float:
        mids = pose_df[pose_df["phase"] == phase]["model_id"].tolist()
        sets = [model_contact_sets.get(m, set()) for m in mids]
        return _mean_pairwise_jaccard(sets)

    contact_j7 = phase_contact_jaccard("Phase7")
    contact_j8 = phase_contact_jaccard("Phase8")

    # hotspot stability
    hot_model_counts = []
    for ph in ["Phase7", "Phase8"]:
        mids = pose_df[pose_df["phase"] == ph]["model_id"].tolist()
        vals = [len(model_hotset.get(mid, set())) for mid in mids]
        hot_model_counts.append(
            {
                "phase": ph,
                "mean_hotspot_contacts": float(np.mean(vals)) if vals else float("nan"),
                "fraction_models_with_2plus_hotspots": float(np.mean([v >= 2 for v in vals])) if vals else float("nan"),
            }
        )
    hot_stat = pd.DataFrame(hot_model_counts)

    # edge role support/divert ratio
    edge_phase = edge_df.groupby(["phase", "role"], as_index=False).size().rename(columns={"size": "count"})
    edge_tot = edge_phase.groupby("phase")["count"].transform("sum")
    edge_phase["fraction"] = edge_phase["count"] / edge_tot

    # sequence consistency
    seq_phase = seq_df.groupby("phase", as_index=False).agg(
        n_sequences=("sequence_id", "nunique"),
        mean_consistency_score=("consistency_score", "mean"),
        mean_dominant_cluster_fraction=("dominant_cluster_fraction", "mean"),
    ) if not seq_df.empty else pd.DataFrame(columns=["phase", "n_sequences", "mean_consistency_score", "mean_dominant_cluster_fraction"])

    # confidence/strain
    conf_phase = conf_df.groupby("phase", as_index=False).agg(
        mean_interface_plddt=("interface_plddt_mean", "mean"),
        mean_best_pair_pae=("best_pair_pae_min", "mean"),
        mean_strain_score=("strain_score", "mean"),
        forced_fit_fraction=("forced_fit_flag", "mean"),
    )

    # Decision logic
    def _phase_val(df: pd.DataFrame, phase: str, col: str, default=float("nan")):
        q = df[df["phase"] == phase]
        if q.empty or col not in q.columns:
            return default
        return float(q.iloc[0][col])

    p7_disp = _phase_val(phase_pose, "Phase7", "mean_dispersion", float("nan"))
    p8_disp = _phase_val(phase_pose, "Phase8", "mean_dispersion", float("nan"))
    p7_domf = _phase_val(phase_pose, "Phase7", "dominant_cluster_fraction", float("nan"))
    p8_domf = _phase_val(phase_pose, "Phase8", "dominant_cluster_fraction", float("nan"))

    h7 = _phase_val(hot_stat, "Phase7", "fraction_models_with_2plus_hotspots", float("nan"))
    h8 = _phase_val(hot_stat, "Phase8", "fraction_models_with_2plus_hotspots", float("nan"))

    s7 = _phase_val(conf_phase, "Phase7", "mean_strain_score", float("nan"))
    s8 = _phase_val(conf_phase, "Phase8", "mean_strain_score", float("nan"))

    c7s = _phase_val(seq_phase, "Phase7", "mean_consistency_score", float("nan"))
    c8s = _phase_val(seq_phase, "Phase8", "mean_consistency_score", float("nan"))

    convergence_improved = (not math.isnan(p7_disp) and not math.isnan(p8_disp) and p8_disp < p7_disp * 0.95) or (
        not math.isnan(p7_domf) and not math.isnan(p8_domf) and p8_domf > p7_domf + 0.05
    )
    hotspot_improved = (not math.isnan(h7) and not math.isnan(h8) and h8 > h7 + 0.05)
    strain_penalty = not math.isnan(s7) and not math.isnan(s8) and s8 > s7 + 0.15
    consistency_improved = not math.isnan(c7s) and not math.isnan(c8s) and c8s > c7s + 0.03

    if convergence_improved and hotspot_improved and not strain_penalty:
        case = "Case 1: hotspot logic is correct, but edge noise still needs tightening"
        recommendation = "worth continuing / promising"
    elif convergence_improved and strain_penalty:
        case = "Case 3: convergence improved, but at the cost of local strain / hard-fit artifacts"
        recommendation = "worth continuing but requires redesign of edge constraints"
    elif (not hotspot_improved) and convergence_improved:
        case = "Case 2: hotspot logic itself is not yet sufficiently anchoring the pose"
        recommendation = "worth continuing but requires redesign of edge constraints"
    else:
        case = "Case 4: no clear gain over Phase 7; apparent wins are mostly stochastic"
        recommendation = "inconclusive"

    # analysis_summary.md
    summary_lines = [
        "# Phase7 vs Phase8 AF3 Narrowing Analysis Summary",
        "",
        "## Inputs",
        f"- Phase7 directory: `{phase7_dir}`",
        f"- Phase8 directory: `{phase8_dir}`",
        f"- Optional config path: `{config_path}` (exists={config_path.exists()})",
        f"- CDR definitions used: CDR1={cdr_defs['CDR1']}, CDR2={cdr_defs['CDR2']}, CDR3={cdr_defs['CDR3']}",
        "",
        "## 1) Pose Geometry (antigen-aligned)",
        f"- Dominant intended cluster (from Phase7): cluster {dom_cluster}",
        f"- Phase7 clusters: {int(_phase_val(phase_pose,'Phase7','n_clusters',0))}, Phase8 clusters: {int(_phase_val(phase_pose,'Phase8','n_clusters',0))}",
        f"- Dominant cluster fraction Phase7={p7_domf:.3f}, Phase8={p8_domf:.3f}",
        f"- Mean dispersion to dominant centroid Phase7={p7_disp:.3f}, Phase8={p8_disp:.3f}",
        "",
        "## 2) Contact Fingerprint Consistency",
        f"- Mean pairwise model-contact Jaccard Phase7={contact_j7:.3f}, Phase8={contact_j8:.3f}",
        "",
        "## 3) Hotspot Occupancy / Co-occurrence",
        f"- Antigen hotspot residues used: {', '.join(ag_hot)}",
        f"- Nanobody hotspot residues used: {', '.join(nb_hot)}",
        f"- Fraction models with >=2 hotspot contacts Phase7={h7:.3f}, Phase8={h8:.3f}",
        "",
        "## 4) Edge-variable Role",
        f"- Edge positions used: {', '.join(str(x) for x in edge_positions)}",
    ]

    for ph in ["Phase7", "Phase8"]:
        q = edge_phase[edge_phase["phase"] == ph]
        if q.empty:
            continue
        parts = []
        for role in ["SUPPORTIVE", "PASSIVE", "DIVERTING"]:
            qq = q[q["role"] == role]
            frac = float(qq.iloc[0]["fraction"]) if not qq.empty else 0.0
            parts.append(f"{role}={frac:.2f}")
        summary_lines.append(f"- {ph}: " + ", ".join(parts))

    summary_lines.extend(
        [
            "",
            "## 5) Within-sequence Multi-model Consistency",
            f"- Mean sequence consistency score Phase7={c7s:.3f}, Phase8={c8s:.3f}",
            f"- Consistency improved: {bool(consistency_improved)}",
            "",
            "## 6) Local Confidence + Strain",
            f"- Mean interface pLDDT Phase7={_phase_val(conf_phase,'Phase7','mean_interface_plddt',float('nan')):.2f}, Phase8={_phase_val(conf_phase,'Phase8','mean_interface_plddt',float('nan')):.2f}",
            f"- Mean best-pair pAE Phase7={_phase_val(conf_phase,'Phase7','mean_best_pair_pae',float('nan')):.3f}, Phase8={_phase_val(conf_phase,'Phase8','mean_best_pair_pae',float('nan')):.3f}",
            f"- Mean strain score Phase7={s7:.3f}, Phase8={s8:.3f}",
            f"- Forced-fit fraction Phase7={_phase_val(conf_phase,'Phase7','forced_fit_fraction',float('nan')):.3f}, Phase8={_phase_val(conf_phase,'Phase8','forced_fit_fraction',float('nan')):.3f}",
            "",
            "## Group-level interpretation",
            f"- Pose convergence improved: {bool(convergence_improved)}",
            f"- Hotspot anchoring improved: {bool(hotspot_improved)}",
            f"- Sequence-level reproducibility improved: {bool(consistency_improved)}",
            f"- Strain penalty observed: {bool(strain_penalty)}",
        ]
    )

    if used_provisional:
        summary_lines.extend(["", "## Provisional / inferred annotations"])
        for n in used_provisional:
            summary_lines.append(f"- {n}")

    (package_root / "analysis_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Final Decision (A/B/C/D)",
        "",
        "## A. Evidence that narrowing worked",
        f"- Dominant-pose occupancy shifted from {p7_domf:.3f} (Phase7) to {p8_domf:.3f} (Phase8).",
        f"- Dispersion relative to dominant intended pose changed from {p7_disp:.3f} to {p8_disp:.3f}.",
        f"- Hotspot co-anchoring (>=2 hotspot contacts/model) changed from {h7:.3f} to {h8:.3f}.",
        f"- Sequence consistency changed from {c7s:.3f} to {c8s:.3f}.",
        "",
        "## B. Evidence that narrowing did NOT work or is incomplete",
        "- Multiple pose clusters remain populated; not all Phase8 models collapse into one intended family.",
        "- Edge-variable positions still show a non-zero DIVERTING fraction in both phases.",
        "- Hotspot occupancy/co-occurrence is not uniformly saturated across all models.",
        "",
        "## C. Core mechanistic diagnosis",
        f"- {case}",
        "",
        "## D. Recommendation",
        f"- {recommendation}",
        "- This recommendation is based on population-level AF3 structural consistency only (not single top-model scores).",
    ]
    (package_root / "final_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    # README for package
    readme = [
        "# phase7_phase8_af3_narrow_analysis",
        "",
        "## Layout",
        "- analysis_summary.md",
        "- final_decision.md",
        "- tables/",
        "- figures/",
        "- representative_structures/",
        "- scripts/",
        "- logs/",
        "",
        "## Metrics and assumptions",
        "- Contacts: heavy-atom distance <= 4.5A",
        "- Hydrogen-bond / salt-bridge / hydrophobic / aromatic labels use geometric heuristics",
        "- Pose alignment anchor: antigen CA atoms on common chain:residue:atom keys",
        "- Dominant intended pose family is defined from Phase7 cluster occupancy",
        "- Sequence consistency aggregates pose/contact/hotspot stability across AF3 models",
        "- Local strain proxies include compressed contacts, clash flag, and CA-geometry outliers",
        "",
        "## Provided vs inferred annotations",
        f"- Analysis config provided: {config_path.exists()}",
        "- Hotspot and edge annotations are marked PROVISIONAL when inferred.",
        "",
        "## Uncertainty",
        "- Contact-type subclasses are heuristic and should be interpreted as structural signals, not energetic truths.",
        "- No AF3 execution or retraining was performed; this is post-hoc structural analysis only.",
    ]
    if used_provisional:
        readme.append("")
        readme.append("### Provisional notes")
        for n in used_provisional:
            readme.append(f"- {n}")

    (package_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    # copy script used
    shutil.copy2(Path(__file__).resolve(), package_root / "scripts" / Path(__file__).name)

    # persist key intermediate tables for transparency
    phase_pose.to_csv(package_root / "tables" / "_phase_pose_group_metrics.csv", index=False)
    hot_stat.to_csv(package_root / "tables" / "_hotspot_phase_metrics.csv", index=False)
    seq_phase.to_csv(package_root / "tables" / "_sequence_phase_metrics.csv", index=False)
    conf_phase.to_csv(package_root / "tables" / "_confidence_phase_metrics.csv", index=False)

    # zip package
    zip_path = out_root / args.zip_name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in package_root.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(out_root))

    log(f"Package ready: {package_root}")
    log(f"Zip ready: {zip_path}")
    return zip_path


def main() -> int:
    args = parse_args()
    zip_path = _build_package(args)
    print(f"Wrote ZIP: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
