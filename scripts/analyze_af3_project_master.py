#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import MMCIFParser
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project-wide AF3 master ranking and wet-lab triage")
    p.add_argument("--af3-root", default="AF3 Results")
    p.add_argument("--outdir", default="results/summaries/af3_project_master_ranking_and_wetlab_tiers")
    p.add_argument("--contact-cutoff", type=float, default=4.5)
    p.add_argument("--hb-cutoff", type=float, default=3.5)
    p.add_argument("--salt-cutoff", type=float, default=4.0)
    p.add_argument("--clash-cutoff", type=float, default=2.0)
    p.add_argument("--sidechain-overclose-cutoff", type=float, default=2.2)
    p.add_argument("--cdr1-start", type=int, default=23)
    p.add_argument("--cdr1-end", type=int, default=34)
    p.add_argument("--cdr2-start", type=int, default=50)
    p.add_argument("--cdr2-end", type=int, default=58)
    p.add_argument("--cdr3-start", type=int, default=97)
    p.add_argument("--cdr3-end", type=int, default=106)
    return p.parse_args()


@dataclass
class RunInfo:
    run_dir: Path
    stage: str
    run_key: str
    job_name: str
    sequence: str
    sequence_hash: str
    is_wt: bool
    lineage_group: str
    mapped_candidate_id: str


def load_json(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        if not obj:
            return {}
        return obj[0]
    return obj


def unique_in_order(seq: Sequence[str]) -> List[str]:
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def is_heavy_atom(atom) -> bool:
    element = (atom.element or "").strip().upper()
    name = atom.get_name().strip().upper()
    if element:
        return element != "H"
    return not name.startswith("H")


def is_backbone_atom(name: str) -> bool:
    return name.strip().upper() in {"N", "CA", "C", "O", "OXT"}


def parse_label(label: str) -> Tuple[str, int]:
    m = re.match(r"^([^:]+):(\d+)", str(label))
    if not m:
        return "", -1
    return m.group(1), int(m.group(2))


def residue_label(chain_id: str, residue) -> str:
    het, resseq, icode = residue.id
    if het != " ":
        return ""
    ins = str(icode).strip()
    return f"{chain_id}:{int(resseq)}{ins}"


def cdr_region(resnum: int, c1: Tuple[int, int], c2: Tuple[int, int], c3: Tuple[int, int]) -> str:
    if c1[0] <= resnum <= c1[1]:
        return "CDR1"
    if c2[0] <= resnum <= c2[1]:
        return "CDR2"
    if c3[0] <= resnum <= c3[1]:
        return "CDR3"
    return "framework"


def extract_chain_atoms_detailed(chain) -> Tuple[np.ndarray, List[dict], Dict[int, np.ndarray]]:
    coords = []
    meta = []
    ca_map: Dict[int, np.ndarray] = {}
    cid = str(chain.id)
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue
        lbl = residue_label(cid, residue)
        if not lbl:
            continue
        _, rnum = parse_label(lbl)
        if "CA" in residue:
            ca_map[rnum] = residue["CA"].coord.astype(float)
        resname = residue.get_resname().strip().upper()
        for atom in residue.get_atoms():
            if not is_heavy_atom(atom):
                continue
            an = atom.get_name().strip().upper()
            coords.append(atom.coord.astype(float))
            meta.append(
                {
                    "chain_id": cid,
                    "residue_label": lbl,
                    "resnum": rnum,
                    "resname": resname,
                    "atom_name": an,
                    "element": (atom.element or "").strip().upper(),
                    "is_backbone": is_backbone_atom(an),
                }
            )
    if not coords:
        return np.empty((0, 3), dtype=float), [], ca_map
    return np.asarray(coords, dtype=float), meta, ca_map


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


def dominant_contact_type(d: dict) -> str:
    types = []
    if d.get("salt_bridge", 0) > 0:
        types.append("salt_bridge")
    if d.get("hydrogen_bond", 0) > 0:
        types.append("hydrogen_bond")
    if d.get("hydrophobic_packing", 0) > 0:
        types.append("hydrophobic_packing")
    if not types:
        return "heavy_atom_contact"
    if len(types) == 1:
        return types[0]
    return "mixed"


def detect_stage(run_dir: Path) -> str:
    p = str(run_dir)
    if "Stage6 AF3" in p:
        return "Stage6"
    if "Stage7 AF3" in p:
        return "Stage7"
    if "Stage 8 AF3" in p:
        return "Stage8"
    if "Stage 9 AF3" in p:
        return "Stage9"
    return "Legacy"


def detect_lineage(job_name: str, stage: str) -> str:
    j = job_name.strip()
    for pat in [r"^(spg\d+)", r"^(p8s\d+)", r"^(p9c)", r"^(fold_test\d+)", r"^(fold_see\d+)"]:
        m = re.match(pat, j, re.IGNORECASE)
        if m:
            return m.group(1)
    if stage == "Stage6":
        if "wt" in j.lower():
            return "WT"
        if "test1" in j.lower():
            return "Test1"
        return "Stage6"
    return stage


def sequence_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()[:12]


def safe_float(x, default=float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def mean_pairwise_jaccard(sets: List[Set[str]]) -> float:
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


def seq_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if len(a) == len(b):
        return sum(x == y for x, y in zip(a, b)) / max(1, len(a))
    return SequenceMatcher(a=a, b=b).ratio()


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
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
    P_fit = P @ R + t
    rmsd = float(np.sqrt(np.mean(np.sum((P_fit - Q) ** 2, axis=1))))
    return R, t, rmsd


def rotation_angle_deg(R: np.ndarray) -> float:
    v = (np.trace(R) - 1.0) / 2.0
    v = max(-1.0, min(1.0, float(v)))
    return float(np.degrees(np.arccos(v)))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def principal_axis(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return np.array([1.0, 0.0, 0.0])
    X = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[0]
    return v / max(1e-8, np.linalg.norm(v))


def load_job_map_candidates(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in root.rglob("*.csv"):
        n = p.name.lower()
        if "map" not in n:
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        if "job_name" not in cols or "candidate_id" not in cols:
            continue
        jcol = cols["job_name"]
        ccol = cols["candidate_id"]
        for _, r in df.iterrows():
            j = str(r.get(jcol, "")).strip()
            c = str(r.get(ccol, "")).strip()
            if not j or not c:
                continue
            if j not in out or (len(c) > len(out[j])):
                out[j] = c
    return out


def parse_run_info(run_dir: Path, job_map: Dict[str, str]) -> Optional[RunInfo]:
    jfiles = sorted(run_dir.glob("*_job_request.json"))
    if not jfiles:
        return None
    job = load_json(jfiles[0])
    job_name = str(job.get("name", run_dir.name)).strip()
    seqs = []
    for item in job.get("sequences", []):
        if isinstance(item, dict) and "proteinChain" in item:
            pc = item.get("proteinChain") or {}
            seq = str(pc.get("sequence", "")).strip()
            if seq:
                seqs.append(seq)
    if not seqs:
        return None
    nb_seq = min(seqs, key=len)

    stage = detect_stage(run_dir)
    is_wt = ("wt" in job_name.lower()) or ("wildtype" in job_name.lower())
    lineage = detect_lineage(job_name, stage)
    mapped = job_map.get(job_name, job_name)

    return RunInfo(
        run_dir=run_dir,
        stage=stage,
        run_key=run_dir.name,
        job_name=job_name,
        sequence=nb_seq,
        sequence_hash=sequence_hash(nb_seq),
        is_wt=is_wt,
        lineage_group=lineage,
        mapped_candidate_id=mapped,
    )


def main() -> int:
    args = parse_args()
    root = Path(".").resolve()
    af3_root = (root / args.af3_root).resolve()
    outdir = (root / args.outdir).resolve()
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    cdr1 = (args.cdr1_start, args.cdr1_end)
    cdr2 = (args.cdr2_start, args.cdr2_end)
    cdr3 = (args.cdr3_start, args.cdr3_end)

    # project hotspot patch (from late-stage consensus)
    hotspot_antigen_positions = {121, 122, 124, 217, 218, 219}

    job_map = load_job_map_candidates(root)

    run_dirs = set()
    for s0 in af3_root.rglob("*_summary_confidences_0.json"):
        d = s0.parent
        if list(d.glob("*_job_request.json")) and list(d.glob("*_model_0.cif")) and list(d.glob("*_full_data_0.json")):
            run_dirs.add(d)
    run_dirs = sorted(run_dirs)
    if not run_dirs:
        raise RuntimeError(f"No AF3 run directories found under {af3_root}")

    parser = MMCIFParser(QUIET=True)

    model_rows = []
    contact_rows = []
    model_contact_sets: Dict[str, Set[str]] = {}
    model_hotspot_res_sets: Dict[str, Set[str]] = {}
    pose_cache: Dict[str, dict] = {}

    total_models = 0
    for rd in run_dirs:
        info = parse_run_info(rd, job_map)
        if info is None:
            continue

        summary_files = sorted(rd.glob("*_summary_confidences_*.json"))
        for sf in summary_files:
            m = re.search(r"_summary_confidences_(\d+)\.json$", sf.name)
            if not m:
                continue
            midx = int(m.group(1))
            fullf = rd / sf.name.replace("_summary_confidences_", "_full_data_")
            ciff = rd / sf.name.replace("_summary_confidences_", "_model_").replace(".json", ".cif")
            if not fullf.exists() or not ciff.exists():
                continue

            summ = load_json(sf)
            full = load_json(fullf)

            chain_order = unique_in_order([str(x) for x in full.get("token_chain_ids", [])])
            counts = Counter(full.get("token_chain_ids", []))
            if not chain_order:
                continue
            nb_chain = min(chain_order, key=lambda c: counts.get(c, 10**9))
            ag_chains = [c for c in chain_order if c != nb_chain]

            pair_iptm = summ.get("chain_pair_iptm", [])
            pair_pae = summ.get("chain_pair_pae_min", [])

            def pair_val(mat, ci, cj):
                if not isinstance(mat, list) or not mat:
                    return float("nan")
                try:
                    i = chain_order.index(ci)
                    j = chain_order.index(cj)
                    return float(mat[i][j])
                except Exception:
                    return float("nan")

            ag_pair_iptm = [pair_val(pair_iptm, nb_chain, c) for c in ag_chains]
            ag_pair_pae = [pair_val(pair_pae, nb_chain, c) for c in ag_chains]

            atom_plddts = np.asarray(full.get("atom_plddts", []), dtype=float)
            mean_plddt = float(np.mean(atom_plddts)) if atom_plddts.size else float("nan")

            structure = parser.get_structure(f"run_{hash(rd)}_{midx}", str(ciff))
            model = next(structure.get_models())
            chains = {str(c.id): c for c in model.get_chains()}
            if nb_chain not in chains:
                continue

            nb_coords, nb_meta, nb_ca = extract_chain_atoms_detailed(chains[nb_chain])

            ag_coords_list = []
            ag_meta = []
            ag_ca_by_chain: Dict[str, Dict[int, np.ndarray]] = {}
            for ac in ag_chains:
                if ac not in chains:
                    continue
                ccoords, cmeta, cca = extract_chain_atoms_detailed(chains[ac])
                if ccoords.size:
                    ag_coords_list.append(ccoords)
                    ag_meta.extend(cmeta)
                ag_ca_by_chain[ac] = cca
            ag_coords = np.vstack(ag_coords_list) if ag_coords_list else np.empty((0, 3), dtype=float)

            res_plddt = {}
            for ch in model.get_chains():
                cid = str(ch.id)
                for res in ch.get_residues():
                    if res.id[0] != " ":
                        continue
                    lbl = residue_label(cid, res)
                    vals = [float(a.bfactor) for a in res.get_atoms() if is_heavy_atom(a)]
                    if vals:
                        res_plddt[lbl] = float(np.mean(vals))

            pair_map = {}
            hotspot_res = set()
            clash_count = 0
            overclose_count = 0
            min_dist = float("nan")

            if nb_coords.size and ag_coords.size:
                tree = cKDTree(ag_coords)
                dists, _ = tree.query(nb_coords, k=1)
                if np.size(dists):
                    min_dist = float(np.min(dists))

                neigh = tree.query_ball_point(nb_coords, r=args.contact_cutoff)
                for i, js in enumerate(neigh):
                    if not js:
                        continue
                    n = nb_meta[i]
                    for j in js:
                        a = ag_meta[j]
                        d = float(np.linalg.norm(nb_coords[i] - ag_coords[j]))
                        k = (n["residue_label"], a["residue_label"])
                        x = pair_map.setdefault(
                            k,
                            {
                                "min_distance": 999.0,
                                "heavy_atom_contacts": 0,
                                "hydrogen_bond": 0,
                                "salt_bridge": 0,
                                "hydrophobic_packing": 0,
                            },
                        )
                        x["min_distance"] = min(x["min_distance"], d)
                        x["heavy_atom_contacts"] += 1

                        if n["element"] in {"N", "O", "S"} and a["element"] in {"N", "O", "S"} and d <= args.hb_cutoff:
                            x["hydrogen_bond"] = 1

                        if d <= args.salt_cutoff:
                            c1 = acidic_atom(n["resname"], n["atom_name"]) and basic_atom(a["resname"], a["atom_name"])
                            c2 = acidic_atom(a["resname"], a["atom_name"]) and basic_atom(n["resname"], n["atom_name"])
                            if c1 or c2:
                                x["salt_bridge"] = 1

                        if (
                            hydrophobic_res(n["resname"])
                            and hydrophobic_res(a["resname"])
                            and n["element"] == "C"
                            and a["element"] == "C"
                        ):
                            x["hydrophobic_packing"] = 1

                        _, ag_resnum = parse_label(a["residue_label"])
                        if ag_resnum in hotspot_antigen_positions:
                            hotspot_res.add(a["residue_label"])

                        if d < args.clash_cutoff:
                            clash_count += 1
                        if (
                            d < args.sidechain_overclose_cutoff
                            and (not n["is_backbone"])
                            and (not a["is_backbone"])
                        ):
                            overclose_count += 1

            interface_nb_res = {x[0] for x in pair_map.keys()}
            interface_ag_res = {x[1] for x in pair_map.keys()}
            nb_interface_plddt = [res_plddt.get(r) for r in interface_nb_res if r in res_plddt]
            ag_interface_plddt = [res_plddt.get(r) for r in interface_ag_res if r in res_plddt]

            # token-level PAE for interface pairs (first token index per residue)
            tchain = full.get("token_chain_ids", [])
            tres = full.get("token_res_ids", [])
            token_idx: Dict[Tuple[str, int], int] = {}
            for ii, (cc, rr) in enumerate(zip(tchain, tres)):
                try:
                    rn = int(rr)
                except Exception:
                    try:
                        rn = int(float(rr))
                    except Exception:
                        continue
                token_idx.setdefault((str(cc), rn), ii)

            pae = np.asarray(full.get("pae", []), dtype=float)
            iface_pae_vals = []
            for nb_lbl, ag_lbl in pair_map.keys():
                nb_c, nb_r = parse_label(nb_lbl)
                ag_c, ag_r = parse_label(ag_lbl)
                i = token_idx.get((nb_c, nb_r))
                j = token_idx.get((ag_c, ag_r))
                if i is None or j is None:
                    continue
                if i < pae.shape[0] and j < pae.shape[1]:
                    iface_pae_vals.append(float((pae[i, j] + pae[j, i]) / 2.0))

            model_uid = f"{info.sequence_hash}|{info.run_key}|{midx}"
            cset = {f"{a}|{b}" for (a, b) in pair_map.keys()}
            model_contact_sets[model_uid] = cset
            model_hotspot_res_sets[model_uid] = set(hotspot_res)

            # model-level contact rows
            for (nb_lbl, ag_lbl), vv in pair_map.items():
                _, nb_r = parse_label(nb_lbl)
                contact_rows.append(
                    {
                        "model_uid": model_uid,
                        "sequence_hash": info.sequence_hash,
                        "variant_hint": info.mapped_candidate_id,
                        "stage": info.stage,
                        "run_key": info.run_key,
                        "job_name": info.job_name,
                        "model_index": midx,
                        "nanobody_residue": nb_lbl,
                        "antigen_residue": ag_lbl,
                        "residue_pair": f"{nb_lbl}|{ag_lbl}",
                        "cdr_region": cdr_region(nb_r, cdr1, cdr2, cdr3),
                        "contact_type": dominant_contact_type(vv),
                        "min_distance": float(vv["min_distance"]),
                        "heavy_atom_contacts": int(vv["heavy_atom_contacts"]),
                        "hydrogen_bond": int(vv["hydrogen_bond"]),
                        "salt_bridge": int(vv["salt_bridge"]),
                        "hydrophobic_packing": int(vv["hydrophobic_packing"]),
                    }
                )

            model_rows.append(
                {
                    "model_uid": model_uid,
                    "sequence_hash": info.sequence_hash,
                    "sequence": info.sequence,
                    "stage": info.stage,
                    "lineage_group": info.lineage_group,
                    "run_key": info.run_key,
                    "run_dir": str(info.run_dir),
                    "job_name": info.job_name,
                    "mapped_candidate_id": info.mapped_candidate_id,
                    "is_wt": int(info.is_wt),
                    "model_index": midx,
                    "summary_json": str(sf),
                    "full_data_json": str(fullf),
                    "cif_path": str(ciff),
                    "ranking_score": safe_float(summ.get("ranking_score")),
                    "ipTM": safe_float(summ.get("iptm")),
                    "pTM": safe_float(summ.get("ptm")),
                    "mean_pLDDT": mean_plddt,
                    "has_clash_summary": safe_float(summ.get("has_clash")),
                    "fraction_disordered": safe_float(summ.get("fraction_disordered")),
                    "interchain_pair_ipTM_best": float(np.nanmax(ag_pair_iptm)) if ag_pair_iptm else float("nan"),
                    "interchain_pair_ipTM_mean": float(np.nanmean(ag_pair_iptm)) if ag_pair_iptm else float("nan"),
                    "interchain_pair_PAE_best": float(np.nanmin(ag_pair_pae)) if ag_pair_pae else float("nan"),
                    "interchain_pair_PAE_mean": float(np.nanmean(ag_pair_pae)) if ag_pair_pae else float("nan"),
                    "interface_nb_mean_pLDDT": float(np.mean(nb_interface_plddt)) if nb_interface_plddt else float("nan"),
                    "interface_ag_mean_pLDDT": float(np.mean(ag_interface_plddt)) if ag_interface_plddt else float("nan"),
                    "interface_pair_PAE_mean": float(np.mean(iface_pae_vals)) if iface_pae_vals else float("nan"),
                    "contact_pair_count": int(len(pair_map)),
                    "interface_nb_res_count": int(len(interface_nb_res)),
                    "interface_ag_res_count": int(len(interface_ag_res)),
                    "hotspot_contact_res_count": int(len(hotspot_res)),
                    "hotspot_contact_present": int(len(hotspot_res) > 0),
                    "clash_count_lt2p0": int(clash_count),
                    "sidechain_overclose_count_lt2p2": int(overclose_count),
                    "min_interchain_heavy_distance": min_dist,
                    "forced_fit_warning_model": int((safe_float(summ.get("has_clash"), 0.0) > 0) or (clash_count > 0) or (overclose_count > 10)),
                    "nanobody_chain": nb_chain,
                    "antigen_chains": ",".join(ag_chains),
                }
            )

            # pose cache
            pose_cache[model_uid] = {
                "sequence_hash": info.sequence_hash,
                "is_wt": int(info.is_wt),
                "stage": info.stage,
                "run_key": info.run_key,
                "job_name": info.job_name,
                "mapped_candidate_id": info.mapped_candidate_id,
                "model_index": midx,
                "ag_ca_by_chain": ag_ca_by_chain,
                "nb_ca": nb_ca,
                "nb_chain": nb_chain,
                "ag_chain_order": list(ag_chains),
            }

            total_models += 1
            if total_models % 150 == 0:
                print(f"[progress] parsed models: {total_models}")

    if not model_rows:
        raise RuntimeError("No model rows parsed")

    model_df = pd.DataFrame(model_rows)
    contact_df = pd.DataFrame(contact_rows)

    # WT fallback by sequence if missing explicit tag
    if model_df["is_wt"].sum() == 0:
        wt_guess = model_df[model_df["job_name"].str.contains("wt|wildtype", case=False, na=False)]
        if not wt_guess.empty:
            seqs = set(wt_guess["sequence_hash"])
            model_df.loc[model_df["sequence_hash"].isin(seqs), "is_wt"] = 1

    # Pose alignment and clustering
    wt_models = model_df[model_df["is_wt"] == 1].copy()
    if wt_models.empty:
        # fallback: choose highest ipTM model as reference
        ref_uid = model_df.sort_values("ipTM", ascending=False)["model_uid"].iloc[0]
    else:
        ref_uid = wt_models.sort_values(["interchain_pair_PAE_best", "ipTM"], ascending=[True, False])["model_uid"].iloc[0]

    ref = pose_cache[ref_uid]
    ref_ag = ref["ag_ca_by_chain"]
    ref_nb = ref["nb_ca"]
    ref_nb_pts = np.asarray(list(ref_nb.values()), dtype=float) if ref_nb else np.empty((0, 3), dtype=float)
    ref_nb_com = ref_nb_pts.mean(axis=0) if ref_nb_pts.size else np.zeros(3)
    ref_axis = principal_axis(ref_nb_pts) if ref_nb_pts.size else np.array([1.0, 0.0, 0.0])

    # epitope center from top WT contacts if possible
    wt_contact = contact_df[contact_df["model_uid"].isin(wt_models["model_uid"])].copy() if not wt_models.empty else pd.DataFrame()
    epitope_labels = []
    if not wt_contact.empty:
        occ = wt_contact.groupby("antigen_residue")["model_uid"].nunique().sort_values(ascending=False)
        epitope_labels = list(occ.head(12).index)

    ref_epi_pts = []
    if epitope_labels:
        for lbl in epitope_labels:
            ch, rn = parse_label(lbl)
            if ch in ref_ag and rn in ref_ag[ch]:
                ref_epi_pts.append(ref_ag[ch][rn])
    if not ref_epi_pts:
        for ch, mp in ref_ag.items():
            ref_epi_pts.extend(list(mp.values()))
    ref_epi_center = np.mean(np.asarray(ref_epi_pts), axis=0) if ref_epi_pts else np.zeros(3)

    pose_rows = []

    for uid, rec in pose_cache.items():
        ag = rec["ag_ca_by_chain"]
        nb = rec["nb_ca"]
        if not ag or not nb or not ref_ag or not ref_nb:
            continue

        model_ag_chains = list(ag.keys())
        ref_ag_chains = list(ref_ag.keys())

        mappings = []
        if len(model_ag_chains) == 2 and len(ref_ag_chains) == 2:
            mappings = [
                {model_ag_chains[0]: ref_ag_chains[0], model_ag_chains[1]: ref_ag_chains[1]},
                {model_ag_chains[0]: ref_ag_chains[1], model_ag_chains[1]: ref_ag_chains[0]},
            ]
        else:
            mappings = [{m: ref_ag_chains[min(i, len(ref_ag_chains) - 1)] for i, m in enumerate(model_ag_chains)}]

        best = None
        for mp in mappings:
            P_list = []
            Q_list = []
            for m_ch, r_ch in mp.items():
                if r_ch not in ref_ag:
                    continue
                common = sorted(set(ag[m_ch].keys()) & set(ref_ag[r_ch].keys()))
                for rr in common:
                    P_list.append(ag[m_ch][rr])
                    Q_list.append(ref_ag[r_ch][rr])
            if len(P_list) < 40:
                continue
            P = np.asarray(P_list, dtype=float)
            Q = np.asarray(Q_list, dtype=float)
            R, t, rmsd_ag = kabsch_align(P, Q)
            if (best is None) or (rmsd_ag < best["rmsd_ag"]):
                best = {"R": R, "t": t, "rmsd_ag": rmsd_ag, "map": mp}

        if best is None:
            continue

        R = best["R"]
        t = best["t"]

        common_nb = sorted(set(nb.keys()) & set(ref_nb.keys()))
        if len(common_nb) < 20:
            continue
        nb_aln = np.asarray([nb[r] @ R + t for r in common_nb], dtype=float)
        nb_ref = np.asarray([ref_nb[r] for r in common_nb], dtype=float)
        pose_rmsd = float(np.sqrt(np.mean(np.sum((nb_aln - nb_ref) ** 2, axis=1))))

        R_nb, _, _ = kabsch_align(nb_aln, nb_ref)
        rot_deg = rotation_angle_deg(R_nb)

        nb_all = np.asarray([v @ R + t for v in nb.values()], dtype=float)
        nb_com = nb_all.mean(axis=0)
        trans = nb_com - ref_nb_com
        trans_norm = float(np.linalg.norm(trans))

        par_res = set(range(cdr1[0], cdr1[1] + 1)) | set(range(cdr2[0], cdr2[1] + 1)) | set(range(cdr3[0], cdr3[1] + 1))
        cdr3_res = set(range(cdr3[0], cdr3[1] + 1))

        par_pts = [nb[r] @ R + t for r in nb.keys() if r in par_res]
        c3_pts = [nb[r] @ R + t for r in nb.keys() if r in cdr3_res]
        par_com = np.mean(np.asarray(par_pts), axis=0) if par_pts else nb_com
        c3_com = np.mean(np.asarray(c3_pts), axis=0) if c3_pts else nb_com

        par_dist = float(np.linalg.norm(par_com - ref_epi_center))
        c3_ang = angle_between(c3_com - nb_com, ref_epi_center - nb_com)
        axis = principal_axis(nb_all)
        axis_ang = min(angle_between(axis, ref_axis), angle_between(-axis, ref_axis))

        pose_rows.append(
            {
                "model_uid": uid,
                "sequence_hash": rec["sequence_hash"],
                "is_wt": rec["is_wt"],
                "pose_antigen_rmsd": best["rmsd_ag"],
                "pose_rmsd_nb_ca": pose_rmsd,
                "translation_x": float(trans[0]),
                "translation_y": float(trans[1]),
                "translation_z": float(trans[2]),
                "translation_norm": trans_norm,
                "rotation_diff_deg": rot_deg,
                "paratope_epitope_distance": par_dist,
                "cdr3_to_epitope_angle_deg": c3_ang,
                "principal_axis_angle_deg": axis_ang,
            }
        )

    pose_df = pd.DataFrame(pose_rows)
    if not pose_df.empty:
        feats = [
            "translation_x",
            "translation_y",
            "translation_z",
            "rotation_diff_deg",
            "paratope_epitope_distance",
            "cdr3_to_epitope_angle_deg",
            "pose_rmsd_nb_ca",
        ]
        X = pose_df[feats].to_numpy(dtype=float)
        col_med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_med, inds[1])
        mu = np.mean(X, axis=0)
        sd = np.std(X, axis=0)
        sd[sd < 1e-8] = 1.0
        Xz = (X - mu) / sd
        if Xz.shape[0] >= 3:
            k = min(8, max(2, int(np.sqrt(Xz.shape[0] / 30))))
            Z = linkage(Xz, method="ward")
            labels = fcluster(Z, t=k, criterion="maxclust")
        else:
            labels = np.ones(Xz.shape[0], dtype=int)
        pose_df["pose_cluster"] = labels
        model_df = model_df.merge(pose_df[["model_uid", "pose_cluster", "pose_rmsd_nb_ca", "translation_norm"]], on="model_uid", how="left")
    else:
        model_df["pose_cluster"] = np.nan
        model_df["pose_rmsd_nb_ca"] = np.nan
        model_df["translation_norm"] = np.nan

    # Build per-run provenance table first
    run_grp = model_df.groupby(["sequence_hash", "run_key", "run_dir", "stage", "lineage_group", "job_name", "mapped_candidate_id"], as_index=False)
    run_prov = run_grp.agg(
        sequence=("sequence", "first"),
        is_wt=("is_wt", "max"),
        n_models=("model_uid", "count"),
        mean_ranking_score_run=("ranking_score", "mean"),
        mean_ipTM_run=("ipTM", "mean"),
        mean_pTM_run=("pTM", "mean"),
        mean_pair_ipTM_run=("interchain_pair_ipTM_mean", "mean"),
        mean_pair_PAE_run=("interchain_pair_PAE_mean", "mean"),
        best_pair_PAE_run=("interchain_pair_PAE_best", "min"),
    )

    # Variant-level aggregation by unique sequence
    master_rows = []
    consensus_contact_sets: Dict[str, Set[str]] = {}

    for seq_hash, g in model_df.groupby("sequence_hash"):
        seq = g["sequence"].iloc[0]
        is_wt = int(g["is_wt"].max())
        stages = sorted(set(g["stage"].dropna().astype(str)))
        lineages = sorted(set(g["lineage_group"].dropna().astype(str)))
        mapped_ids = sorted(set(g["mapped_candidate_id"].dropna().astype(str)))

        if is_wt:
            variant_id = "WT"
        else:
            # prefer long informative id if exists; otherwise first mapped id
            mapped_ids_sorted = sorted(mapped_ids, key=lambda x: (len(x), x), reverse=True)
            variant_id = mapped_ids_sorted[0] if mapped_ids_sorted else f"VAR_{seq_hash}"

        # contact consistency from model contact sets
        model_uids = list(g["model_uid"])
        csets = [model_contact_sets.get(u, set()) for u in model_uids]
        contact_cons = mean_pairwise_jaccard(csets)

        # consensus contacts
        cnt = Counter()
        for s in csets:
            for p in s:
                cnt[p] += 1
        n_mod = len(csets)
        occ = {k: v / max(1, n_mod) for k, v in cnt.items()}
        cons_set = {k for k, v in occ.items() if v >= 0.6}
        consensus_contact_sets[seq_hash] = cons_set
        consensus_occ_mean = float(np.mean(list(occ.values()))) if occ else float("nan")

        # hotspot reproducibility
        hsets = [model_hotspot_res_sets.get(u, set()) for u in model_uids]
        hotspot_model_frac = float(np.mean([1.0 if s else 0.0 for s in hsets])) if hsets else float("nan")

        # pose consistency
        pcls = g["pose_cluster"].dropna().astype(int)
        if not pcls.empty:
            pose_cons = float((pcls.value_counts(normalize=True).iloc[0]))
            assigned_pose = int(pcls.value_counts().idxmax())
        else:
            pose_cons = float("nan")
            assigned_pose = np.nan

        # run concordance
        rg = run_prov[run_prov["sequence_hash"] == seq_hash]
        if rg.shape[0] <= 1:
            run_conc = "single_run"
            run_conc_score = 0.75
        else:
            iptm_rng = float(rg["mean_ipTM_run"].max() - rg["mean_ipTM_run"].min())
            pae_rng = float(rg["mean_pair_PAE_run"].max() - rg["mean_pair_PAE_run"].min())
            if iptm_rng < 0.03 and pae_rng < 1.5:
                run_conc = "high"
                run_conc_score = 1.0
            elif iptm_rng < 0.06 and pae_rng < 3.0:
                run_conc = "mixed"
                run_conc_score = 0.5
            else:
                run_conc = "low"
                run_conc_score = 0.0

        # composite consistency transparency: simple average of explicit terms
        cons_parts = [x for x in [pose_cons, contact_cons] if not pd.isna(x)]
        within_cons = float(np.mean(cons_parts)) if cons_parts else float("nan")

        master_rows.append(
            {
                "variant_id": variant_id,
                "sequence": seq,
                "sequence_hash": seq_hash,
                "originating_phase_stage_run": ";".join(stages),
                "lineage_group": ";".join(lineages),
                "is_wt": int(is_wt),
                "number_of_models_available": int(g.shape[0]),
                "mean_ranking_score": float(g["ranking_score"].mean()),
                "best_ranking_score": float(g["ranking_score"].max()),
                "mean_ipTM": float(g["ipTM"].mean()),
                "best_ipTM": float(g["ipTM"].max()),
                "mean_pTM": float(g["pTM"].mean()),
                "mean_mean_pLDDT": float(g["mean_pLDDT"].mean()),
                "mean_interface_pLDDT": float(g["interface_nb_mean_pLDDT"].mean()),
                "mean_interchain_pair_ipTM": float(g["interchain_pair_ipTM_mean"].mean()),
                "mean_interchain_pair_PAE": float(g["interchain_pair_PAE_mean"].mean()),
                "best_interchain_pair_PAE": float(g["interchain_pair_PAE_best"].min()),
                "within_variant_model_consistency": within_cons,
                "pose_cluster_consistency": pose_cons,
                "contact_fingerprint_consistency": contact_cons,
                "hotspot_reproducibility": hotspot_model_frac,
                "total_interface_contact_count": float(g["contact_pair_count"].mean()),
                "consensus_contact_occupancy": consensus_occ_mean,
                "interface_size_residue_count": float((g["interface_nb_res_count"] + g["interface_ag_res_count"]).mean()),
                "interface_buried_area_if_available": float("nan"),
                "strain_clash_flag_rate": float(((g["has_clash_summary"] > 0) | (g["clash_count_lt2p0"] > 0)).mean()),
                "forced_fit_warning_rate": float(g["forced_fit_warning_model"].mean()),
                "mean_sidechain_overclose_count": float(g["sidechain_overclose_count_lt2p2"].mean()),
                "assigned_pose_family": assigned_pose,
                "run_concordance": run_conc,
                "run_concordance_score": run_conc_score,
            }
        )

    master_df = pd.DataFrame(master_rows)
    master_df = master_df.sort_values(["is_wt", "mean_ipTM", "mean_ranking_score"], ascending=[False, False, False]).reset_index(drop=True)

    # Determine WT baseline row
    wt_rows = master_df[master_df["is_wt"] == 1]
    if wt_rows.empty:
        wt_base = None
    else:
        wt_base = wt_rows.sort_values(["number_of_models_available", "mean_ipTM"], ascending=[False, False]).iloc[0]

    def compare_vs_wt(row: pd.Series, wt: Optional[pd.Series], mode: str) -> str:
        if wt is None:
            return "mixed"
        if int(row["is_wt"]) == 1:
            return "yes"

        if mode == "raw":
            checks = [
                (row["mean_ranking_score"], wt["mean_ranking_score"], "high", 0.005),
                (row["mean_ipTM"], wt["mean_ipTM"], "high", 0.01),
                (row["mean_pTM"], wt["mean_pTM"], "high", 0.01),
                (row["mean_interface_pLDDT"], wt["mean_interface_pLDDT"], "high", 0.5),
                (row["mean_interchain_pair_ipTM"], wt["mean_interchain_pair_ipTM"], "high", 0.01),
                (row["mean_interchain_pair_PAE"], wt["mean_interchain_pair_PAE"], "low", 0.3),
                (row["best_interchain_pair_PAE"], wt["best_interchain_pair_PAE"], "low", 0.3),
            ]
        elif mode == "cons":
            checks = [
                (row["within_variant_model_consistency"], wt["within_variant_model_consistency"], "high", 0.03),
                (row["pose_cluster_consistency"], wt["pose_cluster_consistency"], "high", 0.03),
                (row["contact_fingerprint_consistency"], wt["contact_fingerprint_consistency"], "high", 0.03),
                (row["run_concordance_score"], wt["run_concordance_score"], "high", 0.05),
            ]
        else:  # strain
            checks = [
                (row["strain_clash_flag_rate"], wt["strain_clash_flag_rate"], "low", 0.02),
                (row["forced_fit_warning_rate"], wt["forced_fit_warning_rate"], "low", 0.02),
                (row["mean_sidechain_overclose_count"], wt["mean_sidechain_overclose_count"], "low", 0.5),
            ]

        better = 0
        worse = 0
        for a, b, direction, tol in checks:
            if pd.isna(a) or pd.isna(b):
                continue
            if direction == "high":
                if a > b + tol:
                    better += 1
                elif a < b - tol:
                    worse += 1
            else:
                if a < b - tol:
                    better += 1
                elif a > b + tol:
                    worse += 1

        if better >= max(2, len(checks) // 2) and worse <= 1:
            return "yes"
        if worse >= max(2, len(checks) // 2) and better <= 1:
            return "no"
        return "mixed"

    master_df["better_than_WT_on_raw_AF3_metrics"] = master_df.apply(lambda r: compare_vs_wt(r, wt_base, "raw"), axis=1)
    master_df["more_consistent_than_WT"] = master_df.apply(lambda r: compare_vs_wt(r, wt_base, "cons"), axis=1)
    master_df["lower_strain_than_WT"] = master_df.apply(lambda r: compare_vs_wt(r, wt_base, "strain"), axis=1)

    # WT-like / alternative based on pose family
    wt_pose_family = wt_base["assigned_pose_family"] if wt_base is not None else np.nan

    def pose_tag(row):
        if int(row["is_wt"]) == 1:
            return "WT"
        pf = row["assigned_pose_family"]
        if pd.isna(pf) or pd.isna(wt_pose_family):
            return "ambiguous"
        if int(pf) == int(wt_pose_family):
            return "WT-like"
        if row["pose_cluster_consistency"] >= 0.6:
            return "alternative-pose"
        return "ambiguous"

    master_df["WT_like_or_alternative"] = master_df.apply(pose_tag, axis=1)

    def structural_case(row):
        if int(row["is_wt"]) == 1:
            return "WT_reference"
        raw = row["better_than_WT_on_raw_AF3_metrics"]
        con = row["more_consistent_than_WT"]
        strain = row["lower_strain_than_WT"]
        if raw == "yes" and con == "yes" and strain != "no":
            return "strong_structural_case"
        if raw in {"yes", "mixed"} and con in {"yes", "mixed"} and strain in {"yes", "mixed"}:
            return "plausible_but_mixed"
        if raw == "no" and con == "no":
            return "weaker_than_wt"
        return "uncertain"

    master_df["overall_structural_case_vs_WT"] = master_df.apply(structural_case, axis=1)

    # Rankings (transparent rank-based)
    rank_df = master_df.copy()

    def add_rank(df: pd.DataFrame, col: str, higher_better: bool, outcol: str):
        x = df[col]
        if higher_better:
            r = x.rank(method="average", ascending=False, na_option="bottom")
        else:
            r = x.rank(method="average", ascending=True, na_option="bottom")
        df[outcol] = r

    raw_components = [
        ("mean_ranking_score", True, "rank_mean_ranking_score"),
        ("mean_ipTM", True, "rank_mean_ipTM"),
        ("mean_pTM", True, "rank_mean_pTM"),
        ("mean_mean_pLDDT", True, "rank_mean_mean_pLDDT"),
        ("mean_interface_pLDDT", True, "rank_mean_interface_pLDDT"),
        ("mean_interchain_pair_ipTM", True, "rank_mean_interchain_pair_ipTM"),
        ("mean_interchain_pair_PAE", False, "rank_mean_interchain_pair_PAE"),
        ("best_interchain_pair_PAE", False, "rank_best_interchain_pair_PAE"),
    ]

    robust_components = [
        ("within_variant_model_consistency", True, "rank_within_variant_model_consistency"),
        ("pose_cluster_consistency", True, "rank_pose_cluster_consistency"),
        ("contact_fingerprint_consistency", True, "rank_contact_fingerprint_consistency"),
        ("hotspot_reproducibility", True, "rank_hotspot_reproducibility"),
        ("strain_clash_flag_rate", False, "rank_strain_clash_flag_rate"),
        ("forced_fit_warning_rate", False, "rank_forced_fit_warning_rate"),
        ("run_concordance_score", True, "rank_run_concordance_score"),
    ]

    for c, hb, rc in raw_components:
        add_rank(rank_df, c, hb, rc)
    for c, hb, rc in robust_components:
        add_rank(rank_df, c, hb, rc)

    raw_rank_cols = [x[2] for x in raw_components]
    robust_rank_cols = [x[2] for x in robust_components]

    rank_df["raw_af3_confidence_rank_score"] = rank_df[raw_rank_cols].mean(axis=1)
    rank_df["raw_af3_confidence_rank"] = rank_df["raw_af3_confidence_rank_score"].rank(method="average", ascending=True)

    rank_df["structural_robustness_rank_score"] = rank_df[robust_rank_cols].mean(axis=1)
    rank_df["structural_robustness_rank"] = rank_df["structural_robustness_rank_score"].rank(method="average", ascending=True)

    # integrated transparent rank aggregation
    rank_df["integrated_rank_score"] = (
        rank_df["raw_af3_confidence_rank"] + rank_df["structural_robustness_rank"]
    ) / 2.0
    rank_df["integrated_final_rank"] = rank_df["integrated_rank_score"].rank(method="average", ascending=True)
    rank_df["integration_formula"] = "integrated_rank_score = 0.5*raw_af3_confidence_rank + 0.5*structural_robustness_rank"

    # Tier assignment with diversity-aware logic
    rank_sorted = rank_df.sort_values("integrated_final_rank").reset_index(drop=True)

    non_wt = rank_sorted[rank_sorted["is_wt"] == 0].copy()

    selected_t1 = []

    def can_add(seq: str, selected: List[str], thr: float = 0.98) -> bool:
        for s in selected:
            if seq_identity(seq, s) >= thr:
                return False
        return True

    selected_sequences = []

    # top overall candidates
    for _, r in non_wt.iterrows():
        if len(selected_t1) >= 5:
            break
        if can_add(r["sequence"], selected_sequences, 0.98):
            selected_t1.append(r["variant_id"])
            selected_sequences.append(r["sequence"])

    # ensure WT-like and alternative present
    for tag in ["WT-like", "alternative-pose"]:
        sub = non_wt[non_wt["WT_like_or_alternative"] == tag]
        if not sub.empty and not any(v in selected_t1 for v in sub["variant_id"]):
            rr = sub.sort_values("integrated_final_rank").iloc[0]
            if can_add(rr["sequence"], selected_sequences, 0.98):
                selected_t1.append(rr["variant_id"])
                selected_sequences.append(rr["sequence"])

    # add WT benchmark
    wt_variant_ids = list(rank_sorted[rank_sorted["is_wt"] == 1]["variant_id"].unique())
    if wt_variant_ids:
        selected_t1 = wt_variant_ids[:1] + selected_t1

    # Tier2: next diverse strong candidates
    selected_t2 = []
    for _, r in non_wt.iterrows():
        vid = r["variant_id"]
        if vid in selected_t1:
            continue
        if len(selected_t2) >= 10:
            break
        if can_add(r["sequence"], selected_sequences, 0.95):
            selected_t2.append(vid)
            selected_sequences.append(r["sequence"])

    def tier_of(vid: str) -> str:
        if vid in selected_t1:
            return "Tier 1"
        if vid in selected_t2:
            return "Tier 2"
        row = rank_sorted[rank_sorted["variant_id"] == vid].iloc[0]
        if row["integrated_final_rank"] <= max(15, 0.4 * len(rank_sorted)):
            return "Tier 3"
        return "Hold"

    tier_rows = []
    for _, r in rank_sorted.iterrows():
        vid = r["variant_id"]
        tier = tier_of(vid)
        rec = "yes" if tier in {"Tier 1", "Tier 2"} else "no"

        strengths = []
        risks = []
        if r["better_than_WT_on_raw_AF3_metrics"] == "yes":
            strengths.append("raw_AF3_vs_WT")
        if r["more_consistent_than_WT"] == "yes":
            strengths.append("model_consistency")
        if r["lower_strain_than_WT"] == "yes":
            strengths.append("low_strain")
        if r["WT_like_or_alternative"] == "WT-like":
            strengths.append("WT_like_pose")
        if r["WT_like_or_alternative"] == "alternative-pose":
            strengths.append("alternative_pose_diversity")

        if r["forced_fit_warning_rate"] > 0.2:
            risks.append("forced_fit_risk")
        if r["strain_clash_flag_rate"] > 0.1:
            risks.append("clash_risk")
        if r["more_consistent_than_WT"] == "no":
            risks.append("low_reproducibility")
        if r["better_than_WT_on_raw_AF3_metrics"] == "no":
            risks.append("weak_raw_AF3")

        if not strengths:
            strengths = ["moderate_interface_signal"]
        if not risks:
            risks = ["no_major_structural_alert"]

        rationale = (
            f"raw_vs_WT={r['better_than_WT_on_raw_AF3_metrics']}, "
            f"consistency_vs_WT={r['more_consistent_than_WT']}, "
            f"strain_vs_WT={r['lower_strain_than_WT']}, "
            f"pose={r['WT_like_or_alternative']}"
        )

        tier_rows.append(
            {
                "variant_id": vid,
                "tier": tier,
                "recommended_for_wetlab": rec,
                "rationale_short": rationale,
                "key_strength": ";".join(strengths),
                "key_risk": ";".join(risks),
                "pose_family": r["assigned_pose_family"],
                "WT_like_or_alternative": r["WT_like_or_alternative"],
            }
        )

    tiers_df = pd.DataFrame(tier_rows)

    # Join tier into rankings
    rank_sorted = rank_sorted.merge(tiers_df[["variant_id", "tier", "recommended_for_wetlab"]], on="variant_id", how="left")

    # Save required tables
    master_csv = outdir / "master_variant_table.csv"
    prov_csv = outdir / "variant_provenance_table.csv"
    raw_rank_csv = outdir / "raw_af3_confidence_ranking.csv"
    robust_rank_csv = outdir / "structural_robustness_ranking.csv"
    integrated_csv = outdir / "integrated_final_ranking.csv"
    wetlab_csv = outdir / "wetlab_tier_recommendations.csv"

    master_df.to_csv(master_csv, index=False)

    # provenance table with source files
    run_prov_out = run_prov.copy()
    # map variant id for provenance
    seq_to_vid = dict(zip(master_df["sequence_hash"], master_df["variant_id"]))
    run_prov_out["variant_id"] = run_prov_out["sequence_hash"].map(seq_to_vid)
    run_prov_out["source_summary_confidences_0"] = run_prov_out.apply(
        lambda r: str(Path(r["run_dir"]) / next(iter([f.name for f in Path(r["run_dir"]).glob("*_summary_confidences_0.json")]), "")),
        axis=1,
    )
    run_prov_out.to_csv(prov_csv, index=False)

    raw_cols = [
        "variant_id",
        "is_wt",
        "number_of_models_available",
        "mean_ranking_score",
        "mean_ipTM",
        "mean_pTM",
        "mean_mean_pLDDT",
        "mean_interface_pLDDT",
        "mean_interchain_pair_ipTM",
        "mean_interchain_pair_PAE",
        "best_interchain_pair_PAE",
    ] + [x[2] for x in raw_components] + [
        "raw_af3_confidence_rank_score",
        "raw_af3_confidence_rank",
    ]
    rank_df.sort_values("raw_af3_confidence_rank")[raw_cols].to_csv(raw_rank_csv, index=False)

    robust_cols = [
        "variant_id",
        "is_wt",
        "number_of_models_available",
        "within_variant_model_consistency",
        "pose_cluster_consistency",
        "contact_fingerprint_consistency",
        "hotspot_reproducibility",
        "strain_clash_flag_rate",
        "forced_fit_warning_rate",
        "run_concordance",
    ] + [x[2] for x in robust_components] + [
        "structural_robustness_rank_score",
        "structural_robustness_rank",
    ]
    rank_df.sort_values("structural_robustness_rank")[robust_cols].to_csv(robust_rank_csv, index=False)

    integrated_cols = [
        "variant_id",
        "is_wt",
        "WT_like_or_alternative",
        "tier",
        "recommended_for_wetlab",
        "raw_af3_confidence_rank",
        "structural_robustness_rank",
        "integrated_rank_score",
        "integrated_final_rank",
        "integration_formula",
        "mean_ranking_score",
        "mean_ipTM",
        "mean_pTM",
        "mean_interface_pLDDT",
        "mean_interchain_pair_PAE",
        "best_interchain_pair_PAE",
        "within_variant_model_consistency",
        "pose_cluster_consistency",
        "contact_fingerprint_consistency",
        "strain_clash_flag_rate",
        "forced_fit_warning_rate",
        "better_than_WT_on_raw_AF3_metrics",
        "more_consistent_than_WT",
        "lower_strain_than_WT",
        "overall_structural_case_vs_WT",
    ]
    rank_sorted[integrated_cols].sort_values("integrated_final_rank").to_csv(integrated_csv, index=False)

    tiers_df.to_csv(wetlab_csv, index=False)

    # Figures
    sns.set_theme(style="whitegrid")

    # ranking comparison
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=rank_sorted,
        x="raw_af3_confidence_rank",
        y="structural_robustness_rank",
        hue="tier",
        style="WT_like_or_alternative",
        s=90,
    )
    plt.title("Ranking comparison: Raw AF3 vs Structural Robustness")
    plt.xlabel("Raw AF3 confidence rank (lower better)")
    plt.ylabel("Structural robustness rank (lower better)")
    plt.tight_layout()
    plt.savefig(figdir / "ranking_comparison_plot.png", dpi=180)
    plt.close()

    # WT vs top candidates metrics
    top_non_wt = rank_sorted[rank_sorted["is_wt"] == 0].sort_values("integrated_final_rank").head(8)
    wt_row_df = rank_sorted[rank_sorted["is_wt"] == 1].head(1)
    comp = pd.concat([wt_row_df, top_non_wt], ignore_index=True)
    metrics = ["mean_ipTM", "mean_interchain_pair_PAE", "mean_interface_pLDDT", "within_variant_model_consistency"]
    mlong = comp[["variant_id"] + metrics].melt(id_vars=["variant_id"], var_name="metric", value_name="value")
    g = sns.catplot(data=mlong, x="variant_id", y="value", col="metric", col_wrap=2, kind="bar", sharey=False, height=3.4)
    g.set_xticklabels(rotation=90)
    g.fig.suptitle("WT vs top integrated candidates on key AF3 metrics", y=1.03)
    plt.tight_layout()
    g.savefig(figdir / "wt_vs_top_candidates_key_metrics.png", dpi=180)
    plt.close('all')

    # pose family distribution among top candidates
    top20 = rank_sorted[rank_sorted["is_wt"] == 0].sort_values("integrated_final_rank").head(20)
    plt.figure(figsize=(7, 4))
    sns.countplot(data=top20, x="WT_like_or_alternative")
    plt.title("Pose family distribution among top20 non-WT candidates")
    plt.tight_layout()
    plt.savefig(figdir / "pose_family_distribution_top20.png", dpi=180)
    plt.close()

    # consistency vs confidence scatter
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=rank_sorted,
        x="mean_ipTM",
        y="within_variant_model_consistency",
        hue="tier",
        style="WT_like_or_alternative",
        s=90,
    )
    plt.title("Consistency vs confidence")
    plt.xlabel("mean ipTM")
    plt.ylabel("within-variant model consistency")
    plt.tight_layout()
    plt.savefig(figdir / "consistency_vs_confidence_scatter.png", dpi=180)
    plt.close()

    # optional heatmap: top-candidate contact reproducibility (consensus-set similarity)
    heat_vars = rank_sorted[rank_sorted["is_wt"] == 0].sort_values("integrated_final_rank").head(12)["sequence_hash"].tolist()
    if wt_base is not None:
        heat_vars = [wt_base["sequence_hash"]] + heat_vars
    heat_labels = [master_df.set_index("sequence_hash").loc[h, "variant_id"] for h in heat_vars if h in set(master_df["sequence_hash"])]
    H = np.zeros((len(heat_vars), len(heat_vars)))
    for i, a in enumerate(heat_vars):
        for j, b in enumerate(heat_vars):
            sa = consensus_contact_sets.get(a, set())
            sb = consensus_contact_sets.get(b, set())
            if not sa and not sb:
                H[i, j] = 1.0
            elif not sa or not sb:
                H[i, j] = 0.0
            else:
                H[i, j] = len(sa & sb) / len(sa | sb)
    if len(heat_vars) > 0:
        plt.figure(figsize=(10, 8))
        sns.heatmap(H, xticklabels=heat_labels, yticklabels=heat_labels, cmap="viridis", vmin=0, vmax=1)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title("Top-candidate consensus-contact reproducibility (Jaccard)")
        plt.tight_layout()
        plt.savefig(figdir / "top_candidate_contact_reproducibility_heatmap.png", dpi=180)
        plt.close()

    # Build summary report with explicit final judgments
    top_overall_non_wt = rank_sorted[rank_sorted["is_wt"] == 0].sort_values("integrated_final_rank")
    strongest = top_overall_non_wt.iloc[0] if not top_overall_non_wt.empty else None

    # credible challengers: non-WT with raw=yes and consistency in {yes,mixed} and strain not no
    credible = master_df[
        (master_df["is_wt"] == 0)
        & (master_df["better_than_WT_on_raw_AF3_metrics"] == "yes")
        & (master_df["more_consistent_than_WT"].isin(["yes", "mixed"]))
        & (master_df["lower_strain_than_WT"] != "no")
    ]

    wet_yes = tiers_df[tiers_df["recommended_for_wetlab"] == "yes"]
    diversity_only = tiers_df[
        (tiers_df["tier"] == "Tier 2")
        & (tiers_df["key_strength"].str.contains("alternative_pose_diversity", na=False))
    ]

    report = outdir / "summary_report.md"

    lines = []
    lines.append("# AF3 Project Master Ranking and Wet-Lab Triage")
    lines.append("")
    lines.append("## Overall project summary")
    lines.append(f"- AF3 run directories parsed: **{len(run_dirs)}**")
    lines.append(f"- AF3 models parsed: **{int(model_df.shape[0])}**")
    lines.append(f"- Unique variant sequences: **{int(master_df.shape[0])}**")
    lines.append(f"- WT-like variants: **{int((master_df['WT_like_or_alternative']=='WT-like').sum())}**")
    lines.append(f"- Alternative-pose variants: **{int((master_df['WT_like_or_alternative']=='alternative-pose').sum())}**")
    lines.append("")

    lines.append("## Ranking methodology (transparent)")
    lines.append("- Raw AF3 confidence ranking: rank-based average of mean/best AF3-native metrics.")
    lines.append("- Structural robustness ranking: rank-based average of reproducibility, pose/contact consistency, hotspot reproducibility, and strain flags.")
    lines.append("- Integrated final ranking: **0.5 × raw rank + 0.5 × robustness rank** (explicit, no hidden weighting).")
    lines.append("")

    lines.append("## Tiering strategy")
    lines.append("- Tier 1: strongest primary candidates + mandatory WT benchmark.")
    lines.append("- Tier 2: strong secondary and diversity-preserving candidates.")
    lines.append("- Tier 3: structurally interesting but less convincing.")
    lines.append("- Hold: currently not recommended.")
    lines.append("")

    # one-paragraph rationale for Tier1/Tier2
    lines.append("## Tier 1 / Tier 2 rationale by candidate")
    lines.append("")
    rank_lookup = rank_sorted.set_index("variant_id")
    for tier_name in ["Tier 1", "Tier 2"]:
        subset = tiers_df[tiers_df["tier"] == tier_name]
        if subset.empty:
            continue
        lines.append(f"### {tier_name}")
        for _, tr in subset.iterrows():
            vid = tr["variant_id"]
            if vid not in rank_lookup.index:
                continue
            rr = rank_lookup.loc[vid]
            lines.append(
                f"- **{vid}**: Integrated rank {rr['integrated_final_rank']:.1f}; "
                f"raw-vs-WT={rr['better_than_WT_on_raw_AF3_metrics']}, "
                f"consistency-vs-WT={rr['more_consistent_than_WT']}, "
                f"strain-vs-WT={rr['lower_strain_than_WT']}, "
                f"pose={rr['WT_like_or_alternative']}. "
                f"Strengths: {tr['key_strength']}. Risks: {tr['key_risk']}."
            )
        lines.append("")

    lines.append("## Final judgment questions")
    # Q1
    lines.append(
        f"1. **Did this project produce structurally credible challengers to WT?** "
        f"{('Yes' if not credible.empty else 'Not clearly')} — credible count (strict structural criteria) = {int(credible.shape[0])}."
    )
    # Q2
    if strongest is not None:
        lines.append(
            f"2. **Which candidate currently looks strongest overall?** "
            f"{strongest['variant_id']} (integrated rank {strongest['integrated_final_rank']:.1f})."
        )
    else:
        lines.append("2. **Which candidate currently looks strongest overall?** No non-WT variant parsed.")
    # Q3
    if strongest is not None:
        lines.append(
            f"3. **Is the strongest non-WT candidate WT-like or alternative-pose?** "
            f"{strongest['WT_like_or_alternative']}."
        )
    else:
        lines.append("3. **Is the strongest non-WT candidate WT-like or alternative-pose?** N/A.")
    # Q4
    lines.append(
        f"4. **How many candidates are genuinely worth carrying into wet-lab testing?** "
        f"{int(wet_yes.shape[0])} (Tier 1 + Tier 2, including WT benchmark)."
    )
    # Q5
    lines.append(
        f"5. **Which candidates are included mainly for diversity/interpretability?** "
        f"{', '.join(diversity_only['variant_id'].tolist()) if not diversity_only.empty else 'No explicit diversity-only entries flagged.'}"
    )
    # Q6
    lines.append(
        "6. **Main remaining uncertainties AF3 alone cannot resolve:** "
        "(i) true binding kinetics/thermodynamics, (ii) expression/folding/aggregation in experimental systems, "
        "(iii) whether alternative-pose solutions are physically sampled in reality or modeling artifacts, "
        "(iv) epitope accessibility under biological context dynamics."
    )

    lines.append("")
    lines.append("## Major uncertainties and caution flags")
    lines.append("- Some variants appear in only one AF3 run; cross-run concordance is therefore limited.")
    lines.append("- Sequence-level deduplication is strict (exact sequence), but functional near-neighbors may still behave similarly.")
    lines.append("- Contact-type inference (H-bond/salt/hydrophobic) is geometric and heuristic, not an energy calculation.")

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Save auxiliary raw tables
    model_df.to_csv(outdir / "all_models_evidence_table.csv", index=False)
    contact_df.to_csv(outdir / "all_models_interface_contacts_long.csv", index=False)

    # Package zip
    zip_path = outdir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(outdir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(outdir.parent))

    print(f"Wrote: {master_csv}")
    print(f"Wrote: {prov_csv}")
    print(f"Wrote: {raw_rank_csv}")
    print(f"Wrote: {robust_rank_csv}")
    print(f"Wrote: {integrated_csv}")
    print(f"Wrote: {wetlab_csv}")
    print(f"Wrote: {report}")
    print(f"Wrote figures in: {figdir}")
    print(f"Packaged: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
