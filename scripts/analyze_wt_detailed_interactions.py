#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Bio.PDB import MMCIFParser

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


AROMATIC_RES = {"PHE", "TYR", "TRP", "HIS"}
HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "PRO"}
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


@dataclass
class ModelContext:
    model_id: str
    group: str
    cif_path: Path
    full_json_path: Path
    summary_json_path: Path
    nb_chain: str
    ag_chains: List[str]



def load_json(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj[0] if obj else {}
    return obj



def is_heavy(atom) -> bool:
    e = (atom.element or "").strip().upper()
    n = atom.get_name().strip().upper()
    if e:
        return e != "H"
    return not n.startswith("H")



def residue_label(chain_id: str, residue) -> Optional[str]:
    if residue.id[0] != " ":
        return None
    return f"{chain_id}:{int(residue.id[1])}"



def split_label(label: str) -> Tuple[str, int]:
    ch, num = str(label).split(":", 1)
    return ch, int(num)



def acidic_atom(resname: str, atom_name: str) -> bool:
    return (resname == "ASP" and atom_name.startswith("OD")) or (resname == "GLU" and atom_name.startswith("OE"))



def basic_atom(resname: str, atom_name: str) -> bool:
    if resname == "LYS":
        return atom_name == "NZ"
    if resname == "ARG":
        return atom_name.startswith("NH") or atom_name == "NE"
    if resname == "HIS":
        return atom_name.startswith("ND") or atom_name.startswith("NE")
    return False



def donor_acceptor_atom(atom_name: str, element: str) -> bool:
    if element in {"N", "O", "S"}:
        return True
    if atom_name and atom_name[0] in {"N", "O", "S"}:
        return True
    return False



def cdr_region(pos: int, cdr_defs: Dict[str, Tuple[int, int]]) -> str:
    for name in ("CDR1", "CDR2", "CDR3"):
        s, e = cdr_defs[name]
        if s <= pos <= e:
            return name
    return "framework"



def load_cdr_defs(config_path: Path) -> Dict[str, Tuple[int, int]]:
    # Default consistent with project convention
    defaults = {"CDR1": (23, 34), "CDR2": (50, 58), "CDR3": (97, 106)}
    if yaml is None or not config_path.exists():
        return defaults
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    c = cfg.get("cdr_boundaries", {}) if isinstance(cfg, dict) else {}

    def _pair(key: str, dflt: Tuple[int, int]) -> Tuple[int, int]:
        v = c.get(key)
        if isinstance(v, list) and len(v) == 2:
            try:
                return int(v[0]), int(v[1])
            except Exception:
                return dflt
        return dflt

    return {
        "CDR1": _pair("H1", defaults["CDR1"]),
        "CDR2": _pair("H2", defaults["CDR2"]),
        "CDR3": _pair("H3", defaults["CDR3"]),
    }



def collect_wt_models(wt_dirs: List[Tuple[str, Path]]) -> List[ModelContext]:
    out: List[ModelContext] = []
    for group, d in wt_dirs:
        summary_files = sorted(d.glob("*_summary_confidences_*.json"))
        for sf in summary_files:
            idx = sf.stem.split("_")[-1]
            fullf = d / sf.name.replace("_summary_confidences_", "_full_data_")
            ciff = d / sf.name.replace("_summary_confidences_", "_model_").replace(".json", ".cif")
            if not fullf.exists() or not ciff.exists():
                continue
            full = load_json(fullf)
            chain_ids = list(full.get("token_chain_ids", []))
            if not chain_ids:
                continue
            order = []
            for x in chain_ids:
                if x not in order:
                    order.append(x)
            counts = Counter(chain_ids)
            nb_chain = min(order, key=lambda c: counts.get(c, 10**9))
            ag_chains = [c for c in order if c != nb_chain]
            out.append(
                ModelContext(
                    model_id=f"{group}:{d.name}:m{idx}",
                    group=group,
                    cif_path=ciff,
                    full_json_path=fullf,
                    summary_json_path=sf,
                    nb_chain=nb_chain,
                    ag_chains=ag_chains,
                )
            )
    return out



def detect_contacts_for_model(ctx: ModelContext, parser: MMCIFParser, cdr_defs: Dict[str, Tuple[int, int]]):
    structure = parser.get_structure(ctx.model_id, str(ctx.cif_path))
    model = next(structure.get_models())
    chains = {str(c.id): c for c in model.get_chains()}

    if ctx.nb_chain not in chains:
        return [], {}

    nb_atoms: List[dict] = []
    ag_atoms: List[dict] = []
    residue_atoms: Dict[str, List[dict]] = defaultdict(list)

    for cid, ch in chains.items():
        for res in ch.get_residues():
            lbl = residue_label(cid, res)
            if not lbl:
                continue
            resname = res.get_resname().strip().upper()
            for atom in res.get_atoms():
                if not is_heavy(atom):
                    continue
                an = atom.get_name().strip().upper()
                rec = {
                    "coord": atom.coord.astype(float),
                    "chain": cid,
                    "label": lbl,
                    "resname": resname,
                    "resnum": int(lbl.split(":", 1)[1]),
                    "atom": an,
                    "element": (atom.element or "").strip().upper(),
                    "is_backbone": an in BACKBONE_ATOMS,
                }
                residue_atoms[lbl].append(rec)
                if cid == ctx.nb_chain:
                    nb_atoms.append(rec)
                elif cid in ctx.ag_chains:
                    ag_atoms.append(rec)

    if not nb_atoms or not ag_atoms:
        return [], residue_atoms

    cutoff = 4.5
    hb_cut = 3.5
    hb_loose_cut = 4.2
    salt_cut = 4.0
    salt_loose_cut = 5.2
    cation_pi_cut = 6.0

    pair_info = defaultdict(
        lambda: {
            "min_dist": 999.0,
            "atom_contacts": 0,
            "hbond": 0,
            "hbond_loose": 0,
            "salt": 0,
            "salt_loose": 0,
            "hydrophobic": 0,
            "cation_pi": 0,
            "pi_pi": 0,
        }
    )

    ag_xyz = np.asarray([a["coord"] for a in ag_atoms], dtype=float)

    if cKDTree is not None:
        tree = cKDTree(ag_xyz)
        neighbors = [tree.query_ball_point(a["coord"], r=cutoff) for a in nb_atoms]
    else:
        neighbors = []
        for a in nb_atoms:
            d = np.linalg.norm(ag_xyz - a["coord"], axis=1)
            neighbors.append(np.where(d <= cutoff)[0].tolist())

    # Precompute aromatic centroids by residue for pi-pi
    aromatic_cent = {}
    for lbl, atoms in residue_atoms.items():
        rn = atoms[0]["resname"] if atoms else ""
        if rn not in AROMATIC_RES:
            continue
        pts = [x["coord"] for x in atoms if not x["is_backbone"]]
        if pts:
            aromatic_cent[lbl] = np.mean(np.asarray(pts, dtype=float), axis=0)

    for i, na in enumerate(nb_atoms):
        for j in neighbors[i]:
            aa = ag_atoms[int(j)]
            dist = float(np.linalg.norm(na["coord"] - aa["coord"]))
            key = (na["label"], aa["label"])
            rec = pair_info[key]
            rec["atom_contacts"] += 1
            rec["min_dist"] = min(rec["min_dist"], dist)

            # H-bond-ish
            if donor_acceptor_atom(na["atom"], na["element"]) and donor_acceptor_atom(aa["atom"], aa["element"]):
                if dist <= hb_cut:
                    rec["hbond"] += 1
                elif dist <= hb_loose_cut:
                    rec["hbond_loose"] += 1

            # Salt bridge-ish
            if (acidic_atom(na["resname"], na["atom"]) and basic_atom(aa["resname"], aa["atom"])) or (
                basic_atom(na["resname"], na["atom"]) and acidic_atom(aa["resname"], aa["atom"])
            ):
                if dist <= salt_cut:
                    rec["salt"] += 1
                elif dist <= salt_loose_cut:
                    rec["salt_loose"] += 1

            # hydrophobic contact
            if na["resname"] in HYDROPHOBIC_RES and aa["resname"] in HYDROPHOBIC_RES and dist <= 4.5:
                rec["hydrophobic"] += 1

            # cation-pi-ish
            if dist <= cation_pi_cut:
                na_cation = basic_atom(na["resname"], na["atom"])
                aa_cation = basic_atom(aa["resname"], aa["atom"])
                na_aro = na["resname"] in AROMATIC_RES
                aa_aro = aa["resname"] in AROMATIC_RES
                if (na_cation and aa_aro) or (aa_cation and na_aro):
                    rec["cation_pi"] += 1

    # pi-pi by residue centroid distance
    for (nb_lbl, ag_lbl), rec in pair_info.items():
        if nb_lbl in aromatic_cent and ag_lbl in aromatic_cent:
            dcen = float(np.linalg.norm(aromatic_cent[nb_lbl] - aromatic_cent[ag_lbl]))
            if dcen <= 6.0:
                rec["pi_pi"] += 1

    rows = []
    for (nb_lbl, ag_lbl), rec in pair_info.items():
        _, nb_pos = split_label(nb_lbl)
        reg = cdr_region(nb_pos, cdr_defs)
        if reg not in {"CDR1", "CDR2", "CDR3"}:
            continue
        rows.append(
            {
                "model_id": ctx.model_id,
                "group": ctx.group,
                "nb_label": nb_lbl,
                "ag_label": ag_lbl,
                "nb_pos": nb_pos,
                "cdr_region": reg,
                "ag_pos": split_label(ag_lbl)[1],
                "min_dist": float(rec["min_dist"]),
                "atom_contacts": int(rec["atom_contacts"]),
                "hbond": int(rec["hbond"]),
                "hbond_loose": int(rec["hbond_loose"]),
                "salt": int(rec["salt"]),
                "salt_loose": int(rec["salt_loose"]),
                "hydrophobic": int(rec["hydrophobic"]),
                "cation_pi": int(rec["cation_pi"]),
                "pi_pi": int(rec["pi_pi"]),
            }
        )

    return rows, residue_atoms



def dominant_type(row: pd.Series) -> str:
    scores = {
        "salt_bridge": row["salt_models"],
        "hydrogen_bond": row["hbond_models"],
        "cation_pi": row["cation_pi_models"],
        "pi_pi": row["pi_pi_models"],
        "hydrophobic": row["hydrophobic_models"],
        "heavy_contact": row["n_models_present"],
    }
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[0][0]



def strength_score(row: pd.Series) -> float:
    w = {
        "salt_bridge": 5.0,
        "hydrogen_bond": 4.0,
        "cation_pi": 4.0,
        "pi_pi": 3.0,
        "hydrophobic": 2.0,
        "heavy_contact": 1.0,
    }
    dt = row["dominant_type"]
    occ = row["occupancy"]
    median_dist = row["median_min_dist"]
    strong_frac = row["strong_models"] / max(1.0, row["n_models_present"])
    dist_bonus = 1.0 + 0.30 * max(0.0, min(1.0, (4.5 - float(median_dist)) / 2.5))
    strong_bonus = 1.0 + 0.40 * strong_frac
    return 100.0 * w.get(dt, 1.0) * occ * dist_bonus * strong_bonus



def summarize_pairs(model_df: pd.DataFrame, n_models: int) -> pd.DataFrame:
    g = model_df.groupby(["nb_label", "ag_label", "nb_pos", "cdr_region", "ag_pos"], as_index=False).agg(
        n_models_present=("model_id", "nunique"),
        median_min_dist=("min_dist", "median"),
        mean_min_dist=("min_dist", "mean"),
        mean_atom_contacts=("atom_contacts", "mean"),
        salt_models=("salt", lambda x: int((x > 0).sum())),
        hbond_models=("hbond", lambda x: int((x > 0).sum())),
        cation_pi_models=("cation_pi", lambda x: int((x > 0).sum())),
        pi_pi_models=("pi_pi", lambda x: int((x > 0).sum())),
        hydrophobic_models=("hydrophobic", lambda x: int((x > 0).sum())),
        near_salt_models=("salt_loose", lambda x: int((x > 0).sum())),
        near_hbond_models=("hbond_loose", lambda x: int((x > 0).sum())),
    )
    g["occupancy"] = g["n_models_present"] / float(n_models)
    g["strong_models"] = g[["salt_models", "hbond_models", "cation_pi_models"]].max(axis=1)
    g["dominant_type"] = g.apply(dominant_type, axis=1)
    g["strength_score"] = g.apply(strength_score, axis=1)

    # primary / secondary labels
    primary_mask = (
        (g["occupancy"] >= 0.50)
        & (
            (g["salt_models"] >= 2)
            | (g["hbond_models"] >= 4)
            | (g["cation_pi_models"] >= 2)
            | ((g["dominant_type"].isin(["salt_bridge", "hydrogen_bond", "cation_pi"])) & (g["strength_score"] >= 170.0))
        )
    )

    secondary_mask = (
        (~primary_mask)
        & (g["occupancy"] >= 0.25)
        & (
            (g["hydrophobic_models"] >= 2)
            | (g["pi_pi_models"] >= 2)
            | (g["hbond_models"] >= 1)
            | (g["near_salt_models"] >= 1)
            | (g["near_hbond_models"] >= 1)
        )
    )

    g["interaction_tier"] = "other"
    g.loc[primary_mask, "interaction_tier"] = "primary_strong"
    g.loc[secondary_mask, "interaction_tier"] = "secondary_support"

    g["secondary_subtype"] = ""
    g.loc[(g["interaction_tier"] == "secondary_support") & ((g["near_salt_models"] > 0) | (g["near_hbond_models"] > 0)), "secondary_subtype"] = "incomplete_strong"
    g.loc[(g["interaction_tier"] == "secondary_support") & (g["secondary_subtype"] == ""), "secondary_subtype"] = "auxiliary_support"

    return g



def build_support_links(primary_df: pd.DataFrame, secondary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if primary_df.empty or secondary_df.empty:
        return pd.DataFrame(rows)

    p = primary_df.copy()
    s = secondary_df.copy()

    for _, sr in s.iterrows():
        candidates = p[(p["cdr_region"] == sr["cdr_region"])].copy()
        if candidates.empty:
            candidates = p.copy()

        candidates["delta_nb"] = (candidates["nb_pos"] - sr["nb_pos"]).abs()
        candidates["delta_ag"] = (candidates["ag_pos"] - sr["ag_pos"]).abs()
        candidates["distance_proxy"] = candidates["delta_nb"] + candidates["delta_ag"]
        candidates = candidates.sort_values(["distance_proxy", "delta_ag", "delta_nb", "strength_score"], ascending=[True, True, True, False])

        topk = candidates.head(3)
        for _, pr in topk.iterrows():
            relation = []
            if abs(int(pr["ag_pos"]) - int(sr["ag_pos"])) <= 2:
                relation.append("same_or_adjacent_epitope_patch")
            if abs(int(pr["nb_pos"]) - int(sr["nb_pos"])) <= 2:
                relation.append("nearby_paratope_position")
            if not relation:
                relation.append("longer_range_support")

            rows.append(
                {
                    "secondary_pair": f"{sr['nb_label']}|{sr['ag_label']}",
                    "secondary_subtype": sr["secondary_subtype"],
                    "secondary_cdr": sr["cdr_region"],
                    "secondary_score": sr["strength_score"],
                    "supports_primary_pair": f"{pr['nb_label']}|{pr['ag_label']}",
                    "primary_type": pr["dominant_type"],
                    "primary_score": pr["strength_score"],
                    "relation_basis": ";".join(relation),
                }
            )

    return pd.DataFrame(rows)



def plot_radiation(primary_df: pd.DataFrame, secondary_df: pd.DataFrame, support_df: pd.DataFrame, out_png: Path):
    # Collapse antigen chain symmetry by residue number for cleaner epitope map
    primary = primary_df.copy()
    secondary = secondary_df.copy()

    primary["ag_num"] = primary["ag_pos"].astype(int)
    secondary["ag_num"] = secondary["ag_pos"].astype(int)

    primary_ag = sorted(primary["ag_num"].unique().tolist()) if not primary.empty else []
    secondary_ag = sorted(secondary["ag_num"].unique().tolist()) if not secondary.empty else []
    all_ag = sorted(set(primary_ag) | set(secondary_ag))

    primary_nb = sorted(primary["nb_pos"].unique().tolist()) if not primary.empty else []
    secondary_nb = sorted(secondary["nb_pos"].unique().tolist()) if not secondary.empty else []
    all_nb = sorted(set(primary_nb) | set(secondary_nb))

    if not all_ag:
        all_ag = [1]
    if not all_nb:
        all_nb = [1]

    # angle maps
    ag_angles = {x: 2 * math.pi * i / max(1, len(all_ag)) for i, x in enumerate(all_ag)}
    nb_angles = {x: 2 * math.pi * i / max(1, len(all_nb)) for i, x in enumerate(all_nb)}

    fig = plt.figure(figsize=(13, 13))
    ax = plt.subplot(111, projection="polar")
    ax.set_theta_direction(-1)
    ax.set_theta_offset(math.pi / 2)
    ax.set_ylim(0, 1.45)
    ax.grid(alpha=0.25)

    # nodes
    for ag in all_ag:
        is_p = ag in set(primary_ag)
        is_s = ag in set(secondary_ag)
        r = 0.62
        color = "#d62728" if is_p else ("#ff7f0e" if is_s else "#bdbdbd")
        size = 120 if is_p else 80
        ax.scatter([ag_angles[ag]], [r], c=[color], s=size, zorder=5)
        ax.text(ag_angles[ag], r - 0.06, f"Ag:{ag}", fontsize=7, ha="center", va="center")

    # Nanobody nodes
    for nb in all_nb:
        reg = ""
        if not primary.empty and nb in set(primary["nb_pos"]):
            reg = primary[primary["nb_pos"] == nb]["cdr_region"].iloc[0]
        elif not secondary.empty and nb in set(secondary["nb_pos"]):
            reg = secondary[secondary["nb_pos"] == nb]["cdr_region"].iloc[0]
        color = {"CDR1": "#1f77b4", "CDR2": "#2ca02c", "CDR3": "#9467bd"}.get(reg, "#7f7f7f")
        ax.scatter([nb_angles[nb]], [1.18], c=[color], s=70, zorder=6)
        ax.text(nb_angles[nb], 1.26, f"{reg}:{nb}", fontsize=7, ha="center", va="center")

    # primary edges
    for _, r in primary.iterrows():
        th = [nb_angles[int(r["nb_pos"])], ag_angles[int(r["ag_num"])]]
        rr = [1.18, 0.62]
        lw = 1.3 + 2.2 * float(r["occupancy"])
        ax.plot(th, rr, color="#d62728", lw=lw, alpha=0.7)

    # secondary edges
    for _, r in secondary.iterrows():
        th = [nb_angles[int(r["nb_pos"])], ag_angles[int(r["ag_num"])]]
        rr = [1.18, 0.62]
        lw = 0.8 + 1.8 * float(r["occupancy"])
        ax.plot(th, rr, color="#ff7f0e", lw=lw, alpha=0.65, linestyle="--")

    # support edges (ag -> ag)
    if not support_df.empty:
        # parse antigen residues from pair strings
        def agnum_from_pair(pair: str) -> Optional[int]:
            try:
                ag = pair.split("|", 1)[1]
                return int(ag.split(":", 1)[1])
            except Exception:
                return None

        for _, r in support_df.iterrows():
            ag_s = agnum_from_pair(r["secondary_pair"])
            ag_p = agnum_from_pair(r["supports_primary_pair"])
            if ag_s is None or ag_p is None:
                continue
            if ag_s not in ag_angles or ag_p not in ag_angles:
                continue
            ax.plot([ag_angles[ag_s], ag_angles[ag_p]], [0.62, 0.62], color="#8c564b", lw=0.8, alpha=0.4, linestyle=":")

    ax.set_title("WT CDR1/2/3 - Antigen epitope radiation map\nRed: primary strong interactions | Orange: secondary support | Brown dotted: support-to-primary links", fontsize=11)
    ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)



def draw_pair_image(cif_path: Path, nb_label: str, ag_label: str, out_png: Path, title: str):
    parser = MMCIFParser(QUIET=True)
    st = parser.get_structure(out_png.stem, str(cif_path))
    model = next(st.get_models())

    def collect_res_atoms(label: str):
        ch, rn = split_label(label)
        for chain in model.get_chains():
            if str(chain.id) != ch:
                continue
            for res in chain.get_residues():
                if res.id[0] != " ":
                    continue
                if int(res.id[1]) == rn:
                    arr = []
                    for a in res.get_atoms():
                        if is_heavy(a):
                            arr.append((a.get_name().strip().upper(), a.coord.astype(float), (a.element or "").strip().upper()))
                    return arr
        return []

    nb_atoms = collect_res_atoms(nb_label)
    ag_atoms = collect_res_atoms(ag_label)
    if not nb_atoms or not ag_atoms:
        return

    nb_xyz = np.asarray([x[1] for x in nb_atoms], dtype=float)
    ag_xyz = np.asarray([x[1] for x in ag_atoms], dtype=float)

    # nearest interacting atom pair
    D = np.linalg.norm(nb_xyz[:, None, :] - ag_xyz[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmin(D), D.shape)
    p_nb = nb_xyz[i]
    p_ag = ag_xyz[j]

    center = np.mean(np.vstack([nb_xyz, ag_xyz]), axis=0)

    # gather context atoms within 7A from center (excluding focus residues)
    ctx = []
    for chain in model.get_chains():
        cid = str(chain.id)
        for res in chain.get_residues():
            lbl = residue_label(cid, res)
            if not lbl or lbl in {nb_label, ag_label}:
                continue
            for a in res.get_atoms():
                if not is_heavy(a):
                    continue
                c = a.coord.astype(float)
                if np.linalg.norm(c - center) <= 7.0:
                    ctx.append(c)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # context
    if ctx:
        cxyz = np.asarray(ctx, dtype=float)
        ax.scatter(cxyz[:, 0], cxyz[:, 1], cxyz[:, 2], s=6, c="#c7c7c7", alpha=0.35, depthshade=False)

    # focus residues
    ax.scatter(nb_xyz[:, 0], nb_xyz[:, 1], nb_xyz[:, 2], s=30, c="#1f77b4", depthshade=False)
    ax.scatter(ag_xyz[:, 0], ag_xyz[:, 1], ag_xyz[:, 2], s=30, c="#d62728", depthshade=False)

    # intramolecular pseudo-bonds (distance < 1.9A)
    def draw_bonds(xyz, color):
        for a in range(len(xyz)):
            for b in range(a + 1, len(xyz)):
                d = np.linalg.norm(xyz[a] - xyz[b])
                if d <= 1.9:
                    ax.plot([xyz[a, 0], xyz[b, 0]], [xyz[a, 1], xyz[b, 1]], [xyz[a, 2], xyz[b, 2]], color=color, lw=1.2, alpha=0.8)

    draw_bonds(nb_xyz, "#1f77b4")
    draw_bonds(ag_xyz, "#d62728")

    # interaction line
    ax.plot([p_nb[0], p_ag[0]], [p_nb[1], p_ag[1]], [p_nb[2], p_ag[2]], color="#ffbf00", lw=2.6, alpha=0.95)

    pad = 4.0
    mins = np.min(np.vstack([nb_xyz, ag_xyz]), axis=0) - pad
    maxs = np.max(np.vstack([nb_xyz, ag_xyz]), axis=0) + pad
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=35)

    fig.tight_layout()
    fig.savefig(out_png, dpi=240, facecolor="white")
    plt.close(fig)



def main():
    root = Path('.').resolve()
    outdir = root / 'results' / 'summaries' / 'wt_deep_interaction_analysis'
    outdir.mkdir(parents=True, exist_ok=True)
    imgdir = outdir / 'wt_primary_interaction_structures'
    imgdir.mkdir(parents=True, exist_ok=True)

    cdr_defs = load_cdr_defs(root / 'data' / 'configs' / 'cdr_boundaries.yaml')

    wt_dirs = [
        ("WT_A", root / 'AF3 Results' / 'Stage6 AF3' / 'fold_089_wt'),
        ("WT_B", root / 'AF3 Results' / 'fold_26_wildtype_nanobody_121_aa'),
    ]

    parser = MMCIFParser(QUIET=True)
    models = collect_wt_models(wt_dirs)
    if not models:
        raise SystemExit('No WT models found for analysis.')

    all_rows = []
    model_to_resatoms: Dict[str, Dict[str, List[dict]]] = {}
    for m in models:
        rows, residue_atoms = detect_contacts_for_model(m, parser, cdr_defs)
        all_rows.extend(rows)
        model_to_resatoms[m.model_id] = residue_atoms

    if not all_rows:
        raise SystemExit('No CDR-antigen contacts detected.')

    model_df = pd.DataFrame(all_rows)
    pair_df = summarize_pairs(model_df, n_models=len(models))

    primary_df = pair_df[pair_df['interaction_tier'] == 'primary_strong'].copy().sort_values('strength_score', ascending=False)
    secondary_df = pair_df[pair_df['interaction_tier'] == 'secondary_support'].copy().sort_values('strength_score', ascending=False)

    # support links
    support_df = build_support_links(primary_df, secondary_df)

    # Save tables
    cols_main = [
        'nb_label','ag_label','cdr_region','nb_pos','ag_pos','interaction_tier','secondary_subtype',
        'dominant_type','strength_score','occupancy','n_models_present','median_min_dist','mean_atom_contacts',
        'salt_models','hbond_models','cation_pi_models','pi_pi_models','hydrophobic_models','near_salt_models','near_hbond_models'
    ]
    pair_df[cols_main].sort_values(['interaction_tier','strength_score'], ascending=[True, False]).to_csv(outdir / 'wt_all_cdr_epitope_interactions_ranked.csv', index=False)
    primary_df[cols_main].to_csv(outdir / 'wt_primary_strong_interactions_ranked.csv', index=False)
    secondary_df[cols_main].to_csv(outdir / 'wt_secondary_interactions_ranked.csv', index=False)
    support_df.to_csv(outdir / 'wt_secondary_support_links.csv', index=False)
    model_df.to_csv(outdir / 'wt_model_level_pair_details.csv', index=False)

    # Radiation plot
    plot_radiation(primary_df, secondary_df, support_df, outdir / 'wt_interaction_radiation_map.png')

    # Choose best model per primary pair and render structure image
    # best = min min_dist among model-level rows
    best_rows = (
        model_df.merge(primary_df[['nb_label','ag_label','dominant_type','strength_score']], on=['nb_label','ag_label'], how='inner')
        .sort_values(['nb_label','ag_label','min_dist'])
        .groupby(['nb_label','ag_label'], as_index=False)
        .first()
    )

    # map model_id -> cif path
    model_cif = {m.model_id: m.cif_path for m in models}

    img_meta = []
    for _, r in best_rows.iterrows():
        nb = r['nb_label']
        ag = r['ag_label']
        mid = r['model_id']
        cif = model_cif.get(mid)
        if cif is None:
            continue
        out_png = imgdir / f"{nb.replace(':','_')}__{ag.replace(':','_')}.png"
        title = f"{nb} - {ag} | {r['dominant_type']} | score={r['strength_score']:.1f}"
        draw_pair_image(cif, nb, ag, out_png, title)
        img_meta.append({
            'nb_label': nb,
            'ag_label': ag,
            'model_id': mid,
            'dominant_type': r['dominant_type'],
            'strength_score': r['strength_score'],
            'image_path': str(out_png.relative_to(root)),
        })

    pd.DataFrame(img_meta).sort_values('strength_score', ascending=False).to_csv(outdir / 'wt_primary_structure_images_index.csv', index=False)

    # summary markdown
    lines = []
    lines.append('# WT detailed CDR-epitope interaction analysis')
    lines.append('')
    lines.append(f'- WT models analyzed: **{len(models)}**')
    lines.append(f"- CDR definitions: CDR1={cdr_defs['CDR1']}, CDR2={cdr_defs['CDR2']}, CDR3={cdr_defs['CDR3']}")
    lines.append(f'- Primary strong interactions: **{len(primary_df)}**')
    lines.append(f'- Secondary/support interactions: **{len(secondary_df)}**')
    lines.append('')
    lines.append('## Top 10 primary strong interactions')
    for _, r in primary_df.head(10).iterrows():
        lines.append(
            f"- {r['nb_label']} -> {r['ag_label']} ({r['cdr_region']}): {r['dominant_type']}, "
            f"score={r['strength_score']:.1f}, occ={r['occupancy']:.2f}, median_dist={r['median_min_dist']:.2f}A"
        )
    lines.append('')
    lines.append('## Secondary interactions highlights')
    for _, r in secondary_df.head(12).iterrows():
        lines.append(
            f"- {r['nb_label']} -> {r['ag_label']} ({r['cdr_region']}): {r['secondary_subtype']}, "
            f"type={r['dominant_type']}, score={r['strength_score']:.1f}, occ={r['occupancy']:.2f}"
        )

    (outdir / 'wt_detailed_interaction_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    print('Wrote:', outdir / 'wt_all_cdr_epitope_interactions_ranked.csv')
    print('Wrote:', outdir / 'wt_primary_strong_interactions_ranked.csv')
    print('Wrote:', outdir / 'wt_secondary_interactions_ranked.csv')
    print('Wrote:', outdir / 'wt_secondary_support_links.csv')
    print('Wrote:', outdir / 'wt_model_level_pair_details.csv')
    print('Wrote:', outdir / 'wt_interaction_radiation_map.png')
    print('Wrote:', outdir / 'wt_primary_structure_images_index.csv')
    print('Wrote:', outdir / 'wt_detailed_interaction_summary.md')


if __name__ == '__main__':
    main()
