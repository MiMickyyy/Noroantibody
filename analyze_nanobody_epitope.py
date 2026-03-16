#!/usr/bin/env python3
"""
Analyze AF3-predicted nanobody-antigen interfaces using transparent geometric heuristics.

This script is designed for Norovirus VP1 P-domain / nanobody complexes but can be reused
for similar protein-protein complexes with user-defined chains and CDR residue ranges.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser
    from Bio.PDB.Polypeptide import is_aa
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Biopython is required. Install with: pip install biopython"
    ) from exc

try:
    import yaml
except ImportError:
    yaml = None


ResidueKey = Tuple[str, int, str]  # (chain, resseq, icode)


DEFAULT_CONFIG = {
    "contact_cutoffs": {
        "heavy_atom": 4.5,
        "hydrogen_bond_distance": 3.5,
        "hydrogen_bond_angle": 120.0,  # Kept for transparency; not enforced in AF3 static model.
        "salt_bridge": 4.0,
        "hydrophobic": 4.5,
        "aromatic": 5.0,
    },
    "patch_clustering_distance": 6.0,
    "interface_score_weights": {
        "contact": 1.0,
        "hbond": 2.0,
        "salt_bridge": 2.5,
        "hydrophobic": 1.2,
        "aromatic": 1.5,
        "multi_cdr_bonus": 2.0,
        "cdr3_bonus": 1.5,
    },
}


HYDROPHOBIC_RESIDUES = {
    "ALA",
    "VAL",
    "LEU",
    "ILE",
    "MET",
    "PHE",
    "TRP",
    "PRO",
    "TYR",
    "CYS",
}

AROMATIC_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
}

POSITIVE_SIDECHAIN = {
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "HIS": {"ND1", "NE2"},
}

NEGATIVE_SIDECHAIN = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
}

DONOR_SIDECHAIN = {
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "TRP": {"NE1"},
    "CYS": {"SG"},
}

ACCEPTOR_SIDECHAIN = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"},
    "GLN": {"OE1"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
}

BACKBONE_DONORS = {"N"}
BACKBONE_ACCEPTORS = {"O", "OXT"}
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze nanobody-antigen interface evidence from PDB/mmCIF using "
            "contact-based geometric heuristics."
        )
    )
    parser.add_argument("--structure", type=str, help="Input structure file (.pdb or .cif/.mmcif)")
    parser.add_argument("--config", type=str, help="YAML or JSON config file")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")

    parser.add_argument(
        "--antigen-chains",
        type=str,
        help="Comma-separated antigen chain IDs, e.g. A,B",
    )
    parser.add_argument("--nanobody-chain", type=str, help="Nanobody chain ID, e.g. C")
    parser.add_argument("--cdr1", nargs=2, type=int, metavar=("START", "END"), help="CDR1 residue range")
    parser.add_argument("--cdr2", nargs=2, type=int, metavar=("START", "END"), help="CDR2 residue range")
    parser.add_argument("--cdr3", nargs=2, type=int, metavar=("START", "END"), help="CDR3 residue range")

    parser.add_argument("--heavy-cutoff", type=float, help="Heavy atom contact cutoff (A)")
    parser.add_argument("--hydrophobic-cutoff", type=float, help="Hydrophobic contact cutoff (A)")
    parser.add_argument("--salt-cutoff", type=float, help="Salt bridge cutoff (A)")
    parser.add_argument("--hbond-cutoff", type=float, help="Hydrogen bond distance cutoff (A)")
    parser.add_argument("--hbond-angle", type=float, help="Hydrogen bond angle cutoff (degrees)")
    parser.add_argument("--aromatic-cutoff", type=float, help="Aromatic interaction cutoff (A)")
    parser.add_argument("--patch-cutoff", type=float, help="Patch clustering distance cutoff (A)")
    parser.add_argument("--numbering-offset", type=int, help="Optional structure->full-length residue offset")
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Run a lightweight internal self-test and exit",
    )

    return parser.parse_args()


def deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Optional[str]) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if not config_path:
        return cfg

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required for YAML config files. Install with: pip install pyyaml"
            )
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle) or {}
    else:
        raise ValueError("Config file must be .yaml/.yml or .json")

    if not isinstance(loaded, dict):
        raise ValueError("Config top-level content must be a dictionary/object.")

    return deep_update(cfg, loaded)


def normalize_range(value: Sequence[int], label: str) -> Tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be [start, end].")
    start, end = int(value[0]), int(value[1])
    if start > end:
        start, end = end, start
    return start, end


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.structure:
        cfg["structure"] = args.structure
    if args.outdir:
        cfg["outdir"] = args.outdir
    if args.antigen_chains:
        cfg["antigen_chains"] = [x.strip() for x in args.antigen_chains.split(",") if x.strip()]
    if args.nanobody_chain:
        cfg["nanobody_chain"] = args.nanobody_chain.strip()

    cfg.setdefault("cdrs", {})
    if args.cdr1:
        cfg["cdrs"]["cdr1"] = [int(args.cdr1[0]), int(args.cdr1[1])]
    if args.cdr2:
        cfg["cdrs"]["cdr2"] = [int(args.cdr2[0]), int(args.cdr2[1])]
    if args.cdr3:
        cfg["cdrs"]["cdr3"] = [int(args.cdr3[0]), int(args.cdr3[1])]

    cutoffs = cfg.setdefault("contact_cutoffs", {})
    if args.heavy_cutoff is not None:
        cutoffs["heavy_atom"] = float(args.heavy_cutoff)
    if args.hydrophobic_cutoff is not None:
        cutoffs["hydrophobic"] = float(args.hydrophobic_cutoff)
    if args.salt_cutoff is not None:
        cutoffs["salt_bridge"] = float(args.salt_cutoff)
    if args.hbond_cutoff is not None:
        cutoffs["hydrogen_bond_distance"] = float(args.hbond_cutoff)
    if args.hbond_angle is not None:
        cutoffs["hydrogen_bond_angle"] = float(args.hbond_angle)
    if args.aromatic_cutoff is not None:
        cutoffs["aromatic"] = float(args.aromatic_cutoff)
    if args.patch_cutoff is not None:
        cfg["patch_clustering_distance"] = float(args.patch_cutoff)

    if args.numbering_offset is not None:
        cfg["numbering_map"] = {"mode": "offset", "offset": int(args.numbering_offset)}

    return cfg


def validate_and_finalize_config(cfg: dict) -> dict:
    required = ["structure", "antigen_chains", "nanobody_chain", "cdrs"]
    missing = [x for x in required if x not in cfg]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")

    cdrs = cfg.get("cdrs", {})
    for name in ("cdr1", "cdr2", "cdr3"):
        if name not in cdrs:
            raise ValueError(f"Missing CDR range: {name}")
        cdrs[name] = list(normalize_range(cdrs[name], name))

    cfg["antigen_chains"] = [str(x).strip() for x in cfg["antigen_chains"] if str(x).strip()]
    if not cfg["antigen_chains"]:
        raise ValueError("antigen_chains cannot be empty.")

    cfg["nanobody_chain"] = str(cfg["nanobody_chain"]).strip()
    if not cfg["nanobody_chain"]:
        raise ValueError("nanobody_chain cannot be empty.")

    cfg.setdefault("outdir", "results")
    cfg.setdefault("published_hotspots", {})
    cfg.setdefault("numbering_map", {})
    cfg.setdefault("patch_clustering_distance", DEFAULT_CONFIG["patch_clustering_distance"])

    cfg.setdefault("contact_cutoffs", {})
    for key, val in DEFAULT_CONFIG["contact_cutoffs"].items():
        cfg["contact_cutoffs"].setdefault(key, val)

    cfg.setdefault("interface_score_weights", {})
    for key, val in DEFAULT_CONFIG["interface_score_weights"].items():
        cfg["interface_score_weights"].setdefault(key, val)

    return cfg


def atom_name(atom) -> str:
    return atom.get_name().strip().upper()


def resname(residue) -> str:
    return residue.get_resname().strip().upper()


def is_hydrogen(atom) -> bool:
    elem = (getattr(atom, "element", "") or "").strip().upper()
    if elem == "H":
        return True
    return atom_name(atom).startswith("H")


def is_protein_residue(residue) -> bool:
    hetflag = residue.id[0]
    if hetflag != " ":
        return False
    return bool(is_aa(residue, standard=False))


def residue_key(residue) -> ResidueKey:
    chain = residue.get_parent().id
    resseq = int(residue.id[1])
    icode = residue.id[2].strip() if residue.id[2] else ""
    return chain, resseq, icode


def residue_label(key: ResidueKey) -> str:
    chain, resseq, icode = key
    return f"{chain}:{resseq}{icode}"


def residue_resi_token(key: ResidueKey) -> str:
    _, resseq, icode = key
    return f"{resseq}{icode}"


def distance(coord_a, coord_b) -> float:
    dx = float(coord_a[0]) - float(coord_b[0])
    dy = float(coord_a[1]) - float(coord_b[1])
    dz = float(coord_a[2]) - float(coord_b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def heavy_atoms(residue):
    for atom in residue.get_atoms():
        if not is_hydrogen(atom):
            yield atom


def parse_structure(structure_path: str):
    path = Path(structure_path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    elif suffix == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        # Fall back: try mmCIF first then PDB parser.
        try:
            parser = MMCIFParser(QUIET=True)
            return next(parser.get_structure("complex", str(path)).get_models())
        except Exception:
            parser = PDBParser(QUIET=True)

    structure = parser.get_structure("complex", str(path))
    model = next(structure.get_models())
    return model


def get_chain(model, chain_id: str):
    for chain in model.get_chains():
        if chain.id == chain_id:
            return chain
    return None


def select_chain_residues(model, chain_ids: Sequence[str]) -> Tuple[List, List[str]]:
    residues = []
    warnings = []
    for chain_id in chain_ids:
        chain = get_chain(model, chain_id)
        if chain is None:
            warnings.append(f"Antigen chain '{chain_id}' not found in structure.")
            continue
        chain_res = [r for r in chain.get_residues() if is_protein_residue(r)]
        residues.extend(chain_res)
    return residues, warnings


def select_cdr_residues(model, chain_id: str, cdr_ranges: dict) -> Tuple[Dict[str, List], List[str]]:
    chain = get_chain(model, chain_id)
    if chain is None:
        raise ValueError(f"Nanobody chain '{chain_id}' not found in structure.")

    warnings = []
    cdr_residues = {}
    protein_residues = [r for r in chain.get_residues() if is_protein_residue(r)]
    for cdr_name, (start, end) in cdr_ranges.items():
        selected = [r for r in protein_residues if start <= int(r.id[1]) <= end]
        if not selected:
            warnings.append(
                f"{cdr_name} range {start}-{end} on chain {chain_id} selected 0 residues."
            )
        cdr_residues[cdr_name] = selected
    return cdr_residues, warnings


def residue_is_hydrophobic(residue) -> bool:
    return resname(residue) in HYDROPHOBIC_RESIDUES


def atom_is_hydrophobic(residue, atom) -> bool:
    if not residue_is_hydrophobic(residue):
        return False
    elem = (getattr(atom, "element", "") or "").strip().upper()
    if elem not in {"C", "S"}:
        return False
    return atom_name(atom) not in BACKBONE_ATOMS


def atom_is_positive(residue, atom) -> bool:
    rn = resname(residue)
    return atom_name(atom) in POSITIVE_SIDECHAIN.get(rn, set())


def atom_is_negative(residue, atom) -> bool:
    rn = resname(residue)
    return atom_name(atom) in NEGATIVE_SIDECHAIN.get(rn, set())


def atom_is_donor(residue, atom) -> bool:
    name = atom_name(atom)
    if name in BACKBONE_DONORS:
        return True
    return name in DONOR_SIDECHAIN.get(resname(residue), set())


def atom_is_acceptor(residue, atom) -> bool:
    name = atom_name(atom)
    if name in BACKBONE_ACCEPTORS:
        return True
    return name in ACCEPTOR_SIDECHAIN.get(resname(residue), set())


def is_hbond_pair(res_a, atom_a, res_b, atom_b) -> bool:
    return (atom_is_donor(res_a, atom_a) and atom_is_acceptor(res_b, atom_b)) or (
        atom_is_acceptor(res_a, atom_a) and atom_is_donor(res_b, atom_b)
    )


def is_salt_bridge_pair(res_a, atom_a, res_b, atom_b) -> bool:
    return (atom_is_positive(res_a, atom_a) and atom_is_negative(res_b, atom_b)) or (
        atom_is_negative(res_a, atom_a) and atom_is_positive(res_b, atom_b)
    )


def aromatic_centroid(residue) -> Optional[Tuple[float, float, float]]:
    rn = resname(residue)
    ring_atoms = AROMATIC_ATOMS.get(rn)
    if not ring_atoms:
        return None
    coords = []
    for atom in residue.get_atoms():
        if atom_name(atom) in ring_atoms and not is_hydrogen(atom):
            coords.append(atom.coord)
    if len(coords) < 4:
        return None
    x = sum(float(c[0]) for c in coords) / len(coords)
    y = sum(float(c[1]) for c in coords) / len(coords)
    z = sum(float(c[2]) for c in coords) / len(coords)
    return (x, y, z)


class NumberingMapper:
    def __init__(self, config_map: dict, antigen_chains: Sequence[str], warnings: List[str]):
        self.mode = "none"
        self.offset = 0
        self.by_structure: Dict[ResidueKey, int] = {}
        self.inverse_by_full: Dict[int, List[ResidueKey]] = defaultdict(list)
        self.antigen_chains = list(antigen_chains)

        if not config_map:
            return

        mode = str(config_map.get("mode", "none")).strip().lower()
        self.mode = mode
        if mode == "offset":
            self.offset = int(config_map.get("offset", 0))
        elif mode == "csv":
            csv_path = config_map.get("file")
            if not csv_path:
                warnings.append("numbering_map.mode=csv but no file provided.")
                self.mode = "none"
                return
            path = Path(csv_path)
            if not path.exists():
                warnings.append(f"numbering map CSV not found: {path}")
                self.mode = "none"
                return
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                required = {"structure_chain", "structure_resnum", "full_length_resnum"}
                if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                    warnings.append(
                        "numbering map CSV missing required columns: "
                        "structure_chain, structure_resnum, full_length_resnum"
                    )
                    self.mode = "none"
                    return
                for row in reader:
                    try:
                        chain = str(row["structure_chain"]).strip()
                        s_resnum = int(str(row["structure_resnum"]).strip())
                        icode = str(row.get("structure_icode", "") or "").strip()
                        f_resnum = int(str(row["full_length_resnum"]).strip())
                    except Exception:
                        continue
                    key = (chain, s_resnum, icode)
                    self.by_structure[key] = f_resnum
                    self.inverse_by_full[f_resnum].append(key)
        else:
            self.mode = "none"

    def has_mapping(self) -> bool:
        return self.mode in {"offset", "csv"}

    def to_full_length(self, key: ResidueKey) -> Optional[int]:
        chain, s_resnum, icode = key
        if self.mode == "offset":
            return int(s_resnum + self.offset)
        if self.mode == "csv":
            if key in self.by_structure:
                return self.by_structure[key]
            fallback_key = (chain, s_resnum, "")
            return self.by_structure.get(fallback_key)
        return None

    def structure_keys_for_full(self, full_resnum: int) -> List[ResidueKey]:
        if self.mode == "csv":
            return list(self.inverse_by_full.get(int(full_resnum), []))
        if self.mode == "offset":
            s_resnum = int(full_resnum - self.offset)
            return [(chain, s_resnum, "") for chain in self.antigen_chains]
        return []


def init_residue_record(residue, mapper: NumberingMapper) -> dict:
    key = residue_key(residue)
    mapped = mapper.to_full_length(key)
    return {
        "residue_obj": residue,
        "residue_key": key,
        "antigen_chain": key[0],
        "antigen_resnum": key[1],
        "antigen_icode": key[2],
        "antigen_resname": resname(residue),
        "antigen_full_length_resnum": mapped,
        "contacting_cdrs": set(),
        "min_heavy_atom_distance": None,
        "total_contact_count": 0,
        "cdr1_contact_count": 0,
        "cdr2_contact_count": 0,
        "cdr3_contact_count": 0,
        "hydrogen_bond_count": 0,
        "salt_bridge_count": 0,
        "hydrophobic_contact_count": 0,
        "aromatic_contact_count": 0,
        "interface_score": 0.0,
        "assigned_patch_id": "",
    }


def analyze_contacts(
    antigen_residues: List,
    cdr_residues: Dict[str, List],
    cutoffs: dict,
    mapper: NumberingMapper,
    warnings: List[str],
) -> List[dict]:
    heavy_cutoff = float(cutoffs["heavy_atom"])
    hbond_cutoff = float(cutoffs["hydrogen_bond_distance"])
    salt_cutoff = float(cutoffs["salt_bridge"])
    hydrophobic_cutoff = float(cutoffs["hydrophobic"])
    aromatic_cutoff = float(cutoffs["aromatic"])

    max_cutoff = max(heavy_cutoff, hbond_cutoff, salt_cutoff, hydrophobic_cutoff)

    cdr_heavy_atoms = {}
    for cdr_name, residues in cdr_residues.items():
        atoms = [a for r in residues for a in heavy_atoms(r)]
        cdr_heavy_atoms[cdr_name] = atoms

    cdr_search = {
        cdr_name: NeighborSearch(atoms)
        for cdr_name, atoms in cdr_heavy_atoms.items()
        if atoms
    }

    if not cdr_search:
        warnings.append("No heavy atoms found in any CDR selection; contact analysis will be empty.")

    cdr_aromatic_centroids = {}
    for cdr_name, residues in cdr_residues.items():
        pairs = []
        for r in residues:
            cent = aromatic_centroid(r)
            if cent is not None:
                pairs.append((r, cent))
        cdr_aromatic_centroids[cdr_name] = pairs

    records = []
    for antigen_res in antigen_residues:
        record = init_residue_record(antigen_res, mapper)
        antigen_heavy = list(heavy_atoms(antigen_res))

        for ant_atom in antigen_heavy:
            for cdr_name, ns in cdr_search.items():
                near_atoms = ns.search(ant_atom.coord, max_cutoff, level="A")
                for nb_atom in near_atoms:
                    nb_res = nb_atom.get_parent()
                    dist = ant_atom - nb_atom

                    if dist <= heavy_cutoff:
                        record["total_contact_count"] += 1
                        record[f"{cdr_name}_contact_count"] += 1
                        record["contacting_cdrs"].add(cdr_name.upper())
                        if (
                            record["min_heavy_atom_distance"] is None
                            or dist < record["min_heavy_atom_distance"]
                        ):
                            record["min_heavy_atom_distance"] = float(dist)

                    if dist <= hbond_cutoff and is_hbond_pair(antigen_res, ant_atom, nb_res, nb_atom):
                        record["hydrogen_bond_count"] += 1

                    if dist <= salt_cutoff and is_salt_bridge_pair(
                        antigen_res, ant_atom, nb_res, nb_atom
                    ):
                        record["salt_bridge_count"] += 1

                    if dist <= hydrophobic_cutoff and (
                        atom_is_hydrophobic(antigen_res, ant_atom)
                        and atom_is_hydrophobic(nb_res, nb_atom)
                    ):
                        record["hydrophobic_contact_count"] += 1

        antigen_aromatic = aromatic_centroid(antigen_res)
        if antigen_aromatic is not None:
            for cdr_name, arom_pairs in cdr_aromatic_centroids.items():
                for _, cdr_cent in arom_pairs:
                    if distance(antigen_aromatic, cdr_cent) <= aromatic_cutoff:
                        record["aromatic_contact_count"] += 1

        if record["total_contact_count"] > 0:
            records.append(record)

    if cutoffs.get("hydrogen_bond_angle") is not None:
        warnings.append(
            "Hydrogen bond angle was provided but only distance-based donor/acceptor approximation "
            "is used (no explicit H positions in AF3 model)."
        )

    return records


def compute_interface_score(record: dict, weights: dict) -> float:
    cdr_count = sum(
        1
        for cdr in ("cdr1_contact_count", "cdr2_contact_count", "cdr3_contact_count")
        if record[cdr] > 0
    )

    score = 0.0
    score += float(weights["contact"]) * record["total_contact_count"]
    score += float(weights["hbond"]) * record["hydrogen_bond_count"]
    score += float(weights["salt_bridge"]) * record["salt_bridge_count"]
    score += float(weights["hydrophobic"]) * record["hydrophobic_contact_count"]
    score += float(weights["aromatic"]) * record["aromatic_contact_count"]

    if cdr_count > 1:
        score += float(weights["multi_cdr_bonus"]) * (cdr_count - 1)

    if record["cdr3_contact_count"] > 0:
        score += float(weights["cdr3_bonus"]) * math.log1p(record["cdr3_contact_count"])

    return float(score)


def representative_atom(residue):
    for name in ("CB", "CA"):
        if residue.has_id(name):
            return residue[name]
    for atom in heavy_atoms(residue):
        return atom
    return None


def residues_close(res_a, res_b, cutoff: float) -> bool:
    rep_a = representative_atom(res_a)
    rep_b = representative_atom(res_b)
    if rep_a is not None and rep_b is not None and (rep_a - rep_b) <= cutoff:
        return True

    for a in heavy_atoms(res_a):
        for b in heavy_atoms(res_b):
            if (a - b) <= cutoff:
                return True
    return False


class UnionFind:
    def __init__(self, items: Sequence[int]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def assign_patches(records: List[dict], patch_cutoff: float) -> Dict[str, List[dict]]:
    if not records:
        return {}

    uf = UnionFind(list(range(len(records))))
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if residues_close(records[i]["residue_obj"], records[j]["residue_obj"], patch_cutoff):
                uf.union(i, j)

    comp = defaultdict(list)
    for i in range(len(records)):
        root = uf.find(i)
        comp[root].append(i)

    components = list(comp.values())
    components.sort(
        key=lambda idxs: sum(records[i]["interface_score"] for i in idxs),
        reverse=True,
    )

    patches: Dict[str, List[dict]] = {}
    for i, idxs in enumerate(components, start=1):
        patch_id = f"patch_{i}"
        patch_records = [records[k] for k in idxs]
        for rec in patch_records:
            rec["assigned_patch_id"] = patch_id
        patches[patch_id] = patch_records
    return patches


def normalize_hotspot_sets(raw_hotspots: dict) -> Dict[str, set]:
    out = {}
    for name, values in (raw_hotspots or {}).items():
        if not isinstance(values, (list, tuple, set)):
            continue
        parsed = set()
        for val in values:
            if isinstance(val, int):
                parsed.add(int(val))
                continue
            sval = str(val).strip()
            if not sval:
                continue
            if ":" in sval:
                sval = sval.split(":")[-1]
            try:
                parsed.add(int(sval))
            except ValueError:
                continue
        out[str(name)] = parsed
    return out


def patch_number_set(patch_records: List[dict], use_full_length: bool) -> set:
    nums = set()
    for rec in patch_records:
        if use_full_length and rec["antigen_full_length_resnum"] is not None:
            nums.add(int(rec["antigen_full_length_resnum"]))
        elif not use_full_length:
            nums.add(int(rec["antigen_resnum"]))
    return nums


def summarize_hotspot_overlap(
    patches: Dict[str, List[dict]],
    hotspot_sets: Dict[str, set],
    use_full_length_for_comparison: bool,
) -> dict:
    summary = {
        "comparison_numbering": "full_length" if use_full_length_for_comparison else "structure",
        "patch_overlaps": {},
        "top_patch_best_hotspot": None,
    }

    if not patches or not hotspot_sets:
        return summary

    top_patch_id = sorted(
        patches.keys(),
        key=lambda x: sum(rec["interface_score"] for rec in patches[x]),
        reverse=True,
    )[0]

    best_for_top = None
    for patch_id, patch_records in patches.items():
        pset = patch_number_set(patch_records, use_full_length_for_comparison)
        patch_dict = {}
        for hs_name, hs_set in hotspot_sets.items():
            overlap = sorted(pset & hs_set)
            union = pset | hs_set
            overlap_fraction = (len(overlap) / len(hs_set)) if hs_set else 0.0
            patch_fraction = (len(overlap) / len(pset)) if pset else 0.0
            jaccard = (len(overlap) / len(union)) if union else 0.0
            info = {
                "overlap_residues": overlap,
                "overlap_count": len(overlap),
                "overlap_fraction_of_hotspot": overlap_fraction,
                "overlap_fraction_of_patch": patch_fraction,
                "jaccard_like_score": jaccard,
            }
            patch_dict[hs_name] = info

            if patch_id == top_patch_id:
                current = (jaccard, overlap_fraction, len(overlap))
                if best_for_top is None or current > best_for_top["rank_tuple"]:
                    best_for_top = {
                        "hotspot_set": hs_name,
                        "overlap_residues": overlap,
                        "overlap_fraction_of_hotspot": overlap_fraction,
                        "overlap_fraction_of_patch": patch_fraction,
                        "jaccard_like_score": jaccard,
                        "rank_tuple": current,
                    }

        summary["patch_overlaps"][patch_id] = patch_dict

    if best_for_top is not None:
        best_for_top.pop("rank_tuple", None)
    summary["top_patch_best_hotspot"] = best_for_top
    summary["top_patch_id"] = top_patch_id
    return summary


def dominant_interaction_types(patch_records: List[dict]) -> str:
    counts = {
        "heavy_contact": sum(r["total_contact_count"] for r in patch_records),
        "hydrogen_bond": sum(r["hydrogen_bond_count"] for r in patch_records),
        "salt_bridge": sum(r["salt_bridge_count"] for r in patch_records),
        "hydrophobic": sum(r["hydrophobic_contact_count"] for r in patch_records),
        "aromatic": sum(r["aromatic_contact_count"] for r in patch_records),
    }
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    nonzero = [name for name, val in ranked if val > 0]
    if not nonzero:
        return "none"
    return ",".join(nonzero[:3])


def patch_overlap_compact_string(patch_id: str, hotspot_summary: dict) -> str:
    patch_ov = hotspot_summary.get("patch_overlaps", {}).get(patch_id, {})
    parts = []
    for hs_name, info in patch_ov.items():
        parts.append(
            f"{hs_name}:overlap={info['overlap_count']},"
            f"jaccard={info['jaccard_like_score']:.3f}"
        )
    return "; ".join(parts)


def serialize_record(record: dict) -> dict:
    out = dict(record)
    out.pop("residue_obj", None)
    out.pop("residue_key", None)
    out["contacting_cdrs"] = "|".join(sorted(out["contacting_cdrs"]))
    if out["min_heavy_atom_distance"] is None:
        out["min_heavy_atom_distance"] = ""
    else:
        out["min_heavy_atom_distance"] = round(float(out["min_heavy_atom_distance"]), 3)
    out["interface_score"] = round(float(out["interface_score"]), 3)
    return out


def write_residue_csv(records: List[dict], out_csv: Path):
    fieldnames = [
        "antigen_chain",
        "antigen_resnum",
        "antigen_resname",
        "antigen_full_length_resnum",
        "contacting_cdrs",
        "min_heavy_atom_distance",
        "total_contact_count",
        "cdr1_contact_count",
        "cdr2_contact_count",
        "cdr3_contact_count",
        "hydrogen_bond_count",
        "salt_bridge_count",
        "hydrophobic_contact_count",
        "aromatic_contact_count",
        "interface_score",
        "assigned_patch_id",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: serialize_record(rec).get(k, "") for k in fieldnames})


def write_patch_csv(
    patches: Dict[str, List[dict]],
    weights: dict,
    hotspot_summary: dict,
    out_csv: Path,
):
    rows = []
    for patch_id, patch_records in patches.items():
        patch_total_score = sum(r["interface_score"] for r in patch_records)
        cdr1_score = sum(float(weights["contact"]) * r["cdr1_contact_count"] for r in patch_records)
        cdr2_score = sum(float(weights["contact"]) * r["cdr2_contact_count"] for r in patch_records)
        cdr3_score = sum(
            float(weights["contact"]) * r["cdr3_contact_count"]
            + float(weights["cdr3_bonus"]) * math.log1p(r["cdr3_contact_count"])
            for r in patch_records
            if r["cdr3_contact_count"] > 0
        )
        residues_in_patch = ",".join(
            sorted(residue_label(r["residue_key"]) for r in patch_records)
        )
        full_res = sorted(
            {
                int(r["antigen_full_length_resnum"])
                for r in patch_records
                if r["antigen_full_length_resnum"] is not None
            }
        )
        row = {
            "patch_id": patch_id,
            "residues_in_patch": residues_in_patch,
            "full_length_residues_in_patch": ",".join(str(x) for x in full_res),
            "total_interface_score": round(float(patch_total_score), 3),
            "cdr1_score": round(float(cdr1_score), 3),
            "cdr2_score": round(float(cdr2_score), 3),
            "cdr3_score": round(float(cdr3_score), 3),
            "dominant_interaction_types": dominant_interaction_types(patch_records),
            "overlap_with_hotspot_sets": patch_overlap_compact_string(patch_id, hotspot_summary),
        }
        rows.append(row)

    rows.sort(key=lambda x: x["total_interface_score"], reverse=True)
    fieldnames = [
        "patch_id",
        "residues_in_patch",
        "full_length_residues_in_patch",
        "total_interface_score",
        "cdr1_score",
        "cdr2_score",
        "cdr3_score",
        "dominant_interaction_types",
        "overlap_with_hotspot_sets",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_pymol_selection_from_keys(keys: List[ResidueKey]) -> str:
    if not keys:
        return "none"
    by_chain = defaultdict(list)
    for key in keys:
        by_chain[key[0]].append(residue_resi_token(key))
    parts = []
    for chain_id, tokens in sorted(by_chain.items()):
        uniq_tokens = sorted(set(tokens), key=lambda x: (int("".join(ch for ch in x if ch.isdigit()) or 0), x))
        parts.append(f"(chain {chain_id} and resi {'+'.join(uniq_tokens)})")
    return " or ".join(parts)


def sanitize_selection_name(name: str) -> str:
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch == "_":
            safe.append(ch)
        else:
            safe.append("_")
    text = "".join(safe).strip("_")
    return text or "sel"


def make_pymol_script(
    structure_path: str,
    cfg: dict,
    records: List[dict],
    patches: Dict[str, List[dict]],
    hotspots: Dict[str, set],
    hotspot_summary: dict,
    mapper: NumberingMapper,
    out_pml: Path,
):
    nanobody_chain = cfg["nanobody_chain"]
    cdr1 = cfg["cdrs"]["cdr1"]
    cdr2 = cfg["cdrs"]["cdr2"]
    cdr3 = cfg["cdrs"]["cdr3"]

    contacting_keys = [r["residue_key"] for r in records]
    top10 = sorted(records, key=lambda r: r["interface_score"], reverse=True)[:10]
    top10_keys = [r["residue_key"] for r in top10]

    lines = [
        "reinitialize",
        f'load "{structure_path}", complex',
        f"select nanobody, chain {nanobody_chain}",
        f"select cdr1, chain {nanobody_chain} and resi {cdr1[0]}-{cdr1[1]}",
        f"select cdr2, chain {nanobody_chain} and resi {cdr2[0]}-{cdr2[1]}",
        f"select cdr3, chain {nanobody_chain} and resi {cdr3[0]}-{cdr3[1]}",
        f"select contacting_antigen, {make_pymol_selection_from_keys(contacting_keys)}",
        f"select top10_interface_residues, {make_pymol_selection_from_keys(top10_keys)}",
        "show cartoon, complex",
        "show sticks, nanobody or cdr1 or cdr2 or cdr3 or contacting_antigen",
        "color slate, nanobody",
        "color yellow, cdr1",
        "color orange, cdr2",
        "color red, cdr3",
        "color tv_blue, contacting_antigen",
        "color marine, top10_interface_residues",
    ]

    for patch_id, patch_records in patches.items():
        patch_keys = [r["residue_key"] for r in patch_records]
        lines.append(f"select {patch_id}, {make_pymol_selection_from_keys(patch_keys)}")

    for hs_name, hs_set in hotspots.items():
        hs_sel = sanitize_selection_name(hs_name)
        hs_keys = []
        for num in hs_set:
            hs_keys.extend(mapper.structure_keys_for_full(num))
        hs_keys = sorted(set(hs_keys))
        if not hs_keys and mapper.mode == "none":
            # No mapping: interpret hotspot numbers as structure numbering on antigen chains.
            hs_keys = []
            for chain in cfg["antigen_chains"]:
                for num in hs_set:
                    hs_keys.append((chain, int(num), ""))
        lines.append(f"select hotspot_{hs_sel}, {make_pymol_selection_from_keys(hs_keys)}")

    top_patch_id = hotspot_summary.get("top_patch_id")
    if top_patch_id:
        for hs_name, info in hotspot_summary.get("patch_overlaps", {}).get(top_patch_id, {}).items():
            hs_sel = sanitize_selection_name(hs_name)
            overlap_nums = info.get("overlap_residues", [])
            overlap_keys = []
            if mapper.has_mapping():
                for num in overlap_nums:
                    overlap_keys.extend(mapper.structure_keys_for_full(num))
            else:
                for chain in cfg["antigen_chains"]:
                    for num in overlap_nums:
                        overlap_keys.append((chain, int(num), ""))
            sel_name = sanitize_selection_name(f"overlap_{top_patch_id}_{hs_sel}")
            lines.append(f"select {sel_name}, {make_pymol_selection_from_keys(overlap_keys)}")

    lines.append("zoom contacting_antigen")
    out_pml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_text_report(
    cfg: dict,
    records: List[dict],
    patches: Dict[str, List[dict]],
    hotspot_summary: dict,
    warnings: List[str],
    out_txt: Path,
):
    lines = []
    lines.append("Nanobody-Norovirus Interface Evidence Report")
    lines.append("=" * 46)
    lines.append("")
    lines.append("Interpretation scope:")
    lines.append(
        "This report summarizes structure-based interface evidence from a static AF3 model. "
        "It does not estimate true physical binding energy, true affinity, or experimentally "
        "validated epitope truth."
    )
    lines.append("")

    lines.append(f"Input structure: {cfg['structure']}")
    lines.append(f"Antigen chains: {', '.join(cfg['antigen_chains'])}")
    lines.append(f"Nanobody chain: {cfg['nanobody_chain']}")
    lines.append(
        "CDR ranges: "
        f"CDR1 {tuple(cfg['cdrs']['cdr1'])}, "
        f"CDR2 {tuple(cfg['cdrs']['cdr2'])}, "
        f"CDR3 {tuple(cfg['cdrs']['cdr3'])}"
    )
    lines.append("")

    lines.append(f"Total contacting antigen residues: {len(records)}")
    lines.append("")
    lines.append("Top AF3-supported interface residues:")
    for rec in sorted(records, key=lambda r: r["interface_score"], reverse=True)[:10]:
        full_num = rec["antigen_full_length_resnum"]
        full_txt = f" (full-length {full_num})" if full_num is not None else ""
        lines.append(
            f"- {rec['antigen_chain']}:{rec['antigen_resnum']}{rec['antigen_icode']} "
            f"{rec['antigen_resname']}{full_txt} | score={rec['interface_score']:.3f} | "
            f"CDRs={','.join(sorted(rec['contacting_cdrs'])) or 'NA'}"
        )

    lines.append("")
    lines.append("CDR contribution (heavy-atom contact counts):")
    cdr1_total = sum(r["cdr1_contact_count"] for r in records)
    cdr2_total = sum(r["cdr2_contact_count"] for r in records)
    cdr3_total = sum(r["cdr3_contact_count"] for r in records)
    cdr_totals = {"CDR1": cdr1_total, "CDR2": cdr2_total, "CDR3": cdr3_total}
    for name, val in cdr_totals.items():
        lines.append(f"- {name}: {val}")
    dominant_cdr = max(cdr_totals.items(), key=lambda x: x[1])[0] if records else "NA"
    lines.append(f"Dominant CDR by contact count: {dominant_cdr}")

    lines.append("")
    lines.append("Top epitope patches:")
    patch_ids = sorted(
        patches.keys(),
        key=lambda p: sum(r["interface_score"] for r in patches[p]),
        reverse=True,
    )
    for patch_id in patch_ids[:3]:
        patch_records = patches[patch_id]
        patch_score = sum(r["interface_score"] for r in patch_records)
        residues_txt = ", ".join(
            sorted(residue_label(r["residue_key"]) for r in patch_records)
        )
        lines.append(f"- {patch_id}: score={patch_score:.3f}; residues={residues_txt}")

    best_hotspot = hotspot_summary.get("top_patch_best_hotspot")
    lines.append("")
    if best_hotspot:
        lines.append("Published hotspot overlap (top predicted patch):")
        lines.append(
            f"- Best matching hotspot set: {best_hotspot['hotspot_set']} | "
            f"Jaccard-like={best_hotspot['jaccard_like_score']:.3f} | "
            f"hotspot overlap fraction={best_hotspot['overlap_fraction_of_hotspot']:.3f}"
        )
        lines.append(
            f"- Overlapping residues ({hotspot_summary.get('comparison_numbering')} numbering): "
            f"{best_hotspot['overlap_residues']}"
        )
    else:
        lines.append("Published hotspot overlap: not available (no hotspot sets or no contacting patch).")

    if warnings:
        lines.append("")
        lines.append("Warnings and assumptions:")
        for w in warnings:
            lines.append(f"- {w}")

    lines.append("")
    lines.append(
        "Caution: AF3 interface geometry is best used as a hypothesis generator for candidate "
        "epitope patches; downstream design and prioritization should still be guided by "
        "literature hotspots and experimental validation."
    )

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary_json(
    cfg: dict,
    records: List[dict],
    patches: Dict[str, List[dict]],
    hotspot_summary: dict,
    warnings: List[str],
    out_json: Path,
):
    sorted_records = sorted(records, key=lambda r: r["interface_score"], reverse=True)
    top_residues = []
    for rec in sorted_records[:10]:
        top_residues.append(
            {
                "chain": rec["antigen_chain"],
                "resnum": rec["antigen_resnum"],
                "icode": rec["antigen_icode"],
                "resname": rec["antigen_resname"],
                "full_length_resnum": rec["antigen_full_length_resnum"],
                "contacting_cdrs": sorted(rec["contacting_cdrs"]),
                "interface_score": round(float(rec["interface_score"]), 3),
            }
        )

    patch_list = []
    for patch_id in sorted(
        patches.keys(),
        key=lambda p: sum(r["interface_score"] for r in patches[p]),
        reverse=True,
    ):
        patch_records = patches[patch_id]
        patch_list.append(
            {
                "patch_id": patch_id,
                "residue_count": len(patch_records),
                "residues": [residue_label(r["residue_key"]) for r in patch_records],
                "total_interface_score": round(
                    float(sum(r["interface_score"] for r in patch_records)), 3
                ),
            }
        )

    summary = {
        "input_file_name": cfg["structure"],
        "chain_ids_used": {
            "antigen_chains": cfg["antigen_chains"],
            "nanobody_chain": cfg["nanobody_chain"],
        },
        "cdr_residue_ranges_used": cfg["cdrs"],
        "contact_cutoffs_used": cfg["contact_cutoffs"],
        "patch_clustering_distance": cfg["patch_clustering_distance"],
        "interface_score_weights_used": cfg["interface_score_weights"],
        "total_contacting_antigen_residues": len(records),
        "top_ranked_antigen_residues": top_residues,
        "top_epitope_patches": patch_list[:5],
        "hotspot_overlap_statistics": hotspot_summary,
        "warnings": warnings,
    }

    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def run_internal_test() -> int:
    record = {
        "total_contact_count": 6,
        "hydrogen_bond_count": 2,
        "salt_bridge_count": 1,
        "hydrophobic_contact_count": 3,
        "aromatic_contact_count": 1,
        "cdr1_contact_count": 1,
        "cdr2_contact_count": 0,
        "cdr3_contact_count": 4,
    }
    weights = DEFAULT_CONFIG["interface_score_weights"]
    score = compute_interface_score(record, weights)
    expected_min = 6 + 4 + 2.5 + 3.6 + 1.5  # base terms only
    if score <= expected_min:
        print("Self-test failed: score should include bonuses.")
        return 1
    print(f"Self-test passed: computed score={score:.3f}")
    return 0


def main() -> int:
    args = parse_args()
    if args.run_test:
        return run_internal_test()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    cfg = validate_and_finalize_config(cfg)

    warnings: List[str] = []

    structure_path = cfg["structure"]
    outdir = Path(cfg.get("outdir", "results"))
    outdir.mkdir(parents=True, exist_ok=True)

    model = parse_structure(structure_path)

    antigen_residues, antigen_warnings = select_chain_residues(model, cfg["antigen_chains"])
    warnings.extend(antigen_warnings)
    if not antigen_residues:
        raise RuntimeError("No antigen protein residues were found on requested antigen chains.")

    cdr_ranges = {k: tuple(v) for k, v in cfg["cdrs"].items()}
    cdr_residues, cdr_warnings = select_cdr_residues(model, cfg["nanobody_chain"], cdr_ranges)
    warnings.extend(cdr_warnings)

    mapper = NumberingMapper(cfg.get("numbering_map", {}), cfg["antigen_chains"], warnings)
    if not mapper.has_mapping():
        warnings.append(
            "No residue numbering map provided; hotspot comparison uses structure residue numbering."
        )

    records = analyze_contacts(
        antigen_residues=antigen_residues,
        cdr_residues=cdr_residues,
        cutoffs=cfg["contact_cutoffs"],
        mapper=mapper,
        warnings=warnings,
    )

    if mapper.has_mapping() and records:
        unmapped = sum(1 for r in records if r["antigen_full_length_resnum"] is None)
        if unmapped > 0:
            warnings.append(
                f"{unmapped}/{len(records)} contacting residues could not be mapped to full-length numbering."
            )

    for rec in records:
        rec["interface_score"] = compute_interface_score(rec, cfg["interface_score_weights"])

    records.sort(key=lambda r: r["interface_score"], reverse=True)
    patches = assign_patches(records, float(cfg["patch_clustering_distance"]))

    hotspot_sets = normalize_hotspot_sets(cfg.get("published_hotspots", {}))
    if cfg.get("published_hotspots") and not hotspot_sets:
        warnings.append("published_hotspots provided but no valid residue indices were parsed.")

    use_full_length = mapper.has_mapping()
    hotspot_summary = summarize_hotspot_overlap(
        patches=patches,
        hotspot_sets=hotspot_sets,
        use_full_length_for_comparison=use_full_length,
    )

    residue_csv = outdir / "antigen_interface_residues.csv"
    patch_csv = outdir / "epitope_patches.csv"
    summary_json = outdir / "interface_summary.json"
    pymol_pml = outdir / "visualize_epitope.pml"
    report_txt = outdir / "interface_report.txt"

    write_residue_csv(records, residue_csv)
    write_patch_csv(patches, cfg["interface_score_weights"], hotspot_summary, patch_csv)
    build_summary_json(cfg, records, patches, hotspot_summary, warnings, summary_json)
    make_pymol_script(
        structure_path=structure_path,
        cfg=cfg,
        records=records,
        patches=patches,
        hotspots=hotspot_sets,
        hotspot_summary=hotspot_summary,
        mapper=mapper,
        out_pml=pymol_pml,
    )
    build_text_report(cfg, records, patches, hotspot_summary, warnings, report_txt)

    print(f"Analysis complete. Outputs written to: {outdir}")
    print(f"- Residue table: {residue_csv}")
    print(f"- Patch table: {patch_csv}")
    print(f"- JSON summary: {summary_json}")
    print(f"- PyMOL script: {pymol_pml}")
    print(f"- Text report: {report_txt}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
