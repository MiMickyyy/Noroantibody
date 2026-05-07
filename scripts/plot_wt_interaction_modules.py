#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


COLOR_MAP = {
    "salt_bridge": "#d62728",
    "hydrogen_bond": "#1f77b4",
    "cation_pi": "#9467bd",
    "pi_pi": "#ff7f0e",
    "hydrophobic": "#2ca02c",
    "heavy_contact": "#7f7f7f",
}


@dataclass
class Module:
    module_id: str
    antigen_nodes: List[str]
    nanobody_nodes: List[str]
    edges: pd.DataFrame


class DSU:
    def __init__(self):
        self.p = {}

    def add(self, x):
        if x not in self.p:
            self.p[x] = x

    def find(self, x):
        px = self.p[x]
        if px != x:
            self.p[x] = self.find(px)
        return self.p[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.p[rb] = ra


def parse_pos(lbl: str) -> int:
    return int(str(lbl).split(":", 1)[1])


def build_modules(df: pd.DataFrame) -> List[Module]:
    # Bipartite connected components: nodes = Ag residues + Nb residues, edges = contacts
    dsu = DSU()
    nodes = []
    for _, r in df.iterrows():
        ag = f"ag::{r['ag_label']}"
        nb = f"nb::{r['nb_label']}"
        dsu.add(ag)
        dsu.add(nb)
        dsu.union(ag, nb)
        nodes.extend([ag, nb])

    comp_nodes: Dict[str, List[str]] = {}
    for n in set(nodes):
        root = dsu.find(n)
        comp_nodes.setdefault(root, []).append(n)

    modules: List[Module] = []
    # sort components by number of contact edges descending
    comp_items = []
    for root, nds in comp_nodes.items():
        ag_set = {x.split("::", 1)[1] for x in nds if x.startswith("ag::")}
        nb_set = {x.split("::", 1)[1] for x in nds if x.startswith("nb::")}
        sub = df[df["ag_label"].isin(ag_set) & df["nb_label"].isin(nb_set)].copy()
        comp_items.append((len(sub), root, ag_set, nb_set, sub))

    comp_items.sort(reverse=True, key=lambda x: x[0])
    for i, (_, root, ag_set, nb_set, sub) in enumerate(comp_items, start=1):
        ag_nodes = sorted(list(ag_set), key=parse_pos)
        nb_nodes = sorted(list(nb_set), key=parse_pos)
        modules.append(Module(module_id=f"M{i}", antigen_nodes=ag_nodes, nanobody_nodes=nb_nodes, edges=sub))

    return modules


def plot_modules(modules: List[Module], out_png: Path, title: str):
    n = len(modules)
    if n == 0:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No module data", ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        return

    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig_w = 6.0 * ncols
    fig_h = 5.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, mod in enumerate(modules):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        ax.set_title(f"{mod.module_id} | contacts={len(mod.edges)}", fontsize=11)

        # local layout: antigen top row, nanobody bottom row
        ag = mod.antigen_nodes
        nb = mod.nanobody_nodes

        ag_x = {lbl: (i + 1) / (len(ag) + 1) for i, lbl in enumerate(ag)}
        nb_x = {lbl: (i + 1) / (len(nb) + 1) for i, lbl in enumerate(nb)}

        ag_y = 0.80
        nb_y = 0.20

        # draw edges first
        smin = float(mod.edges["strength_score"].min()) if len(mod.edges) else 0.0
        smax = float(mod.edges["strength_score"].max()) if len(mod.edges) else 1.0
        span = max(1e-6, smax - smin)

        for _, er in mod.edges.iterrows():
            ag_lbl = er["ag_label"]
            nb_lbl = er["nb_label"]
            typ = er.get("dominant_type", "heavy_contact")
            col = COLOR_MAP.get(typ, "#7f7f7f")
            w = 1.2 + 2.8 * ((float(er["strength_score"]) - smin) / span)
            ax.plot([nb_x[nb_lbl], ag_x[ag_lbl]], [nb_y, ag_y], color=col, lw=w, alpha=0.75)

        # nodes + labels
        ax.scatter([ag_x[x] for x in ag], [ag_y] * len(ag), c="#d62728", s=80, marker="s", zorder=3)
        ax.scatter([nb_x[x] for x in nb], [nb_y] * len(nb), c="#1f77b4", s=80, marker="o", zorder=3)

        for x in ag:
            ax.text(ag_x[x], ag_y + 0.06, f"Ag:{parse_pos(x)}", ha="center", va="bottom", fontsize=8)
        for x in nb:
            ax.text(nb_x[x], nb_y - 0.08, x.replace(":", ""), ha="center", va="top", fontsize=8)

        ax.text(0.01, 0.92, "Antigen epitope", transform=ax.transAxes, fontsize=9, color="#d62728")
        ax.text(0.01, 0.05, "Nanobody (CDR)", transform=ax.transAxes, fontsize=9, color="#1f77b4")

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")

        # module note
        types = mod.edges["dominant_type"].value_counts().to_dict()
        short = ", ".join([f"{k}:{v}" for k, v in sorted(types.items(), key=lambda kv: kv[1], reverse=True)[:3]])
        ax.text(0.5, 0.48, short, ha="center", va="center", fontsize=8, color="#333333")

    # hide unused axes
    for j in range(n, nrows * ncols):
        rr = j // ncols
        cc = j % ncols
        axes[rr][cc].axis("off")

    # global legend
    handles = [
        Line2D([0], [0], color=v, lw=3, label=k)
        for k, v in COLOR_MAP.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def export_module_table(modules: List[Module], out_csv: Path):
    rows = []
    for m in modules:
        rows.append(
            {
                "module_id": m.module_id,
                "n_edges": len(m.edges),
                "antigen_residues": ",".join(m.antigen_nodes),
                "nanobody_residues": ",".join(m.nanobody_nodes),
                "dominant_types": ",".join(sorted(m.edges["dominant_type"].value_counts().index.tolist())),
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def run_one(in_csv: Path, out_png: Path, out_module_csv: Path, title: str):
    df = pd.read_csv(in_csv)
    # only keep relevant columns
    need = ["nb_label", "ag_label", "dominant_type", "strength_score", "occupancy", "cdr_region"]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"Missing column {c} in {in_csv}")
    modules = build_modules(df)
    export_module_table(modules, out_module_csv)
    plot_modules(modules, out_png, title)


def main():
    root = Path('.').resolve()
    base = root / 'results' / 'summaries' / 'wt_deep_interaction_analysis'

    core_csv = base / 'wt_primary_strong_interactions_ranked.csv'
    sec_csv = base / 'wt_secondary_interactions_ranked.csv'

    core_out = base / 'wt_core_binding_modules_cluster.png'
    sec_out = base / 'wt_secondary_binding_modules_cluster.png'
    core_mod = base / 'wt_core_binding_modules_table.csv'
    sec_mod = base / 'wt_secondary_binding_modules_table.csv'

    run_one(
        core_csv,
        core_out,
        core_mod,
        title='WT Core (Primary strong) binding modules cluster map',
    )
    run_one(
        sec_csv,
        sec_out,
        sec_mod,
        title='WT Secondary (support/incomplete) binding modules cluster map',
    )

    print('Wrote:', core_out)
    print('Wrote:', sec_out)
    print('Wrote:', core_mod)
    print('Wrote:', sec_mod)


if __name__ == '__main__':
    main()
