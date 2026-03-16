#!/usr/bin/env python3
"""Auto-detect runtime/tooling/checkpoints for RFantibody without guessing."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import yaml
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

RFAB_WEIGHT_FILES = {
    "rfdiffusion_weights": "RFdiffusion_Ab.pt",
    "proteinmpnn_weights": "ProteinMPNN_v48_noise_0.2.pt",
    "rf2_weights": "RF2_ab.pt",
}

CLI_ARGS_REFERENCE = {
    "rfdiffusion": {
        "entrypoint": "rfantibody.cli.inference:rfdiffusion",
        "required": ["--target", "--framework"],
        "common": [
            "--output",
            "--output-quiver",
            "--num-designs",
            "--design-loops",
            "--hotspots",
            "--weights",
            "--diffuser-t",
            "--final-step",
            "--deterministic",
            "--no-trajectory",
            "--extra",
        ],
    },
    "proteinmpnn": {
        "entrypoint": "rfantibody.cli.inference:proteinmpnn",
        "required": ["--input-dir or --input-quiver"],
        "common": [
            "--output-dir",
            "--output-quiver",
            "--loops",
            "--seqs-per-struct",
            "--temperature",
            "--weights",
            "--omit-aas",
            "--augment-eps",
            "--deterministic",
            "--debug",
            "--allow-x",
        ],
    },
    "rf2": {
        "entrypoint": "rfantibody.cli.inference:rf2",
        "required": ["--input-pdb/--input-dir/--input-quiver/--input-json", "--output-dir or --output-quiver"],
        "common": [
            "--num-recycles",
            "--weights",
            "--seed",
            "--cautious",
            "--hotspot-show-prop",
            "--extra",
        ],
    },
}


def run_capture(cmd: List[str]) -> Tuple[int, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        return int(proc.returncode), proc.stdout.strip()
    except Exception as exc:  # pragma: no cover
        return 1, str(exc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-detect VM/runtime and RFantibody tooling paths.")
    p.add_argument("--pipeline-config", default="data/configs/pipeline.yaml")
    p.add_argument("--cdr-config", default="data/configs/cdr_boundaries.yaml")
    p.add_argument("--resolved-inputs", default="data/processed/resolved_inputs.yaml")
    p.add_argument("--complex-structure", default="fold_2026_03_12_12_19_model_0.cif")
    p.add_argument("--scan-root", action="append", default=[])
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_sequence(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines and lines[0].startswith(">"):
        seq = "".join(re.sub(r"[^A-Za-z*]", "", x) for x in lines if not x.startswith(">"))
        return seq.upper().replace("*", "")
    cands = re.findall(r"[ACDEFGHIKLMNPQRSTVWY*]{30,}", text.upper())
    if cands:
        return max(cands, key=len).replace("*", "")
    return ""


def parse_structure(structure_path: Path):
    sfx = structure_path.suffix.lower()
    parser = MMCIFParser(QUIET=True) if sfx in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(structure_path))
    return next(structure.get_models())


def detect_nanobody_chain(complex_path: Path, nanobody_seq: str) -> Optional[str]:
    if not complex_path.exists() or not nanobody_seq:
        return None
    model = parse_structure(complex_path)
    best: Tuple[Optional[str], float] = (None, -1.0)
    for chain in model.get_chains():
        seq = "".join(
            protein_letters_3to1.get(r.get_resname().upper(), "X")
            for r in chain
            if r.id[0] == " "
        )
        if not seq:
            continue
        lmin = min(len(seq), len(nanobody_seq))
        if lmin == 0:
            continue
        ident_prefix = sum(1 for a, b in zip(seq[:lmin], nanobody_seq[:lmin]) if a == b) / lmin
        contains = (nanobody_seq in seq) or (seq in nanobody_seq)
        score = ident_prefix + (1.0 if contains else 0.0)
        if score > best[1]:
            best = (str(chain.id), score)
    return best[0]


def detect_runtime() -> dict:
    out = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "env": "unknown",
        "cuda": None,
        "driver": None,
        "nvidia_gpu": None,
    }
    if os.environ.get("CONDA_DEFAULT_ENV"):
        out["env"] = f"conda:{os.environ.get('CONDA_DEFAULT_ENV')}"
    elif os.environ.get("VIRTUAL_ENV"):
        out["env"] = f"venv:{os.environ.get('VIRTUAL_ENV')}"

    if shutil.which("nvidia-smi"):
        code, txt = run_capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if code == 0 and txt:
            parts = [x.strip() for x in txt.splitlines()[0].split(",")]
            if len(parts) >= 2:
                out["nvidia_gpu"] = parts[0]
                out["driver"] = parts[1]
        code2, txt2 = run_capture(["nvidia-smi"])
        if code2 == 0:
            m = re.search(r"CUDA Version:\s*([0-9.]+)", txt2)
            if m:
                out["cuda"] = m.group(1)

    if out["cuda"] is None and shutil.which("nvcc"):
        code, txt = run_capture(["nvcc", "--version"])
        if code == 0:
            m = re.search(r"release\s+([0-9.]+)", txt)
            if m:
                out["cuda"] = m.group(1)
    return out


def bounded_walk(root: Path, max_depth: int) -> Iterator[Tuple[Path, List[str]]]:
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        return
    root_parts = len(root.parts)
    for current, dirs, files in os.walk(root):
        cur = Path(current)
        depth = len(cur.parts) - root_parts
        if depth >= max_depth:
            dirs[:] = []
        yield cur, files


def is_rfantibody_repo(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "pyproject.toml").exists():
        return False
    if not (path / "scripts" / "rfdiffusion_inference.py").exists():
        return False
    if not (path / "scripts" / "proteinmpnn_interface_design.py").exists():
        return False
    if not (path / "scripts" / "rf2_predict.py").exists():
        return False
    return True


def find_rfantibody_repo(project_root: Path, scan_roots: List[Path], max_depth: int) -> Optional[Path]:
    env_root = os.environ.get("RFANTIBODY_ROOT", "").strip()
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if is_rfantibody_repo(p):
            return p

    known_candidates = [
        project_root / "data/framework/external/RFantibody",
        project_root / "RFantibody",
    ]
    for cand in known_candidates:
        if is_rfantibody_repo(cand):
            return cand.resolve()

    for root in scan_roots:
        for cur, _ in bounded_walk(root, max_depth=max_depth):
            lower = str(cur).lower()
            if any(x in lower for x in ["/.git", "/site-packages", "__pycache__"]):
                continue
            if cur.name.lower() == "rfantibody" and is_rfantibody_repo(cur):
                return cur.resolve()
    return None


def has_importable_rfantibody() -> bool:
    code, _ = run_capture([sys.executable, "-c", "import rfantibody; print('ok')"])
    return code == 0


def detect_tool_command(
    tool_name: str,
    repo_root: Optional[Path],
) -> Tuple[List[str], Optional[str], str, List[str]]:
    """Return (prefix, run_cwd, mode, evidence)."""
    evidence: List[str] = []

    cli_path = shutil.which(tool_name)
    if cli_path:
        evidence.append(f"which {tool_name} -> {cli_path}")
        return [cli_path], None, "installed_cli", evidence

    if repo_root is not None and shutil.which("uv") and (repo_root / "pyproject.toml").exists():
        evidence.append("uv detected; using uv run entrypoint from RFantibody repo")
        return ["uv", "run", tool_name], str(repo_root), "uv_run", evidence

    if repo_root is not None and has_importable_rfantibody():
        script_map = {
            "rfdiffusion": repo_root / "scripts" / "rfdiffusion_inference.py",
            "proteinmpnn": repo_root / "scripts" / "proteinmpnn_interface_design.py",
            "rf2": repo_root / "scripts" / "rf2_predict.py",
        }
        script = script_map.get(tool_name)
        if script and script.exists():
            evidence.append("python can import rfantibody; using scripts/*.py path")
            return [sys.executable, str(script)], str(repo_root), "script", evidence

    return [], None, "unresolved", evidence


def find_weight_file(filename: str, repo_root: Optional[Path], scan_roots: List[Path], max_depth: int) -> Optional[Path]:
    env_weights = os.environ.get("RFANTIBODY_WEIGHTS", "").strip()
    candidates: List[Path] = []
    if env_weights:
        candidates.append(Path(env_weights).expanduser() / filename)
    if repo_root is not None:
        candidates.append(repo_root / "weights" / filename)

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    for root in scan_roots:
        for cur, files in bounded_walk(root, max_depth=max_depth):
            if filename in files:
                path = (cur / filename).resolve()
                return path
    return None


def normalize_scan_roots(project_root: Path, extra: List[str]) -> List[Path]:
    roots = [project_root]
    for c in [project_root.parent, Path.home() / "code", Path.home() / "PycharmProjects"]:
        if c.exists():
            roots.append(c.resolve())
    for e in extra:
        p = Path(e).expanduser().resolve()
        if p.exists():
            roots.append(p)

    dedup: List[Path] = []
    seen = set()
    for p in roots:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            dedup.append(p)
    return dedup


def main() -> int:
    args = parse_args()
    root = Path(".").resolve()

    pipeline_cfg = read_yaml(root / args.pipeline_config)
    cdr_cfg = read_yaml(root / args.cdr_config)
    resolved_inputs_cfg = read_yaml(root / args.resolved_inputs)
    resolved_inputs = resolved_inputs_cfg.get("resolved_inputs", {})

    nanobody_alias = Path(resolved_inputs.get("nanobody_sequence_file", ""))
    nanobody_seq = read_sequence(nanobody_alias) if nanobody_alias.exists() else ""
    detected_chain = detect_nanobody_chain(root / args.complex_structure, nanobody_seq)

    runtime = detect_runtime()
    scan_roots = normalize_scan_roots(root, args.scan_root)
    repo_root = find_rfantibody_repo(root, scan_roots, max_depth=args.max_depth)

    tool_names = ["rfdiffusion", "proteinmpnn", "rf2"]
    tooling: Dict[str, object] = {"execute_real_tools": False}
    tool_evidence: Dict[str, List[str]] = {}

    for tool in tool_names:
        prefix, run_cwd, mode, evidence = detect_tool_command(tool, repo_root)
        tooling[tool] = {
            "command_prefix": prefix,
            "run_cwd": run_cwd,
            "detected_mode": mode,
        }
        tool_evidence[tool] = evidence

    checkpoints = {}
    for key, filename in RFAB_WEIGHT_FILES.items():
        found = find_weight_file(filename=filename, repo_root=repo_root, scan_roots=scan_roots, max_depth=args.max_depth)
        checkpoints[key] = str(found) if found else None

    unresolved: List[str] = []

    chain_in_cfg = str(cdr_cfg.get("nanobody_chain_id", "")).strip()
    if not chain_in_cfg and not detected_chain:
        unresolved.append("cdr.chain")

    framework_pdb_cfg = str(
        resolved_inputs.get("nanobody_framework_pdb_file", "")
        or pipeline_cfg.get("inputs", {}).get("nanobody_framework_pdb_file", "")
    ).strip()
    if framework_pdb_cfg:
        framework_path = Path(framework_pdb_cfg)
        if not framework_path.is_absolute():
            framework_path = (root / framework_path).resolve()
        if not framework_path.exists():
            unresolved.append("inputs.nanobody_framework_pdb_file")
    else:
        unresolved.append("inputs.nanobody_framework_pdb_file")

    for tool in tool_names:
        if not tooling[tool]["command_prefix"]:  # type: ignore[index]
            unresolved.append(f"tools.{tool}_prefix")

    for key, value in checkpoints.items():
        if not value:
            unresolved.append(f"checkpoints.{key}")

    if not runtime.get("nvidia_gpu"):
        unresolved.append("vm.gpu_and_driver")
    if not runtime.get("cuda"):
        unresolved.append("vm.cuda")

    if not unresolved:
        tooling["execute_real_tools"] = True

    outdir = root / "data/processed"
    outdir.mkdir(parents=True, exist_ok=True)

    report = {
        "detected_at": subprocess.getoutput("date '+%Y-%m-%d %H:%M:%S'"),
        "runtime": runtime,
        "rfantibody_repo": str(repo_root) if repo_root else None,
        "chain_detection": {
            "complex_structure": str(root / args.complex_structure),
            "nanobody_alias": str(nanobody_alias),
            "detected_chain": detected_chain,
            "chain_in_cdr_config": chain_in_cfg,
            "nanobody_length": len(nanobody_seq),
        },
        "framework_input": {
            "nanobody_framework_pdb_file": framework_pdb_cfg,
            "exists": bool(framework_pdb_cfg and (root / framework_pdb_cfg).exists()),
        },
        "scan_roots": [str(p) for p in scan_roots],
        "tooling_detected": tooling,
        "tool_detection_evidence": tool_evidence,
        "checkpoints_detected": checkpoints,
        "cli_args_reference": CLI_ARGS_REFERENCE,
        "unresolved_fields": unresolved,
    }

    json_path = outdir / "autodetect_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    tooling_yaml = {
        "execute_real_tools": bool(tooling["execute_real_tools"]),
        "rfdiffusion": tooling["rfdiffusion"],
        "proteinmpnn": tooling["proteinmpnn"],
        "rf2": tooling["rf2"],
        "checkpoints": checkpoints,
        "cli_args_reference": CLI_ARGS_REFERENCE,
        "rfantibody_repo": str(repo_root) if repo_root else "",
    }

    tooling_yaml_path = root / "data/configs/tooling.detected.yaml"
    tooling_yaml_path.write_text(yaml.safe_dump(tooling_yaml, sort_keys=False), encoding="utf-8")

    unresolved_txt = outdir / "unresolved_fields.txt"
    unresolved_txt.write_text("\n".join(unresolved) + ("\n" if unresolved else ""), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {tooling_yaml_path}")
    print(f"Wrote: {unresolved_txt}")

    if unresolved:
        print("\nUNRESOLVED_FIELDS:")
        for field in unresolved:
            print(f"- {field}")
        if args.strict:
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
