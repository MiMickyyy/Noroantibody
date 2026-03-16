#!/usr/bin/env python3
"""Helper utility to inspect/update CDR boundaries config."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_common import PipelineError, log, read_yaml, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage CDR boundaries for nanobody redesign pipeline.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="Show current CDR boundaries")
    p_show.add_argument(
        "--config",
        default="data/configs/cdr_boundaries.yaml",
        help="Path to CDR boundaries config",
    )

    p_set = sub.add_parser("set", help="Set CDR boundaries")
    p_set.add_argument("--config", default="data/configs/cdr_boundaries.yaml")
    p_set.add_argument("--h1", nargs=2, type=int, required=True, metavar=("START", "END"))
    p_set.add_argument("--h2", nargs=2, type=int, required=True, metavar=("START", "END"))
    p_set.add_argument("--h3", nargs=2, type=int, required=True, metavar=("START", "END"))
    p_set.add_argument("--chain", required=True, help="Nanobody chain ID")
    p_set.add_argument("--scheme", default="USER_DEFINED", help="Annotation scheme note")

    return parser.parse_args()


def normalize_pair(pair):
    a, b = int(pair[0]), int(pair[1])
    return [min(a, b), max(a, b)]


def validate_order(h1, h2, h3):
    if not (h1[1] < h2[0] < h2[1] < h3[0]):
        raise PipelineError("Invalid boundary order: expected H1 < H2 < H3")


def show_config(path: Path):
    cfg = read_yaml(path)
    print(f"config: {path}")
    print(f"nanobody_chain_id: {cfg.get('nanobody_chain_id', '')}")
    c = cfg.get("cdr_boundaries", {})
    print(f"H1: {c.get('H1')}")
    print(f"H2: {c.get('H2')}")
    print(f"H3: {c.get('H3')}")
    print(f"annotation_scheme: {cfg.get('annotation_scheme', '')}")


def set_config(path: Path, args: argparse.Namespace):
    h1 = normalize_pair(args.h1)
    h2 = normalize_pair(args.h2)
    h3 = normalize_pair(args.h3)
    validate_order(h1, h2, h3)

    cfg = {
        "nanobody_chain_id": str(args.chain).strip(),
        "cdr_boundaries": {"H1": h1, "H2": h2, "H3": h3},
        "annotation_scheme": args.scheme,
        "notes": "Validated by user. Pipeline will use these boundaries directly.",
    }
    write_yaml(path, cfg)
    log(f"Updated {path}")


def main() -> int:
    args = parse_args()
    path = Path(args.config)
    try:
        if args.cmd == "show":
            show_config(path)
        elif args.cmd == "set":
            set_config(path, args)
        else:
            raise PipelineError(f"Unknown command: {args.cmd}")
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
