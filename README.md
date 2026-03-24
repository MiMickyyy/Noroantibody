# Norovirus CHDC2094 Nanobody Redesign Pipeline (RFantibody Core)

Safety and Ethics Statement:
This study is a computational structural modeling and protein design project focused on nanobody–Norovirus interactions. The work uses Virus-Like Particle (VLP)-related structural information only and does not involve infectious virus, viral propagation, animal experiments, human subjects, clinical samples, or wet-lab experimental procedures. All project activities are conducted under institutional safety and ethics oversight at the University of California, Riverside.

## Scope

This repository implements a complete, execution-ready local pipeline for computational nanobody redesign against the Norovirus CHDC2094 P-domain dimer using the approved RFantibody-style core workflow:

1. antibody-finetuned RFdiffusion (backbone + dock generation)
2. ProteinMPNN (sequence design)
3. antibody-finetuned RF2 (primary filtering)
4. stop after H2 sequence optimization and export final top 25

Not included by design:
- no local AlphaFold 3 deployment
- AF3 will be run manually via web by the user after this pipeline

## Directory layout

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   ├── target/
│   ├── framework/
│   ├── maps/
│   └── configs/
├── phase0_smoke/
├── phase1_coarse_pilot/
├── phase2_focused_pilot/
├── phase3_main_campaign/
├── phase4_h2_refine/
├── results/
│   ├── summaries/
│   ├── rf2_passed/
│   ├── final_25/
│   └── af3_web_exports/
├── logs/
├── scripts/
└── README.md
```

## Environment setup

### Option A: conda

```bash
conda env create -f environment.yml
conda activate noro_rfantibody
```

### Option B: pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Install RFantibody tooling (required for `--execute`)

```bash
cd data/framework/external/RFantibody
pip install -e .
bash include/download_weights.sh
cd -
```

After installation, verify commands:

```bash
rfdiffusion --help
proteinmpnn --help
rf2 --help
```

## Input files

Place these in project root (or update `data/configs/pipeline.yaml`):

- `VP1.prot`
- `P-domain dimer.fasta`
- `Nanobody.fasta`
- `nanobody framework` structure file (PDB/mmCIF), and set:
  - `inputs.nanobody_framework_pdb_file` in `data/configs/pipeline.yaml`

The pipeline handles filenames with spaces by creating documented sanitized symlink aliases under `data/raw/sanitized/`.
It does not silently rename user files.

## Critical manual input: CDR boundaries

You must provide validated H1/H2/H3 residue ranges before phase execution.

Edit:
- `data/configs/cdr_boundaries.yaml`

Or use helper:

```bash
python scripts/manage_cdr_boundaries.py show
python scripts/manage_cdr_boundaries.py set --h1 23 34 --h2 50 58 --h3 97 106 --chain C
```

If boundaries are missing, pipeline fails loudly.

## Step 0: prepare inputs (safe aliases)

```bash
python scripts/prepare_inputs.py
```

This resolves filenames with spaces and writes:

- `data/raw/sanitized/input_aliases.json`
- `data/processed/resolved_inputs.yaml`

## Step 1: target preparation

This prepares:

1. full cleaned P-domain dimer (chains A/B)
2. cropped top-cap dimer for design stage
3. mapping table between structure numbering / crop / P-domain / full-length VP1 (when inferable)

```bash
python scripts/prepare_targets.py \
  --pipeline-config data/configs/pipeline.yaml \
  --campaign-config data/configs/hotspot_campaigns.yaml
```

If no local antigen structure is provided, the script downloads 5IYN biological assembly 1 and records provenance under `data/raw/external_sources.json`.

## Auto-detect tooling/runtime (no guessing)

Before production execution, run:

```bash
python scripts/autodetect_runtime_and_tooling.py --strict
```

Outputs:

- `data/processed/autodetect_report.json`
- `data/configs/tooling.detected.yaml`
- `data/processed/unresolved_fields.txt`

If unresolved fields exist, fix only those fields and rerun detection. Do not proceed to `--execute` until unresolved fields are empty.

### Verified RFantibody CLI contracts used by wrappers

From `RFantibody` repo (`src/rfantibody/cli/inference.py`):

- `rfdiffusion`:
  - required: `--target`, `--framework`
  - used by pipeline: `--output`, `--num-designs`, `--design-loops`, `--hotspots`, `--weights`, `--diffuser-t`, `--final-step`, `--deterministic`, `--no-trajectory`
- `proteinmpnn`:
  - required: `--input-dir` or `--input-quiver`
  - used by pipeline: `--output-dir`, `--loops`, `--seqs-per-struct`, `--temperature`, `--weights`, `--deterministic`
- `rf2`:
  - required: one of `--input-pdb/--input-dir/--input-quiver/--input-json` and one of `--output-dir/--output-quiver`
  - used by pipeline: `--input-pdb`, `--output-dir`, `--num-recycles`, `--weights`, `--seed`, `--cautious`, `--hotspot-show-prop`

## Step 2: run phases

Master orchestrator:

```bash
python scripts/run_pipeline.py --phase phase0_smoke
python scripts/run_pipeline.py --phase phase1_coarse_pilot
python scripts/run_pipeline.py --phase phase2_focused_pilot
python scripts/run_pipeline.py --phase phase3_main_campaign
python scripts/run_pipeline.py --phase phase4_h2_refine
```

Phase4 custom selected table (recommended when you manually pick 25 from phase3):

```bash
python scripts/run_pipeline.py \
  --phase phase4_h2_refine \
  --execute --resume
```

Default auto-detect order for Phase4 input CSV:
- `phase3_selected.csv` (project root)
- `results/summaries/phase3_selected.csv`
- `results/summaries/phase3_top25_pre_h2.csv`

You can still override explicitly with `--phase4-input-csv`.

Accepted input shape:
- full `phase3_top25_pre_h2.csv`, or
- a custom CSV that at least contains `candidate_id`

When only `candidate_id` is provided, the pipeline auto-fills required columns from
`phase3_main_campaign/combinations/*/candidates.csv`.

Manual Phase2 combination override (optional):
- File: `data/configs/phase2_selected_combinations.yaml`
- If present and `enabled: true`, Phase2 uses `selected_combination_ids` instead of `results/summaries/phase1_top8_combinations.csv`.

Manual Phase3 combination override (optional):
- File: `data/configs/phase3_selected_combinations.yaml`
- If present and `enabled: true`, Phase3 uses `selected_combination_ids` instead of `results/summaries/phase2_top2_combinations.csv`.

If your `resolved_targets.yaml` contains relative paths, rerun target prep first so paths are rewritten to absolute:

```bash
python scripts/prepare_targets.py
```

When real tool commands are configured, force non-dry execution:

```bash
python scripts/run_pipeline.py --phase phase1_coarse_pilot --execute
```

Shell wrappers are also provided:

```bash
bash scripts/run_phase0.sh
bash scripts/run_phase1.sh
bash scripts/run_phase2.sh
bash scripts/run_phase3.sh
bash scripts/run_phase4.sh
```

### Dry-run and limited debug batch

```bash
python scripts/run_pipeline.py --phase phase1_coarse_pilot --dry-run
python scripts/run_pipeline.py --phase phase1_coarse_pilot --dry-run --limit-per-combination 2
```

### Resume

```bash
python scripts/run_pipeline.py --phase phase3_main_campaign --resume
```

Completed tasks are skipped based on checkpoint/status manifests.

## Extra utilities

```bash
# one-command local bootstrap + smoke
bash scripts/bootstrap_and_smoke.sh

# detect runtime/tooling/checkpoint fields and unresolved items
python scripts/autodetect_runtime_and_tooling.py --strict

# regenerate ranking tables from existing phase outputs
python scripts/parse_and_rank.py --phase phase3_main_campaign

# residue number mapping lookup
python scripts/map_residue_numbers.py --chain A --full-length-resnum 297

# standalone AF3 export from final table
python scripts/export_af3_web_package.py
```

## Phase plan (exact counts)

- Phase 0:
  - 1 campaign × 1 H1 length × 1 H3 length
  - minimal smoke execution
- Phase 1:
  - 3 campaigns × 3 H1 × 5 H3 = 45 combinations
  - 8 backbones per combination
  - 1 sequence per backbone
- Phase 2:
  - top 8 combinations from phase 1
  - 25 backbones per combination
  - 2 sequences per backbone
- Phase 3:
  - top 2 combinations from phase 2
  - 150 backbones per combination
  - 2 sequences per backbone
  - rank + dedup + select top 25 pre-H2
- Phase 4:
  - H2 sequence-only optimization for top 25
  - keep backbone + H1 + H3 fixed
  - RF2 filter and keep best H2 variant per candidate
  - output final top 25

## Filtering and ranking defaults

Hard filters (configurable):

- RF2 pAE < 10
- design-vs-RF2 RMSD < 2 Å

Ranking priority:

1. RF2 self-consistency
2. hotspot-region agreement
3. docking localization consistency
4. structural plausibility
5. diversity / de-duplication

## Single-L4 optimization strategy

Implemented runtime controls:

- strict single-GPU mode (`max_gpu_parallel_jobs=1`)
- split heavy GPU stages from CPU parsing/reporting
- aggressive reuse of prepared targets and cached stage outputs
- checkpointed/resume-safe per-combination and per-candidate artifacts
- dry-run support for debugging without GPU
- limited-batch mode for pipeline debugging

Expected full run scale with approved settings:

- Phase 1 candidates: 45 × 8 × 1 = 360
- Phase 2 candidates: 8 × 25 × 2 = 400
- Phase 3 candidates: 2 × 150 × 2 = 600
- Phase 4 H2 variants: 25 × configurable (default 4) = 100 RF2 checks

Total expected workload is designed for single-L4 execution with caching and resume support and targeted to remain within approximately <=400 GPU-hours depending on actual RFdiffusion/RF2 throughput and tool configuration.

## AF3 web handoff (manual, external)

No local AF3 is run.
After phase4, use files in:

- `results/final_25/`
- `results/af3_web_exports/`

Generated AF3 handoff package includes:

- final 25 FASTA
- candidate annotation table
- JSON metadata
- antigen context references

Then submit manually to AF3 web.

## Required outputs generated by pipeline

- phase-level summary CSVs
- per-combination pilot summary table
- top 8 combinations table
- top 2 combinations table
- top 25 pre-H2 table
- final 25 H2-optimized table
- final metadata JSON
- final FASTA
- AF3-web submission spreadsheet

## Tool wrapping policy

This code wraps official RFantibody/RFdiffusion/ProteinMPNN/RF2 commands through configurable wrappers.
No model internals are reimplemented.

Configure real command prefixes in:
- `data/configs/tooling.yaml`

Set:
- `execute_real_tools: true`

before production runs.

## RFdiffusion troubleshooting (important)

If RFdiffusion crashes with errors like:
- `Non-positive determinant (left-handed or null coordinate frame)`

this is usually caused by alternate-location (`altLoc`) duplicate atoms in PDB inputs (commonly duplicate `CA` records).

This pipeline now sanitizes RFdiffusion inputs automatically by:
- removing `HETATM`/`ANISOU`
- keeping altloc ` ` and `A` only
- de-duplicating `(chain, resnum, icode, atom)` records
- writing cached `*.rfab_clean.pdb` files next to the original PDBs

If needed, regenerate targets and retry:

```bash
python scripts/prepare_targets.py
python scripts/run_pipeline.py --phase phase0_smoke --execute --no-resume
```

## What you still need to fill in

See:
- `docs/WHAT_YOU_STILL_NEED_TO_FILL_IN.md`
