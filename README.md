# Noroantibody: A Stage-Wise Computational Pipeline for Norovirus Nanobody Redesign

## Abstract
This repository contains the computational redesign pipeline developed for a Norovirus CHDC2094 nanobody optimization project. The project starts from a known nanobody–P-domain dimer system and moves through a staged workflow built around antibody-tuned RFdiffusion, ProteinMPNN, RF2-based surrogate filtering, and downstream AF3-guided analysis. The overall strategy is deliberately funnel-shaped: begin with broad hotspot-guided exploration, narrow the search using fast structural surrogates, then spend computation only on region-focused local maturation lines. The repository therefore serves two roles: it is both the execution framework for the design campaign and the reproducible record of how candidate logic evolved through Stage 9.

The central scientific outcome is not simple recovery of the wild-type (WT) binder. WT remains the stability reference, but later-stage designed candidates repeatedly favor an alternative interface logic. Across the later rounds, CDR1-centered behavior and local support around that region emerge as the main mechanistic differentiator, especially when comparing broader alternative-pose candidates to more compact support-coupled candidates.

## Safety and Scope
This is a computational structural modeling and protein design repository only.

- No infectious virus work
- No viral propagation
- No animal or human experimentation
- No clinical samples
- No wet-lab protocols
- No local AlphaFold 3 deployment in this repository

AF3 in this project was used as an external downstream analysis layer after RF2-based triage.

## What This Repository Includes
This repository is the **pipeline-focused** codebase.

Included:
- RFantibody-style stage orchestration
- hotspot campaign definitions
- target preparation and input sanitation
- ProteinMPNN / RF2 / RFdiffusion wrappers
- local phase runners through Stage 9 logic
- result parsing, filtering, reranking, and export scripts
- AF3 post-analysis helper scripts used after external AF3 runs

Intentionally not versioned here:
- large cloud/VM intermediate outputs
- AF3 raw result folders
- downloaded model weights and large third-party framework payloads
- the separate BIEN225 cellular automaton module (maintained in the companion repository `Noroantibody_AC`)

## Pipeline Framework Through Stage 9
The project is best understood as a compact four-part framework rather than a long chronological log.

### 1. Broad exploration
Goal: identify which antigen-side hotspot logic is worth pursuing.

Core pattern:
- generate backbones with antibody-aware RFdiffusion
- design sequences with ProteinMPNN
- screen designs with RF2

Main search families:
- coarse pilot and focused pilot campaigns
- multi-hotspot main campaign lines, especially the later-successful `campaign_C_A_plus_pocket_rim_HBGA_adjacent_*` family

### 2. Narrowing with surrogate filtering
Goal: avoid expensive full downstream analysis on weak lines.

Primary filters:
- RF2 pAE
- design-vs-RF2 RMSD
- candidate ranking tables
- combination-level summaries

This step turned the project from a broad hotspot search into a practical shortlist-driven workflow.

### 3. Region-focused optimization
Goal: stop doing broad redesign and instead refine the local interface logic.

Main local refinement themes:
- H2 optimization
- CDR1 rescue and rescue-condition ranking
- Test1-centered local maturation
- narrowed champion line refinement
- Phase 9 expansion of the Test1-derived branches

Scientific interpretation:
- the later-stage problem was not “find any binder”
- it was “preserve a viable alternative mode and improve how local support stabilizes it”

### 4. Final shortlisted candidates
By the end of Stage 9, the project had three useful reference points for downstream interpretation:
- `WT`: stability and interface-maturity reference
- `spg3_024`: broader alternative-pose comparator
- `p9c_052 / spg1_020`: compact alternative-pose candidate

## Stage Map
| Stage | Purpose | Main output |
|---|---|---|
| Phase 0 | smoke test and environment validation | minimal executable check |
| Phase 1 | coarse hotspot exploration | top combinations shortlist |
| Phase 2 | focused pilot refinement | narrowed campaign choices |
| Phase 3 | main campaign | top 25 pre-H2 candidates |
| Phase 4 | H2 refinement | H2-optimized shortlist |
| Phase 5 | CDR1 rescue pilot | rescue-condition ranking |
| Phase 6 | CDR1 rescue expansion | rescue candidate ranking |
| Phase 7/8 analysis layer | AF3-guided local maturation interpretation | champion-consensus logic |
| Phase 9 | Test1-derived branch expansion | expanded local maturation panel |

A more detailed stage summary is provided in [`docs/pipeline_stage_summary.md`](docs/pipeline_stage_summary.md).

## Key Scientific Conclusions Carried Forward
These are the project conclusions that mattered most by the end of Stage 9.

1. **WT remains the stability reference.**  
   WT is not trivially displaced by designed sequences when judged using AF3-native confidence and interface consistency.

2. **The designed family does not simply become WT.**  
   The strongest designed candidates repeatedly favor an alternative interface pattern rather than a clean WT-like recovery.

3. **CDR1 behavior is a late-stage control point.**  
   Later optimization lines repeatedly converged on CDR1-support logic as the most useful local lever for improving or destabilizing the alternative pose family.

4. **Compactness and support-coupling matter.**  
   Within the designed family, the stronger candidates are not just “higher scoring”; they tend to be more compact and more support-coupled around the alternative core logic.

## Repository Layout
```text
.
├── data/
│   ├── configs/                 # pipeline, phase, hotspot, tooling, and CDR configs
│   ├── maps/                    # residue mapping tables
│   ├── processed/               # lightweight derived metadata
│   ├── raw/                     # lightweight provenance metadata only in git
│   └── target/                  # prepared target examples and reports
├── docs/
├── scripts/                     # orchestration, parsing, analysis, and export scripts
├── README.md
├── environment.yml
├── requirements.txt
└── config.yaml
```

## Installation
### Option A: conda
```bash
conda env create -f environment.yml
conda activate noro_rfantibody
```

### Option B: venv + pip
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## External Tooling Requirement
The repository expects RFantibody-style tooling to be installed separately.

Typical setup:
```bash
cd data/framework/external/RFantibody
pip install -e .
bash include/download_weights.sh
cd -
```

The external framework directory is intentionally not packaged into git because of size and third-party ownership.

## Minimal Input Files
Project-root examples used during development include:
- `VP1.prot`
- `P-domain dimer.prot`
- `Nanobody.fa`
- framework structure and target structure inputs referenced by config

The input-resolution helpers sanitize filenames with spaces and record exact aliases instead of silently renaming source files.

## Core Run Pattern
Prepare inputs and targets:
```bash
python scripts/prepare_inputs.py
python scripts/prepare_targets.py \
  --pipeline-config data/configs/pipeline.yaml \
  --campaign-config data/configs/hotspot_campaigns.yaml
python scripts/autodetect_runtime_and_tooling.py --strict
```

Run phases:
```bash
python scripts/run_pipeline.py --phase phase0_smoke
python scripts/run_pipeline.py --phase phase1_coarse_pilot
python scripts/run_pipeline.py --phase phase2_focused_pilot
python scripts/run_pipeline.py --phase phase3_main_campaign
python scripts/run_pipeline.py --phase phase4_h2_refine
python scripts/run_pipeline.py --phase phase5_cdr1_rescue_pilot
python scripts/run_pipeline.py --phase phase6_cdr1_rescue_main
python scripts/run_pipeline.py --phase phase9_test1_local_maturation_expand150
```

Phase shell wrappers are also available in `scripts/run_phase*.sh`.

## Important Analysis / Export Scripts
Representative downstream utilities included in this repository:
- `scripts/parse_and_rank.py`
- `scripts/export_af3_web_package.py`
- `scripts/generate_af3_batch_json.py`
- `scripts/rerank_af3_with_rf2.py`
- `scripts/analyze_af3_interface_stability.py`
- `scripts/analyze_af3_project_master.py`
- `scripts/analyze_phase7_phase8_af3_narrow.py`
- `scripts/analyze_wt_detailed_interactions.py`
- `scripts/analyze_wt_vs_p9c052_af3.py`
- `scripts/plot_wt_interaction_modules.py`

These scripts reflect the real workflow that connected the RF2-filtered pipeline to later AF3-based interpretation.

## Recommended Reading Order for New Users
If you are opening this repository for the first time, the fastest way to understand it is:
1. read this README
2. read [`docs/pipeline_stage_summary.md`](docs/pipeline_stage_summary.md)
3. inspect `data/configs/phases.yaml`
4. inspect `data/configs/hotspot_campaigns.yaml`
5. read `scripts/run_pipeline.py`
6. then inspect the analysis/export scripts relevant to your stage of interest

## Reproducibility Notes
- This repository preserves the **logic and structure** of the campaign.
- It does **not** include all raw AF3 folders, VM-only phase outputs, or downloaded model weights.
- Large run directories and cloud outputs were intentionally excluded to keep the git repository usable.
- The curated code/config package here is the right starting point for understanding, extending, or documenting the pipeline.

## Companion Repository
The BIEN225 cellular automaton conversion is maintained separately in:
- `Noroantibody_AC`

That companion repository packages:
- the pipeline foundation needed for the course project
- the CA-ready bridge tables
- the local 2D time-dependent CA implementation

## Citation-Style Summary
If you need one sentence for slides or a manuscript draft:

> This repository implements a staged computational nanobody redesign workflow for Norovirus, beginning with hotspot-guided structural exploration and ending with region-focused local maturation logic in which WT remains the stability reference while designed candidates increasingly favor an alternative, CDR1-sensitive interface mode.
