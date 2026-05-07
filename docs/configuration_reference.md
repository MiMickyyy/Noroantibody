# Configuration Reference

This document explains which YAML files control which parts of the project.

## 1. `data/configs/pipeline.yaml`
This is the highest-level project config.

Controls:
- project name and seed
- input file references
- target-preparation behavior
- default execution behavior (`dry_run`, `resume`, parallelism)
- RF2 hard thresholds
- ranking weights
- postprocessing behavior
- AF3 handoff export count

Edit this file when you need to:
- point to a different framework structure
- change target preparation settings
- change default ranking or threshold policy
- change top-N export behavior

## 2. `data/configs/hotspot_campaigns.yaml`
Defines the antigen-side hotspot families used in broad campaign stages.

Current campaigns:
- `campaign_A_core`
- `campaign_B_A_plus_D_rim_bridge`
- `campaign_C_A_plus_pocket_rim_HBGA_adjacent`

Edit this file when you want to:
- test a different antigen patch logic
- add/remove campaign families
- change which full-length antigen residues count as anchor residues

Important:
- residue numbering here is project-facing full-length numbering
- mapping into cropped target numbering happens later through the residue map tables

## 3. `data/configs/design_matrix.yaml`
Controls the broad loop-design search space.

Current role:
- select which loops are redesigned in broad search
- define H1/H3 length deltas
- keep H2 fixed during the initial design rounds

Edit this file when you want to:
- widen or narrow H1/H3 loop-length exploration
- change which loops are designed in the early stages

## 4. `data/configs/phases.yaml`
Controls how large each phase is.

Examples of fields:
- `backbones_per_combination`
- `sequences_per_backbone`
- `input_top_combinations`
- `input_top_candidates`
- `candidates_per_branch`

Edit this file when you want to:
- make a pilot phase smaller or larger
- scale the main campaign up or down
- alter how many parents are promoted into a later phase

## 5. `data/configs/cdr_boundaries.yaml`
Defines nanobody CDR boundaries.

This file is critical because later logic depends on correct parsing of:
- H1 / CDR1
- H2 / CDR2
- H3 / CDR3

Edit this file when:
- switching to a new framework or numbering scheme
- validating the project on a new nanobody backbone

## 6. `data/configs/tooling.yaml`
Defines command prefixes and checkpoint paths for external tools.

Typical fields:
- `execute_real_tools`
- `rfdiffusion.command_prefix`
- `proteinmpnn.command_prefix`
- `rf2.command_prefix`
- checkpoint paths

This file is environment-specific.

Edit this file when:
- moving from laptop to VM
- moving from one framework checkout to another
- changing CUDA environments or executable prefixes

## 7. Manual selection override configs
Files:
- `data/configs/phase2_selected_combinations.yaml`
- `data/configs/phase3_selected_combinations.yaml`

Purpose:
- manually override which combinations are promoted, instead of relying only on automatic top-k CSV outputs

Use these when you want:
- manuscript-consistent reruns
- fixed selection sets across environments
- manual curation between phases

## 8. Late-stage local refinement configs
### CDR1 rescue
- `data/configs/cdr1_rescue_phase.yaml`
- `data/configs/cdr1_rescue_hotspots.yaml`

### Test1 local maturation
- `data/configs/test1_local_maturation_phase.yaml`
- `data/configs/test1_local_maturation_hotspots.yaml`

### Champion narrowing
- `data/configs/champion_narrow50_phase.yaml`
- `data/configs/champion_narrow50_hotspots.yaml`

These files define:
- parent candidates
- editable positions
- late-stage hotspot subsets
- candidates/branch or candidates/condition scale

These are the correct place to edit if you are reproducing or extending Stages 5–9.

## Practical editing order
If you are starting from a clean clone, the most useful order is:

1. `pipeline.yaml`
2. `cdr_boundaries.yaml`
3. `tooling.yaml`
4. `hotspot_campaigns.yaml`
5. `design_matrix.yaml`
6. `phases.yaml`
7. late-stage phase configs only if you are reproducing later stages
