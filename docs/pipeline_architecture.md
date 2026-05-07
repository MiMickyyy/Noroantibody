# Pipeline Architecture

This document explains how the pipeline is organized conceptually and operationally.

## Core design pattern
The project uses a funnel-shaped computational workflow:

1. broad hotspot-guided exploration
2. RF2-based surrogate narrowing
3. region-focused local optimization
4. AF3 handoff / interpretation outside the main generation loop

This shape is intentional. The code is designed to spend large-scale generation only when the search space is still broad, then move into more restricted and interpretable refinement modes.

## Main components
### Target preparation layer
Files:
- `scripts/prepare_inputs.py`
- `scripts/prepare_targets.py`
- `scripts/map_residue_numbers.py`

Purpose:
- resolve user input files
- fetch or normalize the antigen structure
- crop the target region used for design
- create residue mapping tables that convert full-length numbering into structure-local numbering

### Broad design layer
Files:
- `data/configs/hotspot_campaigns.yaml`
- `data/configs/design_matrix.yaml`
- `scripts/run_pipeline.py`
- `scripts/tool_wrappers.py`

Purpose:
- define campaign hotspot families
- define which loops and loop lengths are explored
- generate backbones and sequences
- evaluate self-consistency with RF2

### Ranking and filtering layer
Files:
- `scripts/run_pipeline.py`
- `scripts/parse_and_rank.py`
- `scripts/check_phase_diversity.py`
- `scripts/pipeline_common.py`

Purpose:
- score combinations and candidates
- enforce RF2 hard thresholds
- deduplicate sequences
- carry only the best lines forward

### Late-stage local refinement layer
Files:
- `data/configs/cdr1_rescue_phase.yaml`
- `data/configs/test1_local_maturation_phase.yaml`
- `data/configs/champion_narrow50_phase.yaml`
- `scripts/run_pipeline.py`
- `scripts/summarize_cdr1_rescue.py`

Purpose:
- stop broad redesign
- fix most of the nanobody and optimize only restricted local regions
- compare mechanistic refinement hypotheses rather than general design quality

### AF3 bridge and interpretation layer
Files:
- `scripts/export_af3_web_package.py`
- `scripts/generate_af3_batch_json.py`
- `scripts/rerank_af3_with_rf2.py`
- `scripts/analyze_af3_interface_stability.py`
- `scripts/analyze_af3_project_master.py`

Purpose:
- export selected candidates for external AF3 runs
- analyze returned AF3 models in the same project language as the RF2 pipeline
- interpret whether candidate logic is WT-like, broader alternative, or compact alternative

## Phase organization
The phase scheduler is configured in `data/configs/phases.yaml` and executed by `scripts/run_pipeline.py`.

Three broad groups exist.

### Group A: discovery phases
- `phase0_smoke`
- `phase1_coarse_pilot`
- `phase2_focused_pilot`
- `phase3_main_campaign`

These phases vary hotspot logic and H1/H3 length/shape more broadly.

### Group B: shortlisted candidate optimization
- `phase4_h2_refine`
- `phase5_cdr1_rescue_pilot`
- `phase6_cdr1_rescue_main`

These phases no longer search broadly. They optimize specific shortlisted parent lines.

### Group C: local maturation after AF3 interpretation
- `phase_next_test1_local_maturation`
- `phase_next_champion_narrow50`
- `phase9_test1_local_maturation_expand150`

These phases operationalize the insight that the project had become a local stabilization problem, especially around CDR1-linked support logic.

## Why the pipeline is staged this way
The pipeline is designed to answer different questions at different scales.

### Early question
Where can a plausible interface form?

### Mid question
Which candidate lines are structurally self-consistent enough to keep?

### Late question
Can a compact alternative interface mode be stabilized without forcing the system back to WT?

That separation is the reason the repository contains both broad search configs and extremely local later-stage configs. They are not duplicates; they reflect different scientific tasks.
