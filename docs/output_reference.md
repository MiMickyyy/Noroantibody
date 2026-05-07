# Output Reference

This document explains the main output files produced by the pipeline and what they are used for.

## Output philosophy
The repository stores outputs in two layers:

1. phase-specific run directories  
   Large or detailed intermediate data, logs, per-combination run assets.

2. compact summary tables  
   Small CSV/JSON/FASTA outputs intended to drive the next stage.

The summary tables are the most important files for understanding the project.

## 1. Early broad-search outputs
### Phase 1
Main summaries:
- `results/summaries/phase1_coarse_pilot_summary.csv`
- `results/summaries/phase1_combination_summary.csv`
- `results/summaries/phase1_top8_combinations.csv`

Meaning:
- one row per evaluated combination / campaign line
- used to choose which hotspot logic moves into Phase 2

### Phase 2
Main summaries:
- `results/summaries/phase2_focused_pilot_summary.csv`
- `results/summaries/phase2_combination_summary.csv`
- `results/summaries/phase2_top2_combinations.csv`

Meaning:
- second narrowing layer on top of Phase 1 winners
- selects the production-scale campaign lines for Phase 3

### Phase 3
Main summaries:
- `results/summaries/phase3_main_campaign_summary.csv`
- `results/summaries/phase3_top25_pre_h2.csv`
- `results/summaries/top25_pre_h2_table.csv`

Meaning:
- candidate-level shortlist before H2 optimization
- starting point for the Phase 4 H2-only stage

## 2. H2 refinement outputs
Main summaries:
- `results/summaries/phase4_h2_refine_summary.csv`
- `results/summaries/final25_h2_optimized_candidates.csv`
- `results/summaries/final_25_h2_optimized_candidates_table.csv`
- `results/final_25/final25_nanobody_sequences.fasta`

Meaning:
- final H2-optimized panel after phase 4 filtering/ranking
- standard AF3 handoff source for the earlier project path

## 3. CDR1 rescue outputs
### Phase 5
Main summaries:
- `results/summaries/phase5_cdr1_rescue_pilot_summary.csv`
- `results/summaries/phase5_cdr1_rescue_condition_ranking.csv`
- `results/summaries/phase5_selected_conditions.csv`

Meaning:
- condition-level comparison across parent x hotspot rescue conditions
- used to decide what enters Phase 6

### Phase 6
Main summaries:
- `results/summaries/phase6_cdr1_rescue_main_summary.csv`
- `results/summaries/phase6_cdr1_rescue_final_ranked_candidates.csv`
- `results/summaries/final25_cdr1_rescue_candidates.csv`

Meaning:
- candidate-level ranked rescue output
- main source for rescue-derived AF3 export

## 4. Test1 / champion local maturation outputs
### `phase_next_test1_local_maturation`
Main summaries:
- `results/summaries/phase_next_test1_local_maturation_rf2_summary.csv`
- `results/summaries/phase_next_test1_local_maturation_strict_pass.csv`
- `results/summaries/phase_next_test1_local_maturation_strict_pass.fasta`
- `results/summaries/phase_next_test1_local_maturation_summary.md`

### `phase_next_champion_narrow50`
Main summaries:
- `results/summaries/phase_next_champion_narrow50_rf2_summary.csv`
- `results/summaries/phase_next_champion_narrow50_strict_pass.csv`
- `results/summaries/phase_next_champion_narrow50_strict_pass.fasta`
- `results/summaries/phase_next_champion_narrow50_summary.md`

### Phase 9
Main summaries:
- `results/summaries/phase9_test1_local_maturation_expand150_rf2_summary.csv`
- strict-pass / FASTA derivatives depending on phase settings

Meaning:
- later-phase branch or champion-specific local maturation results
- best place to inspect compact vs broader alternative-mode behavior before AF3

## 5. AF3 handoff outputs
Representative outputs:
- `results/af3_web_exports/`
- `results/af3_web_exports_cdr1_rescue/`
- `results/af3_web_exports_strict_pass*/`

Meaning:
- JSON/CSV packages for AF3 web submission
- mapping tables that reconnect short AF3 task names to full candidate IDs

## 6. Project-wide summary outputs
Representative files:
- `results/summaries/project_summary.md`
- `results/summaries/final_metadata.json`
- AF3-analysis summary tables written by the downstream analysis scripts

## 7. Logs
Each phase writes logs under:
- `logs/<phase_name>/`

Use these first when debugging.

## 8. Which output should drive which next step?
- After Phase 1: `phase1_top8_combinations.csv`
- After Phase 2: `phase2_top2_combinations.csv`
- After Phase 3: `phase3_top25_pre_h2.csv`
- After Phase 4: `final25_h2_optimized_candidates.csv`
- After Phase 5: `phase5_selected_conditions.csv`
- After Phase 6: `phase6_cdr1_rescue_final_ranked_candidates.csv`
- After later local maturation: the corresponding `*_rf2_summary.csv` and `*_strict_pass.csv`

This repository is designed so that each important stage emits a compact CSV or FASTA that can be used directly by the next stage.
