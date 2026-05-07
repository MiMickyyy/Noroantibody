# Deployment and Quickstart

This document is the practical setup guide for getting the pipeline into a runnable state.

## 1. What you need before running
You need four things:

1. Python environment (`conda` or `venv`)
2. The external RFantibody-style tool stack installed locally
3. Valid model weights / command prefixes configured in `data/configs/tooling.yaml`
4. A real framework structure path set in `data/configs/pipeline.yaml`

This repository does **not** vendor the external framework or model checkpoints.

## 2. Create the environment
### Option A: conda
```bash
conda env create -f environment.yml
conda activate noro_rfantibody
```

### Option B: venv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Install RFantibody-style external tooling
Typical pattern:
```bash
cd data/framework/external/RFantibody
pip install -e .
bash include/download_weights.sh
cd -
```

If your external framework lives elsewhere, that is also fine, but then `tooling.yaml` and your shell environment must point to the correct commands and checkpoints.

## 4. Edit the critical config fields
### `data/configs/pipeline.yaml`
Confirm:
- antigen sequence files
- nanobody sequence file
- `nanobody_framework_pdb_file`
- optional local antigen structure override

### `data/configs/cdr_boundaries.yaml`
Confirm the H1/H2/H3 boundaries are correct for your framework.

### `data/configs/tooling.yaml`
Confirm:
- command prefixes for `rfdiffusion`, `proteinmpnn`, `rf2`
- weights / checkpoint paths
- whether `execute_real_tools: true` is appropriate for your environment

## 5. Prepare inputs and targets
```bash
python scripts/prepare_inputs.py
python scripts/prepare_targets.py \
  --pipeline-config data/configs/pipeline.yaml \
  --campaign-config data/configs/hotspot_campaigns.yaml
```

What this does:
- resolves input filename aliases
- records canonical file locations
- prepares the target structure and cropped target files
- produces residue mapping tables used later for hotspot conversion

## 6. Autodetect the runtime/tooling layer
```bash
python scripts/autodetect_runtime_and_tooling.py --strict
```

This helps populate or validate runtime command prefixes and checkpoint locations.

## 7. Run the smoke / dry-run validation
Recommended:
```bash
python scripts/dry_run_validate.py
```

Equivalent shell helper:
```bash
bash scripts/bootstrap_and_smoke.sh
```

The smoke run is the right first check because it validates:
- config loading
- directory creation
- target preparation compatibility
- wrapper argument construction
- phase orchestration

## 8. Run the broad pipeline
Typical progression:
```bash
python scripts/run_pipeline.py --phase phase1_coarse_pilot
python scripts/run_pipeline.py --phase phase2_focused_pilot
python scripts/run_pipeline.py --phase phase3_main_campaign
python scripts/run_pipeline.py --phase phase4_h2_refine
```

## 9. Run later local-optimization stages
If you are reproducing the later project logic:
```bash
python scripts/run_pipeline.py --phase phase5_cdr1_rescue_pilot
python scripts/run_pipeline.py --phase phase6_cdr1_rescue_main
python scripts/run_pipeline.py --phase phase_next_test1_local_maturation
python scripts/run_pipeline.py --phase phase_next_champion_narrow50
python scripts/run_pipeline.py --phase phase9_test1_local_maturation_expand150
```

## 10. If something fails early
Work through these in order:

1. `python scripts/prepare_inputs.py`
2. `python scripts/prepare_targets.py ...`
3. `python scripts/autodetect_runtime_and_tooling.py --strict`
4. `python scripts/dry_run_validate.py`
5. inspect the corresponding file in `logs/`

The most common causes of failure are:
- missing framework PDB path
- missing RFantibody external install
- unresolved command prefixes / weights
- bad CDR boundaries for the framework
