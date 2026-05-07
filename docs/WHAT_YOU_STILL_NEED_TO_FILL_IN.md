# Deployment Checklist: What You Still Need to Fill In

Safety and Ethics Statement:
This study is a computational structural modeling and protein design project focused on nanobody–Norovirus interactions. The work uses Virus-Like Particle (VLP)-related structural information only and does not involve infectious virus, viral propagation, animal experiments, human subjects, clinical samples, or wet-lab experimental procedures. All project activities are conducted under institutional safety and ethics oversight at the University of California, Riverside.

Use this checklist before attempting a real run.

## Required configuration items
1. Validate CDR boundaries in `data/configs/cdr_boundaries.yaml`.
2. Set `inputs.nanobody_framework_pdb_file` in `data/configs/pipeline.yaml` to a valid framework PDB/mmCIF path.
3. Confirm the input sequence file paths in `data/configs/pipeline.yaml`.
4. Confirm whether `local_antigen_structure_file` should remain empty or point to a local structure override.

## Required environment items
5. Install the external RFantibody-style framework and its Python package.
6. Download or otherwise provide the model weights used by RFdiffusion, ProteinMPNN, and RF2.
7. Confirm GPU/CUDA compatibility for your execution environment.

## Required tooling items
8. Run `python scripts/autodetect_runtime_and_tooling.py --strict`.
9. Review and fix unresolved fields in `data/configs/tooling.yaml` or regenerate `data/configs/tooling.detected.yaml` on the target machine.
10. Set `execute_real_tools: true` in `data/configs/tooling.yaml` only when the runtime is genuinely ready for production execution.

## Required project-logic checks
11. Verify hotspot families in `data/configs/hotspot_campaigns.yaml`.
12. Verify the H1/H3 search space in `data/configs/design_matrix.yaml`.
13. Review RF2 thresholds and ranking weights in `data/configs/pipeline.yaml`.
14. Review phase scale in `data/configs/phases.yaml` before any long run.

## Required execution checks
15. Run `python scripts/prepare_inputs.py`.
16. Run `python scripts/prepare_targets.py --pipeline-config data/configs/pipeline.yaml --campaign-config data/configs/hotspot_campaigns.yaml`.
17. Run `python scripts/dry_run_validate.py`.
18. Run `phase0_smoke` before any broad or expensive phase.

If any of these fail, fix them before launching a large phase. This repository is designed to fail early when critical runtime assumptions are missing.
