# WHAT YOU STILL NEED TO FILL IN

Safety and Ethics Statement:
This study is a computational structural modeling and protein design project focused on nanobody–Norovirus interactions. The work uses Virus-Like Particle (VLP)-related structural information only and does not involve infectious virus, viral propagation, animal experiments, human subjects, clinical samples, or wet-lab experimental procedures. All project activities are conducted under institutional safety and ethics oversight at the University of California, Riverside.

1. Configure validated CDR boundaries in `data/configs/cdr_boundaries.yaml`.
2. Set `inputs.nanobody_framework_pdb_file` in `data/configs/pipeline.yaml` to a valid framework PDB/mmCIF path.
   - For RFantibody RFdiffusion, framework must be HLT-compatible and include chain `H` (and optional `L`).
3. Confirm input sequence file paths in `data/configs/pipeline.yaml`.
4. Run `python scripts/prepare_inputs.py`.
5. Run `python scripts/autodetect_runtime_and_tooling.py --strict` and inspect unresolved fields.
6. Configure unresolved official tool command prefixes in `data/configs/tooling.yaml` (or regenerate `data/configs/tooling.detected.yaml` on VM).
7. Set `execute_real_tools: true` in `data/configs/tooling.yaml` for production runs if autodetect cannot set it automatically.
8. Verify hotspot sets in `data/configs/hotspot_campaigns.yaml`.
9. Review filter/ranking thresholds in `data/configs/pipeline.yaml`.
10. Confirm GPU environment and CUDA compatibility for your RFantibody/RFdiffusion/RF2 stack.
11. Run phase0 smoke before any large run.
