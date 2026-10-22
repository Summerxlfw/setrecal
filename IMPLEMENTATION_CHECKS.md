# Implementation checks

Release 1.2.0. CPU checks recorded 1 October 2026.

- The method's set losses, latent transformation, augmentation and per-view metrics retain the study computations. `SOURCE_PROVENANCE.json` identifies their original symbols and fingerprints.
- The default RIGA manifests match the main comparison: 524 training views and 127 development views from 126 raw-image sources, with zero source overlap. Per-view values are averaged within each source and then equally across sources for checkpoint selection and final evaluation.
- `protocol_initial/` preserves the 131-view RIGA protocol for earlier ablations, repeats and intervals. REFUGE and NPC metadata are unchanged.
- Current RIGA Stage-I checkpoints are bound by SHA-256; configuration files agree with the recorded loss weights, seed and training settings.
- Twenty-one CPU tests passed. They check identity initialization, independent latent inputs, losses, GED, volume aggregation, gradients and frozen weights, data loading, checkpoint round trips, source weighting, source overlap rejection, current and initial protocol selection, and earliest-best source-GED checkpoint selection.
- Six metadata-inspection commands and the synthetic example passed. `TESTED_ENVIRONMENT.json` records the environment.

Software checks use synthetic arrays and a small synthetic backbone. Real-data training and preprocessing were not repeated for this release. The runnable scope and external inputs are listed in `REPRODUCE.md`.
