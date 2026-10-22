# Study data and settings

`configurations.json` defines the four Stage-II tasks. Eight manifests preserve the training/development assignments, sample order, rater order and cache checksums. RIGA cup and disc use the same images. No image arrays are included.

| Task | Training | Development | Raters | Selection unit |
|---|---:|---:|---:|---|
| RIGA cup | 524 views | 127 views / 126 sources | 6 | Source image |
| RIGA disc | 524 views | 127 views / 126 sources | 6 | Source image |
| REFUGE cup | 400 images | 400 images | 7 | Image |
| NPC | 100 patients / 2,405 slices | 20 patients / 487 slices | 4 | Patient volume |

Run the commands in `../REPRODUCE.md` with `--cache-root` pointing to the directory containing each record's `cache_file` path. The CLI binds this root without changing any record. `SOURCE_FINGERPRINTS.json` distinguishes original manifest hashes from portable copies whose root strings were adjusted.

Input preparation is described in [PREPROCESSING.md](../PREPROCESSING.md). Recorded transform specifications are in `preprocessing/`; `SOURCE_FINGERPRINTS.json` identifies original and portable metadata copies.

`UPSTREAM.json` identifies the D-Persona source snapshot. `CONFIGURATION_PROVENANCE.json` identifies the original configuration sources. Stage-I checkpoint inputs and supported training/evaluation commands are described in [REPRODUCE.md](../REPRODUCE.md).

RIGA train and development are disjoint by raw-image SHA-256. Metrics are averaged within each source and then across the 126 sources, at every selection epoch and final evaluation. `../protocol_initial/` retains the initial RIGA split for the earlier analyses.
