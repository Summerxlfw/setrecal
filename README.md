# SetRecal: set supervision and shared gated recalibration for multi-rater segmentation

Official code for ***SetRecal: Set-level supervision and gated latent recalibration for multi-rater medical image segmentation*** by Tian Xia, Zhenqing Wang and Liping Sun. SetRecal trains a single segmentation model against a *set* of raters: a set-summary loss (mean-mask soft Dice plus a pairwise agreement term) replaces one-hot supervision, and a shared gated latent recalibration module (`z + tanh(g) W z`, 37 parameters) recalibrates each expert latent. Evaluated on multi-rater fundus segmentation (RIGA cup/disc, REFUGE cup) and nasopharyngeal CT (NPC-170).

This repository contains the package (`setsup_recal`), preprocessing scripts, evaluation protocols, and Stage-II training and development evaluation on RIGA cup, RIGA disc, REFUGE cup, and NPC-170. It connects to the D-Persona source and task-specific Stage-I checkpoints listed in [REPRODUCE.md](REPRODUCE.md). **No image data, masks, or patient information are included**; dataset files, weights, and third-party comparison implementations are external inputs obtained under their own terms.

## Install and check

Use Python 3.10 or newer. From the repository root:

```sh
python -m pip install '.[reproduction]'
python -m unittest discover -s tests -v
python example.py
python -m setsup_recal.reproduce inspect --task riga_cup --protocol protocol
```

`inspect` verifies metadata and split fingerprints. `TESTED_ENVIRONMENT.json` records the CPU software checks (Python 3.12, PyTorch 2.5.1, 21 tests passed). Training uses the prepared arrays and dependencies described below.

## Repository layout

```text
setsup_recal/          Core package: set loss, recalibration module, training, evaluation,
                       data loading, fingerprint checks, reproduction CLI
preprocessing/         RIGA contour conversion, REFUGE and NPC-170 cache builders, audit receipts
protocol/              Default study protocol: configurations, 8 split manifests, transforms,
                       RIGA source-disjoint split, upstream source fingerprints
protocol_initial/      Initial 131-view RIGA protocol used for earlier ablations
tests/                 21 CPU unit tests (method, protocol metadata, reproduction helpers)
example.py             Synthetic CPU demonstration of the public API
configurations.json    Stage-II task settings (also mirrored per protocol)
REPRODUCE.md           Step-by-step reproduction: inputs, caches, training, evaluation
PREPROCESSING.md       Dataset layouts and cache construction commands
IMPLEMENTATION_CHECKS.md  Release scope and verification record
SOURCE_PROVENANCE.json, RELEASE_PROTOCOL_UPDATE.json, TESTED_ENVIRONMENT.json
```

## Evaluation protocols

| Protocol directory | RIGA development | Aggregation and selection | Manuscript use |
|---|---|---|---|
| `protocol/` (default) | 127 views from 126 sources, disjoint from training | Average within each raw-image source, then across sources; minimum source-macro GED | Main RIGA comparison |
| `protocol_initial/` | Initial 131-view split, including four views with training-source overlap | Image-macro GED | Earlier RIGA ablations, repeats and intervals |

The training records are identical in both protocols. REFUGE and NPC settings are unchanged. REFUGE averages images; NPC computes metrics on complete patient volumes and then averages patients. Both RIGA targets use the same ordered image split.

## Reproduce the manuscript results

After preparing the caches ([PREPROCESSING.md](PREPROCESSING.md)) and the external inputs ([REPRODUCE.md](REPRODUCE.md)):

- Manuscript **Table 2** (RIGA cup/disc comparison, source-disjoint development) and **Table 3** (REFUGE / NPC-170 comparison) are produced by the `train`/`evaluate` commands in [REPRODUCE.md](REPRODUCE.md) with `--protocol protocol`.
- Manuscript **Table 4** (component ablation on the initial RIGA-cup split) uses `--protocol protocol_initial`.
- The recorded training recipe lives in `protocol/configurations.json`: seed 20260840, 100 epochs, batch size 8, learning rate 0.0001, 100 latent candidates, Adam with cosine annealing.

Prepare the caches, obtain the pinned D-Persona source and Stage-I weights, and run:

```sh
python -m setsup_recal.reproduce validate-cache --task riga_cup --protocol protocol --cache-root data/riga_cache/cache
python -m setsup_recal.reproduce train --task riga_cup --protocol protocol --cache-root data/riga_cache/cache --source-root external/D-Persona --stage1-checkpoint weights/riga_cup_stage1.pt --device cuda:0 --output runs/riga_cup
python -m setsup_recal.reproduce evaluate --task riga_cup --protocol protocol --cache-root data/riga_cache/cache --source-root external/D-Persona --stage1-checkpoint weights/riga_cup_stage1.pt --checkpoint runs/riga_cup/best.pt --device cuda:0 --output evaluations/riga_cup.json
```

Use `riga_disc`, `refuge`, or `npc` with the matching inputs. Loss weights and checkpoint fingerprints are task-specific. Selection retains the earliest checkpoint attaining the minimum development GED. Each command writes to a new output path.

## Use the method in another codebase

```python
import torch
from setsup_recal import self_recalibration, set_supervised_loss
recal = self_recalibration(rater_count=6, latent_dim=6)
z = torch.randn(2, 6, 6)
z_prime = recal(z)
assert torch.equal(z_prime, z)
```

The shared module computes `z + tanh(g) W z`, with a 6-by-6 matrix initialized to the identity and a scalar gate initialized to zero. It adds 37 parameters and has no bias. The mean-mask term uses squared-denominator soft Dice; the pairwise term uses mean squared error over off-diagonal soft-IoU entries in both directions. `set_supervised_loss(logits, targets, mean_weight=..., pair_weight=...)` combines these with indexed Dice for arrays of shape `[B,R,H,W]`.

`stage2_adapter(core)` trains the projection heads and recalibration while keeping Stage-I components and decoder weights fixed. The Python package name `setsup_recal` and argument `use_graph=True` are retained for API and checkpoint compatibility; the proposed update acts independently on each expert latent.

## Evaluation API

`evaluate_distribution_batch(probabilities, targets)` accepts `[B,K,H,W]` predictions and `[B,R,H,W]` annotations. It thresholds at `>=0.5`, includes diagonal GED terms, and takes no square root.

`source_macro(rows, groups, metrics)` in `setsup_recal.source_metrics` weights raw-image sources equally. `evaluate_persona_source_loader` applies this reduction to distribution metrics and indexed Dice. Per-view values remain available in the evaluation output.

For NPC, `aggregate_volume_slice_rows` assembles complete ordered slices with stable patient, volume, slice and rater identities. `patient_macro` is the volume endpoint; `sample_macro` is a slice diagnostic. Indexed Dice applies to outputs with matching rater identities.

## Data, weights, and reuse

This repository redistributes **no dataset files or model weights**. Obtain the training data through the providers' official channels and their applicable access terms:

- [RIGA](https://doi.org/10.1117/12.2293584) (orig. images: MESSIDOR; annotations: RIGA release)
- [REFUGE](https://doi.org/10.1016/j.media.2019.101570) (multi-rater development set)
- [NPC-170 / MMIS](https://doi.org/10.1145/3664647.3689150)
- [D-Persona](https://github.com/ycwu1997/D-Persona) (pinned upstream source, commit in `REPRODUCE.md`)

Stage-I checkpoints are required as pinned inputs for Stage-II training and evaluation; their required SHA-256 fingerprints are listed in [REPRODUCE.md](REPRODUCE.md), and the caches they assume are built with [PREPROCESSING.md](PREPROCESSING.md).

The supplied code has an MIT license (see [LICENSE](LICENSE)); external datasets, models and weights retain their own terms. `SOURCE_PROVENANCE.json` records the study implementation, and `RELEASE_PROTOCOL_UPDATE.json` identifies the RIGA configuration used in this release.

Claude and Codex assisted with code preparation. The authors are responsible for the implementation and reported results.
