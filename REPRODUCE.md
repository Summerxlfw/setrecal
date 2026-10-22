# Reproduce Stage-II training and development evaluation

This release implements SetRecal on RIGA cup, RIGA disc, REFUGE cup and NPC-170. The default RIGA protocol matches the main comparison: 127 development views, 126 raw-image sources disjoint from training, and source-macro checkpoint selection. `protocol_initial/` contains the initial 131-view RIGA protocol for the earlier ablations and repeats.

The commands require prepared datasets, the pinned D-Persona source, and task-specific Stage-I weights.

## 1. Prepare software and external inputs

Run commands from this extracted code directory, keeping `protocol/` and `preprocessing/` alongside the installed package:

```sh
python -m pip install '.[reproduction]'
python -m unittest discover -s tests -v
python -m setsup_recal.reproduce --help
```

Obtain the datasets under their providers' access terms and construct the caches using [PREPROCESSING.md](PREPROCESSING.md).

Obtain [D-Persona](https://github.com/ycwu1997/D-Persona) at commit `70689229044dabc006743f439540473df1551abe`, with its own dependencies and license. In the examples, its checkout is `external/D-Persona`; that directory must contain `LICENSE` and `D-Persona/code/`. The CLI checks the source files listed in `protocol/UPSTREAM.json` before model construction.

Stage-II uses the following Stage-I checkpoints, obtained separately. Each checkpoint contains `model_state`; the CLI checks its file fingerprint and the constructed model state. The two RIGA checkpoints below correspond to the source-disjoint main comparison. Initial-split fingerprints are in `protocol_initial/configurations.json`.

| Task | Example Stage-I file | Required SHA-256 |
|---|---|---|
| RIGA cup | `weights/riga_cup_stage1.pt` | `3e819092b9ef3660d0577408ab250d7562e66c02a001f1ee17fd4f41a79beb6f` |
| RIGA disc | `weights/riga_disc_stage1.pt` | `d8bd1272f0f419825607fc8c87e912754595d0be8939f5716edaed58a43ce936` |
| REFUGE cup | `weights/refuge_stage1.pt` | `aae3e3b43b037d9e437ef384be69896dd83ce147cb65675c854085a42df21c1f` |
| NPC-170 | `weights/npc_stage1.pt` | `a7092ba6b5d20da1e8372f82d62bf220400e905c00361284b428444dac7990af` |

## 2. Inspect the protocol and validate caches

| CLI task | Cache root in the examples | Training / development samples | Raters | Evaluation unit |
|---|---|---|---|---|
| `riga_cup` | `data/riga_cache/cache` | 524 / 127 views (126 development sources) | 6 | Raw-image source |
| `riga_disc` | `data/riga_cache/cache` | 524 / 127 views (126 development sources) | 6 | Raw-image source |
| `refuge` | `data/refuge_cache/cache` | 400 / 400 images | 7 | Image |
| `npc` | `data/npc_cache/cache` | 2405 / 487 slices from 100 / 20 patients | 4 | Patient volume |

RIGA cup and disc share image identities and the same split; they are two segmentation tasks on those images. The cache root is the directory containing `optic_cup/`, `optic_disc/`, or `npc_gtvp/`.

Inspect the metadata:

```sh
for task in riga_cup riga_disc refuge npc; do
  python -m setsup_recal.reproduce inspect --task "$task" --protocol protocol
done
```

After constructing the caches, validate them:

```sh
python -m setsup_recal.reproduce validate-cache --task riga_cup --protocol protocol --cache-root data/riga_cache/cache
python -m setsup_recal.reproduce validate-cache --task riga_disc --protocol protocol --cache-root data/riga_cache/cache
python -m setsup_recal.reproduce validate-cache --task refuge --protocol protocol --cache-root data/refuge_cache/cache
python -m setsup_recal.reproduce validate-cache --task npc --protocol protocol --cache-root data/npc_cache/cache
```

Validation reads arrays, checks their structure and finite image values, and verifies every NPZ fingerprint against the supplied manifests. The CLI binds the manifests to the selected cache directory without changing their records. A fingerprint mismatch identifies a difference in source data, conversion, or software environment to resolve before training.

## 3. Train and evaluate all four tasks

The following shell loop starts four real Stage-II training jobs in sequence, then evaluates each selected checkpoint on its development set. Set the device for the available hardware. All output directories and evaluation files must be new; change their names for another run.

```sh
repro_device=cuda:0
repro_source=external/D-Persona
for task in riga_cup riga_disc refuge npc; do
  case "$task" in
    riga_cup|riga_disc) repro_cache=data/riga_cache/cache ;;
    refuge) repro_cache=data/refuge_cache/cache ;;
    npc) repro_cache=data/npc_cache/cache ;;
  esac

  python -m setsup_recal.reproduce train \
    --task "$task" --protocol protocol --cache-root "$repro_cache" \
    --source-root "$repro_source" --stage1-checkpoint "weights/${task}_stage1.pt" \
    --device "$repro_device" --output "runs/${task}"

  python -m setsup_recal.reproduce evaluate \
    --task "$task" --protocol protocol --cache-root "$repro_cache" \
    --source-root "$repro_source" --stage1-checkpoint "weights/${task}_stage1.pt" \
    --checkpoint "runs/${task}/best.pt" --device "$repro_device" \
    --output "evaluations/${task}_validation.json"
done
```

To run one task, use that task alone in the `for task in ...` list. `--device cpu` selects CPU execution. Training and evaluation require both the exact Stage-I checkpoint and the pinned upstream source, including when evaluating an existing Stage-II checkpoint.

`protocol/configurations.json` supplies the recorded settings: seed 20260840, 100 epochs, batch size 8, learning rate 0.0001, and 100 latent candidates. Task-specific mean and pair loss weights are fixed. Training uses Adam and a cosine schedule. It trains the projection heads and shared self-recalibration parameters while keeping Stage-I components and the shared decoder fixed. Selection uses the lowest development GED, retaining the earlier checkpoint in a tie. The CLI uses these recorded settings.

Each training directory contains `config.json`, `epochs.jsonl`, `best.pt`, `last.pt`, `summary.json`, and the selected development metrics and per-view or per-patient records. Current RIGA outputs include `best_validation_source_metrics.json` and `best_validation_per_view.jsonl`; the evaluation JSON reports `source_macro` alongside the per-view diagnostic `image_macro`. Evaluation writes a JSON result to `--output` and prints a report including the checkpoint hash.

## 4. Evaluation definitions

NPC selection and evaluation use complete patient volumes: all slices, including empty slices, are assembled in order before computing each patient's metrics and averaging across patients. Preserve the patient, volume, slice, and rater identities supplied in the manifests. The study has one evaluated volume per patient.

Anonymous generated samples have no rater assignment. Indexed comparisons apply only to predictions with matching rater identities. The supplied distribution metrics use the study's threshold and GED definition, including diagonal terms and no square root.

These commands evaluate the development sets used for checkpoint selection. The manuscript identifies the earlier RIGA analyses separately from the main source-disjoint comparison. Stage-I training, REFUGE test evaluation and third-party comparison pipelines use their respective implementations.
