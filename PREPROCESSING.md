# Prepare the study caches

The scripts in `preprocessing/` convert RIGA contours and construct NPZ caches after checking the input data. `prepare_riga.py` provides a command-line entry point to the two RIGA functions. `preprocessing/SOURCE_PROVENANCE.json` records the script versions.

Obtain the original datasets and annotations from their providers under the applicable access terms, using the links in [README.md](README.md). The commands below read these data in the specified layouts.

Run from the extracted code directory after installing `.[reproduction]` as described in [REPRODUCE.md](REPRODUCE.md). This installs the Python image/cache dependencies. NPC also requires the HDF5 `h5dump` executable on `PATH`. Use new output directories and receipt filenames. Keep generated receipts with their caches.

## RIGA cup and disc

Place the original images and six corresponding contour-marked images under `data/raw/riga/`, preserving the provider's leaf directories and filenames. The expected complete groups are:

| Relative source directory | Complete image groups |
|---|---:|
| `BinRushed/BinRushed1-Corrected` | 50 |
| `BinRushed/BinRushed2` | 47 |
| `BinRushed/BinRushed3` | 47 |
| `BinRushed/BinRushed4` | 51 |
| `MESSIDOR` | 460 |

The source inventory also contains 50 incomplete groups; the converter checks this count. A complete group has an `imageNprime` image and `imageN-1` through `imageN-6` rater images with supported JPEG/TIFF extensions. `protocol/RIGA_INVENTORY.json` records the inventory used by the wrapper.

```sh
python preprocessing/prepare_riga.py \
  --raw-root data/raw/riga --mask-root data/riga_masks \
  --output-root data/riga_cache --protocol protocol
```

This converts the marked contours to binary cup/disc masks, then builds both caches. It uses the retained contour rules, including their fallback cases. The split uses a deterministic source-stratified allocation of 524 training and 131 initial development views. The wrapper compares the generated ordered split with `protocol/RIGA_SPLIT.json`.

The ROI center is the mean of the six disc-mask centroids. A 512 by 512 pixel crop is zero-padded as needed and resized to 256 by 256. Images use bilinear interpolation and division by 255; masks use nearest-neighbor interpolation. Disc annotations locate the crop for both cup and disc segmentation.

The result contains `data/riga_cache/cache/optic_cup/` and `optic_disc/`, with `train/` and `validation/` below each. Use `data/riga_cache/cache` as the reproduction CLI's `--cache-root`.

## REFUGE cup

Use the multi-rater development data in this layout:

```text
data/raw/refuge/
  REFUGE-Multirater/
    REFUGE1Train.csv
    REFUGE1Val.csv
    Training-400/<sample_id>/...
    Validation-400/<sample_id>/...
```

Each sample directory must retain its original image, spreadsheet, consensus cup/disc masks, and all seven cup and disc rater masks. The audit expects filenames such as `<sample_id>_seg_cup_1.png` through `_7.png`. It also checks the source CSVs and image/mask consistency. The preparation directory must contain no `Test-400`, `__MACOSX`, or symlinked content. The public commands cover the 400 training and 400 development images only.

First generate a source-audit receipt from the reader's actual data. Then compute the SHA-256 of that newly written file and pass it to the cache builder:

```sh
python preprocessing/refuge_multirater_dev_audit.py \
  --root data/raw/refuge --receipt data/raw/refuge/receipt.json \
  --expected-per-split 400

refuge_receipt_sha=$(python -c 'import hashlib; from pathlib import Path; print(hashlib.sha256(Path("data/raw/refuge/receipt.json").read_bytes()).hexdigest())')

python preprocessing/refuge_multirater_cup_dev_cache.py \
  --raw-root data/raw/refuge --output-root data/refuge_cache \
  --expected-raw-receipt-sha256 "$refuge_receipt_sha" \
  --expected-per-split 400 --output-size 256
```

The optional audit argument `--source-archive-sha256` can record the actual acquired archive's hash. If supplied, calculate it from the acquired archive.

REFUGE uses a full-image resize to 256 by 256, with bilinear RGB interpolation and division by 255, and nearest-neighbor binary masks. It performs no disc-centered crop. The cache contains `data/refuge_cache/cache/optic_cup/{train,validation}/`; use `data/refuge_cache/cache` as `--cache-root`.

## NPC-170

Obtain the public NPC-170/MMIS HDF5 development data in the layout expected by the upstream release:

```text
data/raw/npc/
  training/Sample_N.h5
  training_2d/Sample_N_slice_M.h5
  validation/Sample_N.h5
```

The expected data contain 100 training volumes with 2405 training slices and 20 development volumes with 487 slices. The three image keys are `t1`, `t1c`, and `t2`; annotation keys are `label_a1` through `label_a4`. Keep the original patient and slice identities. `test/` and `testing/` must be absent from this preparation directory.

Obtain the pinned D-Persona checkout described in [REPRODUCE.md](REPRODUCE.md). The example uses `external/D-Persona`. Generate a new HDF5 audit receipt with an available `h5dump` executable:

```sh
python preprocessing/npc170_h5_admission.py \
  --root data/raw/npc --output data/npc_source_receipt.json \
  --h5dump "$(command -v h5dump)" --workers 8
```

This receipt records data-structure and content checks. Check that its `summary.admission_pass` is true before building the cache. The following commands require that result, calculate the new receipt's hash, and read the data-tree hash that the audit computed:

```sh
npc_receipt_sha=$(python -c 'import hashlib, json; from pathlib import Path; p=Path("data/npc_source_receipt.json"); v=json.loads(p.read_text()); assert v["summary"]["admission_pass"] is True; print(hashlib.sha256(p.read_bytes()).hexdigest())')
npc_tree_sha=$(python -c 'import json; from pathlib import Path; print(json.loads(Path("data/npc_source_receipt.json").read_text())["summary"]["tree_sha256"])')

python preprocessing/npc170_dev_cache.py \
  --raw-root data/raw/npc --output-root data/npc_cache \
  --admission-receipt data/npc_source_receipt.json \
  --expected-admission-receipt-sha256 "$npc_receipt_sha" \
  --expected-tree-sha256 "$npc_tree_sha" \
  --official-source-root external/D-Persona --output-size 128
```

The builder checks that the receipt names the same source root, verifies each recorded HDF5 fingerprint, and checks the upstream source. Generate a fresh receipt after moving the raw tree.

NPC uses the upstream resize convention: `scipy.ndimage.zoom` with order 0 for both images and masks, producing 128 by 128 slices. Images retain their native float32 intensities; this step adds no intensity normalization, crop, or 3D resampling. Empty slices remain included. The output contains `data/npc_cache/cache/npc_gtvp/{train,validation}/`; use `data/npc_cache/cache` as `--cache-root`.

The saved patient, volume, slice, and rater identities are needed to reconstruct full development volumes. The evaluation computes metrics on each assembled patient volume and then averages across patients.

## Check the generated cache against the publication protocol

Each builder writes its own manifests, transform record, and receipt beside `cache/`. Local receipts describe the reader's acquisition and output. The released `protocol/manifests/` provide the study's ordered records and NPZ fingerprints; `protocol/SOURCE_FINGERPRINTS.json` identifies original and portable metadata copies.

Run the four `validate-cache` commands in [REPRODUCE.md](REPRODUCE.md) against the generated cache directories before training. They check the actual NPZ bytes against the published records. Resolve differences in source files, conversion rules, or library versions when fingerprints differ.

Stage-II additionally loads the task-specific Stage-I checkpoint listed in REPRODUCE.md.

## RIGA selection manifests

The preprocessing script creates the initial cache, including all 131 development views. The default `protocol/manifests/riga_*_validation.json` selects the 127 retained views for the main comparison; four views with training-source overlap are excluded. Cached arrays and training order are unchanged. `protocol_initial/` preserves the 131-view selection for the earlier experiments.
