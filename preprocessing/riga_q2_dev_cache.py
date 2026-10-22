#!/usr/bin/env python3
"""Build development-only RIGA disc and cup caches."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from setsup_recal.data import (
    validate_disjoint_manifests,
    validate_multirater_manifest,
)


SPLIT_SALT = "riga_q2_dev_split_v1"
TASKS = ("optic_disc", "optic_cup")
RATER_IDS = [f"rater_{index}" for index in range(1, 7)]


class RigaDevelopmentCacheError(ValueError):
    pass


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def assign_development_split(
    groups,
    *,
    validation_fraction: float = 0.2,
    salt: str = SPLIT_SALT,
) -> dict[tuple[str, str], str]:
    keys = [tuple(group) for group in groups]
    if not keys or len(set(keys)) != len(keys):
        raise RigaDevelopmentCacheError("split groups must be non-empty and unique")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise RigaDevelopmentCacheError("validation fraction must lie between zero and one")

    by_leaf = defaultdict(list)
    for leaf, image_id in keys:
        if not leaf or not image_id:
            raise RigaDevelopmentCacheError("split group identifiers must be non-empty")
        by_leaf[leaf].append((leaf, image_id))

    validation_target = int(math.floor(len(keys) * validation_fraction + 0.5))
    if validation_target < 1 or validation_target >= len(keys):
        raise RigaDevelopmentCacheError("split must contain train and validation images")
    allocations = {
        leaf: int(math.floor(len(values) * validation_fraction))
        for leaf, values in by_leaf.items()
    }
    remaining = validation_target - sum(allocations.values())
    ranked_leaves = sorted(
        by_leaf,
        key=lambda leaf: (
            -(
                len(by_leaf[leaf]) * validation_fraction
                - math.floor(len(by_leaf[leaf]) * validation_fraction)
            ),
            leaf,
        ),
    )
    for leaf in ranked_leaves[:remaining]:
        allocations[leaf] += 1

    assignments = {}
    for leaf in sorted(by_leaf):
        ranked_groups = sorted(
            by_leaf[leaf],
            key=lambda item: hashlib.sha256(
                f"{salt}|{item[0]}/{item[1]}".encode("utf-8")
            ).hexdigest(),
        )
        validation_keys = set(ranked_groups[: allocations[leaf]])
        for key in ranked_groups:
            assignments[key] = "validation" if key in validation_keys else "train"
    return assignments


def transform_roi_sample(
    image,
    disc_masks,
    cup_masks,
    *,
    roi_size: int = 512,
    output_size: int = 256,
) -> dict:
    image = np.asarray(image)
    disc_masks = np.asarray(disc_masks)
    cup_masks = np.asarray(cup_masks)
    if image.ndim != 3 or image.shape[2] != 3:
        raise RigaDevelopmentCacheError("RIGA image must be HWC RGB")
    if (
        disc_masks.ndim != 3
        or cup_masks.shape != disc_masks.shape
        or disc_masks.shape[0] != 6
        or disc_masks.shape[1:] != image.shape[:2]
    ):
        raise RigaDevelopmentCacheError("RIGA masks must be six aligned disc/cup masks")
    if roi_size < 1 or output_size < 1:
        raise RigaDevelopmentCacheError("ROI and output sizes must be positive")

    disc_masks = disc_masks.astype(bool)
    cup_masks = cup_masks.astype(bool)
    if any(not mask.any() for mask in disc_masks) or any(
        not mask.any() for mask in cup_masks
    ):
        raise RigaDevelopmentCacheError("RIGA disc and cup masks must be non-empty")
    if np.any(cup_masks & ~disc_masks):
        raise RigaDevelopmentCacheError("RIGA cup must remain inside disc")

    centroids = []
    for mask in disc_masks:
        yy, xx = np.where(mask)
        centroids.append((float(xx.mean()), float(yy.mean())))
    center_x = int(math.floor(np.mean([item[0] for item in centroids]) + 0.5))
    center_y = int(math.floor(np.mean([item[1] for item in centroids]) + 0.5))
    half = roi_size // 2
    left = center_x - half
    top = center_y - half
    right = left + roi_size
    bottom = top + roi_size
    height, width = image.shape[:2]
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(width, right)
    source_bottom = min(height, bottom)
    destination_left = source_left - left
    destination_top = source_top - top

    image_roi = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
    image_roi[
        destination_top : destination_top + source_bottom - source_top,
        destination_left : destination_left + source_right - source_left,
    ] = image[source_top:source_bottom, source_left:source_right]

    def crop_masks(values):
        cropped = np.zeros((6, roi_size, roi_size), dtype=np.uint8)
        cropped[
            :,
            destination_top : destination_top + source_bottom - source_top,
            destination_left : destination_left + source_right - source_left,
        ] = values[:, source_top:source_bottom, source_left:source_right]
        return np.stack(
            [
                np.asarray(
                    Image.fromarray(mask).resize(
                        (output_size, output_size), Image.Resampling.NEAREST
                    ),
                    dtype=np.uint8,
                )
                for mask in cropped
            ]
        )

    resized_image = np.asarray(
        Image.fromarray(image_roi).resize(
            (output_size, output_size), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    resized_image = (resized_image.transpose(2, 0, 1) / 255.0).astype(
        np.float32, copy=False
    )
    resized_discs = crop_masks(disc_masks.astype(np.uint8))
    resized_cups = crop_masks(cup_masks.astype(np.uint8))
    if not np.isfinite(resized_image).all():
        raise RigaDevelopmentCacheError("transformed RIGA image is non-finite")
    if any(not mask.any() for mask in resized_discs) or any(
        not mask.any() for mask in resized_cups
    ):
        raise RigaDevelopmentCacheError("transformed RIGA masks became empty")
    if np.any(resized_cups.astype(bool) & ~resized_discs.astype(bool)):
        raise RigaDevelopmentCacheError("transformed RIGA cup left disc")

    return {
        "image": resized_image,
        "disc_masks": resized_discs,
        "cup_masks": resized_cups,
        "roi_center_xy": [center_x, center_y],
        "roi_box_xyxy": [left, top, right, bottom],
        "padding_ltrb": [
            max(0, -left),
            max(0, -top),
            max(0, right - width),
            max(0, bottom - height),
        ],
    }


def _load_mask_groups(mask_root: Path, expected_groups_by_leaf: dict[str, int]):
    receipt_path = mask_root / "receipt.json"
    manifest_path = mask_root / "manifest.jsonl"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "COMPLETE":
        raise RigaDevelopmentCacheError("mask package is not complete")
    for field in (
        "magrabia_opened",
        "obsolete_binrushed_opened",
        "test_cache_created",
        "gpu_used",
    ):
        if receipt.get(field) is not False:
            raise RigaDevelopmentCacheError(f"mask package violates sealed state: {field}")
    manifest_sha256 = _sha256(manifest_path)
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise RigaDevelopmentCacheError("mask manifest SHA-256 mismatch")

    groups = defaultdict(list)
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), 1):
        row = json.loads(line)
        if row.get("schema") != "riga_q2_mask_record/1":
            raise RigaDevelopmentCacheError(
                f"unexpected mask record schema at line {line_number}"
            )
        sealed_text = " ".join(
            str(row.get(field, ""))
            for field in ("leaf", "prime_path", "disc_path", "cup_path")
        ).lower()
        if "magrabia" in sealed_text:
            raise RigaDevelopmentCacheError("sealed Magrabia reference is prohibited")
        groups[(row["leaf"], row["image_id"])].append(row)

    actual_groups_by_leaf = dict(Counter(key[0] for key in groups))
    if actual_groups_by_leaf != dict(expected_groups_by_leaf):
        raise RigaDevelopmentCacheError(
            f"development group inventory mismatch: {actual_groups_by_leaf}"
        )
    for key, rows in groups.items():
        rows.sort(key=lambda row: row["rater_id"])
        if [row["rater_id"] for row in rows] != list(range(1, 7)):
            raise RigaDevelopmentCacheError(f"group does not have six raters: {key}")
        if len({row["prime_path"] for row in rows}) != 1 or len(
            {row["prime_sha256"] for row in rows}
        ) != 1:
            raise RigaDevelopmentCacheError(f"group source image is inconsistent: {key}")
    if receipt.get("complete_groups") != len(groups):
        raise RigaDevelopmentCacheError("mask receipt group count mismatch")
    return groups, manifest_sha256


def build_riga_development_cache(
    *,
    raw_root,
    mask_root,
    output_root,
    expected_groups_by_leaf: dict[str, int],
    validation_fraction: float = 0.2,
    roi_size: int = 512,
    output_size: int = 256,
) -> dict:
    raw_root = Path(raw_root).resolve()
    mask_root = Path(mask_root).resolve()
    output_root = Path(output_root).resolve()
    partial_root = output_root.with_name(output_root.name + ".partial")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite cache target: {output_root}")
    if partial_root.exists():
        raise FileExistsError(f"refusing to overwrite partial cache: {partial_root}")

    groups, source_manifest_sha256 = _load_mask_groups(
        mask_root, expected_groups_by_leaf
    )
    assignments = assign_development_split(
        groups, validation_fraction=validation_fraction
    )
    split_rows = [
        {
            "leaf": leaf,
            "image_id": image_id,
            "sample_id": f"{leaf}/{image_id}",
            "split": assignments[(leaf, image_id)],
        }
        for leaf, image_id in sorted(groups)
    ]
    split_counts = dict(Counter(row["split"] for row in split_rows))
    validation_by_leaf = dict(
        Counter(row["leaf"] for row in split_rows if row["split"] == "validation")
    )

    partial_root.mkdir(parents=True)
    current_group = None
    try:
        split_receipt = {
            "schema": "riga_q2_development_split/1",
            "salt": SPLIT_SALT,
            "validation_fraction": validation_fraction,
            "allocation": "source_leaf_stratified_largest_remainder",
            "ranking": "ascending_sha256_of_salt_leaf_image_id",
            "content_blind": True,
            "split_counts": split_counts,
            "validation_by_leaf": validation_by_leaf,
            "records": split_rows,
            "test_opened": False,
        }
        split_path = partial_root / "split.json"
        split_path.write_text(json.dumps(split_receipt, indent=2, sort_keys=True) + "\n")
        split_sha256 = _sha256(split_path)
        transform_receipt = {
            "schema": "riga_q2_development_transform/1",
            "source_mask_manifest_sha256": source_manifest_sha256,
            "split_sha256": split_sha256,
            "roi_center_source": "mean_of_six_disc_centroids",
            "roi_size": roi_size,
            "output_size": output_size,
            "outside_padding": "constant_zero",
            "image_resize": "bilinear",
            "mask_resize": "nearest",
            "normalization": "divide_by_255",
            "image_dtype": "float32",
            "mask_dtype": "uint8",
            "rater_ids": list(RATER_IDS),
            "tasks": list(TASKS),
            "roi_conditioned": True,
            "test_opened": False,
        }
        transform_path = partial_root / "transform_receipt.json"
        transform_path.write_text(
            json.dumps(transform_receipt, indent=2, sort_keys=True) + "\n"
        )
        transform_sha256 = _sha256(transform_path)

        records = {
            (task, split): []
            for task in TASKS
            for split in ("train", "validation")
        }
        cache_bytes = 0
        for leaf, image_id in sorted(groups):
            current_group = f"{leaf}/{image_id}"
            rows = groups[(leaf, image_id)]
            prime_relative = Path(rows[0]["prime_path"])
            prime_path = (raw_root / prime_relative).resolve()
            if not prime_path.is_relative_to(raw_root):
                raise RigaDevelopmentCacheError("source image escapes raw root")
            if _sha256(prime_path) != rows[0]["prime_sha256"]:
                raise RigaDevelopmentCacheError(
                    f"image SHA-256 mismatch: {current_group}"
                )
            with Image.open(prime_path) as source_image:
                image = np.asarray(source_image.convert("RGB"), dtype=np.uint8)

            disc_masks = []
            cup_masks = []
            for row in rows:
                for field, hash_field, destination in (
                    ("disc_path", "disc_sha256", disc_masks),
                    ("cup_path", "cup_sha256", cup_masks),
                ):
                    path = (mask_root / row[field]).resolve()
                    if not path.is_relative_to(mask_root):
                        raise RigaDevelopmentCacheError("mask escapes mask root")
                    if _sha256(path) != row[hash_field]:
                        raise RigaDevelopmentCacheError(
                            f"mask SHA-256 mismatch: {current_group}/{field}"
                        )
                    with Image.open(path) as mask_image:
                        value = np.asarray(mask_image, dtype=np.uint8)
                    if not set(np.unique(value).tolist()).issubset({0, 255}):
                        raise RigaDevelopmentCacheError(
                            f"input mask is not binary: {current_group}/{field}"
                        )
                    destination.append((value > 0).astype(np.uint8))
            transformed = transform_roi_sample(
                image,
                np.stack(disc_masks),
                np.stack(cup_masks),
                roi_size=roi_size,
                output_size=output_size,
            )
            split = assignments[(leaf, image_id)]
            sample_id = f"{leaf}/{image_id}"
            patient_id = f"image_surrogate:{sample_id}"
            cache_name = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]
            for task, mask_key in (
                ("optic_disc", "disc_masks"),
                ("optic_cup", "cup_masks"),
            ):
                cache_relative = Path(task) / split / f"{cache_name}.npz"
                cache_path = partial_root / "cache" / cache_relative
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    image=transformed["image"],
                    masks=transformed[mask_key],
                )
                cache_sha256 = _sha256(cache_path)
                cache_bytes += cache_path.stat().st_size
                source_hashes = {
                    "image": rows[0]["prime_sha256"],
                    **{
                        f"disc_rater_{row['rater_id']}": row["disc_sha256"]
                        for row in rows
                    },
                }
                if task == "optic_cup":
                    source_hashes.update(
                        {
                            f"cup_rater_{row['rater_id']}": row["cup_sha256"]
                            for row in rows
                        }
                    )
                records[(task, split)].append(
                    {
                        "patient_id": patient_id,
                        "patient_id_role": "image_surrogate_no_patient_metadata",
                        "sample_id": sample_id,
                        "split": split,
                        "task_id": task,
                        "cache_file": str(cache_relative),
                        "image_shape": [3, output_size, output_size],
                        "mask_shape": [6, output_size, output_size],
                        "rater_ids": list(RATER_IDS),
                        "spacing": None,
                        "source_sha256": source_hashes,
                        "cache_sha256": cache_sha256,
                        "source_leaf": leaf,
                        "source_image_path": str(prime_relative),
                        "roi_center_xy": transformed["roi_center_xy"],
                        "roi_box_xyxy": transformed["roi_box_xyxy"],
                        "padding_ltrb": transformed["padding_ltrb"],
                    }
                )

        manifests_dir = partial_root / "manifests"
        manifests_dir.mkdir()
        manifest_hashes = {}
        for task in TASKS:
            paths = []
            for split in ("train", "validation"):
                manifest = {
                    "schema": "multirater_seg_manifest/1",
                    "dataset_id": "RIGA",
                    "dataset_version": f"deepblue-mask-v1:{source_manifest_sha256}",
                    "license_id": "CC-BY-NC-4.0",
                    "task_id": task,
                    "modality": "fundus_photography",
                    "split": split,
                    "image_channels": 3,
                    "cache_root": "../cache",
                    "test_opened": False,
                    "transform_receipt_sha256": transform_sha256,
                    "records": records[(task, split)],
                }
                manifest_path = manifests_dir / f"{task}_{split}.json"
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )
                validate_multirater_manifest(manifest_path)
                paths.append(manifest_path)
                manifest_hashes[f"{task}_{split}"] = _sha256(manifest_path)
            validate_disjoint_manifests(paths)

        receipt = {
            "schema": "riga_q2_development_cache/1",
            "status": "COMPLETE",
            "output_root": str(output_root),
            "source_mask_manifest_sha256": source_manifest_sha256,
            "split_sha256": split_sha256,
            "transform_receipt_sha256": transform_sha256,
            "manifest_sha256": manifest_hashes,
            "split_counts": split_counts,
            "validation_by_leaf": validation_by_leaf,
            "task_cache_files": {
                task: sum(len(records[(task, split)]) for split in ("train", "validation"))
                for task in TASKS
            },
            "cache_bytes": cache_bytes,
            "development_split_created": True,
            "final_split_created": False,
            "test_opened": False,
            "test_cache_created": False,
            "magrabia_opened": False,
            "obsolete_binrushed_opened": False,
            "server_transfer_performed": False,
            "gpu_used": False,
            "training_performed": False,
        }
        receipt_path = partial_root / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        partial_root.rename(output_root)
        return receipt
    except Exception as error:
        failure_path = partial_root / "failure.json"
        if not failure_path.exists():
            failure_path.write_text(
                json.dumps(
                    {
                        "schema": "riga_q2_development_cache_failure/1",
                        "status": "FAILED",
                        "current_group": current_group,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "test_opened": False,
                        "magrabia_opened": False,
                        "gpu_used": False,
                        "training_performed": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        raise
