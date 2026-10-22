#!/usr/bin/env python3
"""Build the frozen four-rater NPC-170 development cache."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import subprocess
import zipfile
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import zoom

from setsup_recal.data import validate_disjoint_manifests, validate_multirater_manifest


TASK_ID = "npc_gtvp"
DATASET_ID = "NPC-170-MMIS2024-Task1"
LICENSE_ID = "MMIS2024_TASK1_REGISTERED_ACCESS_REUSE_TERMS_UNSPECIFIED"
IMAGE_KEYS = ("t1", "t1c", "t2")
LABEL_KEYS = ("label_a1", "label_a2", "label_a3", "label_a4")
RATER_IDS = ("rater_a1", "rater_a2", "rater_a3", "rater_a4")
TRAIN_VOLUME_RE = re.compile(r"^training/Sample_(\d+)\.h5$")
VALIDATION_VOLUME_RE = re.compile(r"^validation/Sample_(\d+)\.h5$")
TRAIN_SLICE_RE = re.compile(r"^training_2d/Sample_(\d+)_slice_(\d+)\.h5$")
OFFICIAL_COMMIT = "70689229044dabc006743f439540473df1551abe"
OFFICIAL_FILES = {
    "D-Persona/code/dataloader/dataset.py": "e2badd62e0ea080c829056ba036b9711a7a2bdcdac817119541cb4e7f785bb90",
    "D-Persona/code/configs/config.py": "db98b06d58b9a252a65b331d600eb3f17704f4cca97d49c97c24539ea46d19f5",
    "D-Persona/code/evaluate_dp_npc.py": "25f0cf9eb302b0750e5c03a54d79cc6036c8b323a6b5bbebcd7b60f5fa75f928",
}


class Npc170CacheError(ValueError):
    pass


def _sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_deterministic_npz(path, *, image, masks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for name, array in (("image", image), ("masks", masks)):
            stream = io.BytesIO()
            np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, stream.getvalue(), compresslevel=6)


def patient_macro_relation_prior(mask_volumes, eps=1e-6):
    relations = []
    for masks in mask_volumes:
        masks = np.asarray(masks)
        if masks.ndim != 4 or masks.shape[0] != len(RATER_IDS):
            raise Npc170CacheError("training volume masks must have shape [4,D,H,W]")
        values = set(np.unique(masks).tolist())
        if not values.issubset({0, 1}):
            raise Npc170CacheError("training volume masks must be binary")
        flattened = masks.reshape(len(RATER_IDS), -1).astype(np.float64, copy=False)
        sizes = flattened.sum(axis=1)
        if np.any(sizes <= 0):
            raise Npc170CacheError("training volume contains an empty rater mask")
        intersection = flattened @ flattened.T
        union = sizes[:, None] + sizes[None, :] - intersection
        relations.append((intersection + eps) / (union + eps))
    if not relations:
        raise Npc170CacheError("relation prior requires training volumes")
    relation = np.mean(relations, axis=0)
    np.fill_diagonal(relation, 0.0)
    row_sums = relation.sum(axis=1, keepdims=True)
    if not np.isfinite(relation).all() or np.any(row_sums <= 0):
        raise Npc170CacheError("relation prior is non-finite or contains an empty row")
    return relation / row_sums


def _validate_official_source(official_source_root):
    if official_source_root is None:
        return {
            "verified": False,
            "root": None,
            "commit": OFFICIAL_COMMIT,
            "file_sha256": dict(OFFICIAL_FILES),
        }
    root = Path(official_source_root).resolve()
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != OFFICIAL_COMMIT:
        raise Npc170CacheError("official D-Persona commit mismatch")
    for relative, expected in OFFICIAL_FILES.items():
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise Npc170CacheError(f"official D-Persona source mismatch: {relative}")
    return {
        "verified": True,
        "root": str(root),
        "commit": commit,
        "file_sha256": dict(OFFICIAL_FILES),
    }


def _validate_source(
    *,
    raw_root,
    admission_receipt,
    expected_admission_receipt_sha256,
    expected_tree_sha256,
    expected_train_volumes,
    expected_validation_volumes,
    expected_train_slices,
):
    raw_root = Path(raw_root).resolve()
    admission_receipt = Path(admission_receipt).resolve()
    if (raw_root / "testing").exists() or (raw_root / "test").exists():
        raise Npc170CacheError("test data must remain absent")
    if _sha256(admission_receipt) != expected_admission_receipt_sha256:
        raise Npc170CacheError("admission receipt SHA-256 mismatch")
    value = json.loads(admission_receipt.read_text())
    summary = value.get("summary", {})
    expected_summary = {
        "training_volumes": int(expected_train_volumes),
        "validation_volumes": int(expected_validation_volumes),
        "training_slices": int(expected_train_slices),
    }
    if (
        value.get("schema") != "npc170_h5_development_admission/1"
        or Path(value.get("root", "")).resolve() != raw_root
        or summary.get("admission_pass") is not True
        or summary.get("record_issue_count") != 0
        or summary.get("cross_file_issue_count") != 0
        or summary.get("tree_sha256") != expected_tree_sha256
        or any(summary.get(key) != count for key, count in expected_summary.items())
    ):
        raise Npc170CacheError("admission receipt contract mismatch")

    source_records = {record["path"]: record for record in value.get("records", [])}
    expected_files = int(expected_train_volumes) + int(expected_validation_volumes) + int(
        expected_train_slices
    )
    if len(source_records) != expected_files or summary.get("files") != expected_files:
        raise Npc170CacheError("admission receipt file count mismatch")
    actual_paths = {str(path.relative_to(raw_root)) for path in raw_root.rglob("*.h5")}
    if actual_paths != set(source_records):
        raise Npc170CacheError("source file inventory mismatch")
    for relative, record in source_records.items():
        path = raw_root / relative
        if _sha256(path) != record.get("sha256"):
            raise Npc170CacheError(f"source SHA-256 mismatch: {relative}")
    return raw_root, admission_receipt, value, source_records


def _load_volume(path, *, load_images):
    with h5py.File(path, "r") as source:
        images = (
            np.stack([source[name][...] for name in IMAGE_KEYS]).astype(np.float32)
            if load_images
            else None
        )
        masks = np.stack([source[name][...] for name in LABEL_KEYS]).astype(np.uint8)
        spacing = np.asarray(source["voxel_spacing"][...], dtype=np.float64).tolist()
    return images, masks, spacing


def _load_slice(path):
    with h5py.File(path, "r") as source:
        image = np.stack([source[name][...] for name in IMAGE_KEYS]).astype(np.float32)
        masks = np.stack([source[name][...] for name in LABEL_KEYS]).astype(np.uint8)
    return image, masks


def _resize_official(image, masks, output_size):
    image = np.asarray(image, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.uint8)
    if image.ndim != 3 or masks.ndim != 3 or image.shape[-2:] != masks.shape[-2:]:
        raise Npc170CacheError("slice arrays must have shapes [3,H,W] and [4,H,W]")
    if image.shape[0] != len(IMAGE_KEYS) or masks.shape[0] != len(RATER_IDS):
        raise Npc170CacheError("slice channel or rater count mismatch")
    if not np.isfinite(image).all():
        raise Npc170CacheError("non-finite image values")
    if not set(np.unique(masks).tolist()).issubset({0, 1}):
        raise Npc170CacheError("non-binary mask values")
    height, width = image.shape[-2:]
    factors = (1, int(output_size) / height, int(output_size) / width)
    resized_image = zoom(image, factors, order=0).astype(np.float32, copy=False)
    resized_masks = zoom(masks, factors, order=0).astype(np.uint8, copy=False)
    expected_image_shape = (len(IMAGE_KEYS), int(output_size), int(output_size))
    expected_mask_shape = (len(RATER_IDS), int(output_size), int(output_size))
    if resized_image.shape != expected_image_shape or resized_masks.shape != expected_mask_shape:
        raise Npc170CacheError("official resize produced an unexpected shape")
    if not set(np.unique(resized_masks).tolist()).issubset({0, 1}):
        raise Npc170CacheError("resized masks are not binary")
    return resized_image, resized_masks


def _record_and_cache(
    *,
    partial_root,
    role,
    subject,
    slice_index,
    image,
    masks,
    spacing,
    source_sha256,
    source_path,
    output_size,
):
    source_image_shape = list(np.asarray(image).shape[-2:])
    source_empty_raters = [
        rater_id
        for rater_id, mask in zip(RATER_IDS, np.asarray(masks))
        if not bool(mask.any())
    ]
    image, masks = _resize_official(image, masks, output_size)
    cached_empty_raters = [
        rater_id
        for rater_id, mask in zip(RATER_IDS, masks)
        if not bool(mask.any())
    ]
    lost_raters = sorted(set(cached_empty_raters) - set(source_empty_raters))
    patient_id = f"NPC170:{role}:Sample_{subject}"
    sample_id = f"NPC-170/{role}/Sample_{subject}/slice_{slice_index:03d}"
    cache_name = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]
    cache_relative = Path(TASK_ID) / role / f"{cache_name}.npz"
    cache_path = partial_root / "cache" / cache_relative
    _write_deterministic_npz(cache_path, image=image, masks=masks)
    return {
        "patient_id": patient_id,
        "sample_id": sample_id,
        "volume_id": patient_id,
        "slice_index": int(slice_index),
        "split": role,
        "task_id": TASK_ID,
        "cache_file": str(cache_relative),
        "image_shape": list(image.shape),
        "mask_shape": list(masks.shape),
        "rater_ids": list(RATER_IDS),
        "spacing": [float(value) for value in spacing],
        "source_sha256": dict(source_sha256),
        "cache_sha256": _sha256(cache_path),
        "source_h5_path": source_path,
        "source_image_shape": source_image_shape,
        "spatial_transform": "full_slice_zoom_order_0",
        "all_rater_masks_empty": not bool(masks.any()),
        "any_rater_mask_empty": any(not bool(mask.any()) for mask in masks),
        "source_all_rater_masks_empty": len(source_empty_raters) == len(RATER_IDS),
        "source_any_rater_mask_empty": bool(source_empty_raters),
        "rater_masks_lost_after_resize": lost_raters,
    }


def build_npc170_development_cache(
    *,
    raw_root,
    output_root,
    admission_receipt,
    expected_admission_receipt_sha256,
    expected_tree_sha256,
    expected_train_volumes=100,
    expected_validation_volumes=20,
    expected_train_slices=2405,
    expected_validation_slices=487,
    output_size=128,
    official_source_root=None,
):
    output_root = Path(output_root).resolve()
    partial_root = output_root.with_name(output_root.name + ".partial")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite cache target: {output_root}")
    if partial_root.exists():
        raise FileExistsError(f"refusing to overwrite partial cache: {partial_root}")
    if int(output_size) < 1:
        raise Npc170CacheError("output size must be positive")

    raw_root, admission_receipt, admission, source_records = _validate_source(
        raw_root=raw_root,
        admission_receipt=admission_receipt,
        expected_admission_receipt_sha256=expected_admission_receipt_sha256,
        expected_tree_sha256=expected_tree_sha256,
        expected_train_volumes=expected_train_volumes,
        expected_validation_volumes=expected_validation_volumes,
        expected_train_slices=expected_train_slices,
    )
    official_source = _validate_official_source(official_source_root)

    training_volumes = {}
    validation_volumes = {}
    training_slices = []
    for relative, record in source_records.items():
        if match := TRAIN_VOLUME_RE.match(relative):
            training_volumes[int(match.group(1))] = (relative, record)
        elif match := VALIDATION_VOLUME_RE.match(relative):
            validation_volumes[int(match.group(1))] = (relative, record)
        elif match := TRAIN_SLICE_RE.match(relative):
            training_slices.append((int(match.group(1)), int(match.group(2)), relative, record))
    training_slices.sort()
    if (
        len(training_volumes) != int(expected_train_volumes)
        or len(validation_volumes) != int(expected_validation_volumes)
        or len(training_slices) != int(expected_train_slices)
    ):
        raise Npc170CacheError("source role count mismatch")

    partial_root.mkdir(parents=True)
    current_sample = None
    try:
        transform = {
            "schema": "npc170_official_dpersona_transform/1",
            "source_admission_receipt": str(admission_receipt),
            "source_admission_receipt_sha256": expected_admission_receipt_sha256,
            "source_tree_sha256": expected_tree_sha256,
            "official_dpersona": official_source,
            "task_id": TASK_ID,
            "image_channels": list(IMAGE_KEYS),
            "rater_ids": list(RATER_IDS),
            "output_size": int(output_size),
            "spatial_transform": "full_slice_resize_no_crop_no_3d_resampling",
            "image_resize": "scipy_ndimage_zoom_order_0",
            "mask_resize": "scipy_ndimage_zoom_order_0",
            "normalization": "none_archive_float32_preserved",
            "image_dtype": "float32",
            "mask_dtype": "uint8",
            "empty_slice_policy": "retain_all",
            "validation_representation": "all_volume_slices_in_original_order",
            "spacing_role": "archive_native_metadata_not_used_for_ged_or_dice",
            "spacing_axis_semantics": "release_unspecified",
            "test_opened": False,
            "gpu_used": False,
            "training_performed": False,
        }
        transform_path = partial_root / "transform_receipt.json"
        _write_json(transform_path, transform)
        transform_sha256 = _sha256(transform_path)

        volume_masks = []
        training_metadata = {}
        for subject, (relative, record) in sorted(training_volumes.items()):
            _, masks, spacing = _load_volume(raw_root / relative, load_images=False)
            if list(masks.shape[1:]) != record["shape"]:
                raise Npc170CacheError(f"training volume shape mismatch: {relative}")
            volume_masks.append(masks)
            training_metadata[subject] = {
                "spacing": spacing,
                "volume_sha256": record["sha256"],
            }
        relation = patient_macro_relation_prior(volume_masks)

        records = {"train": [], "validation": []}
        for index, (subject, slice_index, relative, record) in enumerate(training_slices, 1):
            current_sample = f"train/{subject}/{slice_index}"
            image, masks = _load_slice(raw_root / relative)
            metadata = training_metadata.get(subject)
            if metadata is None:
                raise Npc170CacheError(f"training slice has no parent volume: {relative}")
            row = _record_and_cache(
                partial_root=partial_root,
                role="train",
                subject=subject,
                slice_index=slice_index,
                image=image,
                masks=masks,
                spacing=metadata["spacing"],
                source_sha256={
                    "slice_h5": record["sha256"],
                    "volume_h5": metadata["volume_sha256"],
                },
                source_path=relative,
                output_size=output_size,
            )
            records["train"].append(row)
            if index % 500 == 0:
                print(f"cached train {index}/{len(training_slices)}", flush=True)

        validation_count = 0
        for subject, (relative, record) in sorted(validation_volumes.items()):
            images, masks, spacing = _load_volume(raw_root / relative, load_images=True)
            if list(images.shape[1:]) != record["shape"] or masks.shape[1:] != images.shape[1:]:
                raise Npc170CacheError(f"validation volume shape mismatch: {relative}")
            for slice_index in range(images.shape[1]):
                current_sample = f"validation/{subject}/{slice_index}"
                records["validation"].append(
                    _record_and_cache(
                        partial_root=partial_root,
                        role="validation",
                        subject=subject,
                        slice_index=slice_index,
                        image=images[:, slice_index],
                        masks=masks[:, slice_index],
                        spacing=spacing,
                        source_sha256={"volume_h5": record["sha256"]},
                        source_path=relative,
                        output_size=output_size,
                    )
                )
                validation_count += 1
        if validation_count != int(expected_validation_slices):
            raise Npc170CacheError(
                f"validation slice count mismatch: {validation_count} != {expected_validation_slices}"
            )

        manifests_dir = partial_root / "manifests"
        manifests_dir.mkdir()
        manifest_paths = []
        manifest_sha256 = {}
        dataset_version = f"registered-access:{expected_tree_sha256}"
        for role in ("train", "validation"):
            manifest = {
                "schema": "multirater_seg_manifest/1",
                "dataset_id": DATASET_ID,
                "dataset_version": dataset_version,
                "license_id": LICENSE_ID,
                "task_id": TASK_ID,
                "modality": "multiparametric_mri_t1_t1c_t2",
                "split": role,
                "image_channels": len(IMAGE_KEYS),
                "cache_root": "../cache",
                "test_opened": False,
                "transform_receipt_sha256": transform_sha256,
                "records": records[role],
            }
            manifest_path = manifests_dir / f"{TASK_ID}_{role}.json"
            _write_json(manifest_path, manifest)
            validate_multirater_manifest(manifest_path)
            manifest_paths.append(manifest_path)
            manifest_sha256[role] = _sha256(manifest_path)
        validate_disjoint_manifests(manifest_paths)

        relation_value = {
            "schema": "multirater_relation_prior/1",
            "dataset_id": DATASET_ID,
            "dataset_version": dataset_version,
            "task": TASK_ID,
            "split": "train",
            "images": len(records["train"]),
            "relation_units": len(training_volumes),
            "relation_unit": "training_volume_patient_macro",
            "rater_ids": list(RATER_IDS),
            "method": "mean_volume_pairwise_soft_iou_row_stochastic_no_self",
            "row_stochastic_no_self_prior": relation.tolist(),
            "train_manifest_sha256": manifest_sha256["train"],
            "validation_opened": False,
            "test_opened": False,
            "gpu_used": False,
            "training_performed": False,
        }
        relation_path = partial_root / "relation_prior.json"
        _write_json(relation_path, relation_value)
        relation_sha256 = _sha256(relation_path)

        all_records = records["train"] + records["validation"]
        cache_tree_sha256 = hashlib.sha256(
            "".join(
                f"{row['cache_sha256']}  {row['cache_file']}\n"
                for row in sorted(all_records, key=lambda item: item["cache_file"])
            ).encode("utf-8")
        ).hexdigest()
        receipt = {
            "schema": "npc170_multirater_development_cache/1",
            "status": "COMPLETE",
            "output_root": str(output_root),
            "source_admission_receipt_sha256": expected_admission_receipt_sha256,
            "source_tree_sha256": expected_tree_sha256,
            "official_source_verified": official_source["verified"],
            "transform_receipt_sha256": transform_sha256,
            "manifest_sha256": manifest_sha256,
            "relation_prior_sha256": relation_sha256,
            "cache_tree_sha256": cache_tree_sha256,
            "split_counts": {role: len(rows) for role, rows in records.items()},
            "patient_counts": {
                role: len({row["patient_id"] for row in rows})
                for role, rows in records.items()
            },
            "empty_all_rater_slices": {
                role: sum(row["all_rater_masks_empty"] for row in rows)
                for role, rows in records.items()
            },
            "empty_any_rater_slices": {
                role: sum(row["any_rater_mask_empty"] for row in rows)
                for role, rows in records.items()
            },
            "nonempty_rater_masks_lost_after_resize": {
                role: sum(len(row["rater_masks_lost_after_resize"]) for row in rows)
                for role, rows in records.items()
            },
            "cache_bytes": sum(
                (partial_root / "cache" / row["cache_file"]).stat().st_size
                for row in all_records
            ),
            "output_size": int(output_size),
            "rater_ids": list(RATER_IDS),
            "relation_units": len(training_volumes),
            "test_opened": False,
            "server_transfer_performed": False,
            "gpu_used": False,
            "training_performed": False,
        }
        _write_json(partial_root / "receipt.json", receipt)
        partial_root.rename(output_root)
        return receipt
    except Exception as error:
        failure_path = partial_root / "failure.json"
        if not failure_path.exists():
            _write_json(
                failure_path,
                {
                    "schema": "npc170_multirater_cache_failure/1",
                    "status": "FAILED",
                    "current_sample": current_sample,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "test_opened": False,
                    "server_transfer_performed": False,
                    "gpu_used": False,
                    "training_performed": False,
                },
            )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--admission-receipt", required=True)
    parser.add_argument("--expected-admission-receipt-sha256", required=True)
    parser.add_argument("--expected-tree-sha256", required=True)
    parser.add_argument("--official-source-root")
    parser.add_argument("--output-size", type=int, default=128)
    args = parser.parse_args()
    receipt = build_npc170_development_cache(
        raw_root=args.raw_root,
        output_root=args.output_root,
        admission_receipt=args.admission_receipt,
        expected_admission_receipt_sha256=args.expected_admission_receipt_sha256,
        expected_tree_sha256=args.expected_tree_sha256,
        output_size=args.output_size,
        official_source_root=args.official_source_root,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
