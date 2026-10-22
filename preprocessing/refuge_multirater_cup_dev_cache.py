#!/usr/bin/env python3
"""Build the frozen seven-rater REFUGE optic-cup development cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from setsup_recal.data import (
    validate_disjoint_manifests,
    validate_multirater_manifest,
)
from refuge_multirater_dev_audit import audit_refuge_development


SPLITS = {"Training-400": "train", "Validation-400": "validation"}
RATER_IDS = [f"rater_{index}" for index in range(1, 8)]
TASK_ID = "optic_cup"


class RefugeCupCacheError(ValueError):
    pass


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _validate_source(raw_root, expected_raw_receipt_sha256, expected_per_split):
    raw_root = Path(raw_root).resolve()
    dataset_root = raw_root / "REFUGE-Multirater"
    if (dataset_root / "Test-400").exists():
        raise RefugeCupCacheError("Test-400 must remain inaccessible")
    receipt_path = raw_root / "receipt.json"
    if not receipt_path.is_file():
        raise RefugeCupCacheError("source audit receipt is missing")
    actual_receipt_sha256 = _sha256(receipt_path)
    if actual_receipt_sha256 != expected_raw_receipt_sha256:
        raise RefugeCupCacheError("source audit receipt SHA-256 mismatch")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != "refuge_multirater_development_audit/1"
        or receipt.get("status") != "PASS"
        or receipt.get("test_opened") is not False
        or receipt.get("cache_created") is not False
        or receipt.get("split_counts")
        != {"train": int(expected_per_split), "validation": int(expected_per_split)}
    ):
        raise RefugeCupCacheError("source audit receipt contract mismatch")
    try:
        fresh = audit_refuge_development(
            raw_root,
            expected_per_split=expected_per_split,
            source_archive_sha256=receipt.get("source_archive_sha256"),
        )
    except Exception as error:
        raise RefugeCupCacheError(f"source audit failed: {error}") from error
    if fresh["content_ledger_sha256"] != receipt.get("content_ledger_sha256"):
        raise RefugeCupCacheError("source audit content ledger mismatch")
    return raw_root, dataset_root, receipt, actual_receipt_sha256


def _resize_sample(image_path, mask_paths, output_size):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        source_size = list(image.size[::-1])
        resized_image = np.asarray(
            image.resize(
                (output_size, output_size),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )
    resized_image = (resized_image.transpose(2, 0, 1) / 255.0).astype(
        np.float32, copy=False
    )
    masks = []
    for path in mask_paths:
        with Image.open(path) as source:
            mask = np.asarray(source.convert("L"), dtype=np.uint8)
            values = set(np.unique(mask).tolist())
            if not values.issubset({0, 255}) or not np.any(mask):
                raise RefugeCupCacheError(f"invalid source cup mask: {path}")
            resized = np.asarray(
                source.convert("L").resize(
                    (output_size, output_size),
                    Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
        resized = (resized > 0).astype(np.uint8)
        if not np.any(resized):
            raise RefugeCupCacheError(f"resized cup mask became empty: {path}")
        masks.append(resized)
    masks = np.stack(masks)
    if not np.isfinite(resized_image).all():
        raise RefugeCupCacheError(f"resized image is non-finite: {image_path}")
    return resized_image, masks, source_size


def build_refuge_cup_development_cache(
    *,
    raw_root,
    output_root,
    expected_raw_receipt_sha256,
    expected_per_split=400,
    output_size=256,
):
    output_root = Path(output_root).resolve()
    partial_root = output_root.with_name(output_root.name + ".partial")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite cache target: {output_root}")
    if partial_root.exists():
        raise FileExistsError(f"refusing to overwrite partial cache: {partial_root}")
    if int(output_size) < 1:
        raise RefugeCupCacheError("output size must be positive")

    raw_root, dataset_root, source_receipt, source_receipt_sha256 = _validate_source(
        raw_root,
        expected_raw_receipt_sha256,
        expected_per_split,
    )
    partial_root.mkdir(parents=True)
    current_sample = None
    try:
        transform_receipt = {
            "schema": "refuge_multirater_cup_transform/1",
            "source_audit_receipt_sha256": source_receipt_sha256,
            "source_content_ledger_sha256": source_receipt["content_ledger_sha256"],
            "spatial_transform": "full_image_resize_no_crop",
            "output_size": int(output_size),
            "image_resize": "bilinear",
            "mask_resize": "nearest",
            "normalization": "divide_by_255",
            "image_dtype": "float32",
            "mask_dtype": "uint8",
            "task_id": TASK_ID,
            "rater_ids": list(RATER_IDS),
            "split_source": "official_archive_directories",
            "test_opened": False,
            "training_performed": False,
        }
        transform_path = partial_root / "transform_receipt.json"
        _write_json(transform_path, transform_receipt)
        transform_sha256 = _sha256(transform_path)
        records = {role: [] for role in SPLITS.values()}
        cache_bytes = 0

        for split_directory, role in SPLITS.items():
            split_root = dataset_root / split_directory
            samples = sorted(path for path in split_root.iterdir() if path.is_dir())
            if len(samples) != int(expected_per_split):
                raise RefugeCupCacheError(f"unexpected {role} sample count")
            for sample_root in samples:
                sample_id = sample_root.name
                current_sample = f"{role}/{sample_id}"
                image_path = sample_root / f"{sample_id}.jpg"
                mask_paths = [
                    sample_root / f"{sample_id}_seg_cup_{index}.png"
                    for index in range(1, 8)
                ]
                if not image_path.is_file() or any(not path.is_file() for path in mask_paths):
                    raise RefugeCupCacheError(f"source files missing: {current_sample}")
                image, masks, source_image_shape = _resize_sample(
                    image_path,
                    mask_paths,
                    int(output_size),
                )
                logical_sample_id = f"REFUGE-MultiRater/{split_directory}/{sample_id}"
                cache_name = hashlib.sha256(
                    logical_sample_id.encode("utf-8")
                ).hexdigest()[:24]
                cache_relative = Path(TASK_ID) / role / f"{cache_name}.npz"
                cache_path = partial_root / "cache" / cache_relative
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, image=image, masks=masks)
                cache_sha256 = _sha256(cache_path)
                cache_bytes += cache_path.stat().st_size
                source_hashes = {"image": _sha256(image_path)}
                source_hashes.update(
                    {
                        f"cup_rater_{index}": _sha256(mask_path)
                        for index, mask_path in enumerate(mask_paths, 1)
                    }
                )
                records[role].append(
                    {
                        "patient_id": f"image_surrogate:{logical_sample_id}",
                        "patient_id_role": "image_surrogate_no_patient_metadata",
                        "sample_id": logical_sample_id,
                        "split": role,
                        "task_id": TASK_ID,
                        "cache_file": str(cache_relative),
                        "image_shape": [3, int(output_size), int(output_size)],
                        "mask_shape": [7, int(output_size), int(output_size)],
                        "rater_ids": list(RATER_IDS),
                        "spacing": None,
                        "source_sha256": source_hashes,
                        "cache_sha256": cache_sha256,
                        "source_image_path": str(image_path.relative_to(raw_root)),
                        "source_image_shape": source_image_shape,
                        "spatial_transform": "full_image_resize_no_crop",
                    }
                )

        manifests_dir = partial_root / "manifests"
        manifests_dir.mkdir()
        manifest_hashes = {}
        manifest_paths = []
        for role in ("train", "validation"):
            manifest = {
                "schema": "multirater_seg_manifest/1",
                "dataset_id": "REFUGE-MultiRater",
                "dataset_version": (
                    f"hf-f0222b49:{source_receipt['source_archive_sha256']}:"
                    f"{source_receipt['content_ledger_sha256']}"
                ),
                "license_id": (
                    "PROVENANCE-CONFLICT_REFUGE-CC-BY-NC-ND_"
                    "vs_MIRROR-APACHE-2.0"
                ),
                "task_id": TASK_ID,
                "modality": "fundus_photography",
                "split": role,
                "image_channels": 3,
                "cache_root": "../cache",
                "test_opened": False,
                "transform_receipt_sha256": transform_sha256,
                "records": records[role],
            }
            manifest_path = manifests_dir / f"{TASK_ID}_{role}.json"
            _write_json(manifest_path, manifest)
            validate_multirater_manifest(manifest_path)
            manifest_paths.append(manifest_path)
            manifest_hashes[role] = _sha256(manifest_path)
        validate_disjoint_manifests(manifest_paths)

        receipt = {
            "schema": "refuge_multirater_cup_development_cache/1",
            "status": "COMPLETE",
            "output_root": str(output_root),
            "source_audit_receipt_sha256": source_receipt_sha256,
            "source_content_ledger_sha256": source_receipt[
                "content_ledger_sha256"
            ],
            "source_archive_sha256": source_receipt["source_archive_sha256"],
            "transform_receipt_sha256": transform_sha256,
            "manifest_sha256": manifest_hashes,
            "split_counts": {role: len(rows) for role, rows in records.items()},
            "task_cache_files": {TASK_ID: sum(len(rows) for rows in records.values())},
            "cache_bytes": cache_bytes,
            "rater_ids": list(RATER_IDS),
            "output_size": int(output_size),
            "test_opened": False,
            "cache_created": True,
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
                    "schema": "refuge_multirater_cup_cache_failure/1",
                    "status": "FAILED",
                    "current_sample": current_sample,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "test_opened": False,
                    "gpu_used": False,
                    "training_performed": False,
                },
            )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-raw-receipt-sha256", required=True)
    parser.add_argument("--expected-per-split", type=int, default=400)
    parser.add_argument("--output-size", type=int, default=256)
    args = parser.parse_args()
    receipt = build_refuge_cup_development_cache(
        raw_root=args.raw_root,
        output_root=args.output_root,
        expected_raw_receipt_sha256=args.expected_raw_receipt_sha256,
        expected_per_split=args.expected_per_split,
        output_size=args.output_size,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
