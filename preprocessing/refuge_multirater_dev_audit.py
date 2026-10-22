#!/usr/bin/env python3
"""Audit extracted REFUGE-MultiRater training and validation data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = {"Training-400": "train", "Validation-400": "validation"}
RATER_IDS = [f"rater_{index}" for index in range(1, 8)]
SAMPLE_PATTERN = re.compile(r"^\d{4}$")


class RefugeDevelopmentAuditError(ValueError):
    pass


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _add_to_ledger(digest, root: Path, path: Path) -> None:
    relative = str(path.relative_to(root))
    file_sha256 = _sha256(path)
    digest.update(relative.encode("utf-8"))
    digest.update(str(path.stat().st_size).encode("ascii"))
    digest.update(file_sha256.encode("ascii"))


def _read_mask(path: Path, expected_shape, counters, ledger, root):
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
    shape = tuple(mask.shape)
    counters["mask_shapes"][f"{shape[0]}x{shape[1]}"] += 1
    if shape != expected_shape:
        counters["dimension_mismatches"] += 1
        raise RefugeDevelopmentAuditError(f"mask dimension mismatch: {path}")
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 255}):
        counters["nonbinary_masks"] += 1
        raise RefugeDevelopmentAuditError(f"mask is not binary: {path}")
    if not np.any(mask):
        counters["empty_masks"] += 1
        raise RefugeDevelopmentAuditError(f"mask is empty: {path}")
    _add_to_ledger(ledger, root, path)
    return mask > 0


def _read_consensus_mask(
    path: Path,
    expected_shape,
    *,
    allowed_values,
    foreground_value,
    counters,
    ledger,
    root,
):
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
    shape = tuple(mask.shape)
    counters["mask_shapes"][f"{shape[0]}x{shape[1]}"] += 1
    if shape != expected_shape:
        counters["dimension_mismatches"] += 1
        raise RefugeDevelopmentAuditError(
            f"consensus mask dimension mismatch: {path}"
        )
    values = set(np.unique(mask).tolist())
    if not values.issubset(set(allowed_values)):
        raise RefugeDevelopmentAuditError(
            f"unexpected consensus encoding: {path} values={sorted(values)}"
        )
    if not np.any(mask == foreground_value):
        counters["empty_masks"] += 1
        raise RefugeDevelopmentAuditError(f"consensus mask is empty: {path}")
    _add_to_ledger(ledger, root, path)
    return mask == foreground_value


def _audit_csv(dataset_root: Path, root: Path, ledger) -> dict:
    results = {}
    portable = True
    for name in ("REFUGE1Train.csv", "REFUGE1Val.csv"):
        path = dataset_root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RefugeDevelopmentAuditError(f"development CSV missing: {name}")
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
            fields = list(reader.fieldnames or [])
        if not rows:
            raise RefugeDevelopmentAuditError(f"development CSV is empty: {name}")
        path_columns = [
            field for field in ("imgName", "maskName", "multimaskName") if field in fields
        ]
        references = [
            row.get(field, "").strip()
            for row in rows
            for field in path_columns
            if row.get(field, "").strip()
        ]
        if not references or not all((dataset_root / value).is_file() for value in references):
            portable = False
        results[name] = {
            "bytes": path.stat().st_size,
            "columns": fields,
            "rows": len(rows),
            "sha256": _sha256(path),
        }
        _add_to_ledger(ledger, root, path)
    results["portable_loader_manifest"] = portable
    results["role"] = "source_metadata_only" if not portable else "portable_manifest"
    return results


def audit_refuge_development(
    root,
    *,
    expected_per_split=400,
    source_archive_sha256=None,
):
    root = Path(root).resolve()
    dataset_root = root / "REFUGE-Multirater"
    if not dataset_root.is_dir():
        raise RefugeDevelopmentAuditError("REFUGE-Multirater root is missing")
    if (dataset_root / "Test-400").exists():
        raise RefugeDevelopmentAuditError("Test-400 must remain unopened")
    if any(path.name == "__MACOSX" for path in root.rglob("__MACOSX")):
        raise RefugeDevelopmentAuditError("__MACOSX content must not be extracted")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise RefugeDevelopmentAuditError("development tree contains a symlink")
    if source_archive_sha256 is not None and (
        not isinstance(source_archive_sha256, str)
        or len(source_archive_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_archive_sha256)
    ):
        raise RefugeDevelopmentAuditError("source archive SHA-256 is invalid")

    ledger = hashlib.sha256()
    counters = {
        "image_shapes": Counter(),
        "mask_shapes": Counter(),
        "empty_masks": 0,
        "nonbinary_masks": 0,
        "dimension_mismatches": 0,
        "cup_outside_disc_pixels": 0,
        "cup_outside_disc_pairs": 0,
    }
    split_counts = {}
    task_counts = Counter()
    split_sample_ids = {}
    auxiliary_files = []

    for split_directory, role in SPLITS.items():
        split_root = dataset_root / split_directory
        if not split_root.is_dir():
            raise RefugeDevelopmentAuditError(f"split missing: {split_directory}")
        samples = sorted(
            path
            for path in split_root.iterdir()
            if path.is_dir() and SAMPLE_PATTERN.fullmatch(path.name)
        )
        if len(samples) != int(expected_per_split):
            raise RefugeDevelopmentAuditError(
                f"unexpected {role} sample count: {len(samples)}"
            )
        unexpected_directories = sorted(
            path.name for path in split_root.iterdir() if path.is_dir() and path not in samples
        )
        if unexpected_directories:
            raise RefugeDevelopmentAuditError(
                f"unexpected directories in {role}: {unexpected_directories}"
            )
        split_counts[role] = len(samples)
        split_sample_ids[role] = {path.name for path in samples}

        for sample_root in samples:
            sample_id = sample_root.name
            images = [
                path
                for path in (sample_root / f"{sample_id}.jpg", sample_root / f"{sample_id}.png")
                if path.is_file()
            ]
            required = {
                sample_root / f"{sample_id}.xlsx",
                sample_root / f"{sample_id}_cup.bmp",
                sample_root / f"{sample_id}_disc.bmp",
                *(
                    sample_root / f"{sample_id}_seg_cup_{index}.png"
                    for index in range(1, 8)
                ),
                *(
                    sample_root / f"{sample_id}_seg_disc_{index}.png"
                    for index in range(1, 8)
                ),
            }
            missing = sorted(str(path.name) for path in required if not path.is_file())
            if len(images) != 1 or missing:
                raise RefugeDevelopmentAuditError(
                    f"required files missing for {role}/{sample_id}: images={len(images)}, missing={missing}"
                )
            known = required | set(images)
            extras = sorted(path.name for path in sample_root.iterdir() if path.is_file() and path not in known)
            if extras:
                auxiliary_files.append(
                    {"split": role, "sample_id": sample_id, "files": extras}
                )

            with Image.open(images[0]) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            image_shape = tuple(rgb.shape[:2])
            counters["image_shapes"][f"{image_shape[0]}x{image_shape[1]}"] += 1
            _add_to_ledger(ledger, root, images[0])
            task_counts["images"] += 1

            spreadsheet = sample_root / f"{sample_id}.xlsx"
            if spreadsheet.stat().st_size == 0:
                raise RefugeDevelopmentAuditError(f"spreadsheet is empty: {spreadsheet}")
            _add_to_ledger(ledger, root, spreadsheet)
            task_counts["spreadsheets"] += 1

            _read_consensus_mask(
                sample_root / f"{sample_id}_cup.bmp",
                image_shape,
                allowed_values={0, 255},
                foreground_value=0,
                counters=counters,
                ledger=ledger,
                root=root,
            )
            _read_consensus_mask(
                sample_root / f"{sample_id}_disc.bmp",
                image_shape,
                allowed_values={128, 255},
                foreground_value=128,
                counters=counters,
                ledger=ledger,
                root=root,
            )
            task_counts["cup_consensus"] += 1
            task_counts["disc_consensus"] += 1

            for index in range(1, 8):
                cup = _read_mask(
                    sample_root / f"{sample_id}_seg_cup_{index}.png",
                    image_shape,
                    counters,
                    ledger,
                    root,
                )
                disc = _read_mask(
                    sample_root / f"{sample_id}_seg_disc_{index}.png",
                    image_shape,
                    counters,
                    ledger,
                    root,
                )
                outside = int(np.count_nonzero(cup & ~disc))
                counters["cup_outside_disc_pixels"] += outside
                counters["cup_outside_disc_pairs"] += int(outside > 0)
                task_counts["cup_rater"] += 1
                task_counts["disc_rater"] += 1

    if split_sample_ids["train"] & split_sample_ids["validation"]:
        raise RefugeDevelopmentAuditError("train and validation sample IDs overlap")
    csv_receipt = _audit_csv(dataset_root, root, ledger)
    return {
        "schema": "refuge_multirater_development_audit/1",
        "status": "PASS",
        "root": str(root),
        "source_archive_sha256": source_archive_sha256,
        "split_counts": split_counts,
        "rater_ids": list(RATER_IDS),
        "task_file_counts": dict(sorted(task_counts.items())),
        "image_shapes": dict(sorted(counters["image_shapes"].items())),
        "mask_shapes": dict(sorted(counters["mask_shapes"].items())),
        "empty_masks": counters["empty_masks"],
        "nonbinary_masks": counters["nonbinary_masks"],
        "dimension_mismatches": counters["dimension_mismatches"],
        "consensus_encodings": {
            "cup": {"allowed_values": [0, 255], "foreground_value": 0},
            "disc": {"allowed_values": [128, 255], "foreground_value": 128},
        },
        "cup_outside_disc_pixels": counters["cup_outside_disc_pixels"],
        "cup_outside_disc_pairs": counters["cup_outside_disc_pairs"],
        "auxiliary_files": auxiliary_files,
        "csv": csv_receipt,
        "content_ledger_sha256": ledger.hexdigest(),
        "test_opened": False,
        "cache_created": False,
        "server_transfer_performed": False,
        "gpu_used": False,
        "training_performed": False,
    }


def write_refuge_development_receipt(
    *,
    root,
    receipt_path,
    expected_per_split=400,
    source_archive_sha256=None,
):
    receipt_path = Path(receipt_path)
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {receipt_path}")
    receipt = audit_refuge_development(
        root,
        expected_per_split=expected_per_split,
        source_archive_sha256=source_archive_sha256,
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expected-per-split", type=int, default=400)
    parser.add_argument("--source-archive-sha256")
    args = parser.parse_args()
    receipt = write_refuge_development_receipt(
        root=args.root,
        receipt_path=args.receipt,
        expected_per_split=args.expected_per_split,
        source_archive_sha256=args.source_archive_sha256,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
