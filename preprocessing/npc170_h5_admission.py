#!/usr/bin/env python3

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path


IMAGE_KEYS = ("t1", "t1c", "t2")
LABEL_KEYS = ("label_a1", "label_a2", "label_a3", "label_a4")
VOLUME_KEYS = set(IMAGE_KEYS + LABEL_KEYS + ("voxel_spacing",))
SLICE_KEYS = set(IMAGE_KEYS + LABEL_KEYS)
DATASET_RE = re.compile(r'DATASET\s+"/?([^"\\]+)"\s*\{(.*?)(?=^\s*DATASET\s+"|\Z)', re.S | re.M)
DTYPE_RE = re.compile(r"DATATYPE\s+([^\s{]+)")
SHAPE_RE = re.compile(r"DATASPACE\s+SIMPLE\s*\{\s*\(\s*([^)]*?)\s*\)")
DATA_RE = re.compile(r"\bDATA\s*\{(.*?)^\s*\}", re.S | re.M)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
TRAIN_VOLUME_RE = re.compile(r"^training/Sample_(\d+)\.h5$")
VALIDATION_VOLUME_RE = re.compile(r"^validation/Sample_(\d+)\.h5$")
TRAIN_SLICE_RE = re.compile(r"^training_2d/Sample_(\d+)_slice_(\d+)\.h5$")


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_h5dump(text, retain_values=True):
    datasets = {}
    for name, body in DATASET_RE.findall(text):
        dtype_match = DTYPE_RE.search(body)
        shape_match = SHAPE_RE.search(body)
        if not dtype_match or not shape_match:
            continue
        shape_text = shape_match.group(1).strip()
        shape = [] if not shape_text else [int(value.strip()) for value in shape_text.split(",")]
        record = {"dtype": dtype_match.group(1), "shape": shape}
        data_match = DATA_RE.search(body)
        if data_match:
            numbers = NUMBER_RE.finditer(data_match.group(1))
            if retain_values:
                record["values"] = [float(match.group(0)) for match in numbers]
            else:
                count = 0
                foreground_count = 0
                unique = set()
                for match in numbers:
                    value = float(match.group(0))
                    count += 1
                    foreground_count += value == 1.0
                    unique.add(value)
                record["value_summary"] = {
                    "count": count,
                    "foreground_count": foreground_count,
                    "unique": sorted(unique),
                }
        datasets[name] = record
    return datasets


def merge_dataset_values(schema, values):
    merged = {name: dict(record) for name, record in schema.items()}
    for name, value_record in values.items():
        if name not in merged:
            merged[name] = dict(value_record)
            continue
        for key in ("values", "value_summary"):
            if key in value_record:
                merged[name][key] = value_record[key]
    return merged


def validate_sample(datasets, kind, expected_shape=None):
    expected_keys = VOLUME_KEYS if kind == "volume" else SLICE_KEYS
    expected_ndim = 3 if kind == "volume" else 2
    issues = []
    for name in sorted(expected_keys - set(datasets)):
        issues.append(f"missing_dataset:{name}")
    for name in sorted(set(datasets) - expected_keys):
        issues.append(f"unexpected_dataset:{name}")

    shape = datasets.get("t1", {}).get("shape")
    if shape is not None and len(shape) != expected_ndim:
        issues.append(f"expected_ndim:{expected_ndim}:found:{len(shape)}")
    if expected_shape is not None and shape != list(expected_shape):
        issues.append(f"expected_shape:{list(expected_shape)}:found:{shape}")

    foreground_counts = {}
    label_unique = {}
    empty_raters = []
    for name in IMAGE_KEYS:
        record = datasets.get(name)
        if not record:
            continue
        if record["dtype"] != "H5T_IEEE_F32LE":
            issues.append(f"wrong_dtype:{name}:{record['dtype']}")
        if shape is not None and record["shape"] != shape:
            issues.append(f"shape_mismatch:{name}")

    for name in LABEL_KEYS:
        record = datasets.get(name)
        if not record:
            continue
        if record["dtype"] != "H5T_STD_U8LE":
            issues.append(f"wrong_dtype:{name}:{record['dtype']}")
        if shape is not None and record["shape"] != shape:
            issues.append(f"shape_mismatch:{name}")

        if "value_summary" in record:
            summary = record["value_summary"]
            unique = summary["unique"]
            foreground_count = summary["foreground_count"]
            value_count = summary["count"]
        elif "values" in record:
            values = record["values"]
            unique = sorted(set(values))
            foreground_count = sum(value == 1.0 for value in values)
            value_count = len(values)
        else:
            issues.append(f"label_values_missing:{name}")
            continue

        label_unique[name] = unique
        foreground_counts[name] = foreground_count
        if not set(unique).issubset({0.0, 1.0}):
            issues.append(f"non_binary_label:{name}")
        expected_count = math.prod(record["shape"])
        if value_count != expected_count:
            issues.append(f"label_value_count:{name}:{value_count}:{expected_count}")
        if foreground_count == 0:
            empty_raters.append(name)

    spacing = None
    if kind == "volume" and "voxel_spacing" in datasets:
        record = datasets["voxel_spacing"]
        if record["dtype"] != "H5T_IEEE_F64LE":
            issues.append(f"wrong_dtype:voxel_spacing:{record['dtype']}")
        if record["shape"] != [3]:
            issues.append(f"spacing_shape:{record['shape']}")
        spacing = record.get("values")
        if spacing is None or len(spacing) != 3 or not all(
            math.isfinite(value) and value > 0 for value in spacing
        ):
            issues.append("invalid_voxel_spacing")

    return {
        "issues": issues,
        "shape": shape,
        "rater_count": sum(name in datasets for name in LABEL_KEYS),
        "foreground_counts": foreground_counts,
        "empty_raters": empty_raters,
        "label_unique": label_unique,
        "spacing": spacing,
    }


def _run_h5dump(h5dump, path, *args):
    completed = subprocess.run(
        [str(h5dump), *args, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _inspect_file(root, path, kind, h5dump, expected_shape=None):
    schema = parse_h5dump(_run_h5dump(h5dump, path, "-H"))
    label_args = []
    for name in LABEL_KEYS:
        label_args.extend(("-d", f"/{name}"))
    label_values = parse_h5dump(
        _run_h5dump(h5dump, path, "-y", *label_args), retain_values=False
    )
    datasets = merge_dataset_values(schema, label_values)
    if kind == "volume":
        spacing_values = parse_h5dump(
            _run_h5dump(h5dump, path, "-y", "-d", "/voxel_spacing")
        )
        datasets = merge_dataset_values(datasets, spacing_values)

    validation = validate_sample(datasets, kind=kind, expected_shape=expected_shape)
    relative = str(path.relative_to(root))
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "kind": kind,
        **validation,
    }


def _parallel_inspect(jobs, workers):
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(*job) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            records.append(future.result())
            if index % 250 == 0 or index == len(futures):
                print(f"audited {index}/{len(futures)}", flush=True)
    return records


def audit_tree(root, h5dump="h5dump", workers=8):
    root = Path(root)
    h5dump = Path(h5dump)
    train_paths = sorted((root / "training").glob("Sample_*.h5"))
    validation_paths = sorted((root / "validation").glob("Sample_*.h5"))
    slice_paths = sorted((root / "training_2d").glob("Sample_*_slice_*.h5"))

    volume_jobs = [
        (_inspect_file, root, path, "volume", h5dump, None)
        for path in train_paths + validation_paths
    ]
    volumes = _parallel_inspect(volume_jobs, workers)
    volume_by_path = {record["path"]: record for record in volumes}
    training_shapes = {}
    for relative, record in volume_by_path.items():
        match = TRAIN_VOLUME_RE.match(relative)
        if match and record["shape"] and len(record["shape"]) == 3:
            training_shapes[int(match.group(1))] = record["shape"]

    slice_jobs = []
    for path in slice_paths:
        relative = str(path.relative_to(root))
        match = TRAIN_SLICE_RE.match(relative)
        subject = int(match.group(1)) if match else -1
        volume_shape = training_shapes.get(subject)
        expected_shape = tuple(volume_shape[1:]) if volume_shape else None
        slice_jobs.append((_inspect_file, root, path, "slice", h5dump, expected_shape))
    slices = _parallel_inspect(slice_jobs, workers)

    records = sorted(volumes + slices, key=lambda record: record["path"])
    cross_file_issues = []
    slices_by_subject = {}
    for record in slices:
        match = TRAIN_SLICE_RE.match(record["path"])
        if not match:
            cross_file_issues.append(f"unexpected_slice_path:{record['path']}")
            continue
        subject, slice_index = map(int, match.groups())
        slices_by_subject.setdefault(subject, []).append(slice_index)

    for subject in sorted(set(training_shapes) | set(slices_by_subject)):
        volume_shape = training_shapes.get(subject)
        observed = sorted(slices_by_subject.get(subject, []))
        if volume_shape is None:
            cross_file_issues.append(f"slice_subject_without_volume:{subject}")
            continue
        expected = list(range(volume_shape[0]))
        if observed != expected:
            cross_file_issues.append(
                f"slice_indices_mismatch:{subject}:expected_0_{volume_shape[0] - 1}:found_{len(observed)}"
            )

    record_issues = [
        {"path": record["path"], "issues": record["issues"]}
        for record in records
        if record["issues"]
    ]
    tree_hash = hashlib.sha256(
        "".join(
            f"{record['sha256']}  {record['path']}\n" for record in records
        ).encode("utf-8")
    ).hexdigest()
    volume_empty = sum(bool(record["empty_raters"]) for record in volumes)
    slice_empty = sum(bool(record["empty_raters"]) for record in slices)

    return {
        "schema": "npc170_h5_development_admission/1",
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "root": str(root.resolve()),
        "audit_scope": (
            "all HDF5 metadata, label_a1-label_a4 arrays, voxel_spacing arrays, "
            "and whole-file SHA-256; t1/t1c/t2 image arrays were not read"
        ),
        "summary": {
            "files": len(records),
            "training_volumes": len(train_paths),
            "validation_volumes": len(validation_paths),
            "training_slices": len(slice_paths),
            "record_issue_count": len(record_issues),
            "cross_file_issue_count": len(cross_file_issues),
            "volume_samples_with_empty_rater_masks": volume_empty,
            "slice_samples_with_empty_rater_masks": slice_empty,
            "tree_sha256": tree_hash,
            "admission_pass": not record_issues and not cross_file_issues,
        },
        "record_issues": record_issues,
        "cross_file_issues": cross_file_issues,
        "records": records,
    }


def write_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h5dump", type=Path, default=Path("h5dump"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    report = audit_tree(args.root, h5dump=args.h5dump, workers=args.workers)
    write_report(report, args.output)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
