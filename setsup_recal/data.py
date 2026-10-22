"""Manifest validation and cache loading from the study implementation."""
import hashlib,json,math,random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

SCHEMA = "multirater_seg_manifest/1"


REQUIRED_TOP_LEVEL = {
    "schema",
    "dataset_id",
    "dataset_version",
    "license_id",
    "task_id",
    "modality",
    "split",
    "image_channels",
    "cache_root",
    "test_opened",
    "transform_receipt_sha256",
    "records",
}


REQUIRED_RECORD = {
    "patient_id",
    "sample_id",
    "split",
    "task_id",
    "cache_file",
    "image_shape",
    "mask_shape",
    "rater_ids",
    "spacing",
    "source_sha256",
    "cache_sha256",
}


class MultiRaterManifestError(ValueError):
    pass


class RaterCountBatchSampler:
    def __init__(self, rater_counts, batch_size, seed=0, shuffle=True):
        self.rater_counts = tuple(int(value) for value in rater_counts)
        if not self.rater_counts or any(value < 3 for value in self.rater_counts):
            raise MultiRaterManifestError("rater counts must all be at least three")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        groups = {}
        for index, count in enumerate(self.rater_counts):
            groups.setdefault(count, []).append(index)
        generator = random.Random(self.seed + self.epoch)
        batches = []
        for count in sorted(groups):
            indices = groups[count]
            if self.shuffle:
                generator.shuffle(indices)
            batches.extend(
                indices[start : start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
            )
        if self.shuffle:
            generator.shuffle(batches)
        return iter(batches)

    def __len__(self):
        groups = {}
        for count in self.rater_counts:
            groups[count] = groups.get(count, 0) + 1
        return sum(
            (size + self.batch_size - 1) // self.batch_size
            for size in groups.values()
        )


def collate_multirater_samples(samples):
    if not samples:
        raise MultiRaterManifestError("cannot collate an empty batch")
    images = [torch.as_tensor(sample["image"]) for sample in samples]
    masks = [torch.as_tensor(sample["masks"]) for sample in samples]
    rater_counts = {mask.shape[0] for mask in masks}
    if len(rater_counts) != 1:
        raise MultiRaterManifestError("batch rater counts must match")
    if len({tuple(image.shape) for image in images}) != 1:
        raise MultiRaterManifestError("batch image shapes must match")
    if len({tuple(mask.shape) for mask in masks}) != 1:
        raise MultiRaterManifestError("batch mask shapes must match")
    return {
        "image": torch.stack(images),
        "masks": torch.stack(masks),
        "patient_id": [sample["patient_id"] for sample in samples],
        "sample_id": [sample["sample_id"] for sample in samples],
        "volume_id": [sample.get("volume_id") for sample in samples],
        "slice_index": [sample.get("slice_index") for sample in samples],
        "rater_ids": [list(sample["rater_ids"]) for sample in samples],
        "spacing": [sample["spacing"] for sample in samples],
        "task_id": [sample["task_id"] for sample in samples],
        "split": [sample["split"] for sample in samples],
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _read_manifest(path):
    path = Path(path)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MultiRaterManifestError(f"cannot read manifest: {path}") from error
    if not isinstance(value, dict):
        raise MultiRaterManifestError("manifest root must be an object")
    return path, value


def validate_multirater_manifest(path, allow_test_opened=False):
    path, manifest = _read_manifest(path)
    missing = REQUIRED_TOP_LEVEL - set(manifest)
    if missing:
        raise MultiRaterManifestError(f"manifest fields missing: {sorted(missing)}")
    if manifest["schema"] != SCHEMA:
        raise MultiRaterManifestError(f"unexpected manifest schema: {manifest['schema']}")
    if manifest["test_opened"] is not False and not allow_test_opened:
        raise MultiRaterManifestError("test_opened must remain false")
    if not isinstance(manifest["image_channels"], int) or manifest["image_channels"] < 1:
        raise MultiRaterManifestError("image_channels must be positive")
    if not _is_sha256(manifest["transform_receipt_sha256"]):
        raise MultiRaterManifestError("transform receipt SHA-256 is invalid")
    if not isinstance(manifest["records"], list) or not manifest["records"]:
        raise MultiRaterManifestError("manifest records must be non-empty")

    cache_root = Path(manifest["cache_root"])
    if not cache_root.is_absolute():
        cache_root = (path.parent / cache_root).resolve()
    sample_ids = set()
    volume_slices = set()
    volume_specs = {}
    for record in manifest["records"]:
        if not isinstance(record, dict):
            raise MultiRaterManifestError("manifest records must be objects")
        missing = REQUIRED_RECORD - set(record)
        if missing:
            raise MultiRaterManifestError(f"record fields missing: {sorted(missing)}")
        if record["split"] != manifest["split"] or record["task_id"] != manifest["task_id"]:
            raise MultiRaterManifestError("record split/task does not match manifest")
        if record["sample_id"] in sample_ids:
            raise MultiRaterManifestError("sample IDs must be unique")
        sample_ids.add(record["sample_id"])
        raters = record["rater_ids"]
        if not isinstance(raters, list) or len(raters) < 3:
            raise MultiRaterManifestError("each sample requires at least three raters")
        if len(set(raters)) != len(raters):
            raise MultiRaterManifestError("rater IDs must be unique per sample")
        if record["image_shape"][0] != manifest["image_channels"]:
            raise MultiRaterManifestError("image shape does not match image_channels")
        if record["mask_shape"][0] != len(raters):
            raise MultiRaterManifestError("mask shape does not match rater count")
        if record["image_shape"][-2:] != record["mask_shape"][-2:]:
            raise MultiRaterManifestError("image and mask spatial shapes must match")
        spacing = record["spacing"]
        volume_id = record.get("volume_id")
        slice_index = record.get("slice_index")
        if (volume_id is None) != (slice_index is None):
            raise MultiRaterManifestError(
                "volume_id and slice_index must be provided together"
            )
        if volume_id is not None and (
            not isinstance(volume_id, str)
            or not volume_id
            or not isinstance(slice_index, int)
            or isinstance(slice_index, bool)
            or slice_index < 0
        ):
            raise MultiRaterManifestError("volume slice metadata is invalid")
        if volume_id is not None:
            volume_slice = (volume_id, slice_index)
            if volume_slice in volume_slices:
                raise MultiRaterManifestError(
                    "volume slice coordinates must be unique"
                )
            volume_slices.add(volume_slice)
        expected_spacing_dimensions = 3 if volume_id is not None else 2
        if volume_id is not None and spacing is None:
            raise MultiRaterManifestError(
                "volume spacing must contain three positive values"
            )
        if spacing is not None and (
            not isinstance(spacing, list)
            or len(spacing) != expected_spacing_dimensions
            or any(not math.isfinite(float(item)) or float(item) <= 0 for item in spacing)
        ):
            raise MultiRaterManifestError(
                f"spacing must be null or {expected_spacing_dimensions} positive values"
            )
        if volume_id is not None:
            volume_spec = (
                record["patient_id"],
                tuple(raters),
                tuple(record["image_shape"][-2:]),
                tuple(record["mask_shape"][-2:]),
                tuple(spacing) if spacing is not None else None,
            )
            if volume_id in volume_specs and volume_specs[volume_id] != volume_spec:
                raise MultiRaterManifestError(
                    "volume geometry and rater identity must remain consistent"
                )
            volume_specs[volume_id] = volume_spec
        if not isinstance(record["source_sha256"], dict) or not record["source_sha256"]:
            raise MultiRaterManifestError("source_sha256 must be a non-empty object")
        if not all(_is_sha256(item) for item in record["source_sha256"].values()):
            raise MultiRaterManifestError("source SHA-256 is invalid")
        if not _is_sha256(record["cache_sha256"]):
            raise MultiRaterManifestError("cache SHA-256 is invalid")
        cache_path = (cache_root / record["cache_file"]).resolve()
        if not cache_path.is_relative_to(cache_root.resolve()):
            raise MultiRaterManifestError("cache file escapes cache_root")
        if not cache_path.is_file():
            raise MultiRaterManifestError(f"cache file missing: {cache_path}")
        if _sha256(cache_path) != record["cache_sha256"]:
            raise MultiRaterManifestError(f"cache SHA-256 mismatch: {cache_path}")
    return manifest


def validate_disjoint_manifests(paths):
    owners = {}
    manifests = []
    for path in paths:
        manifest = validate_multirater_manifest(path)
        manifests.append(manifest)
        for patient_id in {record["patient_id"] for record in manifest["records"]}:
            if patient_id in owners:
                raise MultiRaterManifestError(
                    f"patient overlap between {owners[patient_id]} and {manifest['split']}: {patient_id}"
                )
            owners[patient_id] = manifest["split"]
    return manifests


def resample_in_plane(image, masks, source_spacing, target_spacing):
    image = np.asarray(image)
    masks = np.asarray(masks)
    if image.ndim != 3 or masks.ndim != 3 or image.shape[-2:] != masks.shape[-2:]:
        raise ValueError("image and masks must have shapes [C,H,W] and [R,H,W]")
    if len(source_spacing) != 2 or len(target_spacing) != 2:
        raise ValueError("source and target spacing must contain two values")
    if any(float(value) <= 0 or not math.isfinite(float(value)) for value in (*source_spacing, *target_spacing)):
        raise ValueError("spacing values must be finite and positive")
    height = max(1, int(round(image.shape[-2] * source_spacing[0] / target_spacing[0])))
    width = max(1, int(round(image.shape[-1] * source_spacing[1] / target_spacing[1])))
    image_tensor = torch.from_numpy(image.astype(np.float32, copy=False)).unsqueeze(0)
    mask_tensor = torch.from_numpy(masks.astype(np.float32, copy=False)).unsqueeze(0)
    resized_image = F.interpolate(
        image_tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    resized_masks = F.interpolate(
        mask_tensor,
        size=(height, width),
        mode="nearest",
    ).squeeze(0)
    return resized_image.numpy(), resized_masks.round().to(torch.uint8).numpy()


class MultiRaterDataset(Dataset):
    def __init__(self, manifest_path, allow_test_opened=False):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = validate_multirater_manifest(
            self.manifest_path,
            allow_test_opened=allow_test_opened,
        )
        self.cache_root = Path(self.manifest["cache_root"])
        if not self.cache_root.is_absolute():
            self.cache_root = (self.manifest_path.parent / self.cache_root).resolve()
        self.records = self.manifest["records"]
        self.split = self.manifest["split"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        cache_path = self.cache_root / record["cache_file"]
        with np.load(cache_path, allow_pickle=False) as cached:
            image = cached["image"]
            masks = cached["masks"]
        if list(image.shape) != record["image_shape"]:
            raise MultiRaterManifestError(f"cached image shape mismatch: {cache_path}")
        if list(masks.shape) != record["mask_shape"]:
            raise MultiRaterManifestError(f"cached mask shape mismatch: {cache_path}")
        if not set(np.unique(masks).tolist()).issubset({0, 1}):
            raise MultiRaterManifestError(f"cached masks are not binary: {cache_path}")
        return {
            "image": torch.from_numpy(image.astype(np.float32, copy=False)),
            "masks": torch.from_numpy(masks.astype(np.float32, copy=False)),
            "patient_id": record["patient_id"],
            "sample_id": record["sample_id"],
            "volume_id": record.get("volume_id"),
            "slice_index": record.get("slice_index"),
            "rater_ids": list(record["rater_ids"]),
            "spacing": record["spacing"],
            "task_id": record["task_id"],
            "split": record["split"],
        }
