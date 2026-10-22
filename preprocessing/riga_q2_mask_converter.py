#!/usr/bin/env python3
"""Format-aware RIGA development contour conversion."""

from __future__ import annotations

import re
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff"}
GROUP_PATTERN = re.compile(r"^(image\d+?)(prime|-([1-6]))$", re.IGNORECASE)
DIFFERENCE_THRESHOLD = 50
LOSSY_RG_CENTER = (46, 49)
LOSSY_RG_TOLERANCE = 32
LOSSY_CLOSE_KERNEL = 5
MINIMUM_COMPONENT_PIXELS = 20
HOLE_MINIMUM_AREA_FRACTION = 0.01
GLOBAL_HULL_MINIMUM_COMPACTNESS = 0.5
GLOBAL_HULL_MAXIMUM_CUP_HULL_FRACTION = 0.8
CONVERSION_RULE = {
    "difference_threshold": DIFFERENCE_THRESHOLD,
    "fallback_order": [
        "direct_contour",
        "hole_topology_fallback",
        "convex_hull_fallback",
        "global_hull_fallback",
    ],
    "global_hull_maximum_cup_hull_fraction": (
        GLOBAL_HULL_MAXIMUM_CUP_HULL_FRACTION
    ),
    "global_hull_minimum_compactness": GLOBAL_HULL_MINIMUM_COMPACTNESS,
    "hole_cup_selection": "maximum_solidity_then_area",
    "hole_minimum_area_fraction": HOLE_MINIMUM_AREA_FRACTION,
    "hole_outer_selection": "largest_root_contour",
    "lossy_close_kernel": LOSSY_CLOSE_KERNEL,
    "lossy_rg_center": list(LOSSY_RG_CENTER),
    "lossy_rg_tolerance": LOSSY_RG_TOLERANCE,
    "minimum_component_pixels": MINIMUM_COMPONENT_PIXELS,
}


class MaskConversionError(ValueError):
    pass


@dataclass(frozen=True)
class GroupRecord:
    leaf: str
    image_id: str
    prime: Path
    raters: dict[int, Path]


@dataclass(frozen=True)
class DiscoveryReport:
    complete: tuple[GroupRecord, ...]
    incomplete_count: int


@dataclass(frozen=True)
class MaskExtraction:
    disc: np.ndarray
    cup: np.ndarray
    disc_area: int
    cup_area: int
    dominant_component_count: int
    reconstruction_mode: str


def discover_groups(root: Path) -> DiscoveryReport:
    root = Path(root).resolve()
    grouped = defaultdict(lambda: {"prime": [], "raters": defaultdict(list)})
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = GROUP_PATTERN.match(path.stem)
        if match is None:
            continue
        key = (str(path.parent.relative_to(root)), match.group(1).lower())
        if match.group(2).lower() == "prime":
            grouped[key]["prime"].append(path)
        else:
            grouped[key]["raters"][int(match.group(3))].append(path)

    complete = []
    for (leaf, image_id), files in sorted(grouped.items()):
        if len(files["prime"]) != 1:
            continue
        if any(len(files["raters"].get(rater, [])) != 1 for rater in range(1, 7)):
            continue
        complete.append(
            GroupRecord(
                leaf=leaf,
                image_id=image_id,
                prime=files["prime"][0],
                raters={rater: files["raters"][rater][0] for rater in range(1, 7)},
            )
        )
    return DiscoveryReport(
        complete=tuple(complete),
        incomplete_count=len(grouped) - len(complete),
    )


def _filled_components(line_mask: np.ndarray) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        line_mask.astype(np.uint8), 8
    )
    components = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < MINIMUM_COMPONENT_PIXELS:
            continue
        component = (labels == index).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        filled = np.zeros_like(component)
        cv2.drawContours(filled, [contour], -1, 1, thickness=-1)
        if int(filled.sum()) > 0:
            components.append(filled.astype(bool))
    return components


def _convex_hull_components(line_mask: np.ndarray) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        line_mask.astype(np.uint8), 8
    )
    components = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < MINIMUM_COMPONENT_PIXELS:
            continue
        yy, xx = np.where(labels == index)
        points = np.column_stack((xx, yy)).astype(np.int32)
        hull = cv2.convexHull(points)
        filled = np.zeros_like(line_mask, dtype=np.uint8)
        cv2.fillConvexPoly(filled, hull, 1)
        if int(filled.sum()) > 0:
            components.append(filled.astype(bool))
    return components


def _nested_candidates(components: list[np.ndarray]) -> list[tuple]:
    candidates = []
    for outer_index, outer in enumerate(components):
        outer_area = int(outer.sum())
        for inner_index, inner in enumerate(components):
            if inner_index == outer_index:
                continue
            inner_area = int(inner.sum())
            if inner_area >= outer_area:
                continue
            if np.any(inner & ~outer):
                continue
            candidates.append((outer_area, inner_area, outer, inner))
    return candidates


def _hole_topology_candidates(line_mask: np.ndarray) -> list[tuple]:
    contours, hierarchy = cv2.findContours(
        line_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    root_indices = [
        index
        for index, item in enumerate(hierarchy)
        if item[3] == -1
        and float(cv2.contourArea(contours[index])) >= MINIMUM_COMPONENT_PIXELS
    ]
    if not root_indices:
        return []
    outer_index = max(root_indices, key=lambda index: cv2.contourArea(contours[index]))
    outer_contour = contours[outer_index]
    outer_contour_area = float(cv2.contourArea(outer_contour))
    child_indices = [
        index
        for index, item in enumerate(hierarchy)
        if item[3] == outer_index
        and float(cv2.contourArea(contours[index]))
        >= outer_contour_area * HOLE_MINIMUM_AREA_FRACTION
    ]
    if len(child_indices) < 2:
        return []

    def solidity_and_area(index: int) -> tuple[float, float]:
        area = float(cv2.contourArea(contours[index]))
        hull_area = float(cv2.contourArea(cv2.convexHull(contours[index])))
        return (area / hull_area if hull_area else 0.0, area)

    cup_index = max(
        child_indices,
        key=solidity_and_area,
    )
    disc = np.zeros_like(line_mask, dtype=np.uint8)
    cup = np.zeros_like(line_mask, dtype=np.uint8)
    cv2.drawContours(disc, [outer_contour], -1, 1, thickness=-1)
    cv2.drawContours(cup, [contours[cup_index]], -1, 1, thickness=-1)
    disc = disc.astype(bool)
    cup = cup.astype(bool)
    disc_area = int(disc.sum())
    cup_area = int(cup.sum())
    if cup_area >= disc_area or np.any(cup & ~disc):
        return []
    return [(disc_area, cup_area, disc, cup)]


def _global_hull_candidates(line_mask: np.ndarray) -> list[tuple]:
    yy, xx = np.where(line_mask)
    if len(xx) < MINIMUM_COMPONENT_PIXELS:
        return []
    all_points = np.column_stack((xx, yy)).astype(np.int32)
    disc_hull = cv2.convexHull(all_points)
    disc = np.zeros_like(line_mask, dtype=np.uint8)
    cv2.fillConvexPoly(disc, disc_hull, 1)
    disc = disc.astype(bool)
    disc_area = int(disc.sum())

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        line_mask.astype(np.uint8), 8
    )
    candidates = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < MINIMUM_COMPONENT_PIXELS:
            continue
        component = (labels == index).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        cup = np.zeros_like(component)
        cv2.drawContours(cup, [contour], -1, 1, thickness=-1)
        cup = cup.astype(bool)
        cup_area = int(cup.sum())

        component_yy, component_xx = np.where(component)
        component_points = np.column_stack((component_xx, component_yy)).astype(
            np.int32
        )
        component_hull = cv2.convexHull(component_points)
        hull_fill = np.zeros_like(component)
        cv2.fillConvexPoly(hull_fill, component_hull, 1)
        hull_area = int(hull_fill.sum())
        compactness = cup_area / hull_area if hull_area else 0.0
        if compactness < GLOBAL_HULL_MINIMUM_COMPACTNESS:
            continue
        if hull_area > disc_area * GLOBAL_HULL_MAXIMUM_CUP_HULL_FRACTION:
            continue
        if cup_area < disc_area * HOLE_MINIMUM_AREA_FRACTION:
            continue
        if cup_area >= disc_area or np.any(cup & ~disc):
            continue
        candidates.append((disc_area, cup_area, disc, cup))
    return candidates


def extract_nested_masks(
    prime_rgb: np.ndarray,
    marked_rgb: np.ndarray,
    *,
    lossless_pair: bool,
) -> MaskExtraction:
    if prime_rgb.shape != marked_rgb.shape or prime_rgb.ndim != 3:
        raise MaskConversionError("prime and marked images must share an RGB shape")
    if prime_rgb.shape[2] != 3:
        raise MaskConversionError("prime and marked images must be RGB")

    prime = prime_rgb.astype(np.int16, copy=False)
    marked = marked_rgb.astype(np.int16, copy=False)
    changed = np.abs(marked - prime).max(axis=2) > DIFFERENCE_THRESHOLD
    if lossless_pair:
        line_mask = changed
    else:
        line_mask = (
            changed
            & (
                np.abs(marked[:, :, 0] - LOSSY_RG_CENTER[0])
                <= LOSSY_RG_TOLERANCE
            )
            & (
                np.abs(marked[:, :, 1] - LOSSY_RG_CENTER[1])
                <= LOSSY_RG_TOLERANCE
            )
        )
        line_mask = cv2.morphologyEx(
            line_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((LOSSY_CLOSE_KERNEL, LOSSY_CLOSE_KERNEL), dtype=np.uint8),
            iterations=1,
        ).astype(bool)

    components = _filled_components(line_mask)
    candidates = _nested_candidates(components)
    reconstruction_mode = "direct_contour"
    if not candidates:
        candidates = _hole_topology_candidates(line_mask)
        reconstruction_mode = "hole_topology_fallback"
    if not candidates:
        candidates = _nested_candidates(_convex_hull_components(line_mask))
        reconstruction_mode = "convex_hull_fallback"
    if not candidates:
        candidates = _global_hull_candidates(line_mask)
        reconstruction_mode = "global_hull_fallback"

    if not candidates:
        raise MaskConversionError("could not recover nested disc/cup contours")
    disc_area, cup_area, disc, cup = max(candidates, key=lambda item: item[:2])
    return MaskExtraction(
        disc=disc,
        cup=cup,
        disc_area=disc_area,
        cup_area=cup_area,
        dominant_component_count=len(components),
        reconstruction_mode=reconstruction_mode,
    )


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _write_binary_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def materialize_development(
    source_root: Path,
    output_root: Path,
    *,
    expected_by_leaf: dict[str, int],
    expected_incomplete_count: int,
) -> dict:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    partial_root = output_root.with_name(output_root.name + ".partial")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite mask target: {output_root}")
    if partial_root.exists():
        raise FileExistsError(f"refusing to overwrite partial mask target: {partial_root}")

    discovery = discover_groups(source_root)
    actual_by_leaf = dict(Counter(group.leaf for group in discovery.complete))
    if (
        actual_by_leaf != dict(expected_by_leaf)
        or discovery.incomplete_count != expected_incomplete_count
    ):
        raise MaskConversionError(
            "group inventory mismatch: "
            f"complete={actual_by_leaf}, incomplete={discovery.incomplete_count}"
        )

    partial_root.mkdir(parents=True)
    manifest_path = partial_root / "manifest.jsonl"
    current_group = None
    rows = []
    try:
        for group in discovery.complete:
            current_group = f"{group.leaf}/{group.image_id}"
            prime_rgb = _read_rgb(group.prime)
            prime_sha256 = _sha256(group.prime)
            for rater_id, marked_path in sorted(group.raters.items()):
                marked_rgb = _read_rgb(marked_path)
                lossless_pair = (
                    group.prime.suffix.lower() in {".tif", ".tiff"}
                    and marked_path.suffix.lower() in {".tif", ".tiff"}
                )
                extraction = extract_nested_masks(
                    prime_rgb,
                    marked_rgb,
                    lossless_pair=lossless_pair,
                )
                relative_dir = Path("masks") / group.leaf / group.image_id
                disc_relative = relative_dir / f"rater_{rater_id}_disc.png"
                cup_relative = relative_dir / f"rater_{rater_id}_cup.png"
                disc_path = partial_root / disc_relative
                cup_path = partial_root / cup_relative
                _write_binary_mask(disc_path, extraction.disc)
                _write_binary_mask(cup_path, extraction.cup)
                rows.append(
                    {
                        "schema": "riga_q2_mask_record/1",
                        "leaf": group.leaf,
                        "image_id": group.image_id,
                        "rater_id": rater_id,
                        "height": int(prime_rgb.shape[0]),
                        "width": int(prime_rgb.shape[1]),
                        "lossless_pair": lossless_pair,
                        "conversion_rule": dict(CONVERSION_RULE),
                        "prime_path": str(group.prime.relative_to(source_root)),
                        "prime_sha256": prime_sha256,
                        "marked_path": str(marked_path.relative_to(source_root)),
                        "source_sha256": _sha256(marked_path),
                        "disc_path": str(disc_relative),
                        "disc_sha256": _sha256(disc_path),
                        "disc_area": extraction.disc_area,
                        "cup_path": str(cup_relative),
                        "cup_sha256": _sha256(cup_path),
                        "cup_area": extraction.cup_area,
                        "cup_disc_ratio": extraction.cup_area / extraction.disc_area,
                        "dominant_component_count": extraction.dominant_component_count,
                        "reconstruction_mode": extraction.reconstruction_mode,
                        "cup_outside_disc": 0,
                    }
                )

        with manifest_path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        receipt = {
            "schema": "riga_q2_mask_materialization/1",
            "status": "COMPLETE",
            "source_root": str(source_root),
            "output_root": str(output_root),
            "converter_sha256": _sha256(Path(__file__)),
            "manifest_sha256": _sha256(manifest_path),
            "conversion_rule": dict(CONVERSION_RULE),
            "complete_groups": len(discovery.complete),
            "incomplete_groups": discovery.incomplete_count,
            "groups_by_leaf": actual_by_leaf,
            "rater_records": len(rows),
            "mask_files": len(rows) * 2,
            "reconstruction_modes": dict(
                Counter(row["reconstruction_mode"] for row in rows)
            ),
            "magrabia_opened": False,
            "obsolete_binrushed_opened": False,
            "split_created": False,
            "test_cache_created": False,
            "gpu_used": False,
        }
        receipt_path = partial_root / "receipt.json"
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
        partial_root.rename(output_root)
        return receipt
    except Exception as error:
        failure_path = partial_root / "failure.json"
        if not failure_path.exists():
            with failure_path.open("x", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema": "riga_q2_mask_materialization_failure/1",
                        "status": "FAILED",
                        "current_group": current_group,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
        raise
