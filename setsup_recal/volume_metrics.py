"""NPC metrics after reconstruction of each complete patient volume."""
import statistics
import torch
from .metrics import METRIC_NAMES, evaluate_distribution_batch

def _metric_dict(probabilities, targets):
    metrics = evaluate_distribution_batch(probabilities, targets)
    return {name: float(metrics[name][0].cpu()) for name in METRIC_NAMES}


def _aligned_rater_dice(generated, targets, eps=1e-6):
    if generated.shape != targets.shape:
        raise ValueError("aligned personas and targets must share shape")
    generated = generated.ge(0.5).to(torch.float32).flatten(start_dim=1)
    targets = targets.ge(0.5).to(torch.float32).flatten(start_dim=1)
    intersection = (generated * targets).sum(dim=1)
    return (
        (2.0 * intersection + eps)
        / (generated.sum(dim=1) + targets.sum(dim=1) + eps)
    ).tolist()


def aggregate_volume_slice_rows(slice_rows, aligned_personas=False):
    if not slice_rows:
        raise ValueError("volume evaluation requires slice rows")
    volumes = {}
    sample_metrics = []
    for row in slice_rows:
        volume_id = row.get("volume_id")
        slice_index = row.get("slice_index")
        if not volume_id or not isinstance(slice_index, int) or slice_index < 0:
            raise ValueError("volume_id and non-negative slice_index are required")
        generated = torch.as_tensor(row["generated"], dtype=torch.float32)
        targets = torch.as_tensor(row["targets"], dtype=torch.float32)
        if generated.ndim != 3 or targets.ndim != 3:
            raise ValueError("slice predictions must have shapes [K,H,W] and [R,H,W]")
        if generated.shape[-2:] != targets.shape[-2:]:
            raise ValueError("generated and target slice geometry differs")
        sample_metrics.append(
            _metric_dict(generated.unsqueeze(0), targets.unsqueeze(0))
        )
        volume = volumes.setdefault(
            volume_id,
            {
                "patient_id": row["patient_id"],
                "rater_ids": list(row["rater_ids"]),
                "slices": {},
            },
        )
        if volume["patient_id"] != row["patient_id"]:
            raise ValueError("patient identity changes within a volume")
        if volume["rater_ids"] != list(row["rater_ids"]):
            raise ValueError("rater order changes within a volume")
        if slice_index in volume["slices"]:
            raise ValueError("duplicate slice index within a volume")
        volume["slices"][slice_index] = (generated, targets)

    volume_rows = []
    rater_rows = []
    for volume_id in sorted(volumes):
        volume = volumes[volume_id]
        indices = sorted(volume["slices"])
        if indices != list(range(len(indices))):
            raise ValueError("volume slice indices must be contiguous from zero")
        generated = torch.stack(
            [volume["slices"][index][0] for index in indices], dim=1
        )
        targets = torch.stack(
            [volume["slices"][index][1] for index in indices], dim=1
        )
        height, width = generated.shape[-2:]
        generated_for_metrics = generated.reshape(
            1, generated.shape[0], generated.shape[1] * height, width
        )
        targets_for_metrics = targets.reshape(
            1, targets.shape[0], targets.shape[1] * height, width
        )
        metrics = _metric_dict(generated_for_metrics, targets_for_metrics)
        per_rater = None
        if aligned_personas:
            per_rater = _aligned_rater_dice(generated, targets)
            rater_rows.append(per_rater)
        volume_rows.append(
            {
                "volume_id": volume_id,
                "patient_id": volume["patient_id"],
                "slice_count": len(indices),
                "slice_indices": indices,
                "rater_ids": volume["rater_ids"],
                "per_rater_dice": per_rater,
                **metrics,
            }
        )

    patient_macro = {
        name: statistics.fmean(row[name] for row in volume_rows)
        for name in METRIC_NAMES
    }
    sample_macro = {
        name: statistics.fmean(row[name] for row in sample_metrics)
        for name in METRIC_NAMES
    }
    per_rater_dice = None
    if aligned_personas:
        per_rater_dice = [
            statistics.fmean(row[index] for row in rater_rows)
            for index in range(len(rater_rows[0]))
        ]
    return {
        "volume_count": len(volume_rows),
        "slice_count": len(slice_rows),
        "patient_macro": patient_macro,
        "sample_macro": sample_macro,
        "per_rater_dice": per_rater_dice,
        "mean_per_rater_dice": (
            statistics.fmean(per_rater_dice) if per_rater_dice is not None else None
        ),
        "rows": volume_rows,
    }




def _collect_volume_rows(model, loader, device, probability_fn):
    device = torch.device(device)
    was_training = model.training
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device)
            probabilities = probability_fn(model, images, batch_index)
            targets = batch["masks"].to(device)
            for item_index in range(images.shape[0]):
                rows.append(
                    {
                        "patient_id": batch["patient_id"][item_index],
                        "volume_id": batch["volume_id"][item_index],
                        "slice_index": batch["slice_index"][item_index],
                        "sample_id": batch["sample_id"][item_index],
                        "rater_ids": batch["rater_ids"][item_index],
                        "generated": probabilities[item_index].cpu(),
                        "targets": targets[item_index].cpu(),
                    }
                )
    if was_training:
        model.train()
    return rows


def evaluate_persona_volume_loader(
    model,
    loader,
    *,
    arm,
    device,
    seed,
    candidate_samples,
):
    device = torch.device(device)

    def sample(current_model, images, batch_index):
        return current_model.sample_personas(
            images,
            candidate_samples=candidate_samples,
            generator=torch.Generator(device=device).manual_seed(seed + batch_index),
            use_graph=arm != "dpersona_stage2",
        )

    rows = _collect_volume_rows(model, loader, device, sample)
    return aggregate_volume_slice_rows(rows, aligned_personas=True)
