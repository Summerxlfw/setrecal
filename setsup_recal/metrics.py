"""Set metrics with hard masks at probability >= 0.5; GED is not square-rooted."""
import torch

METRIC_NAMES = (
    "ged",
    "consensus_dice",
    "brier",
    "variance_mae",
    "best_of_k_dice",
    "generated_diversity",
    "rater_diversity",
)


def _pairwise_iou_and_dice(left, right, eps=1e-6):
    left = left.flatten(start_dim=2).to(torch.float32)
    right = right.flatten(start_dim=2).to(torch.float32)
    intersection = torch.bmm(left, right.transpose(1, 2))
    left_sum = left.sum(dim=-1, keepdim=True)
    right_sum = right.sum(dim=-1).unsqueeze(1)
    union = left_sum + right_sum - intersection
    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (left_sum + right_sum + eps)
    return iou, dice


def _off_diagonal_mean(matrix):
    count = matrix.shape[1]
    if count < 2 or matrix.shape[2] != count:
        raise ValueError("diversity requires a square matrix with at least two samples")
    diagonal = torch.diagonal(matrix, dim1=1, dim2=2).sum(dim=1)
    return (matrix.sum(dim=(1, 2)) - diagonal) / (count * (count - 1))


def _binary_dice(left, right, eps=1e-6):
    left = left.flatten(start_dim=1).to(torch.float32)
    right = right.flatten(start_dim=1).to(torch.float32)
    intersection = (left * right).sum(dim=1)
    return (2.0 * intersection + eps) / (
        left.sum(dim=1) + right.sum(dim=1) + eps
    )


def evaluate_distribution_batch(generated_probabilities, target_masks):
    if generated_probabilities.ndim != 4 or target_masks.ndim != 4:
        raise ValueError("probabilities and masks must have shape [B, K, H, W]")
    if (
        generated_probabilities.shape[0] != target_masks.shape[0]
        or generated_probabilities.shape[-2:] != target_masks.shape[-2:]
    ):
        raise ValueError("probabilities and masks must share batch and spatial shapes")

    generated_hard = (generated_probabilities >= 0.5).to(torch.float32)
    targets_hard = (target_masks >= 0.5).to(torch.float32)
    cross_iou, cross_dice = _pairwise_iou_and_dice(generated_hard, targets_hard)
    generated_iou, _ = _pairwise_iou_and_dice(generated_hard, generated_hard)
    target_iou, _ = _pairwise_iou_and_dice(targets_hard, targets_hard)
    cross_distance = 1.0 - cross_iou
    generated_distance = 1.0 - generated_iou
    target_distance = 1.0 - target_iou
    ged = (
        2.0 * cross_distance.mean(dim=(1, 2))
        - generated_distance.mean(dim=(1, 2))
        - target_distance.mean(dim=(1, 2))
    )

    target_frequency = target_masks.to(torch.float32).mean(dim=1)
    predictive_mean = generated_probabilities.to(torch.float32).mean(dim=1)
    consensus_prediction = predictive_mean >= 0.5
    consensus_target = target_frequency >= 0.5
    brier = (predictive_mean - target_frequency).square().mean(dim=(1, 2))
    generated_variance = generated_probabilities.to(torch.float32).var(
        dim=1, unbiased=False
    )
    target_variance = target_masks.to(torch.float32).var(dim=1, unbiased=False)

    return {
        "ged": ged,
        "consensus_dice": _binary_dice(consensus_prediction, consensus_target),
        "brier": brier,
        "variance_mae": (generated_variance - target_variance)
        .abs()
        .mean(dim=(1, 2)),
        "best_of_k_dice": cross_dice.max(dim=1).values.mean(dim=1),
        "generated_diversity": _off_diagonal_mean(generated_distance),
        "rater_diversity": _off_diagonal_mean(target_distance),
    }


import statistics

def aggregate_metric_rows(rows, metric_names=METRIC_NAMES):
    if not rows:
        raise ValueError("metric aggregation requires at least one row")
    patients = {}
    for row in rows:
        patients.setdefault(row["patient_id"], []).append(row)

    patient_rows = []
    for patient_id in sorted(patients):
        entries = patients[patient_id]
        patient_row = {"patient_id": patient_id, "sample_count": len(entries)}
        for metric in metric_names:
            patient_row[metric] = statistics.fmean(entry[metric] for entry in entries)
        patient_rows.append(patient_row)

    return {
        "sample_count": len(rows),
        "patient_count": len(patient_rows),
        "sample_macro": {
            metric: statistics.fmean(row[metric] for row in rows)
            for metric in metric_names
        },
        "patient_macro": {
            metric: statistics.fmean(row[metric] for row in patient_rows)
            for metric in metric_names
        },
        "patient_metrics": patient_rows,
        "rows": rows,
    }
