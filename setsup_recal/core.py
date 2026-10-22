"""Recalibration and set-summary losses; use the public factories in api.py."""
import torch
from torch import nn
from .indexed_loss import generalized_dice_loss

RG_PERSONA_ARMS = (
    "dpersona_stage2",
    "dpersona_stage2_rrm",
    "rg_persona",
)


def _validate_rater_set(masks):
    if masks.ndim != 4:
        raise ValueError("rater masks must have shape [B,R,H,W]")
    if masks.shape[1] < 2:
        raise ValueError("rater relation modeling requires at least two raters")
    if not bool(torch.isfinite(masks).all()):
        raise ValueError("rater masks must be finite")


def pairwise_soft_iou(masks, eps=1e-6):
    _validate_rater_set(masks)
    flattened = masks.to(torch.float32).flatten(start_dim=2)
    intersection = torch.bmm(flattened, flattened.transpose(1, 2))
    sizes = flattened.sum(dim=2)
    union = sizes.unsqueeze(2) + sizes.unsqueeze(1) - intersection
    return (intersection + eps) / (union + eps)


class RaterRelationMixer(nn.Module):
    def __init__(self, relation_prior, latent_dim):
        super().__init__()
        relation_prior = torch.as_tensor(relation_prior, dtype=torch.float32)
        if relation_prior.ndim != 2 or relation_prior.shape[0] != relation_prior.shape[1]:
            raise ValueError("relation_prior must be a square matrix")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.register_buffer("relation_prior", relation_prior)
        self.projection = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.eye_(self.projection.weight)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, expert_latents):
        if expert_latents.ndim != 3:
            raise ValueError("expert_latents must have shape [B,R,L]")
        if expert_latents.shape[1] != self.relation_prior.shape[0]:
            raise ValueError("expert latent count does not match relation prior")
        messages = torch.matmul(
            self.relation_prior.unsqueeze(0),
            self.projection(expert_latents),
        )
        return expert_latents + torch.tanh(self.gate) * messages


def _soft_center_dice_loss(probabilities, targets, eps=1e-6):
    predicted_center = probabilities.mean(dim=1).flatten(start_dim=1)
    target_center = targets.to(probabilities.dtype).mean(dim=1).flatten(start_dim=1)
    intersection = (predicted_center * target_center).sum(dim=1)
    denominator = predicted_center.square().sum(dim=1) + target_center.square().sum(
        dim=1
    )
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return (1.0 - dice).mean()


def center_relation_calibration(probabilities, targets):
    _validate_rater_set(probabilities)
    _validate_rater_set(targets)
    if probabilities.shape != targets.shape:
        raise ValueError("predicted and target rater sets must share shape")
    predicted_relation = pairwise_soft_iou(probabilities)
    target_relation = pairwise_soft_iou(targets)
    count = probabilities.shape[1]
    off_diagonal = ~torch.eye(
        count,
        dtype=torch.bool,
        device=probabilities.device,
    ).unsqueeze(0)
    relation_loss = (
        predicted_relation.sub(target_relation).square().masked_select(off_diagonal)
    ).mean()
    return {
        "center_loss": _soft_center_dice_loss(probabilities, targets),
        "relation_loss": relation_loss,
        "predicted_relation": predicted_relation,
        "target_relation": target_relation,
    }


def compute_stage2_objective(
    logits,
    targets,
    *,
    arm,
    center_weight=1.0,
    relation_weight=1.0,
):
    if arm not in RG_PERSONA_ARMS:
        raise ValueError(f"unknown RG-Persona arm: {arm}")
    if logits.shape != targets.shape:
        raise ValueError("stage-II logits and targets must share shape")
    if min(float(center_weight), float(relation_weight)) < 0.0:
        raise ValueError("calibration weights must be non-negative")
    _validate_rater_set(targets)
    base = generalized_dice_loss(logits, targets)
    if arm != "rg_persona":
        center = logits.sum() * 0.0
        relation = logits.sum() * 0.0
    else:
        calibration = center_relation_calibration(logits.sigmoid(), targets)
        center = calibration["center_loss"]
        relation = calibration["relation_loss"]
    return {
        "total": base
        + float(center_weight) * center
        + float(relation_weight) * relation,
        "base": base,
        "center": center,
        "relation": relation,
    }
