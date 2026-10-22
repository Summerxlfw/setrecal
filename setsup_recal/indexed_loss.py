"""Indexed Dice objective used by the Stage-II implementation."""
import torch

def generalized_dice_loss(logits, targets, eps=1e-6):
    if logits.shape != targets.shape:
        raise ValueError("dice logits and targets must share shape")
    channel_count = logits.shape[1]
    probabilities = logits.sigmoid().permute(1, 0, 2, 3).contiguous().view(
        channel_count, -1
    )
    targets = (
        targets.to(probabilities.dtype)
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(channel_count, -1)
    )
    if channel_count == 1:
        probabilities = torch.cat((probabilities, 1.0 - probabilities), dim=0)
        targets = torch.cat((targets, 1.0 - targets), dim=0)
    weights = 1.0 / targets.sum(-1).square().clamp(min=eps)
    intersection = (probabilities * targets).sum(-1) * weights
    denominator = ((probabilities + targets).sum(-1) * weights).clamp(min=eps)
    return 1.0 - (2.0 * intersection / denominator).mean()
