"""Public entry points for SetSup+Recal."""
import torch
from .core import RaterRelationMixer, compute_stage2_objective
from .adapter import DPersonaStage2Adapter


def self_recalibration(rater_count, latent_dim=6):
    """Create z_r + tanh(g) W z_r with no cross-rater input."""
    if not isinstance(rater_count, int) or rater_count < 2:
        raise ValueError("rater_count must be an integer >= 2")
    return RaterRelationMixer(torch.eye(rater_count), latent_dim)


def uniform_mixing(rater_count, latent_dim=6):
    """Uniform off-diagonal mixing control, with zero self-edge weights."""
    if not isinstance(rater_count, int) or rater_count < 2:
        raise ValueError("rater_count must be an integer >= 2")
    relation=(torch.ones(rater_count,rater_count)-torch.eye(rater_count))/(rater_count-1)
    return RaterRelationMixer(relation,latent_dim)


def stage2_adapter(core):
    """Wrap an initialized D-Persona core; load the Stage-I weights first.

    The core exposes latent_dim, num_experts, unet, prior, posterior,
    fcomb, proj_heads, and the forward interface in adapter.py.
    """
    return DPersonaStage2Adapter(core,torch.eye(core.num_experts))


def set_supervised_loss(logits, targets, *, mean_weight, pair_weight):
    """Return total, base, center (mean-mask), and relation (pairwise) losses."""
    return compute_stage2_objective(logits,targets,arm="rg_persona",
        center_weight=mean_weight,relation_weight=pair_weight)
