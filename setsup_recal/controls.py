"""Latent controls in the RIGA-cup comparison."""
import torch
from torch import nn

class TwoScalarMixer(nn.Module):
    """Center/dispersion linear adjustment of the latent set; 2 parameters."""

    def __init__(self, rater_count, device=None, dtype=torch.float32):
        super().__init__()
        if int(rater_count) < 2:
            raise ValueError("two-scalar control needs at least two raters")
        self.rater_count = int(rater_count)
        self.center_scale = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
        self.spread_scale = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))

    def forward(self, expert_latents):
        if expert_latents.ndim != 3:
            raise ValueError("expert_latents must have shape [B,R,L]")
        if expert_latents.shape[1] != self.rater_count:
            raise ValueError("expert latent count does not match mixer rater count")
        mean = expert_latents.mean(dim=1, keepdim=True)
        return (
            expert_latents
            + torch.tanh(self.center_scale) * mean
            + torch.tanh(self.spread_scale) * (expert_latents - mean)
        )


class _LinearGateMixer(nn.Module):
    """Shared skeleton of the 37-parameter controls: identity-init unbiased
    linear W plus a zero-init scalar gate; all residual forms are bitwise
    identity at initialization."""

    def __init__(self, latent_dim, device=None, dtype=torch.float32):
        super().__init__()
        self.projection = nn.Linear(latent_dim, latent_dim, bias=False, device=device, dtype=dtype)
        nn.init.eye_(self.projection.weight)
        self.gate = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))

    @property
    def a(self):
        return torch.tanh(self.gate)


class CenterOnlyMixer(_LinearGateMixer):
    """z'_r = z_r + tanh(g) * W mean(z); the ACTION uniform self-loop limit."""

    def forward(self, expert_latents):
        if expert_latents.ndim != 3:
            raise ValueError("expert_latents must have shape [B,R,L]")
        return expert_latents + self.a * self.projection(
            expert_latents.mean(dim=1, keepdim=True)
        )


class DispersionOnlyMixer(_LinearGateMixer):
    """z'_r = z_r - tanh(g)/(R-1) * W (z_r - mean(z)); exact center complement."""

    def __init__(self, latent_dim, rater_count, device=None, dtype=torch.float32):
        super().__init__(latent_dim, device=device, dtype=dtype)
        if int(rater_count) < 2:
            raise ValueError("dispersion control needs at least two raters")
        self.rater_count = int(rater_count)

    def forward(self, expert_latents):
        if expert_latents.ndim != 3:
            raise ValueError("expert_latents must have shape [B,R,L]")
        if expert_latents.shape[1] != self.rater_count:
            raise ValueError("expert latent count does not match mixer rater count")
        center = expert_latents.mean(dim=1, keepdim=True)
        return expert_latents - (self.a / (self.rater_count - 1)) * self.projection(
            expert_latents - center
        )
