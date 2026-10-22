"""Integration with an existing D-Persona core."""
import torch
from .core import RG_PERSONA_ARMS, RaterRelationMixer

def _draw_candidate_set(distribution, sample_count, generator):
    if sample_count < 1:
        raise ValueError("candidate sample count must be positive")
    base = distribution.base_dist
    samples = []
    for _ in range(sample_count):
        noise = torch.randn(
            base.loc.shape,
            generator=generator,
            device=base.loc.device,
            dtype=base.loc.dtype,
        )
        samples.append((base.loc + base.scale * noise).unsqueeze(1))
    return torch.cat(samples, dim=1)


class DPersonaStage2Adapter(torch.nn.Module):
    def __init__(self, core, relation_prior):
        super().__init__()
        self.core = core
        self.relation_mixer = RaterRelationMixer(
            relation_prior,
            latent_dim=core.latent_dim,
        )
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        for parameter in self.core.proj_heads.parameters():
            parameter.requires_grad_(True)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        for component in (
            self.core.unet,
            self.core.prior,
            self.core.posterior,
            self.core.fcomb,
        ):
            component.eval()
        self.core.proj_heads.train(mode)
        self.relation_mixer.train(mode)
        return self

    def persona_logits(
        self,
        images,
        *,
        candidate_samples,
        generator,
        use_graph,
    ):
        with torch.no_grad():
            self.core.forward(images, None, training=False)
            candidate_latents = _draw_candidate_set(
                self.core.prior_latent_space,
                candidate_samples,
                generator,
            )
        devices = [] if images.device.type == "cpu" else [images.device.index]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(generator.initial_seed()) + 1_000_003)
            expert_latents = torch.cat(
                [
                    self.core.proj_heads(
                        self.core.unet_features,
                        candidate_latents,
                        index,
                    )
                    for index in range(self.core.num_experts)
                ],
                dim=1,
            )
        if use_graph:
            expert_latents = self.relation_mixer(expert_latents)
        logits = []
        for index in range(self.core.num_experts):
            logits.append(
                self.core.fcomb.forward(
                    self.core.unet_features,
                    expert_latents[:, index],
                    use_softmax=False,
                ).squeeze(1)
            )
        return torch.stack(logits, dim=1)

    def sample_personas(
        self,
        images,
        *,
        candidate_samples,
        generator,
        use_graph,
    ):
        return self.persona_logits(
            images,
            candidate_samples=candidate_samples,
            generator=generator,
            use_graph=use_graph,
        ).sigmoid()

    def trainable_parameters(self, arm):
        if arm not in RG_PERSONA_ARMS:
            raise ValueError(f"unknown RG-Persona arm: {arm}")
        yield from self.core.proj_heads.parameters()
        if arm != "dpersona_stage2":
            yield from self.relation_mixer.parameters()
