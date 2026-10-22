"""Interface to the separately obtained D-Persona source and Stage-I weights."""
import importlib,sys
from pathlib import Path
import torch

def _draw_from_distribution(distribution, generator):
    base = distribution.base_dist
    noise = torch.randn(
        base.loc.shape,
        generator=generator,
        device=base.loc.device,
        dtype=base.loc.dtype,
    )
    return base.loc + base.scale * noise


class DPersonaStage1Adapter(torch.nn.Module):
    def __init__(self, core, regularizer):
        super().__init__()
        self.core = core
        self._regularizer = regularizer

    def _decode(self, latent):
        return self.core.fcomb.forward(
            self.core.unet_features,
            latent,
            use_softmax=False,
        )

    def training_outputs(
        self,
        images,
        all_masks,
        rater_indices,
        num_samples,
        generator,
    ):
        del rater_indices
        self.core.forward(images, all_masks, training=True)
        posterior_logits = self._decode(
            _draw_from_distribution(self.core.posterior_latent_space, generator)
        )
        prior_logits = torch.cat(
            [
                self._decode(
                    _draw_from_distribution(
                        self.core.prior_latent_space,
                        generator,
                    )
                )
                for _ in range(num_samples)
            ],
            dim=1,
        )
        kl_value = torch.mean(self.core.kl_divergence(analytic=True))
        regularization = self.core.reg_factor * (
            self._regularizer(self.core.posterior)
            + self._regularizer(self.core.prior)
            + self._regularizer(self.core.fcomb.layers)
        )
        return {
            "posterior_logits": posterior_logits,
            "prior_logits": prior_logits,
            "kl": kl_value,
            "regularization": regularization,
        }

    def sample_prior(self, images, num_samples, generator):
        self.core.forward(images, None, training=False)
        logits = torch.cat(
            [
                self._decode(
                    _draw_from_distribution(
                        self.core.prior_latent_space,
                        generator,
                    )
                )
                for _ in range(num_samples)
            ],
            dim=1,
        )
        return logits.sigmoid()

    def trainable_parameters(self):
        for module in (
            self.core.unet,
            self.core.prior,
            self.core.posterior,
            self.core.fcomb,
        ):
            yield from module.parameters()

    def shared_parameters(self):
        for module in (self.core.unet, self.core.prior, self.core.fcomb):
            yield from module.parameters()


def build_dpersona_stage1_adapter(
    *,
    source_root,
    input_channels,
    rater_count,
    device,
):
    source_root = Path(source_root).resolve()
    code_root = source_root / "D-Persona" / "code"
    if not (source_root / "LICENSE").is_file() or not code_root.is_dir():
        raise FileNotFoundError(f"invalid D-Persona source snapshot: {source_root}")
    if rater_count < 2:
        raise ValueError("D-Persona Stage I requires at least two raters")
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    released = importlib.import_module("lib.DPersona")
    regularizer = importlib.import_module(
        "Probabilistic_Unet_Pytorch.utils"
    ).l2_regularisation
    core = released.DPersona(
        input_channels=input_channels,
        num_classes=1,
        latent_dim=6,
        no_convs_fcomb=4,
        num_experts=rater_count,
        reg_factor=0.00001,
        original_backbone=False,
    )
    if rater_count != 4:
        core.posterior = released.AxisAlignedConvGaussian(
            input_channels + rater_count - 4,
            core.num_filters,
            core.no_convs_per_block,
            core.latent_dim,
            core.initializers,
            posterior=True,
        )
    return DPersonaStage1Adapter(core, regularizer).to(device)
