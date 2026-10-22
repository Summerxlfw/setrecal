"""NPC augmentation from the study implementation."""
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate

def _probability(value, name):
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return value


def npc170_training_augment(
    images,
    masks,
    seed,
    *,
    rot_flip_probability=0.5,
    arbitrary_rotation_probability=0.5,
    noise_probability=0.5,
):
    if images.ndim != 4 or masks.ndim != 4:
        raise ValueError("images and masks must have shapes [B,C,H,W] and [B,R,H,W]")
    if images.shape[0] != masks.shape[0] or images.shape[-2:] != masks.shape[-2:]:
        raise ValueError("images and masks must share batch and spatial shapes")
    if not bool(torch.isfinite(images).all()) or not bool(torch.isfinite(masks).all()):
        raise ValueError("images and masks must be finite")
    probabilities = (
        _probability(rot_flip_probability, "rot_flip_probability"),
        _probability(arbitrary_rotation_probability, "arbitrary_rotation_probability"),
        _probability(noise_probability, "noise_probability"),
    )
    generator = torch.Generator(device=images.device).manual_seed(int(seed))
    augmented_images = []
    augmented_masks = []
    for image, rater_masks in zip(images, masks):
        joined = torch.cat((image, rater_masks.to(image.dtype)), dim=0)
        if float(torch.rand((), generator=generator, device=images.device)) < probabilities[0]:
            quarter_turns = int(
                torch.randint(0, 4, (), generator=generator, device=images.device)
            )
            flip_axis = -2 if int(
                torch.randint(0, 2, (), generator=generator, device=images.device)
            ) == 0 else -1
            joined = torch.rot90(joined, quarter_turns, dims=(-2, -1))
            joined = torch.flip(joined, dims=(flip_axis,))
        if float(torch.rand((), generator=generator, device=images.device)) < probabilities[1]:
            angle = int(
                torch.randint(-20, 20, (), generator=generator, device=images.device)
            )
            joined = rotate(
                joined,
                angle=float(angle),
                interpolation=InterpolationMode.NEAREST,
            )
        image = joined[: images.shape[1]]
        rater_masks = joined[images.shape[1] :].to(masks.dtype)
        if float(torch.rand((), generator=generator, device=images.device)) < probabilities[2]:
            noise = torch.randn(
                image.shape,
                generator=generator,
                device=image.device,
                dtype=image.dtype,
            ).mul_(0.1).clamp_(-0.2, 0.2)
            image = image + noise
        augmented_images.append(image)
        augmented_masks.append(rater_masks)
    return torch.stack(augmented_images), torch.stack(augmented_masks)
