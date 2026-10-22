"""Fundus augmentation from the study implementation."""
import torch

def _augment(images, masks, seed):
    augmented_images = []
    augmented_masks = []
    generator = torch.Generator(device=images.device).manual_seed(seed)
    for image, target_masks in zip(images, masks):
        rotation = int(
            torch.randint(0, 4, (1,), generator=generator, device=images.device)
        )
        horizontal = bool(
            torch.randint(0, 2, (1,), generator=generator, device=images.device)
        )
        image = torch.rot90(image, rotation, dims=(-2, -1))
        target_masks = torch.rot90(target_masks, rotation, dims=(-2, -1))
        if horizontal:
            image = torch.flip(image, dims=(-1,))
            target_masks = torch.flip(target_masks, dims=(-1,))
        noise = torch.randn(
            image.shape,
            generator=generator,
            device=image.device,
            dtype=image.dtype,
        ).mul_(0.02).clamp_(-0.04, 0.04)
        augmented_images.append((image + noise).clamp(0.0, 1.0))
        augmented_masks.append(target_masks)
    return torch.stack(augmented_images), torch.stack(augmented_masks)
