from __future__ import annotations
import random
import torch
from torchvision.transforms import functional as TF


def weak_photometric(x: torch.Tensor, strength: float = 0.08) -> torch.Tensor:
    """Light aligned photometric perturbation for weak teacher/student views."""
    if strength <= 0:
        return x
    b = x.shape[0]
    brightness = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * strength
    contrast = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * strength
    mean = x.mean(dim=(2, 3), keepdim=True)
    y = (x - mean) * contrast + mean
    y = y * brightness
    noise = torch.randn_like(y) * (strength * 0.02)
    return (y + noise).clamp(0.0, 1.0)


def _augment_one(img: torch.Tensor, op: str) -> torch.Tensor:
    if op == "color_jitter":
        img = TF.adjust_brightness(img, random.uniform(0.6, 1.4))
        img = TF.adjust_contrast(img, random.uniform(0.6, 1.4))
        img = TF.adjust_saturation(img, random.uniform(0.6, 1.4))
        return img
    if op == "gaussian_blur":
        return TF.gaussian_blur(img, kernel_size=[5, 5], sigma=[0.1, 2.0])
    if op == "sharpness":
        return TF.adjust_sharpness(img, random.uniform(1.5, 2.5))
    raise ValueError(op)


def strong_augment(x: torch.Tensor) -> torch.Tensor:
    """Randomly choose one strong photometric operation for each sample."""
    ops = ("color_jitter", "gaussian_blur", "sharpness")
    out = []
    for img in x:
        out.append(_augment_one(img, random.choice(ops)))
    return torch.stack(out, dim=0).clamp(0.0, 1.0)
