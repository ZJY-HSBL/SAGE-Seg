import torch

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = _IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = _IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std
