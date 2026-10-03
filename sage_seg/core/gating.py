import math
import torch


def normalized_entropy_confidence(prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute sample confidence g = 1 - mean_entropy/log(C).

    Input shape: [B, C, H, W]. Output shape: [B], bounded in [0, 1].
    """
    if prob.ndim != 4:
        raise ValueError("prob must have shape [B, C, H, W]")
    c = prob.shape[1]
    if c <= 1:
        return torch.ones(prob.shape[0], device=prob.device, dtype=prob.dtype)
    entropy = -(prob.clamp_min(eps) * prob.clamp_min(eps).log()).sum(dim=1)
    mean_entropy = entropy.mean(dim=(1, 2))
    confidence = 1.0 - mean_entropy / math.log(c)
    return confidence.clamp_(0.0, 1.0)


def sample_bernoulli_gate(confidence: torch.Tensor) -> torch.Tensor:
    """z~Bernoulli(g). True chooses strong augmentation; False chooses semantic mixing."""
    return torch.bernoulli(confidence.clamp(0, 1)).bool()
