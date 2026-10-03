import torch
from torch.nn import functional as F


def supervised_ce(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255):
    return F.cross_entropy(logits, target.long(), ignore_index=ignore_index)


def per_sample_ce(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    loss = F.cross_entropy(logits, target.long(), ignore_index=ignore_index, reduction="none")
    valid = target.ne(ignore_index)
    num = (loss * valid).flatten(1).sum(1)
    den = valid.flatten(1).sum(1).clamp_min(1)
    return num / den
