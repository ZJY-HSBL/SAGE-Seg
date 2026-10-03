import torch
from sage_seg.core.scd import semantic_consistency_mask


def test_identical_probabilities_are_consistent():
    p = torch.tensor([[[[0.9]], [[0.1]]]])
    mask, sim = semantic_consistency_mask(p, p, threshold=0.7)
    assert bool(mask.item())
    assert sim.item() > 0.999


def test_orthogonal_probabilities_are_inconsistent():
    s = torch.tensor([[[[1.0]], [[0.0]]]])
    t = torch.tensor([[[[0.0]], [[1.0]]]])
    mask, sim = semantic_consistency_mask(s, t, threshold=0.7)
    assert not bool(mask.item())
    assert sim.item() < 1e-6
