import torch
from sage_seg.core.gating import normalized_entropy_confidence, sample_bernoulli_gate


def test_entropy_confidence_range():
    uniform = torch.full((1, 4, 2, 2), 0.25)
    confident = torch.zeros((1, 4, 2, 2))
    confident[:, 0] = 1.0
    g0 = normalized_entropy_confidence(uniform)
    g1 = normalized_entropy_confidence(confident)
    assert g0.item() < 1e-5
    assert g1.item() > 0.999


def test_gate_shape():
    g = torch.tensor([0.0, 1.0])
    z = sample_bernoulli_gate(g)
    assert z.dtype == torch.bool
    assert z.tolist() == [False, True]
