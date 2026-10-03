import torch
from sage_seg.core.mixing import semantic_mix


def test_semantic_mix_obeys_keep_mask():
    u = torch.ones((1, 3, 2, 2))
    l = torch.zeros((1, 3, 2, 2))
    pseudo = torch.ones((1, 2, 2), dtype=torch.long)
    gt = torch.full((1, 2, 2), 2, dtype=torch.long)
    m = torch.tensor([[[True, False], [False, True]]])
    x, y = semantic_mix(u, l, pseudo, gt, m)
    assert x[0, 0].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert y.tolist() == [[[1, 2], [2, 1]]]
