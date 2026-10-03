import torch
from sage_seg.core.srf import sparse_region_filter


def test_small_fragment_is_filtered_but_large_region_is_kept():
    # False regions: one isolated pixel and one 2x2 component.
    e = torch.ones((1, 5, 5), dtype=torch.bool)
    e[0, 0, 0] = False
    e[0, 2:4, 2:4] = False
    m = sparse_region_filter(e, area_threshold=3)
    assert bool(m[0, 0, 0])  # small region removed from replacement set
    assert not bool(m[0, 2, 2])
    assert not bool(m[0, 3, 3])
