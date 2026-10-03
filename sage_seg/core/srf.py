from __future__ import annotations
import numpy as np
import torch
from scipy import ndimage

_FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def sparse_region_filter(consistency_mask: torch.Tensor, area_threshold: int) -> torch.Tensor:
    """Filter fragmented low-consistency regions using 4-connected components.

    Input E: True/1 = consistent, False/0 = inconsistent.
    Output M: True/1 = keep unlabeled content, False/0 = replace with labeled content.
    Inconsistent components smaller than ``area_threshold`` are suppressed (set back to 1).
    """
    if consistency_mask.ndim != 3:
        raise ValueError("consistency_mask must have shape [B, H, W]")
    if area_threshold < 1:
        return consistency_mask.bool()

    device = consistency_mask.device
    masks = consistency_mask.detach().to("cpu").numpy().astype(bool)
    outputs = np.ones_like(masks, dtype=bool)
    for b, e in enumerate(masks):
        low = ~e
        labels, num = ndimage.label(low, structure=_FOUR_CONNECTED)
        if num == 0:
            continue
        areas = np.bincount(labels.reshape(-1), minlength=num + 1)
        keep_labels = np.flatnonzero(areas >= area_threshold)
        keep_labels = keep_labels[keep_labels != 0]
        if keep_labels.size:
            replace = np.isin(labels, keep_labels)
            outputs[b][replace] = False
    return torch.from_numpy(outputs).to(device=device)
