import torch


def semantic_mix(
    unlabeled_image: torch.Tensor,
    labeled_image: torch.Tensor,
    pseudo_label: torch.Tensor,
    ground_truth: torch.Tensor,
    keep_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix aligned images and labels according to M.

    M=1 keeps unlabeled content; M=0 inserts labeled content.
    """
    if keep_mask.ndim != 3:
        raise ValueError("keep_mask must have shape [B,H,W]")
    m_img = keep_mask.unsqueeze(1).to(dtype=unlabeled_image.dtype)
    mixed_image = m_img * unlabeled_image + (1.0 - m_img) * labeled_image
    mixed_label = torch.where(keep_mask, pseudo_label, ground_truth)
    return mixed_image, mixed_label
