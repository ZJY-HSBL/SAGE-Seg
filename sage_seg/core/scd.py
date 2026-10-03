import torch
from torch.nn import functional as F


def semantic_consistency_mask(
    student_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    threshold: float = 0.7,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E and cosine similarity maps.

    E=1 marks semantically consistent pixels; E=0 marks candidate replacement pixels.
    Expected input shape: [B, C, H, W].
    """
    if student_prob.shape != teacher_prob.shape:
        raise ValueError("student_prob and teacher_prob must have identical shapes")
    dot = (student_prob * teacher_prob).sum(dim=1)
    s_norm = torch.linalg.vector_norm(student_prob, dim=1)
    t_norm = torch.linalg.vector_norm(teacher_prob, dim=1)
    similarity = dot / (s_norm * t_norm + eps)
    mask = similarity > threshold
    return mask, similarity
