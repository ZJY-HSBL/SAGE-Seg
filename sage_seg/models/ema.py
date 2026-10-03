from __future__ import annotations
from copy import deepcopy
import torch
from torch import nn


def create_ema_model(student: nn.Module) -> nn.Module:
    teacher = deepcopy(student)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    for key, t_value in teacher_state.items():
        s_value = student_state[key].detach()
        if torch.is_floating_point(t_value):
            t_value.mul_(decay).add_(s_value, alpha=1.0 - decay)
        else:
            t_value.copy_(s_value)
