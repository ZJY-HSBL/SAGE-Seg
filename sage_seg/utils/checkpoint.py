from __future__ import annotations
from pathlib import Path
import torch


def save_checkpoint(path, model, teacher, optimizer, epoch, best_miou, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_miou": float(best_miou),
        "config": cfg,
    }, path)


def load_checkpoint(path, model, teacher=None, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if teacher is not None and "teacher" in ckpt:
        teacher.load_state_dict(ckpt["teacher"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
