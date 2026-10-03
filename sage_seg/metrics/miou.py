from __future__ import annotations
import torch


class MeanIoU:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.detach().to("cpu").long().reshape(-1)
        target = target.detach().to("cpu").long().reshape(-1)
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        pred = pred[valid]
        target = target[valid]
        idx = target * self.num_classes + pred
        hist = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion += hist.reshape(self.num_classes, self.num_classes)

    def compute(self):
        inter = self.confusion.diag()
        union = self.confusion.sum(1) + self.confusion.sum(0) - inter
        iou = inter / union.clamp_min(1)
        valid = union > 0
        miou = iou[valid].mean().item() if valid.any() else 0.0
        return miou, iou.tolist()
