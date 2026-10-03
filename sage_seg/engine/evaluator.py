from __future__ import annotations
import torch
from tqdm import tqdm
from sage_seg.metrics.miou import MeanIoU
from sage_seg.utils.image import imagenet_normalize


@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_index=255, desc="validate"):
    was_training = model.training
    model.eval()
    metric = MeanIoU(num_classes, ignore_index)
    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        logits = model(imagenet_normalize(x))
        metric.update(logits.argmax(1), y)
    if was_training:
        model.train()
    return metric.compute()
