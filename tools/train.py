#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_seg.utils.config import load_config, override_data_root
from sage_seg.utils.seed import seed_everything
from sage_seg.data.factory import build_loaders
from sage_seg.models.deeplabv3plus import DeepLabV3Plus
from sage_seg.models.ema import create_ema_model
from sage_seg.engine.trainer import Trainer
from sage_seg.utils.checkpoint import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = override_data_root(load_config(args.config), args.data_root)
    seed_everything(int(cfg["train"].get("seed", 0)))
    device = torch.device(args.device)

    model = DeepLabV3Plus(
        num_classes=int(cfg["data"]["num_classes"]),
        pretrained_backbone=bool(cfg["model"].get("pretrained_backbone", True)),
        output_stride=int(cfg["model"].get("output_stride", 16)),
        aspp_channels=int(cfg["model"].get("aspp_channels", 256)),
    ).to(device)
    teacher = create_ema_model(model).to(device)

    groups = model.parameter_groups(float(cfg["train"]["lr"]), float(cfg["train"].get("head_lr_multiplier", 10.0)))
    optimizer = torch.optim.SGD(groups, momentum=float(cfg["train"].get("momentum", 0.9)),
                                weight_decay=float(cfg["train"].get("weight_decay", 1e-5)))
    labeled_loader, unlabeled_loader, val_loader = build_loaders(cfg)

    start_epoch, best_miou = 0, 0.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, teacher, optimizer, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0))
        best_miou = float(ckpt.get("best_miou", 0.0))
        print(f"Resumed from epoch {start_epoch}, best mIoU={best_miou:.4f}")

    trainer = Trainer(cfg, model, teacher, optimizer, labeled_loader, unlabeled_loader, val_loader, device)
    trainer.fit(start_epoch=start_epoch, best_miou=best_miou)


if __name__ == "__main__":
    main()
