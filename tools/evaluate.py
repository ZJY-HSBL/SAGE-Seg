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
from sage_seg.data.factory import build_loaders
from sage_seg.models.deeplabv3plus import DeepLabV3Plus
from sage_seg.utils.checkpoint import load_checkpoint
from sage_seg.engine.evaluator import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--use-student", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    cfg = override_data_root(load_config(args.config), args.data_root)
    device = torch.device(args.device)
    model = DeepLabV3Plus(int(cfg["data"]["num_classes"]), False,
                          int(cfg["model"].get("output_stride", 16)), int(cfg["model"].get("aspp_channels", 256))).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    key = "model" if args.use_student else "teacher"
    model.load_state_dict(ckpt[key], strict=True)
    _, _, val_loader = build_loaders(cfg)
    miou, class_iou = evaluate(model, val_loader, device, int(cfg["data"]["num_classes"]),
                               int(cfg["data"].get("ignore_index", 255)))
    print(f"mIoU: {miou:.6f}")
    for i, score in enumerate(class_iou):
        print(f"class {i:02d}: {score:.6f}")


if __name__ == "__main__":
    main()
