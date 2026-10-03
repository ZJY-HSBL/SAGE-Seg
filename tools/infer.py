#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_seg.utils.config import load_config
from sage_seg.data.common import pil_to_tensor
from sage_seg.models.deeplabv3plus import DeepLabV3Plus
from sage_seg.utils.image import imagenet_normalize


def voc_palette(n=256):
    palette = [0] * (n * 3)
    for j in range(n):
        lab, i = j, 0
        while lab:
            palette[j*3+0] |= ((lab >> 0) & 1) << (7-i)
            palette[j*3+1] |= ((lab >> 1) & 1) << (7-i)
            palette[j*3+2] |= ((lab >> 2) & 1) << (7-i)
            i += 1
            lab >>= 3
    return palette


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--use-student", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = DeepLabV3Plus(int(cfg["data"]["num_classes"]), False,
                          int(cfg["model"].get("output_stride", 16)), int(cfg["model"].get("aspp_channels", 256))).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model" if args.use_student else "teacher"], strict=True)
    model.eval()
    image = Image.open(args.input).convert("RGB")
    x = pil_to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(imagenet_normalize(x)).argmax(1)[0].cpu().numpy().astype(np.uint8)
    out = Image.fromarray(pred, mode="P")
    out.putpalette(voc_palette())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
