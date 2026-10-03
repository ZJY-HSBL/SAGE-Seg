from __future__ import annotations
import random
from PIL import Image, ImageOps
import numpy as np
import torch


class TrainSpatialTransform:
    def __init__(self, crop_size, scale_range=(0.5, 2.0), hflip_prob=0.5, ignore_index=255):
        self.crop_h, self.crop_w = map(int, crop_size)
        self.scale_range = tuple(map(float, scale_range))
        self.hflip_prob = float(hflip_prob)
        self.ignore_index = int(ignore_index)

    def __call__(self, image: Image.Image, mask: Image.Image | None = None):
        scale = random.uniform(*self.scale_range)
        new_w = max(1, round(image.width * scale))
        new_h = max(1, round(image.height * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((new_w, new_h), Image.NEAREST)

        pad_w = max(0, self.crop_w - new_w)
        pad_h = max(0, self.crop_h - new_h)
        if pad_w or pad_h:
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
            if mask is not None:
                mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=self.ignore_index)

        x1 = random.randint(0, image.width - self.crop_w)
        y1 = random.randint(0, image.height - self.crop_h)
        box = (x1, y1, x1 + self.crop_w, y1 + self.crop_h)
        image = image.crop(box)
        if mask is not None:
            mask = mask.crop(box)

        if random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image_t = pil_to_tensor(image)
        if mask is None:
            return image_t, None
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_t, mask_t


class EvalTransform:
    def __call__(self, image: Image.Image, mask: Image.Image | None = None):
        image_t = pil_to_tensor(image)
        mask_t = None if mask is None else torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_t, mask_t


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def read_id_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]
