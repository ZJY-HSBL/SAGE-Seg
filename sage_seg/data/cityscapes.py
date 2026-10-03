from __future__ import annotations
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import read_id_list, EvalTransform

# Official labelId -> trainId mapping; ignored/void classes map to 255.
_CITYSCAPES_MAPPING = {
    7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9,
    23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18,
}


def _map_train_ids(mask: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(mask, 255)
    for raw, train in _CITYSCAPES_MAPPING.items():
        out[mask == raw] = train
    return out


class CityscapesSegmentationDataset(Dataset):
    def __init__(self, root, split_file=None, split="train", labeled=True, transform=None):
        self.root = Path(root)
        self.split = split
        self.labeled = bool(labeled)
        self.transform = transform or EvalTransform()
        if split_file:
            self.ids = read_id_list(split_file)
        else:
            base = self.root / "leftImg8bit" / split
            self.ids = []
            for p in sorted(base.glob("*/*_leftImg8bit.png")):
                rel = p.relative_to(base).as_posix()
                self.ids.append(rel[:-len("_leftImg8bit.png")])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        city, stem = sample_id.split("/", 1)
        image_path = self.root / "leftImg8bit" / self.split / city / f"{stem}_leftImg8bit.png"
        image = Image.open(image_path).convert("RGB")
        mask = None
        if self.labeled:
            mask_path = self.root / "gtFine" / self.split / city / f"{stem}_gtFine_labelIds.png"
            mask = Image.open(mask_path)
        image, mask = self.transform(image, mask)
        out = {"image": image, "id": sample_id}
        if mask is not None:
            out["mask"] = _map_train_ids(mask.long())
        return out
