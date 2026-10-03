from __future__ import annotations
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from .common import read_id_list, TrainSpatialTransform, EvalTransform


class VOCSegmentationDataset(Dataset):
    def __init__(self, root, split_file=None, split="train", labeled=True, transform=None):
        self.root = Path(root)
        self.labeled = bool(labeled)
        if split_file:
            self.ids = read_id_list(split_file)
        else:
            self.ids = read_id_list(self.root / "ImageSets" / "Segmentation" / f"{split}.txt")
        self.transform = transform or EvalTransform()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        image = Image.open(self.root / "JPEGImages" / f"{sample_id}.jpg").convert("RGB")
        mask = None
        if self.labeled:
            mask = Image.open(self.root / "SegmentationClass" / f"{sample_id}.png")
        image, mask = self.transform(image, mask)
        out = {"image": image, "id": sample_id}
        if mask is not None:
            out["mask"] = mask
        return out
