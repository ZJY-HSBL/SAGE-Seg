from __future__ import annotations
from torch.utils.data import DataLoader
from .common import TrainSpatialTransform, EvalTransform
from .voc import VOCSegmentationDataset
from .cityscapes import CityscapesSegmentationDataset


def build_datasets(cfg):
    d = cfg["data"]
    transform = TrainSpatialTransform(
        d["crop_size"], d.get("scale_range", [0.5, 2.0]), d.get("hflip_prob", 0.5), d.get("ignore_index", 255)
    )
    if d["dataset"] == "voc":
        labeled = VOCSegmentationDataset(d["root"], d["labeled_list"], "train", True, transform)
        unlabeled = VOCSegmentationDataset(d["root"], d["unlabeled_list"], "train", False, transform)
        val = VOCSegmentationDataset(d["root"], None, "val", True, EvalTransform())
    elif d["dataset"] == "cityscapes":
        labeled = CityscapesSegmentationDataset(d["root"], d["labeled_list"], "train", True, transform)
        unlabeled = CityscapesSegmentationDataset(d["root"], d["unlabeled_list"], "train", False, transform)
        val = CityscapesSegmentationDataset(d["root"], None, "val", True, EvalTransform())
    else:
        raise ValueError(f"Unknown dataset: {d['dataset']}")
    return labeled, unlabeled, val


def build_loaders(cfg):
    labeled, unlabeled, val = build_datasets(cfg)
    total_batch = int(cfg["train"]["batch_size"])
    if total_batch % 2:
        raise ValueError("train.batch_size must be even; each step uses half labeled and half unlabeled")
    half = total_batch // 2
    workers = int(cfg["data"].get("workers", 4))
    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    labeled_loader = DataLoader(labeled, batch_size=half, shuffle=True, drop_last=True, **common)
    unlabeled_loader = DataLoader(unlabeled, batch_size=half, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val, batch_size=int(cfg.get("validation", {}).get("batch_size", 1)), shuffle=False,
                            num_workers=workers, pin_memory=True)
    return labeled_loader, unlabeled_loader, val_loader
