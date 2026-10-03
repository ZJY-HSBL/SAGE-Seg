#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import random


def collect_ids(dataset: str, root: Path):
    if dataset == "voc":
        path = root / "ImageSets" / "Segmentation" / "train.txt"
        return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if dataset == "cityscapes":
        base = root / "leftImg8bit" / "train"
        ids = []
        for p in sorted(base.glob("*/*_leftImg8bit.png")):
            rel = p.relative_to(base).as_posix()
            ids.append(rel[:-len("_leftImg8bit.png")])
        return ids
    raise ValueError(dataset)


def fraction_name(fraction: float):
    inv = round(1.0 / fraction)
    if abs(fraction - 1.0 / inv) < 1e-9:
        return f"1_{inv}"
    return str(fraction).replace(".", "p")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["voc", "cityscapes"], required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--fraction", type=float, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not (0 < args.fraction < 1):
        raise ValueError("--fraction must be between 0 and 1")
    ids = collect_ids(args.dataset, Path(args.root))
    rng = random.Random(args.seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_labeled = max(1, round(len(shuffled) * args.fraction))
    labeled = sorted(shuffled[:n_labeled])
    unlabeled = sorted(shuffled[n_labeled:])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = fraction_name(args.fraction)
    lp = out / f"labeled_{name}.txt"
    up = out / f"unlabeled_{name}.txt"
    lp.write_text("\n".join(labeled) + "\n", encoding="utf-8")
    up.write_text("\n".join(unlabeled) + "\n", encoding="utf-8")
    print(f"total={len(ids)} labeled={len(labeled)} unlabeled={len(unlabeled)}")
    print(lp)
    print(up)


if __name__ == "__main__":
    main()
