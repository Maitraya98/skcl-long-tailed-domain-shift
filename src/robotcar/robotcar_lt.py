"""
robotcar_lt.py

Dataset wrapper for the extracted RobotCar crops (car/pedestrian/cyclist),
matching the same interface IMBALANCECIFAR10/100 exposes (get_cls_num_list(),
etc.) so the existing BCL/SKCL training code can point at this data with
minimal changes.

Unlike IMBALANCECIFAR, this does NOT synthesize an artificial exponential
imbalance -- the long-tailedness here is the real, naturally-occurring class
distribution from actual driving footage.
"""
from __future__ import annotations
import csv
import os
from typing import List, Optional, Callable

from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["Car", "Ped", "Cyc"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

ROBOTCAR_MEAN = (0.485, 0.456, 0.406)
ROBOTCAR_STD = (0.229, 0.224, 0.225)


class RobotCarLT(Dataset):
    def __init__(self, manifest_path: str, split: str, transform: Optional[Callable] = None):
        self.transform = transform
        self.split = split
        self.samples = []

        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                class_idx = CLASS_TO_IDX[row["class"]]
                self.samples.append((row["path"], class_idx))

        if not self.samples:
            raise ValueError(f"No samples found for split={split!r} in {manifest_path}")

        self._build_cls_num_list()

    def _build_cls_num_list(self):
        counts = [0] * len(CLASS_NAMES)
        for _, class_idx in self.samples:
            counts[class_idx] += 1
        self.cls_num_list_ = counts

    def get_cls_num_list(self) -> List[int]:
        return self.cls_num_list_

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, class_idx


def get_robotcar_transforms(crop_size: int = 64):
    from torchvision import transforms

    normalize = transforms.Normalize(ROBOTCAR_MEAN, ROBOTCAR_STD)
    pad = max(2, crop_size // 8)

    train_transform = transforms.Compose([
        transforms.RandomCrop(crop_size, padding=pad),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, test_transform


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    for split in ["day_train", "day_test", "night_test"]:
        ds = RobotCarLT(args.manifest, split=split, transform=None)
        counts = ds.get_cls_num_list()
        print(f"{split}: {len(ds)} samples, per-class {dict(zip(CLASS_NAMES, counts))}")
