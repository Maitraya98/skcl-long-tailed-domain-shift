"""
robotcar_multiview.py

Multi-view wrappers for RobotCarLT, mirroring cifar_lt_multiview.py's design
but adapted for 64x64 real photographic crops (ImageNet-style normalization,
proportionally larger crop/augmentation parameters than CIFAR's 32x32 recipe).
"""
from __future__ import annotations
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from robotcar_lt import RobotCarLT, ROBOTCAR_MEAN, ROBOTCAR_STD


def build_robotcar_bcl_transforms(crop_size: int = 64) -> List[transforms.Compose]:
    normalize = transforms.Normalize(ROBOTCAR_MEAN, ROBOTCAR_STD)
    pad = max(2, crop_size // 8)

    ce_view = transforms.Compose([
        transforms.RandomCrop(crop_size, padding=pad),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    sim_view = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    return [ce_view, sim_view, sim_view]


class MultiViewRobotCar(Dataset):
    def __init__(self, base_dataset, view_transforms):
        self.base_dataset = base_dataset
        self.view_transforms = view_transforms

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        path, class_idx = self.base_dataset.samples[idx]
        img = Image.open(path).convert("RGB")
        views = tuple(t(img) for t in self.view_transforms)
        return views, class_idx

    def get_cls_num_list(self):
        return self.base_dataset.get_cls_num_list()


class MultiViewRobotCarWithCoarse(Dataset):
    def __init__(self, base_dataset, view_transforms, fine_to_coarse):
        self.base_dataset = base_dataset
        self.view_transforms = view_transforms
        self.fine_to_coarse = fine_to_coarse

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        path, fine_class_idx = self.base_dataset.samples[idx]
        img = Image.open(path).convert("RGB")
        views = tuple(t(img) for t in self.view_transforms)
        coarse_class_idx = self.fine_to_coarse[fine_class_idx]
        return views, fine_class_idx, coarse_class_idx

    def get_cls_num_list(self):
        return self.base_dataset.get_cls_num_list()
