"""
cifar_lt_multiview.py

BCL's training loop expects each sample as a tuple of 3 augmented views:
one for the cross-entropy branch, two similarity-view crops for the
contrastive branch. Augmentation recalibrated for 32x32 CIFAR rather than
the original repo's RandAugment pipeline (tuned for 224x224 ImageNet).
"""

from __future__ import annotations

from typing import List, Tuple

from torch.utils.data import Dataset
from torchvision import transforms

from cifar_lt import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD


def build_cifar_bcl_transforms(dataset: str) -> List[transforms.Compose]:
    mean, std = (CIFAR10_MEAN, CIFAR10_STD) if dataset == "cifar10" else (CIFAR100_MEAN, CIFAR100_STD)
    normalize = transforms.Normalize(mean, std)

    ce_view = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    sim_view = transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return [ce_view, sim_view, sim_view]


class MultiViewCIFAR(Dataset):
    def __init__(self, base_dataset, view_transforms: List[transforms.Compose]):
        self.base_dataset = base_dataset
        self.view_transforms = view_transforms

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx) -> Tuple[Tuple, int]:
        img, target = self.base_dataset.data[idx], self.base_dataset.targets[idx]
        from PIL import Image
        img = Image.fromarray(img)
        views = tuple(t(img) for t in self.view_transforms)
        return views, target

    def get_cls_num_list(self):
        return self.base_dataset.get_cls_num_list()
