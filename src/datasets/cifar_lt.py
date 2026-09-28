"""
cifar_lt.py

Long-tailed variants of CIFAR-10 and CIFAR-100, built as a shared
data-loading component for BCL, ConCutMix, and SKCL (Part 1 reproduction).

Follows the standard exponential long-tailed construction protocol used
across the long-tailed recognition literature (Cui et al. 2019, Cao et al.
2019, and subsequently BCL / ConCutMix / SKCL), so that resulting per-class
sample counts match what those papers report their numbers against.

Imbalance factor convention
----------------------------
The paper reports results at imbalance factor beta = 100. In code this is
expressed as `imb_factor = 1 / beta` (i.e. imb_factor=0.01 for beta=100),
since imb_factor is defined as N_min / N_max.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torchvision


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class _LongTailedCIFARMixin:
    cls_num: int

    def __init__(self, *args, imb_type: str = "exp", imb_factor: float = 0.01,
                 rand_number: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        np.random.seed(rand_number)
        img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.img_num_per_cls = img_num_per_cls
        self.num_per_cls_dict: dict = {}
        self._gen_imbalanced_data(img_num_per_cls)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor) -> List[int]:
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == "exp":
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            half = cls_num // 2
            img_num_per_cls.extend([int(img_max)] * half)
            img_num_per_cls.extend([int(img_max * imb_factor)] * (cls_num - half))
        elif imb_type == "none":
            img_num_per_cls.extend([int(img_max)] * cls_num)
        else:
            raise ValueError(f"Unknown imb_type: {imb_type!r}")
        return img_num_per_cls

    def _gen_imbalanced_data(self, img_num_per_cls: Sequence[int]) -> None:
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        if len(classes) != len(img_num_per_cls):
            raise RuntimeError(
                f"Dataset has {len(classes)} classes but img_num_per_cls has "
                f"{len(img_num_per_cls)} entries."
            )
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[int(the_class)] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class] * the_img_num)
        self.data = np.vstack(new_data)
        self.targets = new_targets

    def get_cls_num_list(self) -> List[int]:
        return [self.num_per_cls_dict[i] for i in range(self.cls_num)]


class IMBALANCECIFAR10(_LongTailedCIFARMixin, torchvision.datasets.CIFAR10):
    cls_num = 10


class IMBALANCECIFAR100(_LongTailedCIFARMixin, torchvision.datasets.CIFAR100):
    cls_num = 100


def classify_many_medium_few(cls_num_list: Sequence[int], many_shot_thresh: int = 100,
                              few_shot_thresh: int = 20) -> Tuple[List[int], List[int], List[int]]:
    many, medium, few = [], [], []
    for cls_idx, n in enumerate(cls_num_list):
        if n > many_shot_thresh:
            many.append(cls_idx)
        elif n < few_shot_thresh:
            few.append(cls_idx)
        else:
            medium.append(cls_idx)
    return many, medium, few


def get_transforms(dataset: str):
    from torchvision import transforms
    if dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, test_transform


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--imb_factor", type=float, default=0.01)
    args = parser.parse_args()
    cls = IMBALANCECIFAR100 if args.dataset == "cifar100" else IMBALANCECIFAR10
    ds = cls(root=args.root, imb_type="exp", imb_factor=args.imb_factor, train=True, download=False)
    cls_num_list = ds.get_cls_num_list()
    many, medium, few = classify_many_medium_few(cls_num_list)
    print(f"{args.dataset}-LT  (imb_factor={args.imb_factor}, beta={1/args.imb_factor:.0f})")
    print(f"  Total images: {sum(cls_num_list)}")
    print(f"  Max/min class count: {max(cls_num_list)} / {min(cls_num_list)}")
    print(f"  Many-shot classes:   {len(many)}")
    print(f"  Medium-shot classes: {len(medium)}")
    print(f"  Few-shot classes:    {len(few)}")
