"""
cifar_hierarchy.py -- fine/coarse hierarchy for CIFAR-100 (read from
official data) and CIFAR-10 (paper's Fig. 5 vehicles/animals grouping).
"""
from __future__ import annotations
import os
import pickle
from typing import Dict, List, Tuple

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR10_COARSE_GROUPS = {
    "vehicles": ["airplane", "automobile", "ship", "truck"],
    "animals": ["bird", "cat", "deer", "dog", "frog", "horse"],
}


def get_cifar10_hierarchy() -> Tuple[List[str], List[str], List[int]]:
    coarse_names = list(CIFAR10_COARSE_GROUPS.keys())
    fine_to_coarse = []
    for cls in CIFAR10_CLASSES:
        for coarse_idx, (coarse_name, members) in enumerate(CIFAR10_COARSE_GROUPS.items()):
            if cls in members:
                fine_to_coarse.append(coarse_idx)
                break
        else:
            raise ValueError(f"CIFAR-10 class {cls!r} not found in any coarse group")
    return CIFAR10_CLASSES, coarse_names, fine_to_coarse


def get_cifar100_hierarchy(data_root: str) -> Tuple[List[str], List[str], List[int]]:
    base = os.path.join(data_root, "cifar-100-python")
    meta_path = os.path.join(base, "meta")
    train_path = os.path.join(base, "train")

    if not os.path.isfile(meta_path) or not os.path.isfile(train_path):
        raise FileNotFoundError(f"Expected CIFAR-100 files at {base} (meta, train).")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f, encoding="latin1")
    fine_names = meta["fine_label_names"]
    coarse_names = meta["coarse_label_names"]

    with open(train_path, "rb") as f:
        train = pickle.load(f, encoding="latin1")
    fine_labels = train["fine_labels"]
    coarse_labels = train["coarse_labels"]

    fine_to_coarse: Dict[int, int] = {}
    for f_lbl, c_lbl in zip(fine_labels, coarse_labels):
        if f_lbl in fine_to_coarse:
            assert fine_to_coarse[f_lbl] == c_lbl
        else:
            fine_to_coarse[f_lbl] = c_lbl

    assert len(fine_to_coarse) == 100
    fine_to_coarse_list = [fine_to_coarse[i] for i in range(100)]
    return fine_names, coarse_names, fine_to_coarse_list


def get_hierarchy(dataset: str, data_root: str = None):
    if dataset == "cifar100":
        if data_root is None:
            raise ValueError("data_root required for cifar100")
        return get_cifar100_hierarchy(data_root)
    elif dataset == "cifar10":
        return get_cifar10_hierarchy()
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


if __name__ == "__main__":
    fine, coarse, mapping = get_cifar10_hierarchy()
    print(f"CIFAR-10: {len(fine)} fine classes, {len(coarse)} coarse groups")
    for c_idx, c_name in enumerate(coarse):
        members = [fine[i] for i in range(len(fine)) if mapping[i] == c_idx]
        print(f"  {c_name}: {members}")
