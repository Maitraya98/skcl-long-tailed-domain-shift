"""
robotcar_hierarchy.py

Fine/coarse hierarchy for the RobotCar Car/Ped/Cyc experiment, needed for
SKCL's two-level classifier + semantic graph.

Grouping rationale: "vehicle" (motorized, enclosed) vs "vulnerable_road_user"
(unenclosed, human-scale) is the standard categorical distinction used across
the autonomous driving literature (the same grouping used for "VRU" safety
metrics in real AV safety standards) -- not an arbitrary split.
"""
from __future__ import annotations
from typing import List, Tuple

FINE_CLASSES = ["Car", "Ped", "Cyc"]

COARSE_GROUPS = {
    "vehicle": ["Car"],
    "vulnerable_road_user": ["Ped", "Cyc"],
}


def get_robotcar_hierarchy() -> Tuple[List[str], List[str], List[int]]:
    coarse_names = list(COARSE_GROUPS.keys())
    fine_to_coarse = []
    for cls in FINE_CLASSES:
        for coarse_idx, (coarse_name, members) in enumerate(COARSE_GROUPS.items()):
            if cls in members:
                fine_to_coarse.append(coarse_idx)
                break
        else:
            raise ValueError(f"Class {cls!r} not found in any coarse group")
    return FINE_CLASSES, coarse_names, fine_to_coarse


if __name__ == "__main__":
    fine, coarse, mapping = get_robotcar_hierarchy()
    print(f"Fine classes: {fine}")
    print(f"Coarse groups: {coarse}")
    for c_idx, c_name in enumerate(coarse):
        members = [fine[i] for i in range(len(fine)) if mapping[i] == c_idx]
        print(f"  {c_name}: {members}")
