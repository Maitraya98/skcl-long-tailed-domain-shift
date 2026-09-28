"""
road_annotations.py

Parses ROAD dataset JSON annotations and extracts Car/Ped/Cyc bounding boxes
per video, tagged with day/night condition.
"""
from __future__ import annotations
import json
from typing import Dict, List, Tuple, NamedTuple

TARGET_CLASSES = ["Ped", "Car", "Cyc"]

CONFIRMED_CONDITIONS = {
    "2014-06-25-16-45-34_stereo_centre_02": "day",
    "2014-08-08-13-15-11_stereo_centre_01": "day",
    "2015-02-03-19-43-11_stereo_centre_04": "night",
    "2014-12-10-18-10-50_stereo_centre_02": "night",
}


def infer_condition_from_timestamp(video_name: str) -> str:
    try:
        parts = video_name.split("-")
        hour = int(parts[3])
    except (IndexError, ValueError):
        return "unknown"
    if 8 <= hour < 17:
        return "day"
    elif hour < 6 or hour >= 19:
        return "night_candidate"
    else:
        return "twilight_candidate"


def get_video_condition(video_name: str) -> str:
    if video_name in CONFIRMED_CONDITIONS:
        return CONFIRMED_CONDITIONS[video_name]
    return infer_condition_from_timestamp(video_name)


def load_road_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


class Annotation(NamedTuple):
    frame_num: int
    box_xyxy_px: Tuple[float, float, float, float]
    class_name: str
    tube_uid: str


def extract_annotations_for_video(
    video_data: dict,
    agent_labels: List[str],
    target_classes: List[str] = TARGET_CLASSES,
) -> List[Annotation]:
    target_indices = {agent_labels.index(c): c for c in target_classes if c in agent_labels}
    results: List[Annotation] = []

    frames = video_data.get("frames", {})
    for frame_key, frame_data in frames.items():
        if not frame_data.get("annotated", 0):
            continue
        width = frame_data["width"]
        height = frame_data["height"]
        annos = frame_data.get("annos", {})

        for anno_id, anno in annos.items():
            box = anno.get("box")
            agent_ids = anno.get("agent_ids", [])
            tube_uid = anno.get("tube_uid")
            if box is None or not agent_ids or tube_uid is None:
                continue

            matched_class = None
            for idx in agent_ids:
                if idx in target_indices:
                    matched_class = target_indices[idx]
                    break
            if matched_class is None:
                continue

            x1, y1, x2, y2 = box
            box_px = (x1 * width, y1 * height, x2 * width, y2 * height)
            results.append(Annotation(int(frame_key), box_px, matched_class, tube_uid))

    return results


def subsample_by_tube(annotations: List[Annotation], stride: int = 12) -> List[Annotation]:
    by_tube: Dict[str, List[Annotation]] = {}
    for a in annotations:
        by_tube.setdefault(a.tube_uid, []).append(a)

    kept: List[Annotation] = []
    for tube_uid, anns in by_tube.items():
        anns.sort(key=lambda a: a.frame_num)
        last_kept_frame = None
        for a in anns:
            if last_kept_frame is None or (a.frame_num - last_kept_frame) >= stride:
                kept.append(a)
                last_kept_frame = a.frame_num
    return kept


def filter_by_min_size(annotations: List[Annotation], min_size: int = 20) -> List[Annotation]:
    """
    Discards boxes whose native pixel size (narrower dimension) is below min_size.
    Found via scripts/analyze_box_sizes.py: ~18% of day pedestrians and smaller
    fractions of other classes/conditions fall below this threshold, and are
    genuinely too small to carry usable visual signal regardless of output resize.
    """
    kept = []
    for a in annotations:
        x1, y1, x2, y2 = a.box_xyxy_px
        w, h = x2 - x1, y2 - y1
        if min(w, h) >= min_size:
            kept.append(a)
    return kept


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    data = load_road_json(args.json)
    agent_labels = data["agent_labels"]
    video_name = args.video or list(data["db"].keys())[0]
    video_data = data["db"][video_name]

    anns = extract_annotations_for_video(video_data, agent_labels)
    print(f"Video: {video_name}")
    print(f"Total Car/Ped/Cyc annotations (all frames): {len(anns)}")

    from collections import Counter
    counts = Counter(a.class_name for a in anns)
    print("Per-class counts (before tube subsampling):", dict(counts))

    subsampled = subsample_by_tube(anns, stride=12)
    counts_sub = Counter(a.class_name for a in subsampled)
    print(f"Total after tube subsampling (stride=12): {len(subsampled)}")
    print("Per-class counts (after tube subsampling):", dict(counts_sub))
    print(f"Unique tubes: {len(set(a.tube_uid for a in anns))}")
