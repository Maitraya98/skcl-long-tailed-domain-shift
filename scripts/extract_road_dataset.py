"""
extract_road_dataset.py

Decodes each video sequentially (never seeks -- avoids the H.264 seek-corruption
issue hit earlier in this project), crops every subsampled Car/Ped/Cyc annotation,
resizes to a fixed size, and saves to disk organized by split/class.

Frame-numbering assumption (flagged, not silently trusted): JSON frame keys are
1-indexed ("1", "2", ...) and are assumed to correspond 1:1 with sequential
cv2.VideoCapture reads, i.e. the Nth frame read (1-indexed) == JSON key str(N).
This matches every video's numf exactly equaling its frame-key count (verified
during schema inspection), which is strong evidence for this assumption but not
an explicit confirmation from ROAD's own documentation.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List

import cv2

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "robotcar"))
from road_annotations import (
    load_road_json, extract_annotations_for_video, subsample_by_tube, filter_by_min_size, Annotation
)

DAY_TEST_VIDEOS = {
    "2014-06-26-09-53-12_stereo_centre_02",
    "2014-11-18-13-20-12_stereo_centre_05",
    "2015-02-03-08-45-10_stereo_centre_02",
    "2015-02-24-12-32-19_stereo_centre_04",
    "2015-03-03-11-31-36_stereo_centre_01",
}
NIGHT_TEST_VIDEOS = {
    "2015-02-03-19-43-11_stereo_centre_04",
    "2014-12-10-18-10-50_stereo_centre_02",
}


def find_video_file(video_name: str, video_dirs: List[str]) -> str:
    for d in video_dirs:
        path = os.path.join(d, video_name + ".mp4")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Could not find {video_name}.mp4 in any of {video_dirs}")


def clamp_box(box_px, width, height):
    x1, y1, x2, y2 = box_px
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return int(x1), int(y1), int(x2), int(y2)


def extract_video(
    video_path: str,
    annotations: List[Annotation],
    split: str,
    condition: str,
    video_name: str,
    out_dir: str,
    crop_size: int,
    manifest_writer,
) -> Counter:
    by_frame: Dict[int, List[Annotation]] = defaultdict(list)
    for a in annotations:
        by_frame[a.frame_num].append(a)

    if not by_frame:
        return Counter()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")

    counts = Counter()
    frame_num = 0
    max_frame_needed = max(by_frame.keys())

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num in by_frame:
            h, w = frame.shape[:2]
            for a in by_frame[frame_num]:
                x1, y1, x2, y2 = clamp_box(a.box_xyxy_px, w, h)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

                class_dir = os.path.join(out_dir, split, a.class_name)
                os.makedirs(class_dir, exist_ok=True)
                fname = f"{video_name}_{frame_num}_{a.tube_uid}.png"
                fpath = os.path.join(class_dir, fname)
                cv2.imwrite(fpath, crop_resized)

                counts[a.class_name] += 1
                manifest_writer.writerow([fpath, a.class_name, split, condition, video_name, frame_num, a.tube_uid])

        if frame_num >= max_frame_needed:
            break

    cap.release()
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainval_json", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--trainval_video_dir", required=True)
    parser.add_argument("--test_video_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--min_box_size", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    manifest_file = open(manifest_path, "w", newline="")
    manifest_writer = csv.writer(manifest_file)
    manifest_writer.writerow(["path", "class", "split", "condition", "video", "frame_num", "tube_uid"])

    trainval_data = load_road_json(args.trainval_json)
    test_data = load_road_json(args.test_json)
    agent_labels = trainval_data["agent_labels"]

    all_videos = []
    for video_name, video_data in trainval_data["db"].items():
        all_videos.append((video_name, video_data, [args.trainval_video_dir, args.test_video_dir]))
    for video_name, video_data in test_data["db"].items():
        all_videos.append((video_name, video_data, [args.trainval_video_dir, args.test_video_dir]))

    total_counts = defaultdict(Counter)

    for video_name, video_data, video_dirs in all_videos:
        if video_name in NIGHT_TEST_VIDEOS:
            split = "night_test"
            condition = "night"
        elif video_name in DAY_TEST_VIDEOS:
            split = "day_test"
            condition = "day"
        else:
            split = "day_train"
            condition = "day"

        anns = extract_annotations_for_video(video_data, agent_labels)
        sub = subsample_by_tube(anns, stride=args.stride)
        sub = filter_by_min_size(sub, min_size=args.min_box_size)
        if not sub:
            print(f"[skip] {video_name}: no annotations after subsampling")
            continue

        video_path = find_video_file(video_name, video_dirs)
        print(f"[{split}] Extracting {video_name} ({len(sub)} annotations to crop)...")
        counts = extract_video(
            video_path, sub, split, condition, video_name,
            args.out_dir, args.crop_size, manifest_writer
        )
        total_counts[split].update(counts)
        print(f"    -> {dict(counts)}")

    manifest_file.close()

    print("\n=== FINAL COUNTS ===")
    for split in ["day_train", "day_test", "night_test"]:
        c = total_counts[split]
        print(f"{split}: Car={c['Car']} Ped={c['Ped']} Cyc={c['Cyc']} total={sum(c.values())}")
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
