"""
analyze_box_sizes.py

Checks the native (pre-resize) pixel dimensions of every extracted box, broken
down by class and condition. Tells us whether the low-resolution problem is a
widespread data issue or an isolated case, before deciding on a fix.
"""
from __future__ import annotations
import argparse
import sys
import os
from collections import defaultdict
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "robotcar"))
from road_annotations import load_road_json, extract_annotations_for_video, subsample_by_tube


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainval_json", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--stride", type=int, default=12)
    args = parser.parse_args()

    from extract_road_dataset import DAY_TEST_VIDEOS, NIGHT_TEST_VIDEOS

    sizes = defaultdict(list)

    for json_path in [args.trainval_json, args.test_json]:
        data = load_road_json(json_path)
        agent_labels = data["agent_labels"]
        for video_name, video_data in data["db"].items():
            condition = "night" if video_name in NIGHT_TEST_VIDEOS else "day"
            anns = extract_annotations_for_video(video_data, agent_labels)
            sub = subsample_by_tube(anns, stride=args.stride)
            for a in sub:
                x1, y1, x2, y2 = a.box_xyxy_px
                w, h = x2 - x1, y2 - y1
                sizes[(condition, a.class_name)].append(min(w, h))

    print(f"{'condition':<8} {'class':<6} {'count':>7} {'min':>6} {'p10':>6} {'median':>7} {'p90':>6} {'max':>7} {'%<32px':>8} {'%<20px':>8}")
    print("-" * 80)
    for (condition, class_name), vals in sorted(sizes.items()):
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        p10 = vals_sorted[int(n * 0.10)]
        median = statistics.median(vals_sorted)
        p90 = vals_sorted[int(n * 0.90)]
        pct_under_32 = 100 * sum(1 for v in vals_sorted if v < 32) / n
        pct_under_20 = 100 * sum(1 for v in vals_sorted if v < 20) / n
        print(f"{condition:<8} {class_name:<6} {n:>7} {vals_sorted[0]:>6.0f} {p10:>6.0f} "
              f"{median:>7.0f} {p90:>6.0f} {vals_sorted[-1]:>7.0f} {pct_under_32:>7.1f}% {pct_under_20:>7.1f}%")


if __name__ == "__main__":
    main()
