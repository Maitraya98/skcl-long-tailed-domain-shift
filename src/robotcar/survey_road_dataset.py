"""
survey_road_dataset.py

Counts Car/Ped/Cyc instances (post tube-subsampling) across every video in both
the train-val and test JSON files, broken down by video and condition. Run this
BEFORE full extraction, to see real class coverage per video and make an informed
day-train / day-test / night-test video assignment rather than an arbitrary one.
"""
from __future__ import annotations
import argparse
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from road_annotations import (
    load_road_json, extract_annotations_for_video, subsample_by_tube, get_video_condition
)


def survey_file(json_path: str, stride: int = 12):
    data = load_road_json(json_path)
    agent_labels = data["agent_labels"]

    rows = []
    for video_name, video_data in data["db"].items():
        anns = extract_annotations_for_video(video_data, agent_labels)
        sub = subsample_by_tube(anns, stride=stride)
        counts = Counter(a.class_name for a in sub)
        condition = get_video_condition(video_name)
        rows.append({
            "video": video_name,
            "condition": condition,
            "Car": counts.get("Car", 0),
            "Ped": counts.get("Ped", 0),
            "Cyc": counts.get("Cyc", 0),
            "total": sum(counts.values()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainval_json", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--stride", type=int, default=12)
    args = parser.parse_args()

    all_rows = []
    for json_path, source in [(args.trainval_json, "trainval"), (args.test_json, "test")]:
        rows = survey_file(json_path, args.stride)
        for r in rows:
            r["source"] = source
        all_rows.extend(rows)

    header = f"{'video':<42} {'source':<9} {'condition':<18} {'Car':>6} {'Ped':>6} {'Cyc':>6} {'total':>7}"
    print(header)
    print("-" * len(header))
    for r in sorted(all_rows, key=lambda r: (r["condition"], r["video"])):
        print(f"{r['video']:<42} {r['source']:<9} {r['condition']:<18} "
              f"{r['Car']:>6} {r['Ped']:>6} {r['Cyc']:>6} {r['total']:>7}")

    print()
    totals = Counter()
    by_condition = {}
    for r in all_rows:
        totals["Car"] += r["Car"]
        totals["Ped"] += r["Ped"]
        totals["Cyc"] += r["Cyc"]
        cond = r["condition"]
        by_condition.setdefault(cond, Counter())
        by_condition[cond]["Car"] += r["Car"]
        by_condition[cond]["Ped"] += r["Ped"]
        by_condition[cond]["Cyc"] += r["Cyc"]

    print(f"GRAND TOTAL: Car={totals['Car']}  Ped={totals['Ped']}  Cyc={totals['Cyc']}")
    print()
    print("By condition:")
    for cond, c in by_condition.items():
        print(f"  {cond:<18} Car={c['Car']:<6} Ped={c['Ped']:<6} Cyc={c['Cyc']:<6} total={sum(c.values())}")

    flagged = [r for r in all_rows if r["condition"] in ("night_candidate", "twilight_candidate")]
    if flagged:
        print()
        print("VIDEOS NEEDING VISUAL CONFIRMATION before use (not yet verified as day or night):")
        for r in flagged:
            print(f"  {r['video']}  (heuristic guess: {r['condition']}, source={r['source']})")


if __name__ == "__main__":
    main()
