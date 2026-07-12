"""Batch version of validate_pose_accuracy.py: run across several ASPset-510
clips (varying clip / camera, optionally subject) to check whether the
single-clip finding (100% detection, 68.1% PCK@0.2, legs much more accurate
than arms) holds generally or was an artifact of that one 1e28-0091-left
sample.

Requires each requested subject's videos already extracted via
scripts/download_aspset_sample.py --subject-id <id>.

    python scripts/validate_pose_accuracy_batch.py --subject-id 1e28 --num-clips 5
    python scripts/validate_pose_accuracy_batch.py --subject-id 1e28 --subject-id 8a59 --num-clips 3
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_pose_accuracy import (  # noqa: E402
    DATA_DIR,
    PCK_THRESHOLD,
    build_extractor,
    ground_truth_2d,
    torso_size,
)

from overstride.pose.extract import COMMON_JOINTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = ROOT / "data" / "raw" / "aspset510" / "splits.csv"
FRAME_STRIDE = 5


def list_clips(subject_id: str) -> list[str]:
    clip_ids = []
    seen = set()
    with SPLITS_PATH.open(newline="") as f:
        for subj, clip_id, split, _camera_id in csv.reader(f):
            if subj == subject_id and split == "test" and clip_id not in seen:
                seen.add(clip_id)
                clip_ids.append(clip_id)
    return sorted(clip_ids)


def pick_spread(items: list[str], n: int) -> list[str]:
    """n items spread evenly across the sorted list, rather than just the first n."""
    if n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def run_one(subject_id: str, clip_id: str, camera_id: str, extractor_name: str, frame_stride: int) -> dict | None:
    gt = ground_truth_2d(subject_id, clip_id, camera_id)
    n_frames = next(iter(gt.values())).shape[0]

    video_path = DATA_DIR / "videos" / subject_id / f"{subject_id}-{clip_id}-{camera_id}.mkv"
    if not video_path.exists():
        print(f"  skip {subject_id}-{clip_id}-{camera_id}: video not found")
        return None
    cap = cv2.VideoCapture(str(video_path))

    extractor = build_extractor(extractor_name)
    per_joint_errors: dict[str, list[float]] = {j: [] for j in COMMON_JOINTS}
    per_joint_correct: dict[str, list[bool]] = {j: [] for j in COMMON_JOINTS}
    frames_checked = 0
    frames_detected = 0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx >= n_frames:
            break
        if frame_idx % frame_stride == 0:
            frames_checked += 1
            gt_frame = {j: gt[j][frame_idx] for j in COMMON_JOINTS}
            size = torso_size(gt_frame)
            predicted = extractor.extract(frame)
            if predicted is not None:
                frames_detected += 1
                for joint in COMMON_JOINTS:
                    error = float(np.linalg.norm(np.array(predicted[joint]) - gt_frame[joint]))
                    per_joint_errors[joint].append(error)
                    per_joint_correct[joint].append(error < PCK_THRESHOLD * size)
        frame_idx += 1
    cap.release()
    if hasattr(extractor, "close"):
        extractor.close()

    return {
        "subject_id": subject_id,
        "clip_id": clip_id,
        "camera_id": camera_id,
        "frames_checked": frames_checked,
        "frames_detected": frames_detected,
        "per_joint_errors": per_joint_errors,
        "per_joint_correct": per_joint_correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject-id", action="append", required=True)
    parser.add_argument("--camera-id", default="left", choices=["left", "mid", "right"])
    parser.add_argument("--extractor", default="mediapipe", choices=["mediapipe", "yolo"])
    parser.add_argument("--num-clips", type=int, default=3, help="per subject")
    parser.add_argument("--frame-stride", type=int, default=FRAME_STRIDE)
    args = parser.parse_args()

    results = []
    for subject_id in args.subject_id:
        clips = list_clips(subject_id)
        chosen = pick_spread(clips, args.num_clips)
        print(f"subject {subject_id}: {len(clips)} test clips available, using {chosen}")
        for clip_id in chosen:
            r = run_one(subject_id, clip_id, args.camera_id, args.extractor, args.frame_stride)
            if r is None:
                continue
            results.append(r)
            rate = 100 * r["frames_detected"] / r["frames_checked"] if r["frames_checked"] else 0.0
            print(f"  {subject_id}-{clip_id}-{args.camera_id}: "
                  f"{r['frames_detected']}/{r['frames_checked']} detected ({rate:.1f}%)")

    if not results:
        print("no results")
        return

    print()
    print(f"{'joint':20s} {'mean px error':>15s} {'PCK@0.2':>10s}")
    all_errors: list[float] = []
    all_correct: list[bool] = []
    for joint in COMMON_JOINTS:
        errors = [e for r in results for e in r["per_joint_errors"][joint]]
        correct = [c for r in results for c in r["per_joint_correct"][joint]]
        all_errors.extend(errors)
        all_correct.extend(correct)
        if errors:
            print(f"{joint:20s} {np.mean(errors):15.1f} {100 * np.mean(correct):9.1f}%")
        else:
            print(f"{joint:20s} {'n/a':>15s} {'n/a':>10s}")

    total_checked = sum(r["frames_checked"] for r in results)
    total_detected = sum(r["frames_detected"] for r in results)
    print()
    print(f"Clips run: {len(results)}")
    print(f"Overall detection rate: {total_detected}/{total_checked} "
          f"({100 * total_detected / total_checked:.1f}%)")
    if all_errors:
        print(f"Overall mean per-joint pixel error: {np.mean(all_errors):.1f}")
        print(f"Overall PCK@0.2: {100 * np.mean(all_correct):.1f}%")


if __name__ == "__main__":
    main()
