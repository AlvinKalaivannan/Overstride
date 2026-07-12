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


def run_one(
    subject_id: str,
    clip_id: str,
    camera_id: str,
    extractor_name: str,
    frame_stride: int,
    min_detection_confidence: float | None = None,
) -> dict | None:
    gt = ground_truth_2d(subject_id, clip_id, camera_id)
    n_frames = next(iter(gt.values())).shape[0]

    video_path = DATA_DIR / "videos" / subject_id / f"{subject_id}-{clip_id}-{camera_id}.mkv"
    if not video_path.exists():
        print(f"  skip {subject_id}-{clip_id}-{camera_id}: video not found")
        return None
    cap = cv2.VideoCapture(str(video_path))

    extractor = build_extractor(extractor_name, min_detection_confidence)
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


# Stage 3's actual features (joint angles, stride asymmetry, ground-contact time,
# hip extension, cadence -- see README) are all lower-body/gait-cycle measurements,
# so leg accuracy matters far more than arm accuracy for what this pipeline feeds.
LEG_JOINTS = {"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"}
ARM_JOINTS = {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"}


def summarize(results: list[dict]) -> dict:
    total_checked = sum(r["frames_checked"] for r in results)
    total_detected = sum(r["frames_detected"] for r in results)

    def pck(joints: set[str]) -> float | None:
        correct = [c for r in results for j in joints for c in r["per_joint_correct"][j]]
        return 100 * float(np.mean(correct)) if correct else None

    all_errors = [e for r in results for j in COMMON_JOINTS for e in r["per_joint_errors"][j]]
    return {
        "detection_rate": 100 * total_detected / total_checked if total_checked else None,
        "mean_error": float(np.mean(all_errors)) if all_errors else None,
        "overall_pck": pck(set(COMMON_JOINTS)),
        "leg_pck": pck(LEG_JOINTS),
        "arm_pck": pck(ARM_JOINTS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject-id", action="append", required=True)
    parser.add_argument("--camera-id", default="left", choices=["left", "mid", "right"])
    parser.add_argument("--extractor", default="mediapipe", choices=["mediapipe", "yolo"])
    parser.add_argument("--num-clips", type=int, default=3, help="per subject")
    parser.add_argument("--frame-stride", type=int, default=FRAME_STRIDE)
    parser.add_argument(
        "--min-detection-confidence", type=float, nargs="+", default=[None],
        help="mediapipe only. Pass several values (e.g. 0.1 0.3 0.5) to sweep and compare.",
    )
    args = parser.parse_args()

    clip_specs: list[tuple[str, str]] = []
    for subject_id in args.subject_id:
        clips = list_clips(subject_id)
        chosen = pick_spread(clips, args.num_clips)
        print(f"subject {subject_id}: {len(clips)} test clips available, using {chosen}")
        clip_specs.extend((subject_id, clip_id) for clip_id in chosen)

    sweep_rows = []
    for threshold in args.min_detection_confidence:
        label = "default" if threshold is None else f"{threshold:.2f}"
        print(f"\n--- min_detection_confidence = {label} ---")
        results = []
        for subject_id, clip_id in clip_specs:
            r = run_one(subject_id, clip_id, args.camera_id, args.extractor, args.frame_stride, threshold)
            if r is None:
                continue
            results.append(r)
            rate = 100 * r["frames_detected"] / r["frames_checked"] if r["frames_checked"] else 0.0
            print(f"  {subject_id}-{clip_id}-{args.camera_id}: "
                  f"{r['frames_detected']}/{r['frames_checked']} detected ({rate:.1f}%)")

        if not results:
            print("  no results")
            continue

        print(f"\n  {'joint':20s} {'mean px error':>15s} {'PCK@0.2':>10s}")
        for joint in COMMON_JOINTS:
            errors = [e for r in results for e in r["per_joint_errors"][joint]]
            correct = [c for r in results for c in r["per_joint_correct"][joint]]
            if errors:
                print(f"  {joint:20s} {np.mean(errors):15.1f} {100 * np.mean(correct):9.1f}%")
            else:
                print(f"  {joint:20s} {'n/a':>15s} {'n/a':>10s}")

        summary = summarize(results)
        summary["label"] = label
        summary["clips_run"] = len(results)
        sweep_rows.append(summary)

        print(f"\n  Clips run: {summary['clips_run']}")
        print(f"  Overall detection rate: {summary['detection_rate']:.1f}%")
        print(f"  Overall mean per-joint pixel error: {summary['mean_error']:.1f}")
        print(f"  Overall PCK@0.2: {summary['overall_pck']:.1f}%")

    if len(sweep_rows) > 1:
        print("\n=== sweep comparison ===")
        print(f"{'threshold':>10s} {'detection%':>11s} {'mean px err':>12s} "
              f"{'overall PCK':>12s} {'leg PCK':>9s} {'arm PCK':>9s}")
        for row in sweep_rows:
            print(f"{row['label']:>10s} {row['detection_rate']:11.1f} {row['mean_error']:12.1f} "
                  f"{row['overall_pck']:11.1f}% {row['leg_pck']:8.1f}% {row['arm_pck']:8.1f}%")


if __name__ == "__main__":
    main()
