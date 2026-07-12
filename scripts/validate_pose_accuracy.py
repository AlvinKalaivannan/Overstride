"""Section 4 DoD: validate MediaPipe/YOLO-pose 2D extraction accuracy against
ASPset-510 ground truth before trusting the pipeline on real footage.

Run scripts/download_aspset_sample.py first. Then:

    python scripts/validate_pose_accuracy.py --subject-id 1e28 --clip-id 0091 \
        --camera-id left --extractor mediapipe

ASPset-510 ships 3D mocap ground truth + camera calibration rather than 2D
keypoints directly, so ground truth is projected to 2D via
overstride.pose.camera.project_points before comparison. Its joint names
(left_shoulder, right_knee, etc.) already match overstride.pose.extract's
COMMON_JOINTS exactly, so no translation table is needed -- we just filter
ASPset's 17 joints down to the 12 GT has in common with the extractors.

Metric: PCK@0.2 (percentage of correct keypoints within 20% of torso size --
the shoulder-to-opposite-hip distance -- a standard scale-invariant pose
accuracy metric), plus raw per-joint mean pixel error, computed only on
frames where the extractor found a person at all (the detection rate itself
is reported separately, since a low detection rate is its own finding).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import ezc3d
import numpy as np

from overstride.pose.camera import project_points
from overstride.pose.extract import COMMON_JOINTS

ASPSET_17J_NAMES = [
    "right_ankle", "right_knee", "right_hip",
    "right_wrist", "right_elbow", "right_shoulder",
    "left_ankle", "left_knee", "left_hip",
    "left_wrist", "left_elbow", "left_shoulder",
    "head_top", "head", "neck", "spine", "pelvis",
]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "aspset510" / "test"
PCK_THRESHOLD = 0.2
FRAME_STRIDE = 5  # sample every Nth frame to keep runtime reasonable


def load_ground_truth_3d(subject_id: str, clip_id: str) -> np.ndarray:
    c3d_path = DATA_DIR / "joints_3d" / subject_id / f"{subject_id}-{clip_id}.c3d"
    c3d = ezc3d.c3d(str(c3d_path))
    points = c3d["data"]["points"]  # (4, 17, n_frames): x, y, z, residual
    return points.transpose(2, 1, 0)[..., :3].astype(np.float64)  # (n_frames, 17, 3), mm


def load_camera(subject_id: str, camera_id: str) -> tuple[np.ndarray, np.ndarray]:
    camera_path = DATA_DIR / "cameras" / subject_id / f"{subject_id}-{camera_id}.json"
    data = json.loads(camera_path.read_text())
    intrinsic = np.array(data["intrinsic_matrix"]).reshape(3, 4)
    extrinsic = np.array(data["extrinsic_matrix"]).reshape(4, 4)
    return intrinsic, extrinsic


def ground_truth_2d(subject_id: str, clip_id: str, camera_id: str) -> dict[str, np.ndarray]:
    """joint name -> (n_frames, 2) pixel coords, restricted to COMMON_JOINTS."""
    joints_3d = load_ground_truth_3d(subject_id, clip_id)
    intrinsic, extrinsic = load_camera(subject_id, camera_id)
    pixels = project_points(joints_3d, intrinsic, extrinsic)  # (n_frames, 17, 2)
    name_to_index = {name: i for i, name in enumerate(ASPSET_17J_NAMES)}
    return {joint: pixels[:, name_to_index[joint], :] for joint in COMMON_JOINTS}


def torso_size(gt_frame: dict[str, np.ndarray]) -> float:
    return float(np.linalg.norm(gt_frame["left_shoulder"] - gt_frame["right_hip"]))


def build_extractor(name: str):
    if name == "mediapipe":
        from overstride.pose.extract import MediaPipeExtractor
        return MediaPipeExtractor()
    if name == "yolo":
        from overstride.pose.extract import YoloPoseExtractor
        return YoloPoseExtractor()
    raise ValueError(f"unknown extractor {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--camera-id", required=True, choices=["left", "mid", "right"])
    parser.add_argument("--extractor", required=True, choices=["mediapipe", "yolo"])
    parser.add_argument("--frame-stride", type=int, default=FRAME_STRIDE)
    args = parser.parse_args()

    gt = ground_truth_2d(args.subject_id, args.clip_id, args.camera_id)
    n_frames = next(iter(gt.values())).shape[0]

    video_path = DATA_DIR / "videos" / args.subject_id / f"{args.subject_id}-{args.clip_id}-{args.camera_id}.mkv"
    cap = cv2.VideoCapture(str(video_path))

    extractor = build_extractor(args.extractor)

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
        if frame_idx % args.frame_stride == 0:
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

    print(f"Subject {args.subject_id}, clip {args.clip_id}, camera {args.camera_id}, extractor {args.extractor}")
    print(f"Frames checked: {frames_checked}, frames with a detected person: {frames_detected} "
          f"({100 * frames_detected / frames_checked:.1f}%)")
    print()
    print(f"{'joint':20s} {'mean px error':>15s} {'PCK@0.2':>10s}")
    all_errors = []
    all_correct = []
    for joint in COMMON_JOINTS:
        errors = per_joint_errors[joint]
        correct = per_joint_correct[joint]
        all_errors.extend(errors)
        all_correct.extend(correct)
        if errors:
            print(f"{joint:20s} {np.mean(errors):15.1f} {100 * np.mean(correct):9.1f}%")
        else:
            print(f"{joint:20s} {'n/a':>15s} {'n/a':>10s}")

    print()
    if all_errors:
        print(f"Overall mean per-joint pixel error: {np.mean(all_errors):.1f}")
        print(f"Overall PCK@0.2: {100 * np.mean(all_correct):.1f}%")


if __name__ == "__main__":
    main()
