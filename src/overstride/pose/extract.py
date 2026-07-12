"""2D pose extraction: MediaPipe (single-person) and YOLO-pose (multi-person).

Both extractors return a dict keyed by joint name -> (x, y) pixel
coordinates, restricted to COMMON_JOINTS -- the subset of joints with an
unambiguous correspondence across MediaPipe's BlazePose landmarks, YOLO-pose's
COCO-17 keypoints, and ASPset-510's ground-truth skeleton (used to validate
extraction accuracy; see scripts/validate_pose_accuracy.py). Joints like
head_top, neck, spine, or pelvis exist in one format but not cleanly in
another, so they're left out rather than approximated.
"""

from __future__ import annotations

COMMON_JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# mediapipe.solutions.pose.PoseLandmark indices
_MEDIAPIPE_LANDMARK_INDEX = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

# COCO-17 keypoint indices, as produced by ultralytics YOLO-pose models
_YOLO_COCO_INDEX = {
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


class MediaPipeExtractor:
    """Single-person pose extraction via MediaPipe Pose."""

    def __init__(self, **kwargs):
        import mediapipe as mp

        self._pose = mp.solutions.pose.Pose(static_image_mode=True, **kwargs)

    def extract(self, frame_bgr) -> dict[str, tuple[float, float]] | None:
        import cv2

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks.landmark
        return {
            joint: (landmarks[idx].x * width, landmarks[idx].y * height)
            for joint, idx in _MEDIAPIPE_LANDMARK_INDEX.items()
        }

    def close(self) -> None:
        self._pose.close()


class YoloPoseExtractor:
    """Multi-person pose extraction via YOLO-pose (ultralytics)."""

    def __init__(self, model_name: str = "yolov8n-pose.pt"):
        from ultralytics import YOLO

        self._model = YOLO(model_name)

    def extract(self, frame_bgr) -> dict[str, tuple[float, float]] | None:
        results = self._model(frame_bgr, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return None
        # Highest-confidence detected person only -- multi-person tracking is track.py's job.
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        return {joint: (float(keypoints[idx][0]), float(keypoints[idx][1])) for joint, idx in _YOLO_COCO_INDEX.items()}
