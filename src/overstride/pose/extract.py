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

import urllib.request
from pathlib import Path

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


_POSE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pose_landmarker_full.task"


def _ensure_pose_landmarker_model(model_path: Path) -> Path:
    """Download the PoseLandmarker task bundle if it isn't already cached.

    MediaPipe's legacy `mp.solutions` API was removed upstream (not just
    deprecated -- confirmed via google-ai-edge/mediapipe#6200: "support for
    MediaPipe Solutions has been removed"), so extraction goes through the
    newer Tasks API instead, which requires this model file separately --
    it isn't bundled in the mediapipe pip package.

    Uses the "full" variant, not "lite": the old `mp.solutions.pose.Pose()`
    defaulted to `model_complexity=1` ("full"-equivalent), and this pipeline
    runs offline on submitted footage rather than real-time on mobile, so
    there's no reason to trade the lite model's detection recall for speed.
    """
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_POSE_LANDMARKER_MODEL_URL, model_path)
    return model_path


class MediaPipeExtractor:
    """Single-person pose extraction via MediaPipe's PoseLandmarker (Tasks API).

    Runs in VIDEO mode rather than IMAGE mode: callers feed it consecutive frames
    from a single clip, and VIDEO mode lets BlazePose carry the previous frame's
    ROI forward instead of re-running full-frame person detection from scratch
    every call. On ASPset-510 footage (subject occupies ~2-3% of frame width in
    the source 4K video -- a small, distant figure, not the close-up framing
    BlazePose's defaults assume) this alone raised detection from 8.3% to 29.2%
    at the default 0.5 confidence threshold.

    min_detection_confidence defaults to 0.3, not the model's own 0.5 default and
    not the even-lower 0.1 tried initially. Swept across 3 ASPset-510 clips
    (scripts/validate_pose_accuracy_batch.py --min-detection-confidence 0.1 0.2
    0.3 0.5): 0.5 is unstable -- 93-100% detection on two clips but 0% on a third,
    since the fixed camera framing leaves no margin above threshold on harder
    motion. 0.1 is uniformly worse than 0.2/0.3 on both detection and PCK,
    because min_tracking_confidence (kept equal to the detection threshold) lets
    VIDEO mode keep tracking a drifted ROI instead of forcing re-detection. 0.3
    had the best detection rate (98.3%), lowest pixel error, and best PCK@0.2 of
    all four values tested.
    """

    def __init__(
        self,
        model_path: Path | str = _DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.3,
    ):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        resolved_path = _ensure_pose_landmarker_model(Path(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self._mp = mp
        self._detector = vision.PoseLandmarker.create_from_options(options)
        self._next_timestamp_ms = 0

    def extract(self, frame_bgr) -> dict[str, tuple[float, float]] | None:
        import cv2

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode only requires monotonically increasing timestamps -- frames
        # are processed strictly in order, so a per-call counter is enough; there's
        # no need for the caller's real frame timestamps.
        result = self._detector.detect_for_video(mp_image, self._next_timestamp_ms)
        self._next_timestamp_ms += 1
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]
        return {
            joint: (landmarks[idx].x * width, landmarks[idx].y * height)
            for joint, idx in _MEDIAPIPE_LANDMARK_INDEX.items()
        }

    def close(self) -> None:
        self._detector.close()


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
