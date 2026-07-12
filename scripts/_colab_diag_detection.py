"""One-off diagnostic (not part of the pipeline): why is MediaPipe PoseLandmarker
missing the person on almost every ASPset-510 frame? Run in Colab:

    python scripts/_colab_diag_detection.py --subject-id 1e28 --clip-id 0091 --camera-id left

Checks, per sampled frame:
  1. Default thresholds (matches MediaPipeExtractor) -- confirms the 4.2% detection rate.
  2. Very low thresholds (0.1) -- if this recovers detections, it's a confidence/scale
     problem (subject too small/distant/blurry), not a broken pipeline.
  3. VIDEO running mode with timestamps -- BlazePose can use the previous frame's ROI
     to keep tracking through a dip in detector confidence; IMAGE mode re-runs the
     person detector from scratch every frame with no such help.
  4. Saves one sampled frame to disk so we can eyeball how large the runner actually
     is relative to the frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "aspset510" / "test"
MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"
FRAME_STRIDE = 5


def make_detector(running_mode, min_conf: float):
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=min_conf,
        min_pose_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    return vision.PoseLandmarker.create_from_options(options)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--camera-id", required=True, choices=["left", "mid", "right"])
    parser.add_argument("--frame-stride", type=int, default=FRAME_STRIDE)
    args = parser.parse_args()

    video_path = DATA_DIR / "videos" / args.subject_id / f"{args.subject_id}-{args.clip_id}-{args.camera_id}.mkv"
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {width}x{height}")

    frames = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.frame_stride == 0:
            frames.append((frame_idx, frame))
        frame_idx += 1
    cap.release()
    print(f"sampled {len(frames)} frames")

    # save one mid-clip frame so we can eyeball subject scale
    mid = frames[len(frames) // 2]
    out_path = ROOT / "diag_sample_frame.png"
    cv2.imwrite(str(out_path), mid[1])
    print(f"saved sample frame -> {out_path}")

    for label, running_mode, min_conf in [
        ("default (IMAGE, conf=0.5)", vision.RunningMode.IMAGE, 0.5),
        ("low-threshold (IMAGE, conf=0.1)", vision.RunningMode.IMAGE, 0.1),
        ("VIDEO mode, conf=0.5", vision.RunningMode.VIDEO, 0.5),
        ("VIDEO mode, conf=0.1", vision.RunningMode.VIDEO, 0.1),
    ]:
        detector = make_detector(running_mode, min_conf)
        detected = 0
        bbox_fracs = []
        for i, (frame_idx, frame) in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if running_mode == vision.RunningMode.VIDEO:
                timestamp_ms = int(frame_idx * 1000 / 30)  # approx fps
                result = detector.detect_for_video(mp_image, timestamp_ms)
            else:
                result = detector.detect(mp_image)
            if result.pose_landmarks:
                detected += 1
                lm = result.pose_landmarks[0]
                xs = [p.x for p in lm]
                ys = [p.y for p in lm]
                bbox_fracs.append((max(xs) - min(xs), max(ys) - min(ys)))
        detector.close()
        rate = 100 * detected / len(frames)
        avg_bbox = ""
        if bbox_fracs:
            mean_w = sum(b[0] for b in bbox_fracs) / len(bbox_fracs)
            mean_h = sum(b[1] for b in bbox_fracs) / len(bbox_fracs)
            avg_bbox = f", mean detected bbox = {mean_w:.2f}w x {mean_h:.2f}h of frame"
        print(f"{label}: {detected}/{len(frames)} detected ({rate:.1f}%){avg_bbox}")


if __name__ == "__main__":
    main()
