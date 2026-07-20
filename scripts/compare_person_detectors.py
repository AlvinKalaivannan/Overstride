"""Compare our own YOLO-pose detector against Roboflow-hosted YOLO-World on
real marathon crowd footage, before committing to either for the marathon-
footage population-baseline pipeline.

YOLO-World (Tencent AI Lab) is one of Roboflow's own curated foundation
models -- a real, published open-vocabulary detector, not an unverified
community Universe upload -- so it's called with a plain-text class prompt
("person") rather than a project/version ID.

There's no ground truth for marathon crowds (unlike ASPset-510's mocap), so
this is deliberately qualitative: detection counts/confidence side by side,
plus annotated frames saved for visual inspection -- not a PCK-style number.

    python scripts/compare_person_detectors.py --video race.mp4 --num-frames 8

Requires ROBOFLOW_API_KEY in the environment and the `cv` + `inference-sdk`
extras installed (pip install -e ".[cv]").
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "interim" / "detector_comparison"


def sample_frame_indices(cap: cv2.VideoCapture, num_frames: int) -> list[int]:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError("could not read frame count from video")
    return [int(i * total / num_frames) for i in range(num_frames)]


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"could not read frame {frame_idx}")
    return frame


def detect_yolo(frame: np.ndarray, model) -> list[tuple[float, float, float, float, float]]:
    """All detected people (yolov8n-pose.pt only has a person class) -> (x1,y1,x2,y2,conf)."""
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    return [(*box, c) for box, c in zip(xyxy.tolist(), conf.tolist())]


def detect_roboflow(
    frame_path: Path, client, class_names: list[str], model_version: str, confidence: float
) -> list[tuple[float, float, float, float, float]]:
    """Open-vocabulary detection via Roboflow-hosted YOLO-World -> (x1,y1,x2,y2,conf)."""
    import supervision as sv

    results = client.infer_from_yolo_world(
        inference_input=[str(frame_path)],
        class_names=class_names,
        model_version=model_version,
        confidence=confidence,
    )
    detections = sv.Detections.from_inference(results[0])
    if len(detections) == 0:
        return []
    return [
        (x1, y1, x2, y2, float(conf))
        for (x1, y1, x2, y2), conf in zip(detections.xyxy.tolist(), detections.confidence.tolist())
    ]


def draw_boxes(frame: np.ndarray, boxes: list[tuple[float, float, float, float, float]], color) -> np.ndarray:
    annotated = frame.copy()
    for x1, y1, x2, y2, conf in boxes:
        pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(annotated, pt1, pt2, color, 2)
        cv2.putText(annotated, f"{conf:.2f}", (pt1[0], max(0, pt1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated


def summarize(boxes: list[tuple[float, float, float, float, float]]) -> str:
    if not boxes:
        return "0 detections"
    confs = [b[4] for b in boxes]
    return f"{len(boxes)} detections, confidence mean={np.mean(confs):.2f} min={np.min(confs):.2f} max={np.max(confs):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--yolo-model", default="yolov8n-pose.pt")
    parser.add_argument("--roboflow-classes", nargs="+", default=["person"])
    parser.add_argument("--roboflow-model-version", default="v2-s",
                         choices=["s", "m", "l", "x", "v2-s", "v2-m", "v2-l", "v2-x"])
    parser.add_argument("--roboflow-confidence", type=float, default=0.1)
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY is not set in the environment")

    from inference_sdk import InferenceHTTPClient
    from ultralytics import YOLO

    yolo_model = YOLO(args.yolo_model)
    rf_client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)

    cap = cv2.VideoCapture(str(args.video))
    frame_indices = sample_frame_indices(cap, args.num_frames)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'frame':>8s}  {'yolo-pose':<45s}  roboflow")

    with tempfile.TemporaryDirectory() as tmpdir:
        for frame_idx in frame_indices:
            frame = read_frame(cap, frame_idx)

            yolo_boxes = detect_yolo(frame, yolo_model)

            tmp_path = Path(tmpdir) / f"frame_{frame_idx}.png"
            cv2.imwrite(str(tmp_path), frame)
            try:
                rf_boxes = detect_roboflow(
                    tmp_path, rf_client, args.roboflow_classes, args.roboflow_model_version, args.roboflow_confidence
                )
            except Exception as exc:  # noqa: BLE001 -- surface any Roboflow API error, keep comparing other frames
                print(f"  Roboflow inference failed for frame {frame_idx}: {exc}")
                rf_boxes = []

            print(f"{frame_idx:>8d}  {summarize(yolo_boxes):<45s}  {summarize(rf_boxes)}")

            cv2.imwrite(str(OUTPUT_DIR / f"frame_{frame_idx}_yolo.png"), draw_boxes(frame, yolo_boxes, (0, 255, 0)))
            cv2.imwrite(str(OUTPUT_DIR / f"frame_{frame_idx}_roboflow.png"), draw_boxes(frame, rf_boxes, (0, 0, 255)))

    cap.release()
    print(f"\nAnnotated frames saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
