"""ByteTrack wrapper: holds one athlete's identity across a single continuous
shot. Per the README's scope decision, re-identification across cuts,
occlusion-heavy crowds, or camera changes is explicitly out of scope --
this only needs to keep IDs stable within one uninterrupted shot.
"""

from __future__ import annotations

import numpy as np


class ByteTrackWrapper:
    def __init__(self, **kwargs):
        import supervision as sv

        self._tracker = sv.ByteTrack(**kwargs)

    def update(
        self,
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
    ) -> np.ndarray:
        """One frame's detections in -> per-detection tracker IDs out."""
        import supervision as sv

        detections = sv.Detections(xyxy=boxes_xyxy, confidence=confidences, class_id=class_ids)
        tracked = self._tracker.update_with_detections(detections)
        return tracked.tracker_id
