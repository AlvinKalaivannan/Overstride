import numpy as np

from overstride.pose.track import ByteTrackWrapper


def test_tracker_id_stable_for_smoothly_moving_box():
    tracker = ByteTrackWrapper()
    base_box = np.array([[10.0, 10.0, 50.0, 50.0]])
    confidence = np.array([0.9])
    class_id = np.array([0])

    ids = []
    for step in range(5):
        boxes = base_box + step * 2  # small per-frame motion
        tracker_ids = tracker.update(boxes, confidence, class_id)
        ids.append(tracker_ids[0])

    assert len(set(ids)) == 1, f"expected a single stable ID, got {ids}"


def test_tracker_assigns_distinct_ids_to_separated_boxes():
    tracker = ByteTrackWrapper()
    boxes = np.array([[10.0, 10.0, 50.0, 50.0], [500.0, 500.0, 540.0, 540.0]])
    confidence = np.array([0.9, 0.9])
    class_id = np.array([0, 0])

    tracker_ids = tracker.update(boxes, confidence, class_id)

    assert len(set(tracker_ids)) == 2
