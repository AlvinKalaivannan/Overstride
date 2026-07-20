from overstride.pose.extract import COMMON_JOINTS, _MEDIAPIPE_LANDMARK_INDEX, _YOLO_COCO_INDEX


def test_mediapipe_index_covers_all_common_joints():
    assert set(_MEDIAPIPE_LANDMARK_INDEX) == set(COMMON_JOINTS)


def test_yolo_index_covers_all_common_joints():
    assert set(_YOLO_COCO_INDEX) == set(COMMON_JOINTS)


def test_mediapipe_indices_are_unique():
    indices = list(_MEDIAPIPE_LANDMARK_INDEX.values())
    assert len(indices) == len(set(indices))


def test_yolo_indices_are_unique():
    indices = list(_YOLO_COCO_INDEX.values())
    assert len(indices) == len(set(indices))
