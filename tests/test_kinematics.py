import numpy as np
import pytest

from overstride.pose.kinematics import (
    _fill_gaps,
    _side_features,
    _symmetry_index,
    aggregate_session,
    detect_foot_strikes,
    detect_stance_phases,
    joint_angle,
    torso_size,
)

# Two clean gait cycles for one foot: a lead-in ramp, then a 10-frame cycle
# (foot-strike at the peak, a brief near-ground decay, a sharp drop into
# swing, a swing trough, and a rise back toward the next strike) repeated
# twice. Peaks land at indices 2 and 12 -- both have strictly lower
# neighbors on both sides, so find_peaks reports them cleanly with no
# plateau ambiguity.
_CYCLE = [100, 99, 98, 95, 85, 60, 75, 90, 97, 99]
ANKLE_Y = np.array([70, 85] + _CYCLE + _CYCLE, dtype=float)
FPS = 10.0


def test_joint_angle_right_angle():
    a, b, c = (1.0, 0.0), (0.0, 0.0), (0.0, 1.0)
    assert joint_angle(a, b, c) == pytest.approx(90.0)


def test_joint_angle_straight_line():
    a, b, c = (1.0, 0.0), (0.0, 0.0), (-1.0, 0.0)
    assert joint_angle(a, b, c) == pytest.approx(180.0)


def test_joint_angle_vectorized_over_frames():
    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 0.0]])
    c = np.array([[0.0, 1.0], [-1.0, 0.0]])
    angles = joint_angle(a, b, c)
    np.testing.assert_allclose(angles, [90.0, 180.0])


def test_detect_foot_strikes_finds_both_peaks():
    strikes = detect_foot_strikes(ANKLE_Y, FPS)
    np.testing.assert_array_equal(strikes, [2, 12])


def test_detect_foot_strikes_distance_filters_noise():
    # two independent local maxima (each with strictly lower neighbors, so
    # both are valid peak candidates on their own) too close together to be
    # two different footfalls at the given cadence bound -- only the taller
    # one should survive.
    y = np.array([0.0, 10.0, 0.0, 8.0, 0.0])
    # min_stride_seconds = 120/60 = 2s -> min_distance = round(2*2) = 4 frames,
    # comfortably more than the 2-frame gap between these two candidate peaks
    strikes = detect_foot_strikes(y, fps=2.0, max_cadence_spm=60.0)
    np.testing.assert_array_equal(strikes, [1])


def test_detect_stance_phases_toe_off_matches_hand_computation():
    strikes = detect_foot_strikes(ANKLE_Y, FPS)
    phases = detect_stance_phases(ANKLE_Y, strikes)

    # hand-derived: window = ANKLE_Y[2:12], amplitude = 100-60 = 40,
    # threshold = 100 - 0.1*40 = 96; first value below 96 after the peak
    # is ANKLE_Y[5] = 95, so the last in-contact frame is index 4.
    assert phases == [(2, 4), (12, 14)]


def test_torso_size_matches_hand_computation():
    frame = {"left_shoulder": (0.0, 0.0), "right_hip": (3.0, 4.0)}
    assert torso_size(frame) == pytest.approx(5.0)


def test_symmetry_index_matches_formula():
    assert _symmetry_index(100.0, 100.0) == pytest.approx(0.0)
    assert _symmetry_index(90.0, 110.0) == pytest.approx(abs(90 - 110) / (0.5 * (90 + 110)) * 100)


def test_symmetry_index_nan_when_either_side_missing():
    assert np.isnan(_symmetry_index(float("nan"), 100.0))


def test_fill_gaps_interpolates_missing_frames():
    frames = [
        {"left_ankle": (0.0, 0.0)},
        None,
        {"left_ankle": (2.0, 4.0)},
    ]
    joints = _fill_gaps(frames, ("left_ankle",))
    np.testing.assert_allclose(joints["left_ankle"][1], [1.0, 2.0])


def test_side_features_hand_computed():
    ankle_x = 5.0 * np.arange(len(ANKLE_Y))
    torso_series = np.full(len(ANKLE_Y), 50.0)
    hip_angle = np.full(len(ANKLE_Y), 170.0)
    knee_angle = np.zeros(len(ANKLE_Y))
    knee_angle[2:12] = np.linspace(90.0, 150.0, 10)
    knee_angle[12:22] = np.linspace(90.0, 150.0, 10)

    result = _side_features(ankle_x, ANKLE_Y, hip_angle, knee_angle, torso_series, FPS)

    assert result["n_strikes"] == 2
    # ground contact = (toe_off - strike)/fps*1000 = (4-2)/10*1000 for both phases
    assert result["ground_contact_time_ms"] == pytest.approx(200.0)
    # hip angle is constant, so its mean at either toe-off frame is trivially 170
    assert result["hip_extension_deg"] == pytest.approx(170.0)
    # each stride window's knee-angle range is exactly 150-90=60
    assert result["knee_rom_deg"] == pytest.approx(60.0)
    # ankle_x displacement between strikes (10 frames * 5.0/frame = 50) / torso 50
    assert result["stride_length_norm"] == pytest.approx(1.0)


def test_aggregate_session_end_to_end_symmetric_scenario():
    n = len(ANKLE_Y)
    ankle_x = 5.0 * np.arange(n)
    frames = []
    for i in range(n):
        ankle_xy = (float(ankle_x[i]), float(ANKLE_Y[i]) + 100.0)  # keep ankle below the knee
        frame = {
            "left_shoulder": (0.0, 0.0), "right_shoulder": (0.0, 0.0),
            "left_hip": (0.0, 50.0), "right_hip": (0.0, 50.0),
            "left_knee": (0.0, 90.0), "right_knee": (0.0, 90.0),
            "left_ankle": ankle_xy, "right_ankle": ankle_xy,
        }
        frames.append(frame)

    result = aggregate_session(frames, FPS)

    # shoulder/hip/knee are colinear and never move -> hip angle is a
    # constant 180 degrees at every frame, on both (identical) sides.
    assert result["hip_extension_left_deg"] == pytest.approx(180.0)
    assert result["hip_extension_right_deg"] == pytest.approx(180.0)

    # left and right sides are identical inputs, so every asymmetry is zero
    assert result["ground_contact_asymmetry_pct"] == pytest.approx(0.0)
    assert result["stride_length_asymmetry_pct"] == pytest.approx(0.0)
    assert result["ground_contact_time_left_ms"] == result["ground_contact_time_right_ms"]
    assert result["knee_rom_left_deg"] == pytest.approx(result["knee_rom_right_deg"])

    # cadence: 2 strikes per side over a 22-frame / 10fps clip
    session_minutes = n / FPS / 60.0
    assert result["cadence_spm"] == pytest.approx(4 / session_minutes)

    # independently re-derive the knee-angle series from the same raw
    # geometry (hip=(0,50), knee=(0,90), ankle=(ankle_x, ANKLE_Y+100)) and
    # confirm the per-stride range this module reports matches.
    hip_minus_knee = np.array([0.0, -40.0])
    knee_rom_expected = []
    strikes = [2, 12]
    for i in range(len(strikes) - 1):
        window_angles = []
        for f in range(strikes[i], strikes[i + 1]):
            ankle_minus_knee = np.array([ankle_x[f], ANKLE_Y[f] + 100.0 - 90.0])
            cos_angle = np.dot(hip_minus_knee, ankle_minus_knee) / (
                np.linalg.norm(hip_minus_knee) * np.linalg.norm(ankle_minus_knee)
            )
            window_angles.append(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
        knee_rom_expected.append(max(window_angles) - min(window_angles))
    assert result["knee_rom_left_deg"] == pytest.approx(float(np.mean(knee_rom_expected)))

    expected_keys = {
        "cadence_spm",
        "ground_contact_time_left_ms", "ground_contact_time_right_ms", "ground_contact_asymmetry_pct",
        "hip_extension_left_deg", "hip_extension_right_deg",
        "knee_rom_left_deg", "knee_rom_right_deg",
        "stride_length_norm_left", "stride_length_norm_right", "stride_length_asymmetry_pct",
    }
    assert set(result) == expected_keys
