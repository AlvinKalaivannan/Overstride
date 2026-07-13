"""Tracked pose sequence -> the named kinematic feature vector Stage 3 scores
against an athlete's baseline (joint angles, stride asymmetry, ground-contact
time, hip extension, cadence -- see the README's Stage 3 section).

Mirrors ingest/aggregate.py's role for Stage 2 (raw input -> named feature
dict), but the input here is a tracked pose sequence -- the output of
MediaPipeExtractor/YoloPoseExtractor plus ByteTrackWrapper over one
continuous shot -- rather than logged sessions.

There's no camera calibration for arbitrary user-submitted footage the way
ASPset-510 provides it, so distances are expressed relative to torso size
(shoulder-to-opposite-hip pixel distance in the same frame) rather than real
-world units. Angles are already scale-invariant. Time-based features use
real seconds, from the video's fps. This matches the project's own-baseline
philosophy: an athlete is compared against their own past sessions, shot
with roughly the same framing, not against an absolute population norm.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

REQUIRED_JOINTS = (
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

STANCE_BAND_FRAC = 0.1
MAX_CADENCE_SPM = 220.0  # plausible fastest whole-body (both feet) running cadence


def joint_angle(a, b, c) -> np.ndarray:
    """Interior angle at vertex b, in degrees, given points/point-sequences a, b, c.

    Vectorized over frames: a, b, c may each be a single (2,) point or an
    (n, 2) array of points.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    v1 = a - b
    v2 = c - b
    dot = np.sum(v1 * v2, axis=-1)
    norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    cos_angle = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def torso_size(frame: dict[str, tuple[float, float]]) -> float:
    """Shoulder-to-opposite-hip pixel distance, used as this frame's scale reference."""
    shoulder = np.array(frame["left_shoulder"], dtype=float)
    hip = np.array(frame["right_hip"], dtype=float)
    return float(np.linalg.norm(shoulder - hip))


def _fill_gaps(
    frames: list[dict[str, tuple[float, float]] | None],
    joint_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Per joint, an (n_frames, 2) array with None frames linearly interpolated.

    Assumes sparse single/few-frame dropouts, not extended occlusion --
    matches the ~95-98% per-frame detection rates measured against
    ASPset-510 at MediaPipeExtractor's tuned confidence threshold.
    """
    n = len(frames)
    idx = np.arange(n)
    result = {}
    for joint in joint_names:
        xy = np.full((n, 2), np.nan)
        for i, frame in enumerate(frames):
            if frame is not None:
                xy[i] = frame[joint]
        valid = ~np.isnan(xy[:, 0])
        if not valid.any():
            raise ValueError(f"no frames contain joint {joint!r}")
        if not valid.all():
            xy[~valid, 0] = np.interp(idx[~valid], idx[valid], xy[valid, 0])
            xy[~valid, 1] = np.interp(idx[~valid], idx[valid], xy[valid, 1])
        result[joint] = xy
    return result


def detect_foot_strikes(
    ankle_y: np.ndarray, fps: float, max_cadence_spm: float = MAX_CADENCE_SPM
) -> np.ndarray:
    """Frame indices where this foot is nearest the ground (local maxima of
    image-y, since image y increases downward).

    `max_cadence_spm` bounds the whole-body running cadence (both feet); the
    minimum spacing enforced between consecutive strikes of the *same* foot
    is derived by halving it, so real strides aren't rejected as noise.
    """
    min_stride_seconds = 120.0 / max_cadence_spm
    min_distance = max(1, int(round(min_stride_seconds * fps)))
    peaks, _ = find_peaks(ankle_y, distance=min_distance)
    return peaks


def detect_stance_phases(
    ankle_y: np.ndarray, foot_strikes: np.ndarray, stance_band_frac: float = STANCE_BAND_FRAC
) -> list[tuple[int, int]]:
    """(foot_strike_frame, toe_off_frame) pairs, one per stride where a toe-off
    could be identified before the next foot-strike (or the sequence's end).

    Toe-off is the last frame still within `stance_band_frac` of that
    stride's peak-to-trough amplitude below the foot-strike height -- a
    standard threshold-based markerless gait-event heuristic (there's no
    force-plate ground truth to fit a more precise method against).
    """
    phases = []
    for i, strike in enumerate(foot_strikes):
        end = foot_strikes[i + 1] if i + 1 < len(foot_strikes) else len(ankle_y)
        window = ankle_y[strike:end]
        if len(window) < 2:
            continue
        amplitude = window.max() - window.min()
        if amplitude <= 0:
            continue
        threshold = window[0] - stance_band_frac * amplitude
        below = np.where(window[1:] < threshold)[0]
        if len(below) == 0:
            continue
        phases.append((int(strike), int(strike + below[0])))
    return phases


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _symmetry_index(left: float, right: float) -> float:
    """Standard gait-asymmetry metric: |L-R| / (0.5*(L+R)) * 100."""
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    denom = 0.5 * (left + right)
    if denom == 0:
        return float("nan")
    return abs(left - right) / denom * 100.0


def _side_features(
    ankle_x: np.ndarray,
    ankle_y: np.ndarray,
    hip_angle: np.ndarray,
    knee_angle: np.ndarray,
    torso_series: np.ndarray,
    fps: float,
) -> dict[str, float]:
    strikes = detect_foot_strikes(ankle_y, fps)
    phases = detect_stance_phases(ankle_y, strikes)

    contact_times_ms = [(toe_off - strike) / fps * 1000.0 for strike, toe_off in phases]
    hip_extension = [float(hip_angle[toe_off]) for _, toe_off in phases]

    knee_rom = []
    stride_lengths = []
    for i in range(len(strikes) - 1):
        window = knee_angle[strikes[i]:strikes[i + 1]]
        if len(window):
            knee_rom.append(float(window.max() - window.min()))
        scale = torso_series[strikes[i]]
        if scale > 0:
            stride_lengths.append(abs(float(ankle_x[strikes[i + 1]] - ankle_x[strikes[i]])) / scale)

    return {
        "n_strikes": len(strikes),
        "ground_contact_time_ms": _safe_mean(contact_times_ms),
        "hip_extension_deg": _safe_mean(hip_extension),
        "knee_rom_deg": _safe_mean(knee_rom),
        "stride_length_norm": _safe_mean(stride_lengths),
    }


def aggregate_session(
    frames: list[dict[str, tuple[float, float]] | None], fps: float
) -> dict[str, float]:
    """One tracked running clip -> the named kinematic feature vector."""
    joints = _fill_gaps(frames, REQUIRED_JOINTS)

    hip_angle_left = joint_angle(joints["left_shoulder"], joints["left_hip"], joints["left_knee"])
    hip_angle_right = joint_angle(joints["right_shoulder"], joints["right_hip"], joints["right_knee"])
    knee_angle_left = joint_angle(joints["left_hip"], joints["left_knee"], joints["left_ankle"])
    knee_angle_right = joint_angle(joints["right_hip"], joints["right_knee"], joints["right_ankle"])
    torso_series = np.linalg.norm(joints["left_shoulder"] - joints["right_hip"], axis=-1)

    left = _side_features(
        joints["left_ankle"][:, 0], joints["left_ankle"][:, 1],
        hip_angle_left, knee_angle_left, torso_series, fps,
    )
    right = _side_features(
        joints["right_ankle"][:, 0], joints["right_ankle"][:, 1],
        hip_angle_right, knee_angle_right, torso_series, fps,
    )

    session_minutes = len(frames) / fps / 60.0
    cadence_spm = (left["n_strikes"] + right["n_strikes"]) / session_minutes if session_minutes > 0 else float("nan")

    return {
        "cadence_spm": cadence_spm,
        "ground_contact_time_left_ms": left["ground_contact_time_ms"],
        "ground_contact_time_right_ms": right["ground_contact_time_ms"],
        "ground_contact_asymmetry_pct": _symmetry_index(
            left["ground_contact_time_ms"], right["ground_contact_time_ms"]
        ),
        "hip_extension_left_deg": left["hip_extension_deg"],
        "hip_extension_right_deg": right["hip_extension_deg"],
        "knee_rom_left_deg": left["knee_rom_deg"],
        "knee_rom_right_deg": right["knee_rom_deg"],
        "stride_length_norm_left": left["stride_length_norm"],
        "stride_length_norm_right": right["stride_length_norm"],
        "stride_length_asymmetry_pct": _symmetry_index(
            left["stride_length_norm"], right["stride_length_norm"]
        ),
    }
