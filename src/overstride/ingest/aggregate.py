"""Raw session logs -> the same 22-feature weekly vector (+ 3 relative km
ratios) that Sections 1/2 were built against.

Dataset #1's columns come from a finer zone taxonomy (Z1-Z5 + two threshold
zones T1/T2) and 0-1-scaled exertion/success/recovery scores than the live
schema captures (Z1-Z4, RPE 1-10, perceived success 1-5, no recovery score
at all). Where there's no clean equivalent, we use the closest available
proxy so the live pipeline stays pluggable into the frozen Stage 2 model
today; each proxy is called out below. A future recalibration against real
athlete data would be the point to close these gaps properly.

    dataset column                              -> live computation
    ------------------------------------------------------------------
    nr. sessions                                -> count(distance_km > 0)
    nr. rest days                                -> count(rest_day)
    total kms                                    -> sum(distance_km)
    max km one day                               -> max(distance_km)
    total km Z3-4                                -> sum(Z3_km + Z4_km)
    max km Z3-4 one day                          -> max(Z3_km + Z4_km per day)
    total hours alternative training             -> sum(cross_training_hours)
    nr. strength trainings                       -> sum(strength_training_count)
    avg/min/max exertion                         -> avg/min/max(rpe / 10)            [PROXY: rescaled]
    avg/min/max training success                 -> avg/min/max(success / 5)         [PROXY: rescaled]
    total km Z3-Z4-Z5-T1-T2                      -> sum(Z3_km + Z4_km)               [PROXY: no Z5/T1/T2 data]
    nr. tough sessions (effort in Z5, T1 or T2)  -> count(Z4_km > 0)                 [PROXY]
    nr. days with interval session               -> 0.0                              [GAP: not captured live]
    total km Z5-T1-T2                            -> 0.0                              [GAP]
    max km Z5-T1-T2 one day                      -> 0.0                              [GAP]
    avg/min/max recovery                         -> 0.0                              [GAP: no recovery score live]
"""

from __future__ import annotations

from overstride.ingest.schema import SessionLog
from overstride.risk.calibration import RELATIVE_FEATURE_COLUMNS, WEEK0_FEATURE_COLUMNS

EPSILON = 1e-6


def _tough_km(session: SessionLog) -> float:
    return session.zone_km.get("Z3", 0.0) + session.zone_km.get("Z4", 0.0)


def aggregate_week(sessions: list[SessionLog]) -> dict[str, float]:
    """Aggregate one week's SessionLogs into the 22 dataset-aligned features."""
    running_days = [s for s in sessions if s.distance_km > 0]
    tough_km_per_day = [_tough_km(s) for s in sessions]

    features = {
        "nr. sessions": float(len(running_days)),
        "nr. rest days": float(sum(1 for s in sessions if s.rest_day)),
        "total kms": sum(s.distance_km for s in sessions),
        "max km one day": max((s.distance_km for s in sessions), default=0.0),
        "total km Z3-Z4-Z5-T1-T2": sum(tough_km_per_day),
        "nr. tough sessions (effort in Z5, T1 or T2)": float(
            sum(1 for s in sessions if s.zone_km.get("Z4", 0.0) > 0)
        ),
        "nr. days with interval session": 0.0,
        "total km Z3-4": sum(tough_km_per_day),
        "max km Z3-4 one day": max(tough_km_per_day, default=0.0),
        "total km Z5-T1-T2": 0.0,
        "max km Z5-T1-T2 one day": 0.0,
        "total hours alternative training": sum(s.cross_training_hours for s in sessions),
        "nr. strength trainings": float(sum(s.strength_training_count for s in sessions)),
        "avg recovery": 0.0,
        "min recovery": 0.0,
        "max recovery": 0.0,
    }

    if running_days:
        exertion = [s.rpe / 10 for s in running_days]
        success = [s.perceived_training_success / 5 for s in running_days]
        features["avg exertion"] = sum(exertion) / len(exertion)
        features["min exertion"] = min(exertion)
        features["max exertion"] = max(exertion)
        features["avg training success"] = sum(success) / len(success)
        features["min training success"] = min(success)
        features["max training success"] = max(success)
    else:
        features["avg exertion"] = 0.0
        features["min exertion"] = 0.0
        features["max exertion"] = 0.0
        features["avg training success"] = 0.0
        features["min training success"] = 0.0
        features["max training success"] = 0.0

    assert set(features) == set(WEEK0_FEATURE_COLUMNS)
    return features


def relative_features(
    week0_total_km: float, week1_total_km: float, week2_total_km: float
) -> dict[str, float]:
    """Week-over-week km ratios, matching Dataset #1's own safe-division
    convention (divide by denominator + 1e-6 rather than branching on zero).
    """
    features = {
        "rel total kms week 0_1": week0_total_km / (week1_total_km + EPSILON),
        "rel total kms week 0_2": week0_total_km / (week2_total_km + EPSILON),
        "rel total kms week 1_2": week1_total_km / (week2_total_km + EPSILON),
    }
    assert set(features) == set(RELATIVE_FEATURE_COLUMNS)
    return features


def aggregate_athlete_week(
    week0_sessions: list[SessionLog],
    week1_sessions: list[SessionLog],
    week2_sessions: list[SessionLog],
) -> dict[str, float]:
    """Full 22 + 3 feature vector for one athlete-week, given the current
    week's sessions and the two preceding weeks' sessions (for the relative
    km features).
    """
    week0 = aggregate_week(week0_sessions)
    week1_total_km = sum(s.distance_km for s in week1_sessions)
    week2_total_km = sum(s.distance_km for s in week2_sessions)
    return {**week0, **relative_features(week0["total kms"], week1_total_km, week2_total_km)}
