from datetime import date, timedelta

import pytest

from overstride.ingest.aggregate import aggregate_athlete_week, aggregate_week, relative_features
from overstride.ingest.schema import SessionLog
from overstride.risk.calibration import FEATURE_COLUMNS

START = date(2026, 1, 5)


def day(n: int) -> date:
    return START + timedelta(days=n)


def build_week() -> list[SessionLog]:
    return [
        SessionLog(
            date=day(0), distance_km=10.0, zone_km={"Z1": 6.0, "Z3": 4.0},
            rpe=5, perceived_training_success=4, strength_training_count=1,
        ),
        SessionLog(date=day(1), distance_km=0.0, rest_day=True),
        SessionLog(
            date=day(2), distance_km=8.0, zone_km={"Z1": 8.0},
            rpe=3, perceived_training_success=3, cross_training_hours=0.5,
        ),
        SessionLog(
            date=day(3), distance_km=12.0, zone_km={"Z1": 6.0, "Z4": 6.0},
            rpe=8, perceived_training_success=5,
        ),
        SessionLog(date=day(4), distance_km=0.0, rest_day=True),
        SessionLog(
            date=day(5), distance_km=5.0, zone_km={"Z2": 5.0},
            rpe=4, perceived_training_success=4,
        ),
        SessionLog(date=day(6), distance_km=0.0, rest_day=True),
    ]


def test_aggregate_week_matches_hand_computed_vector():
    features = aggregate_week(build_week())

    expected = {
        "nr. sessions": 4.0,
        "nr. rest days": 3.0,
        "total kms": 35.0,
        "max km one day": 12.0,
        "total km Z3-Z4-Z5-T1-T2": 10.0,
        "nr. tough sessions (effort in Z5, T1 or T2)": 1.0,
        "nr. days with interval session": 0.0,
        "total km Z3-4": 10.0,
        "max km Z3-4 one day": 6.0,
        "total km Z5-T1-T2": 0.0,
        "max km Z5-T1-T2 one day": 0.0,
        "total hours alternative training": 0.5,
        "nr. strength trainings": 1.0,
        "avg exertion": 0.5,
        "min exertion": 0.3,
        "max exertion": 0.8,
        "avg training success": 0.8,
        "min training success": 0.6,
        "max training success": 1.0,
        "avg recovery": 0.0,
        "min recovery": 0.0,
        "max recovery": 0.0,
    }

    assert features.keys() == expected.keys()
    for key, value in expected.items():
        assert features[key] == pytest.approx(value), key


def test_aggregate_week_empty_week_is_all_zero():
    features = aggregate_week([])
    assert all(v == 0.0 for v in features.values())


def test_relative_features_matches_dataset_epsilon_convention():
    result = relative_features(week0_total_km=35.0, week1_total_km=30.0, week2_total_km=0.0)

    eps = 1e-6
    assert result["rel total kms week 0_1"] == pytest.approx(35.0 / (30.0 + eps), rel=1e-9)
    assert result["rel total kms week 0_2"] == pytest.approx(35.0 / eps, rel=1e-9)
    assert result["rel total kms week 1_2"] == pytest.approx(30.0 / eps, rel=1e-9)


def test_aggregate_athlete_week_produces_full_feature_set():
    week0 = build_week()
    week1 = [SessionLog(date=day(-7), distance_km=30.0, zone_km={"Z1": 30.0}, rpe=5, perceived_training_success=4)]
    week2: list[SessionLog] = []

    features = aggregate_athlete_week(week0, week1, week2)

    assert set(features) == set(FEATURE_COLUMNS)
    assert features["total kms"] == 35.0
    assert features["rel total kms week 0_1"] == pytest.approx(35.0 / (30.0 + 1e-6), rel=1e-9)
    assert features["rel total kms week 0_2"] == pytest.approx(35.0 / 1e-6, rel=1e-9)
