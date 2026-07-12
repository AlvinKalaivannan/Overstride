from datetime import date

import pytest

from overstride.ingest.schema import SessionLog


def make_valid_log(**overrides) -> dict:
    defaults = dict(
        date=date(2026, 1, 5),
        distance_km=8.5,
        zone_km={"Z1": 6.0, "Z3": 2.5},
        rpe=7,
        perceived_training_success=4,
        strength_training_count=1,
        cross_training_hours=0.5,
        rest_day=False,
    )
    defaults.update(overrides)
    return defaults


def test_valid_session_log_constructs():
    log = SessionLog(**make_valid_log())
    assert log.distance_km == 8.5


def test_valid_rest_day_constructs():
    log = SessionLog(date=date(2026, 1, 6), distance_km=0, rest_day=True)
    assert log.rest_day is True


@pytest.mark.parametrize("rpe", [0, 11, -1])
def test_rpe_out_of_range_rejected(rpe):
    with pytest.raises(ValueError, match="rpe"):
        SessionLog(**make_valid_log(rpe=rpe))


@pytest.mark.parametrize("success", [0, 6, -1])
def test_perceived_success_out_of_range_rejected(success):
    with pytest.raises(ValueError, match="perceived_training_success"):
        SessionLog(**make_valid_log(perceived_training_success=success))


def test_negative_distance_rejected():
    with pytest.raises(ValueError, match="distance_km"):
        SessionLog(**make_valid_log(distance_km=-1.0))


def test_unknown_zone_key_rejected():
    with pytest.raises(ValueError, match="zone_km"):
        SessionLog(**make_valid_log(zone_km={"Z9": 1.0}))


def test_negative_zone_km_rejected():
    with pytest.raises(ValueError, match="zone_km"):
        SessionLog(**make_valid_log(zone_km={"Z1": -1.0}))


def test_zone_km_exceeding_distance_rejected():
    with pytest.raises(ValueError, match="exceeds distance_km"):
        SessionLog(**make_valid_log(distance_km=5.0, zone_km={"Z1": 4.0, "Z3": 4.0}))


def test_negative_strength_count_rejected():
    with pytest.raises(ValueError, match="strength_training_count"):
        SessionLog(**make_valid_log(strength_training_count=-1))


def test_negative_cross_training_hours_rejected():
    with pytest.raises(ValueError, match="cross_training_hours"):
        SessionLog(**make_valid_log(cross_training_hours=-0.5))


def test_rest_day_with_distance_rejected():
    with pytest.raises(ValueError, match="rest_day"):
        SessionLog(date=date(2026, 1, 6), distance_km=5.0, rest_day=True)


def test_rest_day_with_rpe_rejected():
    with pytest.raises(ValueError, match="rest_day"):
        SessionLog(
            date=date(2026, 1, 6), distance_km=0.0, rest_day=True, rpe=5,
            perceived_training_success=None,
        )


def test_running_day_without_rpe_rejected():
    log = make_valid_log(rpe=None)
    with pytest.raises(ValueError, match="rpe and perceived_training_success are required"):
        SessionLog(**log)


def test_running_day_without_perceived_success_rejected():
    log = make_valid_log(perceived_training_success=None)
    with pytest.raises(ValueError, match="rpe and perceived_training_success are required"):
        SessionLog(**log)
