import numpy as np
import pytest
from scipy.stats import chi2

from overstride.baseline.welford import WelfordBaseline
from overstride.biomechanics.score import ANOMALY_ALPHA, COLD_START_MIN_SESSIONS, process_session

FEATURE_COLUMNS = ["a", "b", "c"]


def make_features(a, b, c):
    return {"a": a, "b": b, "c": c}


def test_cold_start_before_min_sessions_never_scores():
    baseline = WelfordBaseline()
    population_cov = np.eye(3)
    for session_num in range(COLD_START_MIN_SESSIONS):
        result = process_session(
            baseline, make_features(float(session_num), 0.0, 0.0), FEATURE_COLUMNS, population_cov
        )
        assert result.status == "building_baseline"
        assert result.d2 is None
        assert result.flagged is None
    assert baseline.n == COLD_START_MIN_SESSIONS


def test_scoring_starts_once_min_clean_sessions_reached():
    baseline = WelfordBaseline()
    population_cov = np.eye(3)
    for _ in range(COLD_START_MIN_SESSIONS):
        baseline.update([0.0, 0.0, 0.0])

    result = process_session(baseline, make_features(1.0, 2.0, 3.0), FEATURE_COLUMNS, population_cov)

    assert result.status == "scored"
    assert result.d2 is not None
    assert result.flagged is not None


def test_process_session_reproduces_expected_d2_and_flag():
    baseline = WelfordBaseline()
    population_cov = np.eye(3)
    prior_strength = 1e6
    for _ in range(COLD_START_MIN_SESSIONS):
        baseline.update([0.0, 0.0, 0.0])

    result = process_session(
        baseline, make_features(1.0, 2.0, 3.0), FEATURE_COLUMNS, population_cov, prior_strength=prior_strength
    )

    # independently re-derive from the same shrinkage + Mahalanobis + chi-squared formulas
    lam = prior_strength / (COLD_START_MIN_SESSIONS + prior_strength)
    expected_cov = lam * population_cov + (1 - lam) * np.zeros((3, 3))
    diff = np.array([1.0, 2.0, 3.0])
    expected_d2 = diff @ np.linalg.inv(expected_cov) @ diff
    expected_critical = chi2.ppf(1 - ANOMALY_ALPHA, df=3)

    assert result.d2 == pytest.approx(expected_d2, rel=1e-9)
    assert result.critical_value == pytest.approx(expected_critical, rel=1e-9)
    assert result.flagged == bool(expected_d2 >= expected_critical)
    # hand-computed anchor: cov ~= I (prior_strength dwarfs n=5), so D^2 ~= 1^2+2^2+3^2 = 14
    assert result.d2 == pytest.approx(14.0, rel=1e-3)


def test_feature_contributions_sum_to_d2():
    baseline = WelfordBaseline()
    population_cov = np.eye(3)
    for _ in range(COLD_START_MIN_SESSIONS):
        baseline.update([0.0, 0.0, 0.0])

    result = process_session(
        baseline, make_features(1.0, 2.0, 3.0), FEATURE_COLUMNS, population_cov, prior_strength=1e6
    )

    assert set(result.feature_contributions) == set(FEATURE_COLUMNS)
    assert sum(result.feature_contributions.values()) == pytest.approx(result.d2, rel=1e-9)


def test_flagged_session_excluded_from_baseline_update():
    baseline = WelfordBaseline()
    population_cov = np.eye(3) * 0.01  # tight population prior -> large deviations are easily flagged
    for _ in range(COLD_START_MIN_SESSIONS):
        baseline.update([0.0, 0.0, 0.0])
    n_before = baseline.n

    result = process_session(
        baseline, make_features(10.0, 10.0, 10.0), FEATURE_COLUMNS, population_cov, prior_strength=1e6
    )

    assert result.flagged is True
    assert baseline.n == n_before


def test_normal_session_included_in_baseline_update():
    baseline = WelfordBaseline()
    population_cov = np.eye(3)
    for _ in range(COLD_START_MIN_SESSIONS):
        baseline.update([0.0, 0.0, 0.0])
    n_before = baseline.n

    result = process_session(
        baseline, make_features(0.01, 0.0, 0.0), FEATURE_COLUMNS, population_cov, prior_strength=1e6
    )

    assert result.flagged is False
    assert baseline.n == n_before + 1


def test_nan_feature_raises():
    baseline = WelfordBaseline()
    population_cov = np.eye(3)

    with pytest.raises(ValueError):
        process_session(baseline, make_features(float("nan"), 0.0, 0.0), FEATURE_COLUMNS, population_cov)
