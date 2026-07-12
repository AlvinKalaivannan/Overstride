import json

import numpy as np
import pytest

from overstride.baseline.welford import WelfordBaseline
from overstride.risk.score import ANOMALY_THRESHOLD, COLD_START_MIN_WEEKS, FrozenModel, process_week

FEATURE_COLUMNS = ["a", "b", "c"]


def make_model(prior_strength=1e6, intercept=-5.0, coef_d2=1.0):
    return FrozenModel(
        feature_columns=FEATURE_COLUMNS,
        prior_strength=prior_strength,
        population_cov=np.eye(3),
        intercept=intercept,
        coef_d2=coef_d2,
    )


def test_cold_start_before_min_weeks_never_scores():
    model = make_model()
    baseline = WelfordBaseline()
    for week_num in range(COLD_START_MIN_WEEKS):
        result = process_week(baseline, [float(week_num), 0.0, 0.0], model)
        assert result.status == "building_baseline"
        assert result.probability is None
        assert result.d2 is None
    # every cold-start week is incorporated into the baseline
    assert baseline.n == COLD_START_MIN_WEEKS


def test_scoring_starts_once_min_clean_weeks_reached():
    model = make_model()
    baseline = WelfordBaseline()
    for _ in range(COLD_START_MIN_WEEKS):
        baseline.update([0.0, 0.0, 0.0])

    result = process_week(baseline, [1.0, 2.0, 3.0], model)

    assert result.status == "scored"
    assert result.d2 is not None
    assert result.probability is not None


def test_process_week_reproduces_expected_d2_and_probability():
    model = make_model()
    baseline = WelfordBaseline()
    for _ in range(COLD_START_MIN_WEEKS):
        baseline.update([0.0, 0.0, 0.0])

    features = np.array([1.0, 2.0, 3.0])
    result = process_week(baseline, features, model)

    # independently re-derive the expected value from the shrinkage + Mahalanobis + logistic formulas
    lam = model.prior_strength / (COLD_START_MIN_WEEKS + model.prior_strength)
    expected_cov = lam * model.population_cov + (1 - lam) * np.zeros((3, 3))
    diff = features - np.zeros(3)
    expected_d2 = diff @ np.linalg.inv(expected_cov) @ diff
    expected_prob = 1.0 / (1.0 + np.exp(-(model.intercept + model.coef_d2 * expected_d2)))

    assert result.d2 == pytest.approx(expected_d2, rel=1e-9)
    assert result.probability == pytest.approx(expected_prob, rel=1e-9)
    # hand-computed anchor: cov ~= I (prior_strength dwarfs n=8), so D^2 ~= 1^2+2^2+3^2 = 14
    assert result.d2 == pytest.approx(14.0, rel=1e-3)


def test_feature_contributions_sum_to_d2():
    model = make_model()
    baseline = WelfordBaseline()
    for _ in range(COLD_START_MIN_WEEKS):
        baseline.update([0.0, 0.0, 0.0])

    result = process_week(baseline, [1.0, 2.0, 3.0], model)

    assert set(result.feature_contributions) == set(FEATURE_COLUMNS)
    assert sum(result.feature_contributions.values()) == pytest.approx(result.d2, rel=1e-9)


def test_anomalous_week_excluded_from_baseline_update():
    model = make_model()
    baseline = WelfordBaseline()
    for _ in range(COLD_START_MIN_WEEKS):
        baseline.update([0.0, 0.0, 0.0])
    n_before = baseline.n

    result = process_week(baseline, [10.0, 10.0, 10.0], model)

    assert result.probability >= ANOMALY_THRESHOLD
    assert baseline.n == n_before


def test_normal_week_included_in_baseline_update():
    model = make_model()
    baseline = WelfordBaseline()
    for _ in range(COLD_START_MIN_WEEKS):
        baseline.update([0.0, 0.0, 0.0])
    n_before = baseline.n

    result = process_week(baseline, [0.05, 0.0, 0.0], model)

    assert result.probability < ANOMALY_THRESHOLD
    assert baseline.n == n_before + 1


def test_frozen_model_load_roundtrip(tmp_path):
    payload = {
        "feature_columns": FEATURE_COLUMNS,
        "prior_strength": 10.0,
        "population_cov": np.eye(3).tolist(),
        "intercept": -1.0,
        "coef_d2": 0.5,
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    model = FrozenModel.load(path)

    assert model.feature_columns == FEATURE_COLUMNS
    assert model.prior_strength == 10.0
    np.testing.assert_allclose(model.population_cov, np.eye(3))
    assert model.intercept == -1.0
    assert model.coef_d2 == 0.5
