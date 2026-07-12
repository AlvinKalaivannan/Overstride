"""Stage 2 runtime scoring: incrementally-updated per-athlete baseline ->
Mahalanobis D^2 -> frozen logistic regression -> probability + feature
contribution breakdown.

Consumes the artifact frozen by scripts/run_stage2_calibration.py
(models/stage2_logistic_coeffs.json) — never refits it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from overstride.baseline.mahalanobis import shrink_covariance
from overstride.baseline.welford import WelfordBaseline

COLD_START_MIN_WEEKS = 8
ANOMALY_THRESHOLD = 0.5


@dataclass
class FrozenModel:
    feature_columns: list[str]
    prior_strength: float
    population_cov: np.ndarray
    intercept: float
    coef_d2: float

    @classmethod
    def load(cls, path: str | Path) -> "FrozenModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_columns=data["feature_columns"],
            prior_strength=data["prior_strength"],
            population_cov=np.array(data["population_cov"]),
            intercept=data["intercept"],
            coef_d2=data["coef_d2"],
        )


@dataclass
class ScoreResult:
    status: str  # "building_baseline" or "scored"
    clean_weeks: int
    d2: float | None = None
    probability: float | None = None
    feature_contributions: dict[str, float] | None = None


def _logistic(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def _feature_contributions(
    diff: np.ndarray, cov_inv: np.ndarray, feature_columns: list[str]
) -> dict[str, float]:
    """Exact additive decomposition of D^2 = diff^T @ cov_inv @ diff into
    per-feature terms diff_i * (cov_inv @ diff)_i, which sum to D^2.
    Sorted by descending magnitude so the top entry is the "driver".
    """
    terms = diff * (cov_inv @ diff)
    pairs = sorted(zip(feature_columns, terms.tolist()), key=lambda p: abs(p[1]), reverse=True)
    return dict(pairs)


def process_week(
    baseline: WelfordBaseline,
    features: np.ndarray,
    model: FrozenModel,
    cold_start_min_weeks: int = COLD_START_MIN_WEEKS,
    anomaly_threshold: float = ANOMALY_THRESHOLD,
) -> ScoreResult:
    """Score one athlete-week and update the baseline in place.

    Cold-start weeks (before `cold_start_min_weeks` clean weeks have been
    seen) are always incorporated into the baseline — there's no score yet
    to judge them anomalous by. Once scoring starts, a week is excluded
    from the baseline update whenever its own predicted probability meets
    or exceeds `anomaly_threshold` (baseline integrity: a risky week
    shouldn't redefine what "normal" means going forward).
    """
    features = np.asarray(features, dtype=float)

    if baseline.n < cold_start_min_weeks:
        result = ScoreResult(status="building_baseline", clean_weeks=baseline.n)
        baseline.update(features)
        return result

    cov = shrink_covariance(baseline.covariance, model.population_cov, baseline.n, model.prior_strength)
    cov_inv = np.linalg.inv(cov)
    diff = features - baseline.mean
    d2 = float(diff @ cov_inv @ diff)
    probability = float(_logistic(model.intercept + model.coef_d2 * d2))
    contributions = _feature_contributions(diff, cov_inv, model.feature_columns)

    result = ScoreResult(
        status="scored",
        clean_weeks=baseline.n,
        d2=d2,
        probability=probability,
        feature_contributions=contributions,
    )

    if probability < anomaly_threshold:
        baseline.update(features)

    return result
