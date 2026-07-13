"""Stage 3 runtime scoring: incrementally-updated per-athlete kinematic
baseline -> Mahalanobis D^2 -> chi-squared anomaly flag + feature-contribution
breakdown.

Unlike Stage 2, there's no injury-labeled biomechanics dataset to calibrate a
deviation-to-risk mapping against (see the README), so this reports a flag,
not a probability -- there's no frozen logistic regression here, and no
FrozenModel-style artifact to load. Under the null hypothesis that a session's
features are normal for this athlete, D^2 is chi-squared distributed with k
degrees of freedom (k = the number of features), which gives a principled
significance-level threshold instead of an arbitrary cutoff.

`population_cov` is passed in directly (same convention as
baseline.mahalanobis.fit_baseline/shrink_covariance) rather than loaded from a
frozen artifact -- the real value will come from the marathon-footage
population baseline once that dataset exists; until then callers can supply
any population covariance (e.g. identity, or one estimated from whatever
footage is on hand) and get correct scoring against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from overstride.baseline.mahalanobis import feature_contributions, shrink_covariance
from overstride.baseline.welford import WelfordBaseline

# Fewer than Stage 2's 8 clean weeks: the README notes footage arrives at a
# lower cadence than session logs ("filming has real friction"), so requiring
# as many clean sessions before scoring starts would leave athletes without a
# score for a long time. Shrinkage toward the population prior compensates for
# the noisier per-athlete covariance this implies at low n.
COLD_START_MIN_SESSIONS = 5
ANOMALY_ALPHA = 0.05  # flag when D^2 exceeds the (1 - alpha) chi-squared critical value


@dataclass
class SessionScoreResult:
    status: str  # "building_baseline" or "scored"
    clean_sessions: int
    d2: float | None = None
    critical_value: float | None = None
    flagged: bool | None = None
    feature_contributions: dict[str, float] | None = None


def process_session(
    baseline: WelfordBaseline,
    features: dict[str, float],
    feature_columns: list[str],
    population_cov: np.ndarray,
    prior_strength: float = 10.0,
    cold_start_min_sessions: int = COLD_START_MIN_SESSIONS,
    alpha: float = ANOMALY_ALPHA,
) -> SessionScoreResult:
    """Score one footage session (e.g. overstride.pose.kinematics.aggregate_session's
    output) and update the baseline in place.

    Cold-start sessions (before `cold_start_min_sessions` clean sessions have
    been seen) are always incorporated into the baseline -- there's no score
    yet to judge them anomalous by. Once scoring starts, a session is excluded
    from the baseline update whenever it's flagged (baseline integrity: an
    anomalous session shouldn't redefine what "normal" means going forward --
    mirrors Stage 2's rule in overstride.risk.score.process_week).
    """
    vector = np.array([features[name] for name in feature_columns], dtype=float)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"session features contain NaN/inf: {dict(zip(feature_columns, vector))}")

    if baseline.n < cold_start_min_sessions:
        result = SessionScoreResult(status="building_baseline", clean_sessions=baseline.n)
        baseline.update(vector)
        return result

    cov = shrink_covariance(baseline.covariance, population_cov, baseline.n, prior_strength)
    cov_inv = np.linalg.inv(cov)
    diff = vector - baseline.mean
    d2 = float(diff @ cov_inv @ diff)
    critical_value = float(chi2.ppf(1 - alpha, df=len(feature_columns)))
    flagged = d2 >= critical_value
    contributions = feature_contributions(diff, cov_inv, feature_columns)

    result = SessionScoreResult(
        status="scored",
        clean_sessions=baseline.n,
        d2=d2,
        critical_value=critical_value,
        flagged=flagged,
        feature_contributions=contributions,
    )

    if not flagged:
        baseline.update(vector)

    return result
