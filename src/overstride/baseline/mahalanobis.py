"""Per-athlete Mahalanobis baseline: mean/covariance estimation and distance scoring.

Shared by Stage 2 (training load) and Stage 3 (biomechanics) — both measure
deviation from an athlete's own baseline the same way, just over different
feature spaces.
"""

from __future__ import annotations

import numpy as np


def shrink_covariance(
    athlete_cov: np.ndarray,
    population_cov: np.ndarray,
    n_athlete: int,
    prior_strength: float = 10.0,
) -> np.ndarray:
    """Blend an athlete's covariance toward a population-level covariance.

    Weight on the population prior shrinks as the athlete accumulates more
    clean weeks: lam = prior_strength / (n_athlete + prior_strength).
    Guarantees a full-rank, invertible result even when n_athlete is smaller
    than the number of features, as long as population_cov is full rank.
    """
    lam = prior_strength / (n_athlete + prior_strength)
    return lam * population_cov + (1 - lam) * athlete_cov


def fit_baseline(
    clean_features: np.ndarray,
    population_cov: np.ndarray,
    prior_strength: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an athlete's (mean, shrunk covariance) baseline from their clean weeks."""
    if clean_features.ndim != 2:
        raise ValueError("clean_features must be a 2D (n_weeks, n_features) array")
    n_athlete = clean_features.shape[0]
    mean = clean_features.mean(axis=0)
    if n_athlete > 1:
        athlete_cov = np.cov(clean_features, rowvar=False)
    else:
        athlete_cov = np.zeros_like(population_cov)
    cov = shrink_covariance(athlete_cov, population_cov, n_athlete, prior_strength)
    return mean, cov


def mahalanobis_sq(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """D^2 = (x - mean)^T Sigma^-1 (x - mean), vectorized over rows of x."""
    diff = np.atleast_2d(x) - mean
    cov_inv = np.linalg.inv(cov)
    d2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    return d2 if x.ndim > 1 else d2[0]


def feature_contributions(
    diff: np.ndarray, cov_inv: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    """Exact additive decomposition of D^2 = diff^T @ cov_inv @ diff into
    per-feature terms diff_i * (cov_inv @ diff)_i, which sum to D^2.
    Sorted by descending magnitude so the top entry is the "driver".

    Shared by Stage 2 and Stage 3's scoring -- this is generic Mahalanobis
    decomposition, not specific to either feature space.
    """
    terms = diff * (cov_inv @ diff)
    pairs = sorted(zip(feature_names, terms.tolist()), key=lambda p: abs(p[1]), reverse=True)
    return dict(pairs)
