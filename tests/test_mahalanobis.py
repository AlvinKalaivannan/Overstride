import numpy as np

from overstride.baseline.mahalanobis import fit_baseline, mahalanobis_sq, shrink_covariance


def test_mahalanobis_zero_at_mean():
    mean = np.array([1.0, 2.0, 3.0])
    cov = np.eye(3)
    assert mahalanobis_sq(mean, mean, cov) == 0.0


def test_mahalanobis_increases_with_distance():
    mean = np.zeros(3)
    cov = np.eye(3)
    near = np.array([0.1, 0.0, 0.0])
    far = np.array([5.0, 0.0, 0.0])
    assert mahalanobis_sq(far, mean, cov) > mahalanobis_sq(near, mean, cov)


def test_mahalanobis_vectorized_matches_scalar():
    rng = np.random.default_rng(0)
    mean = rng.normal(size=4)
    cov = np.eye(4) * 2
    x = rng.normal(size=(5, 4))
    batch = mahalanobis_sq(x, mean, cov)
    scalar = np.array([mahalanobis_sq(row, mean, cov) for row in x])
    np.testing.assert_allclose(batch, scalar)


def test_shrink_covariance_toward_population_for_small_n():
    population_cov = np.eye(2) * 10
    athlete_cov = np.eye(2) * 0.001
    shrunk = shrink_covariance(athlete_cov, population_cov, n_athlete=1, prior_strength=10.0)
    # n=1 with prior_strength=10 -> lam=10/11, dominated by population
    np.testing.assert_allclose(shrunk, population_cov * (10 / 11) + athlete_cov * (1 / 11))


def test_shrink_covariance_favors_athlete_as_n_grows():
    population_cov = np.eye(2) * 10
    athlete_cov = np.eye(2) * 1
    shrunk_small_n = shrink_covariance(athlete_cov, population_cov, n_athlete=2, prior_strength=10.0)
    shrunk_large_n = shrink_covariance(athlete_cov, population_cov, n_athlete=1000, prior_strength=10.0)
    # more athlete history -> covariance closer to the athlete's own (further from population)
    assert np.linalg.norm(shrunk_large_n - athlete_cov) < np.linalg.norm(shrunk_small_n - athlete_cov)


def test_fit_baseline_returns_invertible_covariance_for_single_week():
    population_cov = np.eye(3)
    clean_features = np.array([[1.0, 2.0, 3.0]])  # only one observed week
    mean, cov = fit_baseline(clean_features, population_cov, prior_strength=5.0)
    np.testing.assert_allclose(mean, [1.0, 2.0, 3.0])
    # must be invertible despite n=1 < n_features
    np.linalg.inv(cov)
