import numpy as np

from overstride.baseline.welford import WelfordBaseline


def test_welford_matches_batch_mean_and_covariance():
    rng = np.random.default_rng(42)
    batch = rng.normal(size=(50, 4))

    baseline = WelfordBaseline()
    for row in batch:
        baseline.update(row)

    np.testing.assert_allclose(baseline.mean, batch.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(baseline.covariance, np.cov(batch, rowvar=False), rtol=1e-10)


def test_welford_n_increments_per_update():
    baseline = WelfordBaseline()
    for i in range(5):
        baseline.update(np.array([float(i), float(i) * 2]))
    assert baseline.n == 5


def test_welford_covariance_zero_matrix_for_single_observation():
    baseline = WelfordBaseline()
    baseline.update(np.array([1.0, 2.0, 3.0]))
    assert baseline.n == 1
    np.testing.assert_array_equal(baseline.covariance, np.zeros((3, 3)))


def test_welford_roundtrip_serialization():
    rng = np.random.default_rng(1)
    baseline = WelfordBaseline()
    for row in rng.normal(size=(10, 3)):
        baseline.update(row)

    restored = WelfordBaseline.from_dict(baseline.to_dict())

    assert restored.n == baseline.n
    np.testing.assert_allclose(restored.mean, baseline.mean)
    np.testing.assert_allclose(restored.covariance, baseline.covariance)

    # updates continue seamlessly after a save/load round trip
    next_week = rng.normal(size=3)
    baseline.update(next_week)
    restored.update(next_week)
    np.testing.assert_allclose(restored.mean, baseline.mean)
    np.testing.assert_allclose(restored.covariance, baseline.covariance)
