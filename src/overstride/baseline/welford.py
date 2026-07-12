"""Incremental per-athlete (mean, covariance) baseline, updated one week at a time.

Generalizes Welford's single-pass variance algorithm to the multivariate
covariance case: track a running mean and an M2 accumulator (sum of
outer-product deviations), from which the sample covariance is recovered
in O(1) at any point without revisiting past weeks.
"""

from __future__ import annotations

import numpy as np


class WelfordBaseline:
    def __init__(self, n: int = 0, mean: np.ndarray | None = None, m2: np.ndarray | None = None):
        self.n = n
        self.mean = mean
        self.m2 = m2

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        if self.mean is None:
            self.mean = np.zeros_like(x)
            self.m2 = np.zeros((x.size, x.size))
        self.n += 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = x - self.mean
        self.m2 = self.m2 + np.outer(delta, delta2)

    @property
    def covariance(self) -> np.ndarray:
        """Sample covariance (ddof=1). Zero matrix if fewer than 2 observations."""
        if self.n < 2:
            return np.zeros_like(self.m2) if self.m2 is not None else None
        return self.m2 / (self.n - 1)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "m2": self.m2.tolist() if self.m2 is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WelfordBaseline":
        mean = np.array(d["mean"]) if d.get("mean") is not None else None
        m2 = np.array(d["m2"]) if d.get("m2") is not None else None
        return cls(n=d.get("n", 0), mean=mean, m2=m2)
