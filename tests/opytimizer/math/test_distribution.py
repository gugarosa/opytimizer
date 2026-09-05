import numpy as np

from opytimizer.math import distribution


def test_generate_levy_distribution_applies_mantegna_step(monkeypatch):
    draws = iter([np.array([2.0]), np.array([-4.0])])
    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: next(draws))

    sample = distribution.generate_levy_distribution(beta=1, size=1)

    np.testing.assert_allclose(sample, [0.5])


def test_generate_levy_distribution_uses_mantegna_standard_deviation(monkeypatch):
    calls = []

    def normal(loc=0.0, scale=1.0, size=None):
        calls.append((loc, scale, size))
        return np.full(size, scale)

    monkeypatch.setattr(np.random, "normal", normal)

    sample = distribution.generate_levy_distribution(beta=1.5, size=3)

    sigma = 0.6965745025576967
    np.testing.assert_allclose(calls[0], [0, sigma, 3])
    assert calls[1] == (0.0, 1.0, 3)
    np.testing.assert_allclose(sample, np.full(3, sigma))
