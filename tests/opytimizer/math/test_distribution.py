import numpy as np

from opytimizer.math import distribution


def test_generate_levy_distribution_applies_mantegna_step(monkeypatch):
    draws = iter([np.array([2.0]), np.array([-4.0])])
    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: next(draws))

    sample = distribution.generate_levy_distribution(beta=1, size=1)

    np.testing.assert_allclose(sample, [0.5])
