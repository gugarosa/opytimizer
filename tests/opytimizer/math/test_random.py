import numpy as np

from opytimizer.math import random


def test_integer_preserves_scalar_and_array_behavior():
    assert np.isscalar(random.integer(0, 2))
    assert random.integer(0, 2, size=(2, 3)).shape == (2, 3)


def test_integer_excludes_without_redrawing(monkeypatch):
    calls = []

    def randint(low, high, size):
        calls.append((low, high, size))
        return np.array([0, 1, 0, 1])

    monkeypatch.setattr(np.random, "randint", randint)

    values = random.integer(0, 3, exclude=1, size=4)

    assert calls == [(0, 2, 4)]
    np.testing.assert_array_equal(values, [0, 2, 0, 2])
