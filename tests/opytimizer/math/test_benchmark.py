import numpy as np
import pytest
from opytimizer.math import benchmark


def test_exponential():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.exponential(x)

    assert y < 0


def test_sphere():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.sphere(x)

    assert y > 0
