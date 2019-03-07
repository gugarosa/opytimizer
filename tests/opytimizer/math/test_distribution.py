import pytest
from opytimizer.math import distribution


def test_generate_levy_distribution():
    levy_array = distribution.generate_levy_distribution(beta=0.1, size=5)

    assert levy_array.shape == (5, )
