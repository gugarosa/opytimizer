import pytest

from opytimizer.math import random


def test_generate_uniform():
    uniform_array = random.generate_uniform_random_number(0, 1, 5)

    assert uniform_array.shape == (5, )


def test_generate_gaussian():
    gaussian_array = random.generate_gaussian_random_number(0, 1, 3)

    assert gaussian_array.shape == (3, )
