import pytest
from opytimizer.math import random


def test_generate_uniform_random_number():
    uniform_array = random.generate_uniform_random_number(low=0.0, high=1.0, size=5)

    assert uniform_array.shape == (5, )


def test_generate_gaussian_random_number():
    gaussian_array = random.generate_gaussian_random_number(mean=0.0, variance=1.0, size=3)

    assert gaussian_array.shape == (3, )
