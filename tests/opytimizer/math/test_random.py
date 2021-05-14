import numpy as np

from opytimizer.math import random

np.random.seed(0)


def test_generate_binary_random_number():
    binary_array = random.generate_binary_random_number(5)

    assert binary_array.shape == (5, )


def test_generate_exponential_random_number():
    exponential_array = random.generate_exponential_random_number(1, 5)

    assert exponential_array.shape == (5, )


def test_generate_gamma_random_number():
    gamma_array = random.generate_gamma_random_number(1, 1, 5)

    assert gamma_array.shape == (5, )


def test_generate_integer_random_number():
    integer_array = random.generate_integer_random_number(0, 1, None, 5)

    assert integer_array.shape == (5, )

    integer_array = random.generate_integer_random_number(0, 10, 1, 9)

    assert integer_array.shape == (9, )


def test_generate_uniform_random_number():
    uniform_array = random.generate_uniform_random_number(0, 1, 5)

    assert uniform_array.shape == (5, )


def test_generate_gaussian_random_number():
    gaussian_array = random.generate_gaussian_random_number(0, 1, 3)

    assert gaussian_array.shape == (3, )
