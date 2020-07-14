from opytimizer.math import random


def test_generate_binary_random_number():
    binary_array = random.generate_binary_random_number(5)

    assert binary_array.shape == (5, )


def test_generate_uniform_random_number():
    uniform_array = random.generate_uniform_random_number(0, 1, 5)

    assert uniform_array.shape == (5, )


def test_generate_gaussian_random_number():
    gaussian_array = random.generate_gaussian_random_number(0, 1, 3)

    assert gaussian_array.shape == (3, )
