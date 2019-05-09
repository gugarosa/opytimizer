import numpy as np
import pytest

from opytimizer.math import benchmark


def test_alpine1():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.alpine1(x)

    assert y > 0


def test_alpine2():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.alpine2(x)

    assert y < 0


def test_csendes():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.csendes(x)

    assert y > 0


def test_exponential():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.exponential(x)

    assert y < 0


def test_rastringin():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.rastringin(x)

    assert y > 0


def test_salomon():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.salomon(x)

    assert y > 0


def test_schwefel():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.schwefel(x)

    assert y > 0


def test_sphere():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.sphere(x)

    assert y > 0
