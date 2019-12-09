import numpy as np
import pytest

from opytimizer.math import benchmark


def test_ackley1():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.ackley1(x)

    assert np.round(y, 2) == 4.64


def test_alpine1():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.alpine1(x)

    assert np.round(y, 2) == 2.46


def test_alpine2():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.alpine2(x)

    assert np.round(y, 2) == -0.08


def test_brown():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.brown(x)

    assert np.round(y, 2) == 3.42


def test_chung_reynolds():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.chung_reynolds(x)

    assert np.round(y, 2) == 6.25


def test_cosine_mixture():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.cosine_mixture(x)

    assert np.round(y, 2) == -2.7


def test_csendes():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.csendes(x)

    assert np.round(y, 2) == 5.77


def test_deb1():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.deb1(x)

    assert np.round(y, 2) == -0.5


def test_deb2():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.deb2(x)

    assert np.round(y, 2) == -0.16


def test_exponential():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.exponential(x)

    assert np.round(y, 2) == -0.29


def test_quintic():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.quintic(x)

    assert np.round(y, 2) == 36.31


def test_rastringin():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.rastringin(x)

    assert np.round(y, 2) == 42.5


def test_salomon():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.salomon(x)

    assert np.round(y, 2) == 2.03


def test_schwefel():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.schwefel(x)

    assert np.round(y, 2) == 1673.6


def test_schumer_steiglitz():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.schumer_steiglitz(x)

    assert np.round(y, 2) == 2.12


def test_sphere():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.sphere(x)

    assert np.round(y, 2) == 2.5


def test_styblinski_tang():
    x = np.array([0.5, 0.5, 1, 1])

    y = benchmark.styblinski_tang(x)

    assert np.round(y, 2) == -11.44
