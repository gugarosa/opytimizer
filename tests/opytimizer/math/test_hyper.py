import numpy as np

from opytimizer.math import hyper


def test_norm():
    array = np.array([[1, 1]])

    norm_array = hyper.norm(array)

    assert norm_array > 0


def test_span():
    array = np.array([[0.5, 0.75, 0.5, 0.9]])

    lb = [0]

    ub = [10]

    span_array = hyper.span(array, lb, ub)

    assert span_array > 0


def test_span_to_hyper_value():
    lb = np.full(1, 10)
    ub = np.full(1, 20)

    @hyper.span_to_hyper_value(lb, ub)
    def call(x):
        return np.sum(x)

    y = call(np.array([[0.5], [0.5]]))

    assert y == 30
