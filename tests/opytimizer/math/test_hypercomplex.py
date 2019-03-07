import numpy as np
import pytest
from opytimizer.math import hypercomplex


def test_norm():
    array = np.array([[1, 1]])

    norm_array = hypercomplex.norm(array)

    assert norm_array > 0


def test_span():
    array = np.array([[0.5, 0.75, 0.5, 0.9]])

    lb = [0]

    ub = [10]

    span_array = hypercomplex.span(array, lb, ub)

    assert span_array > 0
