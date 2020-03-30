import numpy as np
import pytest

from opytimizer.functions import weighted


def test_weighted_functions():
    new_weighted = weighted.WeightedFunction()

    assert type(new_weighted.functions) == list


def test_weighted_functions_setter():
    new_weighted = weighted.WeightedFunction()

    try:
        new_weighted.functions = None
    except:
        new_weighted.functions = [1, 2]

    assert len(new_weighted.functions) == 2


def test_weighted_weights():
    new_weighted = weighted.WeightedFunction()

    assert type(new_weighted.weights) == list


def test_weighted_weights_setter():
    new_weighted = weighted.WeightedFunction()

    try:
        new_weighted.weights = None
    except:
        new_weighted.weights = [0.5, 0.5]

    assert len(new_weighted.weights) == 2


def test_weighted_build():
    new_weighted = weighted.WeightedFunction()

    assert type(new_weighted.functions) == list

    assert type(new_weighted.pointer).__name__ == 'function'

    assert new_weighted.built == True


def test_weighted_create_multi_objective():
    def square(x):
        return x**2

    assert square(2) == 4

    def cube(x):
        return x**3

    assert cube(2) == 8

    new_weighted = weighted.WeightedFunction(
        functions=[square, cube], weights=[0.5, 0.5])

    new_weighted._create_multi_objective()

    assert new_weighted.pointer(2) == 6
