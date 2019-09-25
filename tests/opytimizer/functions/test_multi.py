import numpy as np
import pytest

from opytimizer.functions import multi


def test_multi_functions():
    new_multi = multi.MultiFunction()

    assert type(new_multi.functions) == list


def test_multi_functions_setter():
    new_multi = multi.MultiFunction()

    try:
        new_multi.functions = None
    except:
        new_multi.functions = [1, 2]

    assert len(new_multi.functions) == 2


def test_multi_weights():
    new_multi = multi.MultiFunction()

    assert type(new_multi.weights) == list


def test_multi_weights_setter():
    new_multi = multi.MultiFunction()

    try:
        new_multi.weights = None
    except:
        new_multi.weights = [0.5, 0.5]

    assert len(new_multi.weights) == 2


def test_multi_method():
    new_multi = multi.MultiFunction()

    assert new_multi.method == 'weight_sum'


def test_multi_method_setter():
    new_multi = multi.MultiFunction()

    try:
        new_multi.method = 'x'
    except:
        new_multi.method = 'weight_sum'

    assert new_multi.method == 'weight_sum'


def test_multi_build():
    new_multi = multi.MultiFunction()

    assert type(new_multi.functions) == list

    assert type(new_multi.pointer).__name__ == 'function'

    assert new_multi.built == True


def test_multi_create_strategy():
    def square(x):
        return x**2

    assert square(2) == 4

    def cube(x):
        return x**3

    assert cube(2) == 8

    new_multi = multi.MultiFunction(
        functions=[square, cube], weights=[0.5, 0.5])

    new_multi.pointer = new_multi._create_strategy('weight_sum')

    assert new_multi.pointer(2) == 6
