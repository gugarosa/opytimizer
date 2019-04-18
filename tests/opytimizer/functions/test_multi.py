import numpy as np
import pytest
from opytimizer.functions import multi


def test_multi_functions():
    new_multi = multi.Multi()

    assert type(new_multi.functions) == list


def test_multi_functions_setter():
    new_multi = multi.Multi()

    new_multi.functions = [1, 2]

    assert len(new_multi.functions) == 2


def test_multi_weights():
    new_multi = multi.Multi(weights=[0.5, 0.5])

    assert len(new_multi.weights) == 2


def test_multi_method():
    new_multi = multi.Multi()

    assert new_multi.method == 'weight_sum'


def test_multi_build():
    new_multi = multi.Multi()

    assert type(new_multi.functions) == list

    assert type(new_multi.pointer).__name__ == 'function'

    assert new_multi.built == True


def test_multi_create_functions():
    def square(x):
        return x**2

    assert square(2) == 4

    def cube(x):
        return x**3

    assert cube(2) == 8

    new_multi = multi.Multi()

    try:
        new_multi.functions = new_multi._create_functions('')
    except:
        new_multi.functions = new_multi._create_functions([square, cube])

    assert len(new_multi.functions) == 2


def test_multi_create_strategy():
    def square(x):
        return x**2

    assert square(2) == 4

    def cube(x):
        return x**3

    assert cube(2) == 8

    new_multi = multi.Multi(functions=[square, cube], weights=[0.5, 0.5])

    new_multi.pointer = new_multi._create_strategy('weight_sum')

    assert new_multi.pointer(2) == 6
