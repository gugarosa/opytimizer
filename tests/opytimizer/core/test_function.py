import numpy as np
import pytest

from opytimizer.core import function


def test_function_pointer():
    new_function = function.Function()

    assert new_function.pointer == callable


def test_function_pointer_setter():
    def square(x):
        return x**2

    assert square(2) == 4

    try:
        new_function = function.Function(pointer=0)
    except:
        new_function = function.Function(pointer=square)

    def square2(x, y):
        return x**2 + y**2

    assert square2(2, 2) == 8

    try:
        new_function = function.Function(pointer=square2)
    except:
        new_function = function.Function(pointer=square)

    assert new_function.pointer == square


def test_function_built():
    new_function = function.Function()

    assert new_function.built == True


def test_function_built_setter():
    new_function = function.Function()

    new_function.built = False

    assert new_function.built == False


def test_function_build():
    new_function = function.Function()

    assert new_function.built == True
