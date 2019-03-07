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

    new_function = function.Function(pointer=square)

    assert new_function.pointer == square


def test_function_built():
    try:
        new_function = function.Function(pointer=None)
    except:
        new_function = function.Function()

    assert new_function.built == True
