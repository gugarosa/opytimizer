import numpy as np
import pytest
from opytimizer.core import function


def test_function_pointer():
    new_function = function.Function()

    assert new_function.pointer == callable


def test_function_pointer_setter():
    new_function = function.Function()

    def square(x):
        return x**2

    new_function.pointer = square

    assert new_function.pointer != None


def test_function_built():
    new_function = function.Function()

    assert new_function.built == True