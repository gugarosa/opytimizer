import numpy as np

from opytimizer.core import function
from opytimizer.utils import constant


def pointer(x):
    return x


assert pointer(1) == 1


def test_function_name():
    new_function = function.Function(pointer)

    assert new_function.name == "pointer"


def test_function_name_setter():
    new_function = function.Function(pointer)

    try:
        new_function.name = 1
    except:
        new_function.name = "pointer"

    assert new_function.name == "pointer"


def test_function_pointer():
    new_function = function.Function(pointer)

    assert new_function.pointer.__name__ == "pointer"


def test_function_pointer_setter():
    new_function = function.Function(pointer)

    try:
        new_function.pointer = "a"
    except:
        new_function.pointer = callable

    assert (
        new_function.pointer.__class__.__name__ == "builtin_function_or_method"
        or "builtin_function"
    )


def test_function_built():
    new_function = function.Function(pointer)

    assert new_function.built is True


def test_function_built_setter():
    new_function = function.Function(pointer)

    new_function.built = False

    assert new_function.built is False


def test_function_call():
    def square(x):
        return np.sum(x**2)

    assert square(2) == 4

    def square2(x, y):
        return x**2 + y**2

    assert square2(2, 2) == 8

    new_function = function.Function(square)

    assert new_function(np.zeros(2)) == 0

    try:
        new_function = function.Function(square2)
    except:
        new_function = function.Function(square)

    assert new_function.name == "square"


def test_function():
    class Square:
        def __call__(self, x):
            return np.sum(x**2)

    s = Square()

    assert s(2) == 4

    new_function = function.Function(s)

    assert new_function.name == "Square"
