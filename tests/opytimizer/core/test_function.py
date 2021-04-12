import numpy as np

from opytimizer.core import function
from opytimizer.utils import constants


def test_function_name():
    new_function = function.Function()

    assert new_function.name == 'callable'


def test_function_name_setter():
    new_function = function.Function()

    try:
        new_function.name = 1
    except:
        new_function.name = 'callable'

    assert new_function.name == 'callable'


def test_function_constraints():
    new_function = function.Function()

    assert new_function.constraints == []


def test_function_constraints_setter():
    def c_1(x):
        return x**2

    assert c_1(2) == 4

    try:
        new_function = function.Function(constraints=c_1)
    except:
        new_function = function.Function(constraints=[c_1])

    assert len(new_function.constraints) == 1


def test_function_penalty():
    new_function = function.Function()

    assert new_function.penalty == 0.0


def test_function_penalty_setter():
    new_function = function.Function()

    try:
        new_function.penalty = 'a'
    except:
        new_function.penalty = 1

    try:
        new_function.penalty = -1
    except:
        new_function.penalty = 1

    assert new_function.penalty == 1


def test_function_pointer():
    new_function = function.Function()

    assert new_function.pointer.__name__ == '_constrain_pointer'


def test_function_pointer_setter():
    new_function = function.Function()

    try:
        new_function.pointer = 'a'
    except:
        new_function.pointer = callable

    assert new_function.pointer.__class__.__name__ == 'builtin_function_or_method' or 'builtin_function'


def test_function_built():
    new_function = function.Function()

    assert new_function.built == True


def test_function_built_setter():
    new_function = function.Function()

    new_function.built = False

    assert new_function.built == False


def test_function_create_pointer():
    def square(x):
        return np.sum(x**2)

    assert square(2) == 4

    def square2(x, y):
        return x**2 + y**2

    assert square2(2, 2) == 8

    def c_1(x):
        return x[0] + x[1] <= 0

    assert c_1(np.zeros(2)) == True

    new_function = function.Function(pointer=square, constraints=[c_1], penalty=100)

    assert new_function(np.zeros(2)) == 0

    assert new_function(np.ones(2)) == 202

    try:
        new_function = function.Function(pointer=square2)
    except:
        new_function = function.Function()

    assert new_function.name == 'callable'


def test_function():
    class Square():
        def __call__(self, x):
            return np.sum(x**2)

    s = Square()

    assert s(2) == 4

    new_function = function.Function(pointer=s)

    assert new_function.name == 'Square'
