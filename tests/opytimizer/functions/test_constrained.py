import numpy as np

from opytimizer.functions import constrained
from opytimizer.utils import constant


def pointer(x):
    return x


assert pointer(1) == 1


def test_constrained_function_name():
    new_constrained_function = constrained.ConstrainedFunction(pointer, [])

    assert new_constrained_function.name == "pointer"


def test_constrained_function_name_setter():
    new_constrained_function = constrained.ConstrainedFunction(pointer, [])

    try:
        new_constrained_function.name = 1
    except:
        new_constrained_function.name = "callable"

    assert new_constrained_function.name == "callable"


def test_constrained_function_constraints():
    new_constrained_function = constrained.ConstrainedFunction(pointer, [])

    assert new_constrained_function.constraints == []


def test_constrained_function_constraints_setter():
    def c_1(x):
        return x**2

    assert c_1(2) == 4

    try:
        new_constrained_function = constrained.ConstrainedFunction(
            pointer, constraints=c_1
        )
    except:
        new_constrained_function = constrained.ConstrainedFunction(
            pointer, constraints=[c_1]
        )

    assert len(new_constrained_function.constraints) == 1


def test_constrained_function_penalty():
    new_constrained_function = constrained.ConstrainedFunction(pointer, [])

    assert new_constrained_function.penalty == 0.0


def test_constrained_function_penalty_setter():
    new_constrained_function = constrained.ConstrainedFunction(pointer, [])

    try:
        new_constrained_function.penalty = "a"
    except:
        new_constrained_function.penalty = 1

    try:
        new_constrained_function.penalty = -1
    except:
        new_constrained_function.penalty = 1

    assert new_constrained_function.penalty == 1


def test_constrained_function_pointer():
    new_constrained_function = constrained.ConstrainedFunction(pointer, [])

    assert new_constrained_function.pointer.__name__ == "pointer"


def test_constrained_function_pointer_setter():
    new_constrained_function = constrained.ConstrainedFunction(pointer, [])

    try:
        new_constrained_function.pointer = "a"
    except:
        new_constrained_function.pointer = callable

    assert (
        new_constrained_function.pointer.__class__.__name__
        == "builtin_constrained_function_or_method"
        or "builtin_constrained_function"
    )


def test_constrained_function_built():
    new_constrained_function = constrained.ConstrainedFunction(pointer, [])

    assert new_constrained_function.built is True


def test_constrained_function_built_setter():
    new_constrained_function = constrained.ConstrainedFunction(pointer, [])

    new_constrained_function.built = False

    assert new_constrained_function.built is False


def test_constrained_call():
    def square(x):
        return np.sum(x**2)

    assert square(2) == 4

    def square2(x, y):
        return x**2 + y**2

    assert square2(2, 2) == 8

    def c_1(x):
        return x[0] + x[1] <= 0

    new_constrained_function = constrained.ConstrainedFunction(square, [c_1], 100)

    assert new_constrained_function(np.zeros(2)) == 0
    assert new_constrained_function(np.ones(2)) == 202
