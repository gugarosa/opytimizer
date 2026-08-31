import numpy as np
import pytest

from opytimizer.functions import ConstrainedFunction


def square(x):
    return np.sum(x**2)


def test_constrained_function_keeps_raw_callable_and_state():
    constraint = lambda x: x[0] <= 0
    function = ConstrainedFunction(square, [constraint], penalty=2)

    assert function.function is square
    assert function.constraints == [constraint]
    assert function.penalty == 2
    assert not hasattr(function, "pointer")
    assert not hasattr(function, "name")
    assert not hasattr(function, "built")

    assert function(np.zeros(2)) == 0
    assert function(np.ones(2)) == 6

    function.penalty = 1
    assert function(np.ones(2)) == 4


def test_constrained_function_applies_each_failed_constraint():
    function = ConstrainedFunction(square, [lambda x: False, lambda x: False], 1)

    assert function(np.array([2])) == 16


@pytest.mark.parametrize(
    "args,error",
    [
        ((1, []), TypeError),
        ((square, None), TypeError),
        ((square, [1]), TypeError),
        ((square, [], "x"), TypeError),
        ((square, [], -1), ValueError),
    ],
)
def test_constrained_function_validates_constructor_inputs(args, error):
    with pytest.raises(error):
        ConstrainedFunction(*args)
