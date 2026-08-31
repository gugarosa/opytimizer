import pytest

from opytimizer.functions.multi_objective import MultiObjectiveFunction


def square(x):
    return x**2


def cube(x):
    return x**3


def test_multi_objective_function_invokes_raw_callables():
    function = MultiObjectiveFunction([square, cube])

    assert function.functions == [square, cube]
    assert function(2) == [4, 8]
    assert not hasattr(function, "built")


@pytest.mark.parametrize("functions", [None, [square, 1]])
def test_multi_objective_function_validates_callables(functions):
    with pytest.raises(TypeError):
        MultiObjectiveFunction(functions)
