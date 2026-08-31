import pytest

from opytimizer.functions.multi_objective import MultiObjectiveWeightedFunction


def square(x):
    return x**2


def cube(x):
    return x**3


def test_weighted_function_keeps_mutable_weighted_behavior():
    function = MultiObjectiveWeightedFunction([square, cube], [0.5, 0.5])

    assert function(2) == 6

    function.weights = [1, 0]
    assert function(2) == 4


@pytest.mark.parametrize(
    "functions,weights,error",
    [
        ([square], None, TypeError),
        ([square], [], ValueError),
        ([1], [1], TypeError),
    ],
)
def test_weighted_function_validates_constructor_inputs(functions, weights, error):
    with pytest.raises(error):
        MultiObjectiveWeightedFunction(functions, weights)
