# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opytimizer.functions import ConstrainedFunction


def square(x):
    return np.sum(x**2)


def test_constrained_function_keeps_raw_callable_and_state():
    def constraint(x):
        return x[0] <= 0

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


@pytest.mark.parametrize("penalty,expected", [(0, -4), (0.5, -2), (1, 0), (2, 4)])
def test_constrained_function_does_not_reward_negative_fitness(penalty, expected):
    function = ConstrainedFunction(lambda x: -square(x), [lambda x: False], penalty)

    assert function(np.array([2])) == expected


def test_constrained_function_penalizes_each_violation_with_negative_fitness():
    function = ConstrainedFunction(lambda x: -square(x), [lambda x: False, lambda x: False], 0.5)

    assert function(np.array([2])) == -1


def test_constrained_function_preserves_feasible_and_zero_fitness():
    feasible = ConstrainedFunction(lambda x: -square(x), [lambda x: True], 2)
    zero = ConstrainedFunction(square, [lambda x: False], 2)

    assert feasible(np.array([2])) == -4
    assert zero(np.zeros(2)) == 0


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
