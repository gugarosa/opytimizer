import pytest

from opytimizer.core import function
from opytimizer.core import agent


def test_function_creation():
    expression = 'x1 + x2 + x3'
    lower_bound = [1, 1, 1]
    upper_bound = [3, 3, 3]
    new_function = function.Function(expression=expression,
                                     lower_bound=lower_bound, upper_bound=upper_bound)
    assert new_function.expression


def test_function_instanciate():
    expression = 'x1 + x2 + x3'
    lower_bound = [1, 1, 1]
    upper_bound = [3, 3, 3]
    new_function = function.Function(expression=expression,
                                     lower_bound=lower_bound, upper_bound=upper_bound)
    new_function.instanciate()
    assert len(new_function.variables) == 3


def test_function_evaluate():
    expression = 'x1 + x2 + x3'
    lower_bound = [1, 1, 1]
    upper_bound = [3, 3, 3]
    new_function = function.Function(expression=expression,
                                     lower_bound=lower_bound, upper_bound=upper_bound)

    n_variables = 3
    n_dimensions = 2
    new_agent = agent.Agent(n_variables=n_variables, n_dimensions=n_dimensions)
    new_agent.position[0] = [1, 1]
    new_agent.position[1] = [1, 1]
    new_agent.position[2] = [1, 1]

    new_function.instanciate()
    new_function.evaluate(new_agent)

    assert new_agent.fit != 0
