import sys

import numpy as np
import pytest
from opytimizer.core import function, optimizer
from opytimizer.spaces import search


def test_optimizer_algorithm():
    new_optimizer = optimizer.Optimizer(algorithm='PSO')

    assert new_optimizer.algorithm == 'PSO'


def test_optimizer_hyperparams():
    new_optimizer = optimizer.Optimizer()

    assert new_optimizer.hyperparams == None


def test_optimizer_hyperparams_setter():
    new_optimizer = optimizer.Optimizer()

    new_optimizer.hyperparams = {
        'w': 1.5
    }

    assert new_optimizer.hyperparams['w'] == 1.5


def test_optimizer_built():
    new_optimizer = optimizer.Optimizer()

    assert new_optimizer.built == False


def test_optimizer_built_setter():
    new_optimizer = optimizer.Optimizer()

    new_optimizer.built = True

    assert new_optimizer.built == True


def test_optimizer_update():
    def square(x):
        return np.sum(x**2)

    assert square(2) == 4

    new_function = function.Function(pointer=square)

    new_optimizer = optimizer.Optimizer()

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    with pytest.raises(NotImplementedError):
        new_optimizer._update(search_space.agents, search_space.best_agent, new_function)


def test_optimizer_evaluate():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    search_space = search.SearchSpace(n_agents=20, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_optimizer = optimizer.Optimizer()

    new_optimizer._evaluate(search_space, new_function)

    assert search_space.best_agent.fit < sys.float_info.max


def test_optimizer_run():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_optimizer = optimizer.Optimizer()

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    with pytest.raises(NotImplementedError):
        history = new_optimizer.run(search_space, new_function)
