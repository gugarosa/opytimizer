import sys

import numpy as np

from opytimizer.core import function, optimizer
from opytimizer.spaces import search


def test_optimizer_algorithm():
    new_optimizer = optimizer.Optimizer()

    assert new_optimizer.algorithm == 'Optimizer'


def test_optimizer_algorithm_setter():
    new_optimizer = optimizer.Optimizer()

    try:
        new_optimizer.algorithm = 0
    except:
        new_optimizer.algorithm = 'Optimizer'

    assert new_optimizer.algorithm == 'Optimizer'


def test_optimizer_params():
    new_optimizer = optimizer.Optimizer()

    assert new_optimizer.params == None


def test_optimizer_params_setter():
    new_optimizer = optimizer.Optimizer()

    try:
        new_optimizer.params = 1
    except:
        new_optimizer.params = {
            'w': 1.5
        }

    assert new_optimizer.params['w'] == 1.5


def test_optimizer_built():
    new_optimizer = optimizer.Optimizer()

    assert new_optimizer.built == False


def test_optimizer_built_setter():
    new_optimizer = optimizer.Optimizer()

    try:
        new_optimizer.built = 1
    except:
        new_optimizer.built = True

    assert new_optimizer.built == True


def test_optimizer_build():
    new_optimizer = optimizer.Optimizer()

    new_optimizer.build({'w': 1.5})


def test_optimizer_update():
    new_optimizer = optimizer.Optimizer()

    new_optimizer.update()


def test_optimizer_evaluate():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(square)
    new_search_space = search.SearchSpace(n_agents=1, n_variables=2,
                                          lower_bound=[0, 0], upper_bound=[10, 10])

    new_optimizer = optimizer.Optimizer()
    new_optimizer.evaluate(new_search_space, new_function)

    assert new_search_space.best_agent.fit < sys.float_info.max
