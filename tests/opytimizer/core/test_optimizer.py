import sys

import numpy as np
import pytest

from opytimizer import Opytimizer
from opytimizer.spaces import search
from opytimizer.optimizers.pso import PSO
from opytimizer.spaces.search import SearchSpace
from opytimizer.core.function import Function
from opytimizer.core import function, optimizer


def test_optimizer_algorithm():
    new_optimizer = optimizer.Optimizer(algorithm='PSO')

    assert new_optimizer.algorithm == 'PSO'


def test_optimizer_hyperparams():
    new_optimizer = optimizer.Optimizer()

    assert new_optimizer.hyperparams == {}


def test_optimizer_hyperparams_setter():
    new_optimizer = optimizer.Optimizer()

    try:
        new_optimizer.hyperparams = 1
    except:
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
    new_optimizer = optimizer.Optimizer()

    with pytest.raises(NotImplementedError):
        new_optimizer._update()


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
    new_optimizer = optimizer.Optimizer()

    with pytest.raises(NotImplementedError):
        target_fn = Function(lambda x: x)
        search_space = SearchSpace()
        new_optimizer.run(search_space, target_fn)


def test_store_best_agent_only():
    pso = PSO()
    n_iters = 10
    target_fn = Function(lambda x: x**2)
    space = SearchSpace(lower_bound=[-10], upper_bound=[10], n_iterations=n_iters)

    history = Opytimizer(space, pso, target_fn).start(store_best_only=True)
    assert not hasattr(history, 'agents')

    assert hasattr(history, 'best_agent')
    assert len(history.best_agent) == n_iters


def test_store_all_agents():
    pso = PSO()
    n_iters = 10
    n_agents = 2
    target_fn = Function(lambda x: x**2)
    space = SearchSpace(lower_bound=[-10], upper_bound=[10], n_iterations=n_iters, n_agents=n_agents)

    history = Opytimizer(space, pso, target_fn).start()
    assert hasattr(history, 'agents')

    # Ensuring that the amount of entries is the same as the amount of iterations and
    # that for each iteration all agents are kept
    assert len(history.agents) == n_iters
    assert all(len(iter_agents) == n_agents for iter_agents in history.agents)

    assert hasattr(history, 'best_agent')
    assert len(history.best_agent) == n_iters
