import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.evolutionary import es
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_es_params():
    params = {
        'child_ratio': 0.5
    }

    new_es = es.ES(params=params)

    assert new_es.child_ratio == 0.5


def test_es_params_setter():
    new_es = es.ES()

    try:
        new_es.child_ratio = 'a'
    except:
        new_es.child_ratio = 0.5

    try:
        new_es.child_ratio = -1
    except:
        new_es.child_ratio = 0.5

    assert new_es.child_ratio == 0.5


def test_es_build():
    new_es = es.ES()

    assert new_es.built == True


def test_es_mutate_parent():
    def square(x):
        return np.sum(x**2)

    new_es = es.ES()

    search_space = search.SearchSpace(n_agents=4, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    strategy = np.zeros(4)

    agent = new_es._mutate_parent(search_space.agents[0], square, strategy[0])

    assert agent.position[0][0] > 0


def test_es_update_strategy():
    new_es = es.ES()

    strategy = np.ones((4, 1))

    new_strategy = new_es._update_strategy(strategy)

    assert new_strategy[0][0] > 0


def test_es_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    params = {
        'child_ratio': 0.5
    }

    new_es = es.ES(params=params)

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_es.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm de failed to converge.'
