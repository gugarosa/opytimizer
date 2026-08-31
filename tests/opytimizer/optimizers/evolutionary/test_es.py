import numpy as np

from opytimizer.optimizers.evolutionary import es
from opytimizer.spaces import search


def test_es_params():
    params = {"child_ratio": 0.5}

    new_es = es.ES(params=params)

    assert new_es.child_ratio == 0.5


def test_es_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_es = es.ES()
    new_es.compile(search_space)


def test_es_mutate_parent():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_es = es.ES()
    new_es.compile(search_space)

    agent = new_es._mutate_parent(search_space.agents[0], 0, square)

    assert agent.position[0][0] > 0


def test_es_update_strategy():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_es = es.ES()
    new_es.compile(search_space)

    new_es._update_strategy(0)

    assert new_es.strategy[0][0] > 0


def test_es_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_es = es.ES()
    new_es.compile(search_space)

    new_es.update(search_space, square)
