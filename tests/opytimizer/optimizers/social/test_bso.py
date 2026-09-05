import numpy as np

from opytimizer.optimizers.social import bso
from opytimizer.spaces import search


def test_bso_params():
    params = {
        "m": 5,
        "p_replacement_cluster": 0.2,
        "p_single_cluster": 0.8,
        "p_single_best": 0.4,
        "p_double_best": 0.5,
        "k": 20,
    }

    new_bso = bso.BSO(params=params)

    assert new_bso.m == 5

    assert new_bso.p_replacement_cluster == 0.2

    assert new_bso.p_single_cluster == 0.8

    assert new_bso.p_single_best == 0.4

    assert new_bso.p_double_best == 0.5

    assert new_bso.k == 20


def test_bso_clusterize():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_bso = bso.BSO()

    new_bso._clusterize(search_space.agents)


def test_bso_clusterize_single_cluster_retains_every_agent(monkeypatch):
    search_space = search.SearchSpace(3, 1, [0], [10])
    for agent, position in zip(search_space.agents, [0.0, 5.0, 10.0]):
        agent.position[:] = position
        agent.fit = position**2
    monkeypatch.setattr(np.random, "randint", lambda low, high: 2)

    indexes, best_indexes = bso.BSO({"m": 1})._clusterize(search_space.agents)

    assert len(indexes) == 1
    np.testing.assert_array_equal(indexes[0], [0, 1, 2])
    assert best_indexes == [0]


def test_bso_sigmoid():
    new_bso = bso.BSO()

    x = 0.5

    y = new_bso._sigmoid(x)

    assert y == 0.6224593312018546


def test_bso_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=50, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_bso = bso.BSO()
    new_bso.evaluate(search_space, square)

    new_bso.update(search_space, square, 1, 10)

    new_bso.p_replacement_cluster = 1
    new_bso.update(search_space, square, 1, 10)
