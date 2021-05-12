import numpy as np

from opytimizer.optimizers.swarm import cs
from opytimizer.spaces import search


def test_cs_params():
    params = {
        'alpha': 1.0,
        'beta': 1.5,
        'p': 0.2
    }

    new_cs = cs.CS(params=params)

    assert new_cs.alpha == 1.0
    assert new_cs.beta == 1.5
    assert new_cs.p == 0.2


def test_cs_params_setter():
    new_cs = cs.CS()

    try:
        new_cs.alpha = 'a'
    except:
        new_cs.alpha = 0.001

    try:
        new_cs.alpha = -1
    except:
        new_cs.alpha = 0.001

    assert new_cs.alpha == 0.001

    try:
        new_cs.beta = 'b'
    except:
        new_cs.beta = 0.75

    try:
        new_cs.beta = -1
    except:
        new_cs.beta = 0.75

    assert new_cs.beta == 0.75

    try:
        new_cs.p = 'c'
    except:
        new_cs.p = 0.25

    try:
        new_cs.p = -1
    except:
        new_cs.p = 0.25

    assert new_cs.p == 0.25


def test_cs_generate_new_nests():
    search_space = search.SearchSpace(n_agents=20, n_variables=2,
                                      lower_bound=[-10, -10], upper_bound=[10, 10])

    new_cs = cs.CS()

    new_agents = new_cs._generate_new_nests(
        search_space.agents, search_space.best_agent)

    assert len(new_agents) == 20


def test_cs_generate_abandoned_nests():
    search_space = search.SearchSpace(n_agents=20, n_variables=2,
                                      lower_bound=[-10, -10], upper_bound=[10, 10])

    new_cs = cs.CS()

    new_agents = new_cs._generate_abandoned_nests(search_space.agents, 0.5)

    assert len(new_agents) == 20


def test_cs_evaluate_nests():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=20, n_variables=2,
                                      lower_bound=[-10, -10], upper_bound=[10, 10])

    new_cs = cs.CS()

    new_agents = new_cs._generate_abandoned_nests(search_space.agents, 0.5)
    new_cs._evaluate_nests(search_space.agents, new_agents, square)


def test_cs_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=20, n_variables=2,
                                      lower_bound=[-10, -10], upper_bound=[10, 10])

    new_cs = cs.CS()

    new_cs.update(search_space, square)
