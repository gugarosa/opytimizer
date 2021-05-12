import numpy as np

from opytimizer.optimizers.swarm import mrfo
from opytimizer.spaces import search

np.random.seed(0)


def test_mrfo_params():
    params = {
        'S': 2.0
    }

    new_mrfo = mrfo.MRFO(params=params)

    assert new_mrfo.S == 2.0


def test_mrfo_params_setter():
    new_mrfo = mrfo.MRFO()

    try:
        new_mrfo.S = 'a'
    except:
        new_mrfo.S = 2.0

    try:
        new_mrfo.S = -1
    except:
        new_mrfo.S = 2.0

    assert new_mrfo.S == 2.0


def test_mrfo_cyclone_foraging():
    new_mrfo = mrfo.MRFO()

    search_space = search.SearchSpace(n_agents=5, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    cyclone = new_mrfo._cyclone_foraging(
        search_space.agents, search_space.best_agent.position, 1, 1, 20)

    assert cyclone[0] != 0


def test_mrfo_chain_foraging():
    new_mrfo = mrfo.MRFO()

    search_space = search.SearchSpace(n_agents=5, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    chain = new_mrfo._chain_foraging(
        search_space.agents, search_space.best_agent.position, 1)

    assert chain[0] != 0


def test_mrfo_somersault_foraging():
    new_mrfo = mrfo.MRFO()

    somersault = new_mrfo._somersault_foraging(1, 1)

    assert somersault != 0


def test_mrfo_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=5, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_mrfo = mrfo.MRFO()

    new_mrfo.update(search_space, square, 1, 10)
