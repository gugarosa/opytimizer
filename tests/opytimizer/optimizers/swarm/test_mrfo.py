import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import mrfo
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_mrfo_hyperparams():
    hyperparams = {
        'S': 2.0
    }

    new_mrfo = mrfo.MRFO(hyperparams=hyperparams)

    assert new_mrfo.S == 2.0


def test_mrfo_hyperparams_setter():
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


def test_mrfo_build():
    new_mrfo = mrfo.MRFO()

    assert new_mrfo.built == True


def test_mrfo_cyclone_foraging():
    new_mrfo = mrfo.MRFO()

    search_space = search.SearchSpace(n_agents=5, n_iterations=20,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    cyclone = new_mrfo._cyclone_foraging(search_space.agents, search_space.best_agent.position, 1, 1, 20)

    assert cyclone[0] != 0


def test_mrfo_chain_foraging():
    new_mrfo = mrfo.MRFO()

    search_space = search.SearchSpace(n_agents=5, n_iterations=20,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    chain = new_mrfo._chain_foraging(search_space.agents, search_space.best_agent.position, 1)

    assert chain[0] != 0


def test_mrfo__somersault_foraging():
    new_mrfo = mrfo.MRFO()

    somersault = new_mrfo._somersault_foraging(1, 1)

    assert somersault != 0


def test_mrfo_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_mrfo = mrfo.MRFO()

    search_space = search.SearchSpace(n_agents=5, n_iterations=20,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_mrfo.run(search_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm mrfo failed to converge.'
