import numpy as np

from opytimizer.core import function
from opytimizer.optimizers import hs
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_hs_hyperparams():
    hyperparams = {
        'HMCR': 0.7,
        'PAR': 0.7,
        'bw': 10.0
    }

    new_hs = hs.HS(hyperparams=hyperparams)

    assert new_hs.HMCR == 0.7

    assert new_hs.PAR == 0.7

    assert new_hs.bw == 10.0


def test_hs_hyperparams_setter():
    new_hs = hs.HS()

    try:
        new_hs.HMCR = 'a'
    except:
        new_hs.HMCR = 0.5

    try:
        new_hs.HMCR = -1
    except:
        new_hs.HMCR = 0.5

    assert new_hs.HMCR == 0.5

    try:
        new_hs.PAR = 'b'
    except:
        new_hs.PAR = 0.5

    try:
        new_hs.PAR = -1
    except:
        new_hs.PAR = 0.5

    assert new_hs.PAR == 0.5

    try:
        new_hs.bw = 'c'
    except:
        new_hs.bw = 5

    try:
        new_hs.bw = -1
    except:
        new_hs.bw = 5

    assert new_hs.bw == 5

    assert new_hs.bw == 5


def test_hs_build():
    new_hs = hs.HS()

    assert new_hs.built == True


def test_hs_generate_new_harmony():
    new_hs = hs.HS()

    search_space = search.SearchSpace(n_agents=2, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    agent = new_hs._generate_new_harmony(search_space.agents[0])

    assert agent.fit > 0


def test_hs_run():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    hyperparams = {
        'HMCR': 0.7,
        'PAR': 0.7,
        'bw': 10.0
    }

    new_hs = hs.HS(hyperparams=hyperparams)

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_hs.run(search_space, new_function)

    assert len(history.agents) > 0
    assert len(history.best) > 0
    assert len(history.best_index) > 0

    best_fitness = history.best[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm hs failed to converge.'
