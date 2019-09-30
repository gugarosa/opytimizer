import numpy as np

from opytimizer.core import function
from opytimizer.optimizers import ihs
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_ihs_hyperparams():
    hyperparams = {
        'PAR_min': 0.5,
        'PAR_max': 1,
        'bw_min': 2,
        'bw_max': 5
    }

    new_ihs = ihs.IHS(hyperparams=hyperparams)

    assert new_ihs.PAR_min == 0.5

    assert new_ihs.PAR_max == 1

    assert new_ihs.bw_min == 2

    assert new_ihs.bw_max == 5


def test_ihs_hyperparams_setter():
    new_ihs = ihs.IHS()

    try:
        new_ihs.PAR_min = 'a'
    except:
        new_ihs.PAR_min = 0.5

    try:
        new_ihs.PAR_min = -1
    except:
        new_ihs.PAR_min = 0.5

    assert new_ihs.PAR_min == 0.5

    try:
        new_ihs.PAR_max = 'b'
    except:
        new_ihs.PAR_max = 1.0

    try:
        new_ihs.PAR_max = -1
    except:
        new_ihs.PAR_max = 1.0

    try:
        new_ihs.PAR_max = 0
    except:
        new_ihs.PAR_max = 1.0

    assert new_ihs.PAR_max == 1.0

    try:
        new_ihs.bw_min = 'c'
    except:
        new_ihs.bw_min = 1.0

    try:
        new_ihs.bw_min = -1
    except:
        new_ihs.bw_min = 1.0

    assert new_ihs.bw_min == 1.0

    try:
        new_ihs.bw_max = 'd'
    except:
        new_ihs.bw_max = 10.0

    try:
        new_ihs.bw_max = -1
    except:
        new_ihs.bw_max = 10.0

    try:
        new_ihs.bw_max = 0
    except:
        new_ihs.bw_max = 10.0

    assert new_ihs.bw_max == 10.0


def test_ihs_rebuild():
    new_ihs = ihs.IHS()

    assert new_ihs.built == True


def test_ihs_run():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_ihs = ihs.IHS()

    search_space = search.SearchSpace(n_agents=20, n_iterations=50,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[5, 5])

    history = new_ihs.run(search_space, new_function)

    assert len(history.agents) > 0
    assert len(history.best) > 0

    best_fitness = history.best[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm ihs failed to converge.'
