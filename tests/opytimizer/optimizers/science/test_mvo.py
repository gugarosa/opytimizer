import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.science import mvo
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_mvo_params():
    params = {
        'WEP_min': 0.2,
        'WEP_max': 1.0,
        'p': 0.5
    }

    new_mvo = mvo.MVO(params=params)

    assert new_mvo.WEP_min == 0.2

    assert new_mvo.WEP_max == 1.0

    assert new_mvo.p == 0.5


def test_mvo_params_setter():
    new_mvo = mvo.MVO()

    try:
        new_mvo.WEP_min = 'a'
    except:
        new_mvo.WEP_min = 0.75

    try:
        new_mvo.WEP_min = -1
    except:
        new_mvo.WEP_min = 0.75

    assert new_mvo.WEP_min == 0.75

    try:
        new_mvo.WEP_max = 'b'
    except:
        new_mvo.WEP_max = 0.9

    try:
        new_mvo.WEP_max = 0.1
    except:
        new_mvo.WEP_max = 0.9

    try:
        new_mvo.WEP_max = -1
    except:
        new_mvo.WEP_max = 0.9

    assert new_mvo.WEP_max == 0.9

    try:
        new_mvo.p = 'c'
    except:
        new_mvo.p = 0.25

    try:
        new_mvo.p = -1
    except:
        new_mvo.p = 0.25

    assert new_mvo.p == 0.25


def test_mvo_build():
    new_mvo = mvo.MVO()

    assert new_mvo.built == True


def test_mvo_update():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_mvo = mvo.MVO()

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[1, 1],
                                      upper_bound=[10, 10])

    new_mvo._update(search_space.agents, search_space.best_agent, new_function, 1, 1)

    assert search_space.agents[0].position[0] != 0


def test_mvo_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_mvo = mvo.MVO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=30,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_mvo.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm mvo failed to converge.'
