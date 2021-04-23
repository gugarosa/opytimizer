import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import abo
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_abo_params():
    params = {
        'sunspot_ratio': 0.9,
        'a': 2.0
    }

    new_abo = abo.ABO(params=params)

    assert new_abo.sunspot_ratio == 0.9
    
    assert new_abo.a == 2.0


def test_abo_params_setter():
    new_abo = abo.ABO()

    try:
        new_abo.sunspot_ratio = 'a'
    except:
        new_abo.sunspot_ratio = 0.9

    try:
        new_abo.sunspot_ratio = -1
    except:
        new_abo.sunspot_ratio = 0.9

    assert new_abo.sunspot_ratio == 0.9

    try:
        new_abo.a = 'b'
    except:
        new_abo.a = 2.0

    try:
        new_abo.a = -1
    except:
        new_abo.a = 2.0

    assert new_abo.a == 2.0


def test_abo_build():
    new_abo = abo.ABO()

    assert new_abo.built == True


def test_abo_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_abo = abo.ABO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_abo.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm abo failed to converge.'
