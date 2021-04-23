import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.population import epo
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_epo_params():
    params = {
        'f': 2.0,
        'l': 1.5
    }

    new_epo = epo.EPO(params=params)

    assert new_epo.f == 2.0

    assert new_epo.l == 1.5


def test_epo_params_setter():
    new_epo = epo.EPO()

    try:
        new_epo.f = 'a'
    except:
        new_epo.f = 2.0

    try:
        new_epo.l = 'b'
    except:
        new_epo.l = 1.5


def test_epo_build():
    new_epo = epo.EPO()

    assert new_epo.built == True


def test_epo_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_epo = epo.EPO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_epo.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm epo failed to converge.'
