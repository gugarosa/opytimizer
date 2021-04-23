import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import abc
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_abc_params():
    params = {
        'n_trials': 5
    }

    new_abc = abc.ABC(params=params)

    assert new_abc.n_trials == 5


def test_abc_params_setter():
    new_abc = abc.ABC()

    try:
        new_abc.n_trials = 0.0
    except:
        new_abc.n_trials = 10

    try:
        new_abc.n_trials = 0
    except:
        new_abc.n_trials = 10

    assert new_abc.n_trials == 10


def test_abc_build():
    new_abc = abc.ABC()

    assert new_abc.built == True


def test_abc_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    params = {
        'n_trials': 1
    }

    new_abc = abc.ABC(params=params)

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_abc.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm abc failed to converge.'
