import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import boa
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_boa_hyperparams():
    hyperparams = {
        'c': 0.01,
        'a': 0.1,
        'p': 0.8
    }

    new_boa = boa.BOA(hyperparams=hyperparams)

    assert new_boa.c == 0.01
    assert new_boa.a == 0.1
    assert new_boa.p == 0.8


def test_boa_hyperparams_setter():
    new_boa = boa.BOA()

    try:
        new_boa.c = 'a'
    except:
        new_boa.c = 0.01

    try:
        new_boa.c = -1
    except:
        new_boa.c = 0.01

    assert new_boa.c == 0.01

    try:
        new_boa.a = 'b'
    except:
        new_boa.a = 0.1

    try:
        new_boa.a = -1
    except:
        new_boa.a = 0.1

    assert new_boa.a == 0.1

    try:
        new_boa.p = 'c'
    except:
        new_boa.p = 0.8

    try:
        new_boa.p = -1
    except:
        new_boa.p = 0.8

    assert new_boa.p == 0.8


def test_boa_build():
    new_boa = boa.BOA()

    assert new_boa.built == True


def test_boa_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_boa = boa.BOA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_boa.run(search_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm boa failed to converge.'
