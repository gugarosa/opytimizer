import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import sso
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_sso_hyperparams():
    hyperparams = {
        'C_w': 0.1,
        'C_p': 0.4,
        'C_g': 0.9
    }

    new_sso = sso.SSO(hyperparams=hyperparams)

    assert new_sso.C_w == 0.1

    assert new_sso.C_p == 0.4

    assert new_sso.C_g == 0.9


def test_sso_hyperparams_setter():
    new_sso = sso.SSO()

    try:
        new_sso.C_w = 'a'
    except:
        new_sso.C_w = 0.1

    try:
        new_sso.C_w = -1
    except:
        new_sso.C_w = 0.1

    assert new_sso.C_w == 0.1

    try:
        new_sso.C_p = 'b'
    except:
        new_sso.C_p = 0.4

    try:
        new_sso.C_p = 0.05
    except:
        new_sso.C_p = 0.4

    assert new_sso.C_p == 0.4

    try:
        new_sso.C_g = 'c'
    except:
        new_sso.C_g = 0.9

    try:
        new_sso.C_g = 0.35
    except:
        new_sso.C_g = 0.9

    assert new_sso.C_g == 0.9


def test_sso_build():
    new_sso = sso.SSO()

    assert new_sso.built == True


def test_sso_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_sso = sso.SSO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_sso.run(search_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm sso failed to converge.'
