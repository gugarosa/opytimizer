import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import sbo
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_sbo_hyperparams():
    hyperparams = {
        'alpha': 0.9,
        'p_mutation': 0.05,
        'z': 0.02
    }

    new_sbo = sbo.SBO(hyperparams=hyperparams)

    assert new_sbo.alpha == 0.9

    assert new_sbo.p_mutation == 0.05

    assert new_sbo.z == 0.02


def test_sbo_hyperparams_setter():
    new_sbo = sbo.SBO()

    try:
        new_sbo.alpha = 'a'
    except:
        new_sbo.alpha = 0.75

    try:
        new_sbo.alpha = -1
    except:
        new_sbo.alpha = 0.75

    assert new_sbo.alpha == 0.75

    try:
        new_sbo.p_mutation = 'b'
    except:
        new_sbo.p_mutation = 0.05

    try:
        new_sbo.p_mutation = 1.5
    except:
        new_sbo.p_mutation = 0.05

    assert new_sbo.p_mutation == 0.05

    try:
        new_sbo.z = 'c'
    except:
        new_sbo.z = 0.02

    try:
        new_sbo.z = 1.5
    except:
        new_sbo.z = 0.02

    assert new_sbo.z == 0.02


def test_sbo_build():
    new_sbo = sbo.SBO()

    assert new_sbo.built == True


def test_sbo_update():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_sbo = sbo.SBO()

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[1, 1],
                                      upper_bound=[10, 10])

    new_sbo._update(search_space.agents,
                    search_space.best_agent, new_function, np.array([0.5, 0.5]))

    assert search_space.agents[0].position[0] != 0


def test_sbo_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_sbo = sbo.SBO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=30,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_sbo.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm sbo failed to converge.'
