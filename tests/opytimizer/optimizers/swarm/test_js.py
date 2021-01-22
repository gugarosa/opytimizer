import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import js
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_js_hyperparams():
    hyperparams = {
        'eta': 4.0,
        'beta': 3.0,
        'gamma': 0.1
    }

    new_js = js.JS(hyperparams=hyperparams)

    assert new_js.eta == 4.0

    assert new_js.beta == 3.0

    assert new_js.gamma == 0.1


def test_js_hyperparams_setter():
    new_js = js.JS()

    try:
        new_js.eta = 'a'
    except:
        new_js.eta = 4.0

    try:
        new_js.eta = -1
    except:
        new_js.eta = 4.0

    assert new_js.eta == 4.0

    try:
        new_js.beta = 'b'
    except:
        new_js.beta = 2.0

    try:
        new_js.beta = 0
    except:
        new_js.beta = 3.0

    assert new_js.beta == 3.0

    try:
        new_js.gamma = 'c'
    except:
        new_js.gamma = 0.1

    try:
        new_js.gamma = -1
    except:
        new_js.gamma = 0.1

    assert new_js.gamma == 0.1


def test_js_build():
    new_js = js.JS()

    assert new_js.built == True


def test_js_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_js = js.JS()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_js.run(search_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm js failed to converge.'


def test_nbjs_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_nbjs = js.NBJS()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_nbjs.run(search_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm nbjs failed to converge.'
