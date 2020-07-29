import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.evolutionary import ep
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_ep_hyperparams():
    hyperparams = {
        'bout_size': 0.1,
        'clip_ratio': 0.05
    }

    new_ep = ep.EP(hyperparams=hyperparams)

    assert new_ep.bout_size == 0.1

    assert new_ep.clip_ratio == 0.05


def test_ep_hyperparams_setter():
    new_ep = ep.EP()

    try:
        new_ep.bout_size = 'a'
    except:
        new_ep.bout_size = 0.5

    try:
        new_ep.bout_size = -1
    except:
        new_ep.bout_size = 0.5

    assert new_ep.bout_size == 0.5

    try:
        new_ep.clip_ratio = 'b'
    except:
        new_ep.clip_ratio = 0.5

    try:
        new_ep.clip_ratio = -1
    except:
        new_ep.clip_ratio = 0.5

    assert new_ep.clip_ratio == 0.5


def test_ep_build():
    new_ep = ep.EP()

    assert new_ep.built == True


def test_ep_mutate_parent():
    def square(x):
        return np.sum(x**2)

    new_ep = ep.EP()

    search_space = search.SearchSpace(n_agents=4, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    strategy = np.zeros(4)

    agent = new_ep._mutate_parent(search_space.agents[0], square, strategy[0])

    assert agent.position[0][0] > 0


def test_ep_update_strategy():
    new_ep = ep.EP()

    strategy = np.zeros((4, 1))

    new_strategy = new_ep._update_strategy(strategy, [1], [2])

    assert new_strategy[0][0] > 0


def test_ep_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    hyperparams = {
        'bout_size': 0.1,
        'clip_ratio': 0.05
    }

    new_ep = ep.EP(hyperparams=hyperparams)

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_ep.run(search_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm de failed to converge.'
