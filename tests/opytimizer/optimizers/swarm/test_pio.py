import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import pio
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_pio_hyperparams():
    hyperparams = {
        'n_c1': 150,
        'n_c2': 200,
        'R': 0.2
    }

    new_pio = pio.PIO(hyperparams=hyperparams)

    assert new_pio.n_c1 == 150

    assert new_pio.n_c2 == 200

    assert new_pio.R == 0.2


def test_pio_hyperparams_setter():
    new_pio = pio.PIO()

    try:
        new_pio.n_c1 = 0.0
    except:
        new_pio.n_c1 = 150

    try:
        new_pio.n_c1 = 0
    except:
        new_pio.n_c1 = 150

    assert new_pio.n_c1 == 150

    try:
        new_pio.n_c2 = 0.0
    except:
        new_pio.n_c2 = 200

    try:
        new_pio.n_c2 = 0
    except:
        new_pio.n_c2 = 200

    assert new_pio.n_c2 == 200

    try:
        new_pio.R = 'a'
    except:
        new_pio.R = 0.2

    try:
        new_pio.R = -1
    except:
        new_pio.R = 0.2

    assert new_pio.R == 0.2


def test_pio_build():
    new_pio = pio.PIO()

    assert new_pio.built == True


def test_pio_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_pio = pio.PIO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=175,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_pio.run(search_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm pio failed to converge.'
