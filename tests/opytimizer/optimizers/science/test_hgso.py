import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.science import hgso
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_hgso_params():
    params = {
        'n_clusters': 2,
        'l1': 0.0005,
        'l2': 100,
        'l3': 0.001,
        'alpha': 1.0,
        'beta': 1.0,
        'K': 1.0
    }

    new_hgso = hgso.HGSO(params=params)

    assert new_hgso.n_clusters == 2

    assert new_hgso.l1 == 0.0005

    assert new_hgso.l2 == 100

    assert new_hgso.l3 == 0.001

    assert new_hgso.alpha == 1.0

    assert new_hgso.beta == 1.0

    assert new_hgso.K == 1.0


def test_hgso_params_setter():
    new_hgso = hgso.HGSO()

    try:
        new_hgso.n_clusters = 'a'
    except:
        new_hgso.n_clusters = 2

    try:
        new_hgso.n_clusters = -1
    except:
        new_hgso.n_clusters = 2

    assert new_hgso.n_clusters == 2

    try:
        new_hgso.l1 = 'b'
    except:
        new_hgso.l1 = 0.0005

    try:
        new_hgso.l1 = -1
    except:
        new_hgso.l1 = 0.0005

    assert new_hgso.l1 == 0.0005

    try:
        new_hgso.l2 = 'c'
    except:
        new_hgso.l2 = 100

    try:
        new_hgso.l2 = -1
    except:
        new_hgso.l2 = 100

    assert new_hgso.l2 == 100

    try:
        new_hgso.l3 = 'd'
    except:
        new_hgso.l3 = 0.001

    try:
        new_hgso.l3 = -1
    except:
        new_hgso.l3 = 0.001

    assert new_hgso.l3 == 0.001

    try:
        new_hgso.alpha = 'e'
    except:
        new_hgso.alpha = 1.0

    try:
        new_hgso.alpha = -1
    except:
        new_hgso.alpha = 1.0

    assert new_hgso.alpha == 1.0

    try:
        new_hgso.beta = 'f'
    except:
        new_hgso.beta = 1.0

    try:
        new_hgso.beta = -1
    except:
        new_hgso.beta = 1.0

    assert new_hgso.beta == 1.0

    try:
        new_hgso.K = 'g'
    except:
        new_hgso.K = 1.0

    try:
        new_hgso.K = -1
    except:
        new_hgso.K = 1.0

    assert new_hgso.K == 1.0


def test_hgso_build():
    new_hgso = hgso.HGSO()

    assert new_hgso.built == True


def test_hgso_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_hgso = hgso.HGSO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_hgso.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm hgso failed to converge.'
