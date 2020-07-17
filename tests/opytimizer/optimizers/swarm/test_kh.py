import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import kh
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_kh_hyperparams():
    hyperparams = {
        'N_max': 0.01,
        'w_n': 0.42,
        'NN': 5,
        'V_f': 0.02,
        'w_f': 0.38,
        'D_max': 0.002,
        'C_t': 0.5,
        'Cr': 0.2,
        'Mu': 0.05
    }

    new_kh = kh.KH(hyperparams=hyperparams)

    assert new_kh.N_max == 0.01
    assert new_kh.w_n == 0.42
    assert new_kh.NN == 5
    assert new_kh.V_f == 0.02
    assert new_kh.w_f == 0.38
    assert new_kh.D_max == 0.002
    assert new_kh.C_t == 0.5
    assert new_kh.Cr == 0.2
    assert new_kh.Mu == 0.05


def test_kh_hyperparams_setter():
    new_kh = kh.KH()

    try:
        new_kh.N_max = 'a'
    except:
        new_kh.N_max = 0.01

    assert new_kh.N_max == 0.01

    try:
        new_kh.N_max = -1
    except:
        new_kh.N_max = 0.01

    assert new_kh.N_max == 0.01

    try:
        new_kh.w_n = 'a'
    except:
        new_kh.w_n = 0.42

    assert new_kh.w_n == 0.42

    try:
        new_kh.w_n = 1.01
    except:
        new_kh.w_n = 0.42

    assert new_kh.w_n == 0.42

    try:
        new_kh.NN = 0.5
    except:
        new_kh.NN = 5

    assert new_kh.NN == 5

    try:
        new_kh.NN = -1
    except:
        new_kh.NN = 5

    assert new_kh.NN == 5

    try:
        new_kh.V_f = 'a'
    except:
        new_kh.V_f = 0.02

    assert new_kh.V_f == 0.02

    try:
        new_kh.V_f = -1
    except:
        new_kh.V_f = 0.02

    assert new_kh.V_f == 0.02

    try:
        new_kh.w_f = 'a'
    except:
        new_kh.w_f = 0.38

    assert new_kh.w_f == 0.38

    try:
        new_kh.w_f = 1.01
    except:
        new_kh.w_f = 0.38

    assert new_kh.w_f == 0.38

    try:
        new_kh.D_max = 'a'
    except:
        new_kh.D_max = 0.02

    assert new_kh.D_max == 0.02

    try:
        new_kh.D_max = -1
    except:
        new_kh.D_max = 0.02

    assert new_kh.D_max == 0.02

    try:
        new_kh.C_t = 'a'
    except:
        new_kh.C_t = 0.5

    assert new_kh.C_t == 0.5

    try:
        new_kh.C_t = 2.01
    except:
        new_kh.C_t = 0.5

    assert new_kh.C_t == 0.5

    try:
        new_kh.Cr = 'a'
    except:
        new_kh.Cr = 0.2

    assert new_kh.Cr == 0.2

    try:
        new_kh.Cr = 1.1
    except:
        new_kh.Cr = 0.2

    assert new_kh.Cr == 0.2

    try:
        new_kh.Mu = 'a'
    except:
        new_kh.Mu = 0.05

    assert new_kh.Mu == 0.05

    try:
        new_kh.Mu = 1.1
    except:
        new_kh.Mu = 0.05

    assert new_kh.Mu == 0.05


def test_kh_build():
    new_kh = kh.KH()

    assert new_kh.built == True


def test_kh_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_kh = kh.KH()

    search_space = search.SearchSpace(n_agents=5, n_iterations=20,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_kh.run(search_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm kh failed to converge.'
