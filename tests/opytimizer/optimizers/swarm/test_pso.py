import sys

import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import pso
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_pso_hyperparams():
    hyperparams = {
        'w': 2,
        'c1': 1.7,
        'c2': 1.7
    }

    new_pso = pso.PSO(hyperparams=hyperparams)

    assert new_pso.w == 2

    assert new_pso.c1 == 1.7

    assert new_pso.c2 == 1.7


def test_pso_hyperparams_setter():
    new_pso = pso.PSO()

    try:
        new_pso.w = 'a'
    except:
        new_pso.w = 1

    try:
        new_pso.w = -1
    except:
        new_pso.w = 1

    assert new_pso.w == 1

    try:
        new_pso.c1 = 'b'
    except:
        new_pso.c1 = 1.5

    try:
        new_pso.c1 = -1
    except:
        new_pso.c1 = 1.5

    assert new_pso.c1 == 1.5

    try:
        new_pso.c2 = 'c'
    except:
        new_pso.c2 = 1.5

    try:
        new_pso.c2 = -1
    except:
        new_pso.c2 = 1.5

    assert new_pso.c2 == 1.5


def test_pso_build():
    new_pso = pso.PSO()

    assert new_pso.built == True


def test_pso_update_velocity():
    new_pso = pso.PSO()

    velocity = new_pso._update_velocity(1, 1, 1, 1)

    assert velocity != 0


def test_pso_update_position():
    new_pso = pso.PSO()

    position = new_pso._update_position(1, 1)

    assert position == 2


def test_pso_evaluate():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_pso = pso.PSO()

    local_position = np.zeros((2, 2, 1))

    new_pso._evaluate(search_space, new_function, local_position)

    assert search_space.best_agent.fit < sys.float_info.max


def test_pso_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_pso = pso.PSO()

    search_space = search.SearchSpace(n_agents=5, n_iterations=20,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_pso.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0
    assert len(history.local) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm pso failed to converge.'


def test_aiwpso_hyperparams():
    hyperparams = {
        'w_min': 1,
        'w_max': 3,
    }

    new_aiwpso = pso.AIWPSO(hyperparams=hyperparams)

    assert new_aiwpso.w_min == 1

    assert new_aiwpso.w_max == 3


def test_aiwpso_hyperparams_setter():
    new_aiwpso = pso.AIWPSO()

    try:
        new_aiwpso.w_min = 'a'
    except:
        new_aiwpso.w_min = 0.5

    try:
        new_aiwpso.w_min = -1
    except:
        new_aiwpso.w_min = 0.5

    assert new_aiwpso.w_min == 0.5

    try:
        new_aiwpso.w_max = 'b'
    except:
        new_aiwpso.w_max = 1.0

    try:
        new_aiwpso.w_max = -1
    except:
        new_aiwpso.w_max = 1.0

    try:
        new_aiwpso.w_max = 0
    except:
        new_aiwpso.w_max = 1.0

    assert new_aiwpso.w_max == 1.0


def test_aiwpso_rebuild():
    new_aiwpso = pso.AIWPSO()

    assert new_aiwpso.built == True


def test_aiwpso_compute_success():
    n_agents = 2

    search_space = search.SearchSpace(n_agents=n_agents, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_aiwpso = pso.AIWPSO()

    new_fitness = np.zeros(n_agents)

    new_aiwpso._compute_success(search_space.agents, new_fitness)

    assert new_aiwpso.w != 0


def test_aiwpso_evaluate():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_aiwpso = pso.AIWPSO()

    local_position = np.zeros((2, 2, 1))

    new_aiwpso._evaluate(search_space, new_function, local_position)

    assert search_space.best_agent.fit < sys.float_info.max


def test_aiwpso_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_aiwpso = pso.AIWPSO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_aiwpso.run(search_space, new_function,
                             pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0
    assert len(history.local) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm aiwpso failed to converge.'


def test_rpso_hyperparams():
    hyperparams = {
        'c1': 1.7,
        'c2': 1.7
    }

    new_rpso = pso.RPSO(hyperparams=hyperparams)

    assert new_rpso.c1 == 1.7
    assert new_rpso.c2 == 1.7


def test_rpso_hyperparams_setter():
    new_rpso = pso.RPSO()

    try:
        new_rpso.c1 = 'a'
    except:
        new_rpso.c1 = 1.5

    try:
        new_rpso.c1 = -1
    except:
        new_rpso.c1 = 1.5

    assert new_rpso.c1 == 1.5

    try:
        new_rpso.c2 = 'b'
    except:
        new_rpso.c2 = 1.5

    try:
        new_rpso.c2 = -1
    except:
        new_rpso.c2 = 1.5

    assert new_rpso.c2 == 1.5


def test_rpso_build():
    new_rpso = pso.RPSO()

    assert new_rpso.built == True


def test_rpso_update_velocity():
    new_rpso = pso.RPSO()

    velocity = new_rpso._update_velocity(1, 1, 1, 10, 1, 1)

    assert velocity != 0


def test_rpso_update_position():
    new_rpso = pso.RPSO()

    position = new_rpso._update_position(1, 1)

    assert position == 2


def test_rpso_evaluate():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_rpso = pso.RPSO()

    local_position = np.zeros((2, 2, 1))

    new_rpso._evaluate(search_space, new_function, local_position)

    assert search_space.best_agent.fit < sys.float_info.max


def test_rpso_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_rpso = pso.RPSO()

    search_space = search.SearchSpace(n_agents=5, n_iterations=20,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_rpso.run(search_space, new_function,
                           pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0
    assert len(history.local) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm rpso failed to converge.'


def test_savpso_hyperparams():
    hyperparams = {
        'w': 2,
        'c1': 1.7,
        'c2': 1.7
    }

    new_savpso = pso.SAVPSO(hyperparams=hyperparams)

    assert new_savpso.w == 2

    assert new_savpso.c1 == 1.7

    assert new_savpso.c2 == 1.7


def test_savpso_hyperparams_setter():
    new_savpso = pso.SAVPSO()

    try:
        new_savpso.w = 'a'
    except:
        new_savpso.w = 1

    try:
        new_savpso.w = -1
    except:
        new_savpso.w = 1

    assert new_savpso.w == 1

    try:
        new_savpso.c1 = 'b'
    except:
        new_savpso.c1 = 1.5

    try:
        new_savpso.c1 = -1
    except:
        new_savpso.c1 = 1.5

    assert new_savpso.c1 == 1.5

    try:
        new_savpso.c2 = 'c'
    except:
        new_savpso.c2 = 1.5

    try:
        new_savpso.c2 = -1
    except:
        new_savpso.c2 = 1.5

    assert new_savpso.c2 == 1.5


def test_savpso_update_velocity():
    new_savpso = pso.SAVPSO()

    velocity = new_savpso._update_velocity(1, 1, 1, 1, 1)

    assert velocity == 0


def test_savpso_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_savpso = pso.SAVPSO()

    search_space = search.SearchSpace(n_agents=5, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_savpso.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0
    assert len(history.local) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm pso failed to converge.'


def test_vpso_hyperparams():
    hyperparams = {
        'w': 2,
        'c1': 1.7,
        'c2': 1.7
    }

    new_vpso = pso.VPSO(hyperparams=hyperparams)

    assert new_vpso.w == 2

    assert new_vpso.c1 == 1.7

    assert new_vpso.c2 == 1.7


def test_vpso_hyperparams_setter():
    new_vpso = pso.VPSO()

    try:
        new_vpso.w = 'a'
    except:
        new_vpso.w = 1

    try:
        new_vpso.w = -1
    except:
        new_vpso.w = 1

    assert new_vpso.w == 1

    try:
        new_vpso.c1 = 'b'
    except:
        new_vpso.c1 = 1.5

    try:
        new_vpso.c1 = -1
    except:
        new_vpso.c1 = 1.5

    assert new_vpso.c1 == 1.5

    try:
        new_vpso.c2 = 'c'
    except:
        new_vpso.c2 = 1.5

    try:
        new_vpso.c2 = -1
    except:
        new_vpso.c2 = 1.5

    assert new_vpso.c2 == 1.5


def test_vpso_update_velocity():
    new_vpso = pso.VPSO()

    velocity, v_velocity = new_vpso._update_velocity(1, 1, 1, 1, 1)

    assert velocity == 0.7
    assert v_velocity == 0


def test_vpso_update_position():
    new_vpso = pso.VPSO()

    position = new_vpso._update_position(1, 1, 1)

    assert position == 2


def test_vpso_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_vpso = pso.VPSO()

    search_space = search.SearchSpace(n_agents=5, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[1, 1])

    history = new_vpso.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0
    assert len(history.local) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm pso failed to converge.'
