import numpy as np

from opytimizer.optimizers.social import ssd
from opytimizer.spaces import search
from opytimizer.utils import constant


def test_ssd_params():
    params = {"c": 2.0, "decay": 0.99}

    new_ssd = ssd.SSD(params=params)

    assert new_ssd.c == 2.0

    assert new_ssd.decay == 0.99


def test_ssd_params_setter():
    new_ssd = ssd.SSD()

    try:
        new_ssd.c = "a"
    except:
        new_ssd.c = 0.5

    try:
        new_ssd.c = -1
    except:
        new_ssd.c = 0.5

    assert new_ssd.c == 0.5

    try:
        new_ssd.decay = "b"
    except:
        new_ssd.decay = 0.99

    try:
        new_ssd.decay = -1
    except:
        new_ssd.decay = 0.99

    assert new_ssd.decay == 0.99


def test_ssd_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_ssd = ssd.SSD()
    new_ssd.compile(search_space)

    try:
        new_ssd.local_position = 1
    except:
        new_ssd.local_position = np.array([1])

    assert new_ssd.local_position == 1

    try:
        new_ssd.velocity = 1
    except:
        new_ssd.velocity = np.array([1])

    assert new_ssd.velocity == 1


def test_ssd_mean_global_solution():
    new_ssd = ssd.SSD()

    mean = new_ssd._mean_global_solution(1, 2, 3)

    assert mean != 0


def test_ssd_update_position():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_ssd = ssd.SSD()
    new_ssd.compile(search_space)

    position = new_ssd._update_position(1, 1)

    assert position[0][0] != 0


def test_ssd_update_velocity():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_ssd = ssd.SSD()
    new_ssd.compile(search_space)

    velocity = new_ssd._update_velocity(0.5, 10, 1)

    assert velocity[0] != 0


def test_ssd_evaluate():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_ssd = ssd.SSD()
    new_ssd.compile(search_space)

    new_ssd.evaluate(search_space, square)

    assert search_space.best_agent.fit != constant.FLOAT_MAX


def test_ssd_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_ssd = ssd.SSD()
    new_ssd.compile(search_space)

    new_ssd.update(search_space, square)
