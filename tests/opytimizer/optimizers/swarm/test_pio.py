import numpy as np

from opytimizer.optimizers.swarm import pio
from opytimizer.spaces import search


def test_pio_params():
    params = {
        'n_c1': 150,
        'n_c2': 200,
        'R': 0.2
    }

    new_pio = pio.PIO(params=params)

    assert new_pio.n_c1 == 150

    assert new_pio.n_c2 == 200

    assert new_pio.R == 0.2


def test_pio_params_setter():
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


def test_pio_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_pio = pio.PIO()
    new_pio.compile(search_space)

    try:
        new_pio.n_p = 'a'
    except:
        new_pio.n_p = 1

    assert new_pio.n_p == 1

    try:
        new_pio.n_p = -1
    except:
        new_pio.n_p = 1

    assert new_pio.n_p == 1

    try:
        new_pio.velocity = 1
    except:
        new_pio.velocity = np.array([1])

    assert new_pio.velocity == np.array([1])


def test_pio_calculate_center():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_pio = pio.PIO()
    new_pio.compile(search_space)

    center = new_pio._calculate_center(search_space.agents)


def test_pio_update_center_position():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_pio = pio.PIO()
    new_pio.compile(search_space)

    center = new_pio._calculate_center(search_space.agents)
    position = new_pio._update_center_position(
        search_space.agents[0].position, center)


def test_pio_update():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_pio = pio.PIO()
    new_pio.compile(search_space)

    new_pio.update(search_space, 1)
    new_pio.update(search_space, 175)
