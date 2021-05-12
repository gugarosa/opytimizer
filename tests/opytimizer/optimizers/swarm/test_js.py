import numpy as np

from opytimizer.optimizers.swarm import js
from opytimizer.spaces import search

np.random.seed(0)


def test_js_params():
    params = {
        'eta': 4.0,
        'beta': 3.0,
        'gamma': 0.1
    }

    new_js = js.JS(params=params)

    assert new_js.eta == 4.0

    assert new_js.beta == 3.0

    assert new_js.gamma == 0.1


def test_js_params_setter():
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


def test_js_initialize_chaotic_map():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_js = js.JS()
    new_js._initialize_chaotic_map(search_space.agents)


def test_js_create_additional_attrs():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_js = js.JS()
    new_js.create_additional_attrs(search_space)


def test_js_ocean_current():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_js = js.JS()
    new_js.create_additional_attrs(search_space)

    trend = new_js._ocean_current(search_space.agents, search_space.best_agent)

    assert trend[0][0] != 0


def test_js_motion_a():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_js = js.JS()
    new_js.create_additional_attrs(search_space)

    motion = new_js._motion_a(0, 1)

    assert motion[0][0] == 0


def test_js_motion_b():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_js = js.JS()
    new_js.create_additional_attrs(search_space)

    motion = new_js._motion_b(search_space.agents[0], search_space.agents[1])

    assert motion[0][0] != 0


def test_js_motion_a():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_js = js.JS()
    new_js.create_additional_attrs(search_space)

    new_js.update(search_space, 1, 10)


def test_nbjs_motion_a():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_nbjs = js.NBJS()
    new_nbjs.create_additional_attrs(search_space)

    motion = new_nbjs._motion_a(0, 1)

    assert motion[0] != 0
