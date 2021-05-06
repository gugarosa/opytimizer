import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.misc import cem
from opytimizer.spaces import search


def test_cem_params():
    params = {
        'n_updates': 5,
        'alpha': 0.7,
    }

    new_cem = cem.CEM(params=params)

    assert new_cem.n_updates == 5

    assert new_cem.alpha == 0.7


def test_cem_params_setter():
    new_cem = cem.CEM()

    try:
        new_cem.n_updates = 'a'
    except:
        new_cem.n_updates = 10

    try:
        new_cem.n_updates = -1
    except:
        new_cem.n_updates = 10

    assert new_cem.n_updates == 10

    try:
        new_cem.alpha = 'b'
    except:
        new_cem.alpha = 0.5

    try:
        new_cem.alpha = -1
    except:
        new_cem.alpha = 0.5

    assert new_cem.alpha == 0.5


def test_cem_build():
    new_cem = cem.CEM()

    assert new_cem.built == True


def test_cem_create_additional_attrs():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_cem = cem.CEM()
    new_cem.create_additional_attrs(search_space)

    try:
        new_cem.mean = 1
    except:
        new_cem.mean = np.array([1])

    assert new_cem.mean == 1

    try:
        new_cem.std = 1
    except:
        new_cem.std = np.array([1])

    assert new_cem.std == 1


def test_cem_create_new_samples():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_cem = cem.CEM()
    new_cem.create_additional_attrs(search_space)

    new_cem._create_new_samples(search_space.agents, square)


def test_cem_update_mean():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_cem = cem.CEM()
    new_cem.create_additional_attrs(search_space)

    new_cem._update_mean(np.array([1, 1]))

    assert new_cem.mean[0] != 0


def test_cem_update_std():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_cem = cem.CEM()
    new_cem.create_additional_attrs(search_space)

    new_cem._update_std(np.array([1, 1]))

    assert new_cem.std[0] != 0


def test_cem_update():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_cem = cem.CEM()
    new_cem.create_additional_attrs(search_space)

    new_cem.update(search_space, new_function)
