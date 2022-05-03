import numpy as np

from opytimizer.optimizers.science import weo
from opytimizer.spaces import search


def test_weo_params():
    params = {
        "E_min": -3.5,
        "E_max": -0.5,
        "theta_min": -np.pi / 3.6,
        "theta_max": -np.pi / 9,
    }

    new_weo = weo.WEO(params=params)

    assert new_weo.E_min == -3.5

    assert new_weo.E_max == -0.5

    assert new_weo.theta_min == -np.pi / 3.6

    assert new_weo.theta_max == -np.pi / 9


def test_weo_params_setter():
    new_weo = weo.WEO()

    try:
        new_weo.E_min = "a"
    except:
        new_weo.E_min = -3.5

    assert new_weo.E_min == -3.5

    try:
        new_weo.E_max = "b"
    except:
        new_weo.E_max = -0.5

    assert new_weo.E_max == -0.5

    try:
        new_weo.E_max = -5.0
    except:
        new_weo.E_max = -0.5

    assert new_weo.E_max == -0.5

    try:
        new_weo.theta_min = "c"
    except:
        new_weo.theta_min = -np.pi / 3.6

    assert new_weo.theta_min == -np.pi / 3.6

    try:
        new_weo.theta_max = "d"
    except:
        new_weo.theta_max = -np.pi / 9

    assert new_weo.theta_max == -np.pi / 9

    try:
        new_weo.theta_max = -np.pi / 3
    except:
        new_weo.theta_max = -np.pi / 9

    assert new_weo.theta_max == -np.pi / 9


def test_weo_evaporation_flux():
    new_weo = weo.WEO()

    theta = np.pi

    J = new_weo._evaporation_flux(theta)

    assert J == 0.6349860094028128


def test_weo_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_weo = weo.WEO()

    new_weo.update(search_space, square, 1, 10)
    new_weo.update(search_space, square, 10, 10)
