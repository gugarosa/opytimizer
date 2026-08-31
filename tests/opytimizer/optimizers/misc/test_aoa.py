from opytimizer.optimizers.misc import aoa
from opytimizer.spaces import search


def test_aoa_params():
    params = {"a_min": 0.2, "a_max": 1.0, "alpha": 5.0, "mu": 0.499}

    new_aoa = aoa.AOA(params=params)

    assert new_aoa.a_min == 0.2

    assert new_aoa.a_max == 1.0

    assert new_aoa.alpha == 5.0

    assert new_aoa.mu == 0.499


def test_aoa_update():
    new_aoa = aoa.AOA()

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_aoa.update(search_space, 1, 10)
