from opytimizer.optimizers.science import teo
from opytimizer.spaces import search


def test_teo_params():
    params = {"c1": True, "c2": True, "pro": 0.05, "n_TM": 4}

    new_teo = teo.TEO(params=params)

    assert new_teo.c1 is True

    assert new_teo.c2 is True

    assert new_teo.pro == 0.05

    assert new_teo.n_TM == 4


def test_teo_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_teo = teo.TEO()
    new_teo.compile(search_space)


def test_teo_update():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_teo = teo.TEO()
    new_teo.compile(search_space)
    new_teo.pro = 1.0

    new_teo.update(search_space, 1, 10)
