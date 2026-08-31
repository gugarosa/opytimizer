from opytimizer.optimizers.misc import doa
from opytimizer.spaces import search


def test_doa_params():
    params = {"r": 1.0}

    new_doa = doa.DOA(params=params)

    assert new_doa.r == 1.0


def test_doa_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_doa = doa.DOA()
    new_doa.compile(search_space)


def test_doa_calculate_chaotic_map():
    new_doa = doa.DOA()

    c_map = new_doa._calculate_chaotic_map(0, 1)

    assert c_map.shape == ()


def test_doa_update():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_doa = doa.DOA()
    new_doa.compile(search_space)

    new_doa.update(search_space)
