from opytimizer.optimizers.science import moa
from opytimizer.spaces import search


def test_moa_params():
    params = {
        'alpha': 1.0,
        'rho': 2.0,
    }

    new_moa = moa.MOA(params=params)

    assert new_moa.alpha == 1.0

    assert new_moa.rho == 2.0


def test_moa_params_setter():
    new_moa = moa.MOA()

    try:
        new_moa.alpha = 'a'
    except:
        new_moa.alpha = 1.0

    assert new_moa.alpha == 1.0

    try:
        new_moa.alpha = -1
    except:
        new_moa.alpha = 1.0

    assert new_moa.alpha == 1.0

    try:
        new_moa.rho = 'b'
    except:
        new_moa.rho = 2.0

    assert new_moa.rho == 2.0

    try:
        new_moa.rho = -1
    except:
        new_moa.rho = 2.0

    assert new_moa.rho == 2.0


def test_moa_create_additional_attrs():
    try:
        search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                          lower_bound=[0, 0], upper_bound=[10, 10])

        new_moa = moa.MOA()
        new_moa.create_additional_attrs(search_space)
    except:
        search_space = search.SearchSpace(n_agents=9, n_variables=2,
                                          lower_bound=[0, 0], upper_bound=[10, 10])

        new_moa = moa.MOA()
        new_moa.create_additional_attrs(search_space)


def test_moa_update():
    search_space = search.SearchSpace(n_agents=9, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_moa = moa.MOA()
    new_moa.create_additional_attrs(search_space)

    new_moa.update(search_space)
