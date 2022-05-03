from opytimizer.optimizers.population import osa
from opytimizer.spaces import search


def test_osa_params():
    params = {"beta": 1.9}

    new_osa = osa.OSA(params=params)

    assert new_osa.beta == 1.9


def test_osa_params_setter():
    new_osa = osa.OSA()

    try:
        new_osa.beta = "a"
    except:
        new_osa.beta = 1.9

    try:
        new_osa.beta = -1
    except:
        new_osa.beta = 1.9

    assert new_osa.beta == 1.9


def test_osa_update():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[-10, -10], upper_bound=[10, 10]
    )

    new_osa = osa.OSA()

    new_osa.update(search_space, 1, 10)
