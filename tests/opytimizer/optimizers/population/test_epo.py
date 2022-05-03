from opytimizer.optimizers.population import epo
from opytimizer.spaces import search


def test_epo_params():
    params = {"f": 2.0, "l": 1.5}

    new_epo = epo.EPO(params=params)

    assert new_epo.f == 2.0

    assert new_epo.l == 1.5


def test_epo_params_setter():
    new_epo = epo.EPO()

    try:
        new_epo.f = "a"
    except:
        new_epo.f = 2.0

    try:
        new_epo.l = "b"
    except:
        new_epo.l = 1.5


def test_epo_update():
    new_epo = epo.EPO()

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_epo.update(search_space, 1, 10)
