from opytimizer.optimizers.science import teo
from opytimizer.spaces import search


def test_teo_params():
    params = {"c1": True, "c2": True, "pro": 0.05, "n_TM": 4}

    new_teo = teo.TEO(params=params)

    assert new_teo.c1 is True

    assert new_teo.c2 is True

    assert new_teo.pro == 0.05

    assert new_teo.n_TM == 4


def test_teo_params_setter():
    new_teo = teo.TEO()

    try:
        new_teo.c1 = 1
    except:
        new_teo.c1 = True

    assert new_teo.c1 is True

    try:
        new_teo.c2 = 1
    except:
        new_teo.c2 = True

    assert new_teo.c2 is True

    try:
        new_teo.pro = "a"
    except:
        new_teo.pro = 0.05

    assert new_teo.pro == 0.05

    try:
        new_teo.pro = -1
    except:
        new_teo.pro = 0.05

    assert new_teo.pro == 0.05

    try:
        new_teo.n_TM = "b"
    except:
        new_teo.n_TM = 4

    assert new_teo.n_TM == 4

    try:
        new_teo.n_TM = -1
    except:
        new_teo.n_TM = 4

    assert new_teo.n_TM == 4

    try:
        new_teo.TM = 1
    except:
        new_teo.TM = []

    assert new_teo.TM == []


def test_teo_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_teo = teo.TEO()
    new_teo.compile(search_space)

    try:
        new_teo.environment = 1
    except:
        new_teo.environment = []

    assert new_teo.environment == []


def test_teo_update():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_teo = teo.TEO()
    new_teo.compile(search_space)
    new_teo.pro = 1.0

    new_teo.update(search_space, 1, 10)
