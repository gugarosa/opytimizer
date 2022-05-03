import numpy as np

from opytimizer.optimizers.evolutionary import foa
from opytimizer.spaces import search


def test_foa_params():
    params = {
        "life_time": 6,
        "area_limit": 30,
        "LSC": 1,
        "GSC": 1,
        "transfer_rate": 0.1,
    }

    new_foa = foa.FOA(params=params)

    assert new_foa.life_time == 6

    assert new_foa.area_limit == 30

    assert new_foa.LSC == 1

    assert new_foa.GSC == 1

    assert new_foa.transfer_rate == 0.1


def test_foa_params_setter():
    new_foa = foa.FOA()

    try:
        new_foa.life_time = "a"
    except:
        new_foa.life_time = 6

    assert new_foa.life_time == 6

    try:
        new_foa.life_time = -1
    except:
        new_foa.life_time = 6

    assert new_foa.life_time == 6

    try:
        new_foa.area_limit = "b"
    except:
        new_foa.area_limit = 30

    assert new_foa.area_limit == 30

    try:
        new_foa.area_limit = -1
    except:
        new_foa.area_limit = 30

    assert new_foa.area_limit == 30

    try:
        new_foa.LSC = "c"
    except:
        new_foa.LSC = 1

    assert new_foa.LSC == 1

    try:
        new_foa.LSC = -1
    except:
        new_foa.LSC = 1

    assert new_foa.LSC == 1

    try:
        new_foa.GSC = "d"
    except:
        new_foa.GSC = 1

    assert new_foa.GSC == 1

    try:
        new_foa.GSC = -1
    except:
        new_foa.GSC = 1

    assert new_foa.GSC == 1

    try:
        new_foa.transfer_rate = "e"
    except:
        new_foa.transfer_rate = 0.1

    assert new_foa.transfer_rate == 0.1

    try:
        new_foa.transfer_rate = -1
    except:
        new_foa.transfer_rate = 0.1

    assert new_foa.transfer_rate == 0.1


def test_foa_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_foa = foa.FOA()
    new_foa.compile(search_space)

    try:
        new_foa.age = 1
    except:
        new_foa.age = []

    assert new_foa.age == []


def test_foa_local_seeding():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_foa = foa.FOA()
    new_foa.compile(search_space)

    new_foa._local_seeding(search_space, square)


def test_foa_population_limiting():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_foa = foa.FOA()
    new_foa.compile(search_space)

    candidate = new_foa._population_limiting(search_space)

    assert len(candidate) == 0

    new_foa.life_time = 1
    new_foa.area_limit = 1
    new_foa.age = [2] * 10
    candidate = new_foa._population_limiting(search_space)

    assert len(candidate) == 9


def test_foa_global_seeding():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_foa = foa.FOA()
    new_foa.compile(search_space)

    new_foa.life_time = 1
    new_foa.area_limit = 1
    new_foa.transfer_rate = 0.5
    new_foa.age = [2] * 10
    candidate = new_foa._population_limiting(search_space)

    new_foa._global_seeding(search_space, square, candidate)


def test_foa_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_foa = foa.FOA()
    new_foa.compile(search_space)

    new_foa.update(search_space, square)
