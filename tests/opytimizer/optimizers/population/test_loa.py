from types import new_class

import numpy as np

from opytimizer.optimizers.population import loa
from opytimizer.spaces import search

np.random.seed(0)


def test_lion_params():
    new_lion = loa.Lion(1, 1, [0], [1], np.array([0]), 0)

    assert new_lion.best_position[0] == 0

    assert new_lion.p_fit == 0

    assert new_lion.nomad == False

    assert new_lion.female == False

    assert new_lion.pride == 0

    assert new_lion.group == 0


def test_lion_params_setter():
    new_lion = loa.Lion(1, 1, [0], [1], np.array([0]), 0)

    try:
        new_lion.best_position = 0
    except:
        new_lion.best_position = np.array([0])

    assert new_lion.best_position == np.array([0])

    try:
        new_lion.p_fit = 'a'
    except:
        new_lion.p_fit = 0

    assert new_lion.p_fit == 0

    try:
        new_lion.nomad = 'b'
    except:
        new_lion.nomad = False

    assert new_lion.nomad == False

    try:
        new_lion.female = 'c'
    except:
        new_lion.female = False

    assert new_lion.female == False

    try:
        new_lion.pride = 'd'
    except:
        new_lion.pride = 0

    assert new_lion.pride == 0

    try:
        new_lion.pride = -1
    except:
        new_lion.pride = 0

    assert new_lion.pride == 0

    try:
        new_lion.group = 'e'
    except:
        new_lion.group = 0

    assert new_lion.group == 0

    try:
        new_lion.group = -1
    except:
        new_lion.group = 0

    assert new_lion.group == 0


def test_loa_params():
    params = {
        'N': 0.2,
        'P': 4,
        'S': 0.8,
        'R': 0.2,
        'I': 0.4,
        'Ma': 0.3,
        'Mu': 0.2
    }

    new_loa = loa.LOA(params=params)

    assert new_loa.N == 0.2

    assert new_loa.P == 4

    assert new_loa.S == 0.8

    assert new_loa.R == 0.2

    assert new_loa.I == 0.4

    assert new_loa.Ma == 0.3

    assert new_loa.Mu == 0.2


def test_loa_params_setter():
    new_loa = loa.LOA()

    try:
        new_loa.N = 'a'
    except:
        new_loa.N = 0.2

    assert new_loa.N == 0.2

    try:
        new_loa.N = -1
    except:
        new_loa.N = 0.2

    assert new_loa.N == 0.2

    try:
        new_loa.P = 'b'
    except:
        new_loa.P = 4

    assert new_loa.P == 4

    try:
        new_loa.P = -1
    except:
        new_loa.P = 4

    assert new_loa.P == 4

    try:
        new_loa.S = 'c'
    except:
        new_loa.S = 0.8

    assert new_loa.S == 0.8

    try:
        new_loa.S = -1
    except:
        new_loa.S = 0.8

    assert new_loa.S == 0.8

    try:
        new_loa.R = 'd'
    except:
        new_loa.R = 0.2

    assert new_loa.R == 0.2

    try:
        new_loa.R = -1
    except:
        new_loa.R = 0.2

    assert new_loa.R == 0.2

    try:
        new_loa.I = 'e'
    except:
        new_loa.I = 0.4

    assert new_loa.I == 0.4

    try:
        new_loa.I = -1
    except:
        new_loa.I = 0.4

    assert new_loa.I == 0.4

    try:
        new_loa.Ma = 'f'
    except:
        new_loa.Ma = 0.3

    assert new_loa.Ma == 0.3

    try:
        new_loa.Ma = -1
    except:
        new_loa.Ma = 0.3

    assert new_loa.Ma == 0.3

    try:
        new_loa.Mu = 'g'
    except:
        new_loa.Mu = 0.2

    assert new_loa.Mu == 0.2

    try:
        new_loa.Mu = -1
    except:
        new_loa.Mu = 0.2

    assert new_loa.Mu == 0.2


def test_loa_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)


def test_loa_get_nomad_lions():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)

    nomads = new_loa._get_nomad_lions(search_space.agents)


def test_loa_get_pride_lions():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)

    prides = new_loa._get_pride_lions(search_space.agents)


def test_loa_hunting():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)

    prides = new_loa._get_pride_lions(search_space.agents)

    new_loa._hunting(prides, square)


def test_loa_moving_safe_place():
    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)

    prides = new_loa._get_pride_lions(search_space.agents)

    new_loa._moving_safe_place(prides)


def test_loa_roaming():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)

    prides = new_loa._get_pride_lions(search_space.agents)

    new_loa._roaming(prides, square)


def test_loa_mating_operator():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)
    new_loa.Mu = 1

    nomads = new_loa._get_nomad_lions(search_space.agents)
    males_nomads = [nomad for nomad in nomads if not nomad.female]

    a1, a2 = new_loa._mating_operator(
        search_space.agents[0], males_nomads, square)

    assert type(a1).__name__ == 'Lion'
    assert type(a2).__name__ == 'Lion'


def test_loa_mating():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)

    prides = new_loa._get_pride_lions(search_space.agents)

    prides_cubs = new_loa._mating(prides, square)


def test_loa_defense():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)

    nomads = new_loa._get_nomad_lions(search_space.agents)
    prides = new_loa._get_pride_lions(search_space.agents)
    cubs = new_loa._mating(prides, square)

    new_nomads, new_prides = new_loa._defense(nomads, prides, cubs)


def test_loa_nomad_roaming():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)

    nomads = new_loa._get_nomad_lions(search_space.agents)

    new_loa._nomad_roaming(nomads, square)


def test_loa_nomad_mating():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)

    nomads = new_loa._get_nomad_lions(search_space.agents)

    new_nomads = new_loa._nomad_mating(nomads, square)


def test_loa_nomad_attack():
    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)

    nomads = new_loa._get_nomad_lions(search_space.agents)
    prides = new_loa._get_pride_lions(search_space.agents)

    new_nomads, new_prides = new_loa._nomad_attack(nomads, prides)


def test_loa_migrating():
    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)

    nomads = new_loa._get_nomad_lions(search_space.agents)
    prides = new_loa._get_pride_lions(search_space.agents)

    new_nomads, new_prides = new_loa._migrating(nomads, prides)


def test_loa_equilibrium():
    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)

    nomads = new_loa._get_nomad_lions(search_space.agents)
    prides = new_loa._get_pride_lions(search_space.agents)
    prides[0] = prides[0] + prides[0]

    new_nomads, new_prides = new_loa._equilibrium(nomads, prides, 100)


def test_loa_check_prides_for_males():
    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)

    prides = new_loa._get_pride_lions(search_space.agents)

    for agent in prides[0]:
        agent.female = True

    new_loa._check_prides_for_males(prides)


def test_loa_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_loa = loa.LOA()
    new_loa.compile(search_space)

    new_loa.update(search_space, square)
