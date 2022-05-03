import numpy as np

from opytimizer.optimizers.swarm import kh
from opytimizer.spaces import search

np.random.seed(0)


def test_kh_params():
    params = {
        "N_max": 0.01,
        "w_n": 0.42,
        "NN": 5,
        "V_f": 0.02,
        "w_f": 0.38,
        "D_max": 0.002,
        "C_t": 0.5,
        "Cr": 0.2,
        "Mu": 0.05,
    }

    new_kh = kh.KH(params=params)

    assert new_kh.N_max == 0.01
    assert new_kh.w_n == 0.42
    assert new_kh.NN == 5
    assert new_kh.V_f == 0.02
    assert new_kh.w_f == 0.38
    assert new_kh.D_max == 0.002
    assert new_kh.C_t == 0.5
    assert new_kh.Cr == 0.2
    assert new_kh.Mu == 0.05


def test_kh_params_setter():
    new_kh = kh.KH()

    try:
        new_kh.N_max = "a"
    except:
        new_kh.N_max = 0.01

    assert new_kh.N_max == 0.01

    try:
        new_kh.N_max = -1
    except:
        new_kh.N_max = 0.01

    assert new_kh.N_max == 0.01

    try:
        new_kh.w_n = "a"
    except:
        new_kh.w_n = 0.42

    assert new_kh.w_n == 0.42

    try:
        new_kh.w_n = 1.01
    except:
        new_kh.w_n = 0.42

    assert new_kh.w_n == 0.42

    try:
        new_kh.NN = 0.5
    except:
        new_kh.NN = 5

    assert new_kh.NN == 5

    try:
        new_kh.NN = -1
    except:
        new_kh.NN = 5

    assert new_kh.NN == 5

    try:
        new_kh.V_f = "a"
    except:
        new_kh.V_f = 0.02

    assert new_kh.V_f == 0.02

    try:
        new_kh.V_f = -1
    except:
        new_kh.V_f = 0.02

    assert new_kh.V_f == 0.02

    try:
        new_kh.w_f = "a"
    except:
        new_kh.w_f = 0.38

    assert new_kh.w_f == 0.38

    try:
        new_kh.w_f = 1.01
    except:
        new_kh.w_f = 0.38

    assert new_kh.w_f == 0.38

    try:
        new_kh.D_max = "a"
    except:
        new_kh.D_max = 0.02

    assert new_kh.D_max == 0.02

    try:
        new_kh.D_max = -1
    except:
        new_kh.D_max = 0.02

    assert new_kh.D_max == 0.02

    try:
        new_kh.C_t = "a"
    except:
        new_kh.C_t = 0.5

    assert new_kh.C_t == 0.5

    try:
        new_kh.C_t = 2.01
    except:
        new_kh.C_t = 0.5

    assert new_kh.C_t == 0.5

    try:
        new_kh.Cr = "a"
    except:
        new_kh.Cr = 0.2

    assert new_kh.Cr == 0.2

    try:
        new_kh.Cr = 1.1
    except:
        new_kh.Cr = 0.2

    assert new_kh.Cr == 0.2

    try:
        new_kh.Mu = "a"
    except:
        new_kh.Mu = 0.05

    assert new_kh.Mu == 0.05

    try:
        new_kh.Mu = 1.1
    except:
        new_kh.Mu = 0.05

    assert new_kh.Mu == 0.05


def test_kh_compile():
    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    try:
        new_kh.motion = 1
    except:
        new_kh.motion = np.array([1])

    assert new_kh.motion == np.array([1])

    try:
        new_kh.foraging = 1
    except:
        new_kh.foraging = np.array([1])

    assert new_kh.foraging == np.array([1])


def test_kh_food_location():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    food = new_kh._food_location(search_space.agents, square)

    assert food.fit >= 0


def test_kh_sensing_distance():
    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    distance, eucl_distance = new_kh._sensing_distance(search_space.agents, 0)

    assert distance >= 0
    assert len(eucl_distance) >= 0


def test_kh_get_neighbours():
    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    distance, eucl_distance = new_kh._sensing_distance(search_space.agents, 0)

    neighbours = new_kh._get_neighbours(search_space.agents, 0, distance, eucl_distance)

    assert len(neighbours) >= 0


def test_kh_local_alpha():
    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    distance, eucl_distance = new_kh._sensing_distance(search_space.agents, 0)

    neighbours = new_kh._get_neighbours(search_space.agents, 0, distance, eucl_distance)

    alpha = new_kh._local_alpha(
        search_space.agents[0],
        search_space.agents[-1],
        search_space.agents[0],
        neighbours,
    )

    assert alpha.shape == (2, 1) or alpha == 0


def test_kh_target_alpha():
    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    alpha = new_kh._target_alpha(
        search_space.agents[0], search_space.agents[-1], search_space.agents[0], 1
    )

    assert alpha.shape == (2, 1) or alpha == 0


def test_kh_neighbour_motion():
    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    motion = np.zeros((5, 2, 1))

    new_motion = new_kh._neighbour_motion(search_space.agents, 0, 1, 20, motion)

    assert new_motion.shape == (5, 2, 1)


def test_kh_food_beta():
    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    beta = new_kh._food_beta(
        search_space.agents[0],
        search_space.agents[-1],
        search_space.agents[0],
        search_space.agents[0],
        1,
    )

    assert beta.shape == (2, 1)


def test_kh_best_beta():
    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    beta = new_kh._best_beta(
        search_space.agents[0], search_space.agents[-1], search_space.agents[0]
    )

    assert beta.shape == (2, 1)


def test_kh_foraging_motion():
    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    foraging = np.zeros((5, 2, 1))

    new_foraging = new_kh._foraging_motion(
        search_space.agents, 0, 1, 20, search_space.agents[0], foraging
    )

    assert new_foraging.shape == (5, 2, 1)


def test_kh_physical_diffusion():
    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    new_physical = new_kh._physical_diffusion(1, 1, 1, 20)

    assert new_physical.shape == (1, 1)


def test_kh_update_position():
    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    motion = np.zeros((2, 1))

    foraging = np.zeros((2, 1))

    new_position = new_kh._update_position(
        search_space.agents, 0, 1, 20, search_space.agents[0], motion, foraging
    )

    assert new_position.shape == (2, 1)


def test_kh_crossover():
    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    crossover = new_kh._crossover(search_space.agents, 0)

    assert crossover.position.shape == (2, 1)


def test_kh_mutation():
    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    mutation = new_kh._mutation(search_space.agents, 0)

    assert mutation.position.shape == (2, 1)


def test_kh_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_kh = kh.KH()
    new_kh.compile(search_space)

    new_kh.update(search_space, square, 1, 10)
