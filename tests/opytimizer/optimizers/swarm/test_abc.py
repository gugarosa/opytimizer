import numpy as np

from opytimizer.optimizers.swarm import abc
from opytimizer.spaces import search


def test_abc_params():
    params = {
        'n_trials': 5
    }

    new_abc = abc.ABC(params=params)

    assert new_abc.n_trials == 5


def test_abc_params_setter():
    new_abc = abc.ABC()

    try:
        new_abc.n_trials = 0.0
    except:
        new_abc.n_trials = 10

    try:
        new_abc.n_trials = 0
    except:
        new_abc.n_trials = 10

    assert new_abc.n_trials == 10


def test_abc_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_abc = abc.ABC()
    new_abc.compile(search_space)

    try:
        new_abc.trial = 1
    except:
        new_abc.trial = np.array([1])

    assert new_abc.trial == np.array([1])


def test_abc_evaluate_location():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_abc = abc.ABC()
    new_abc.compile(search_space)

    new_abc._evaluate_location(
        search_space.agents[0], search_space.agents[1], square, 0)


def test_abc_send_employee():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_abc = abc.ABC()
    new_abc.compile(search_space)

    new_abc._send_employee(search_space.agents, square)


def test_abc_send_onlooker():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_abc = abc.ABC()
    new_abc.compile(search_space)

    new_abc._send_onlooker(search_space.agents, square)


def test_abc_send_scout():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_abc = abc.ABC()
    new_abc.compile(search_space)

    new_abc._send_scout(search_space.agents, square)

    new_abc.trial[0] = 5
    new_abc.n_trials = 1
    new_abc._send_scout(search_space.agents, square)


def test_abc_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_abc = abc.ABC()
    new_abc.compile(search_space)

    new_abc.update(search_space, square)
