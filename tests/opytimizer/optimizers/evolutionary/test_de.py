import numpy as np

from opytimizer.optimizers.evolutionary import de
from opytimizer.spaces import search


def test_de_params():
    params = {
        'CR': 0.9,
        'F': 0.7
    }

    new_de = de.DE(params=params)

    assert new_de.CR == 0.9

    assert new_de.F == 0.7


def test_de_params_setter():
    new_de = de.DE()

    try:
        new_de.CR = 'a'
    except:
        new_de.CR = 0.5

    try:
        new_de.CR = -1
    except:
        new_de.CR = 0.5

    assert new_de.CR == 0.5

    try:
        new_de.F = 'b'
    except:
        new_de.F = 0.5

    try:
        new_de.F = -1
    except:
        new_de.F = 0.5

    assert new_de.F == 0.5


def test_de_mutate_agent():
    new_de = de.DE()

    search_space = search.SearchSpace(n_agents=4, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    agent = new_de._mutate_agent(
        search_space.agents[0], search_space.agents[1], search_space.agents[2], search_space.agents[3])

    assert agent.position[0][0] != 0


def test_de_run():
    def square(x):
        return np.sum(x**2)

    new_de = de.DE()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_de.update(search_space, square)
