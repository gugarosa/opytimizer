import numpy as np

from opytimizer.optimizers.swarm import woa
from opytimizer.spaces import search


def test_woa_params():
    params = {
        'b': 1
    }

    new_woa = woa.WOA(params=params)

    assert new_woa.b == 1


def test_woa_params_setter():
    new_woa = woa.WOA()

    try:
        new_woa.b = 'a'
    except:
        new_woa.b = 1


def test_woa_generate_random_agent():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_woa = woa.WOA()
    agent = new_woa._generate_random_agent(search_space.agents[0])

    assert type(agent).__name__ == 'Agent'


def test_woa_update():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_woa = woa.WOA()

    new_woa.update(search_space, 1, 10)
