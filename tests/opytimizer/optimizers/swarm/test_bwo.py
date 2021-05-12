import numpy as np

from opytimizer.optimizers.swarm import bwo
from opytimizer.spaces import search


def test_bwo_params():
    params = {
        'pp': 0.6,
        'cr': 0.44,
        'pm': 0.4,
    }

    new_bwo = bwo.BWO(params=params)

    assert new_bwo.pp == 0.6

    assert new_bwo.cr == 0.44

    assert new_bwo.pm == 0.4


def test_bwo_params_setter():
    new_bwo = bwo.BWO()

    try:
        new_bwo.pp = 'a'
    except:
        new_bwo.pp = 0.6

    try:
        new_bwo.pp = -1
    except:
        new_bwo.pp = 0.6

    assert new_bwo.pp == 0.6

    try:
        new_bwo.cr = 'b'
    except:
        new_bwo.cr = 0.44

    try:
        new_bwo.cr = -1
    except:
        new_bwo.cr = 0.44

    assert new_bwo.cr == 0.44

    try:
        new_bwo.pm = 'c'
    except:
        new_bwo.pm = 0.4

    try:
        new_bwo.pm = -1
    except:
        new_bwo.pm = 0.4

    assert new_bwo.pm == 0.4


def test_bwo_procreating():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[1, 1], upper_bound=[10, 10])

    new_bwo = bwo.BWO()

    y1, y2 = new_bwo._procreating(
        search_space.agents[0], search_space.agents[1])

    assert type(y1).__name__ == 'Agent'
    assert type(y2).__name__ == 'Agent'


def test_bwo_mutation():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[1, 1], upper_bound=[10, 10])

    new_bwo = bwo.BWO()

    alpha = new_bwo._mutation(search_space.agents[0])

    assert type(alpha).__name__ == 'Agent'


def test_bwo_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[1, 1], upper_bound=[10, 10])

    new_bwo = bwo.BWO()

    new_bwo.update(search_space, square)
