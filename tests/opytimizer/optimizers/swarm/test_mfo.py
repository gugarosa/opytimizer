import numpy as np

from opytimizer.optimizers.swarm import mfo
from opytimizer.spaces import search


def test_mfo_params():
    params = {
        'b': 1
    }

    new_mfo = mfo.MFO(params=params)

    assert new_mfo.b == 1


def test_mfo_params_setter():
    new_mfo = mfo.MFO()

    try:
        new_mfo.b = 'a'
    except:
        new_mfo.b = 1

    try:
        new_mfo.b = -1
    except:
        new_mfo.b = 1

    assert new_mfo.b == 1


def test_mfo_update():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])
    new_mfo = mfo.MFO()

    new_mfo.update(search_space, 1, 10)
