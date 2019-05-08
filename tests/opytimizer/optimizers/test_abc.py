import sys

import numpy as np
import pytest

from opytimizer.core import function
from opytimizer.optimizers import abc
from opytimizer.spaces import search


def test_abc_hyperparams():
    hyperparams = {
        'n_trials': 5
    }

    new_abc = abc.ABC(hyperparams=hyperparams)

    assert new_abc.n_trials == 5


def test_abc_hyperparams_setter():
    new_abc = abc.ABC()

    new_abc.n_trials = 10
    assert new_abc.n_trials == 10


def test_abc_build():
    new_abc = abc.ABC()

    assert new_abc.built == True


def test_abc_run():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    hyperparams = {
        'n_trials': 10
    }

    new_abc = abc.ABC(hyperparams=hyperparams)

    search_space = search.SearchSpace(n_agents=2, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_abc.run(search_space, new_function)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0
