import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.misc import gs
from opytimizer.spaces import grid
from opytimizer.utils import constant

np.random.seed(0)


def test_gs_build():
    new_gs = gs.GS()

    assert new_gs.built == True


def test_gs_run():
    def square(x):
        return np.sum(x)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_gs = gs.GS()

    grid_space = grid.GridSpace(n_variables=2, step=(0.1, 0.1), lower_bound=(
                                0, 0), upper_bound=(5, 5))

    history = new_gs.run(grid_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm gs failed to converge.'
