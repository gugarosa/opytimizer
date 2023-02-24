import sys

import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.evolutionary import gsgp
from opytimizer.spaces import tree


def test_gsgp_mutation():
    new_gsgp = gsgp.GSGP()

    tree_space = tree.TreeSpace(
        n_agents=10,
        n_terminals=2,
        n_variables=1,
        min_depth=1,
        max_depth=5,
        functions=["SUM"],
        lower_bound=[0],
        upper_bound=[10],
    )

    new_gsgp._mutation(tree_space)


def test_gsgp_mutate():
    new_gsgp = gsgp.GSGP()

    tree_space = tree.TreeSpace(
        n_agents=10,
        n_terminals=2,
        n_variables=1,
        min_depth=1,
        max_depth=5,
        functions=["SUM"],
        lower_bound=[0],
        upper_bound=[10],
    )

    new_gsgp._mutate(tree_space.trees[0], tree_space.n_variables, 1)


def test_gsgp_crossover():
    new_gsgp = gsgp.GSGP()

    tree_space = tree.TreeSpace(
        n_agents=10,
        n_terminals=2,
        n_variables=1,
        min_depth=1,
        max_depth=5,
        functions=["SUM"],
        lower_bound=[0],
        upper_bound=[10],
    )

    new_gsgp._crossover(tree_space)


def test_gsgp_cross():
    new_gsgp = gsgp.GSGP()

    tree_space = tree.TreeSpace(
        n_agents=10,
        n_terminals=2,
        n_variables=1,
        min_depth=1,
        max_depth=5,
        functions=["SUM"],
        lower_bound=[0],
        upper_bound=[10],
    )

    new_gsgp._cross(
        tree_space.trees[0], tree_space.trees[1], tree_space.n_variables, 1, 1
    )
