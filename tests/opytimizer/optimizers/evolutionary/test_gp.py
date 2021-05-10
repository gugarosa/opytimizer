import sys

import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.evolutionary import gp
from opytimizer.spaces import tree


def test_gp_params():
    params = {
        'p_reproduction': 1.0,
        'p_mutation': 0.5,
        'p_crossover': 0.5,
        'prunning_ratio': 0.5
    }

    new_gp = gp.GP(params=params)

    assert new_gp.p_reproduction == 1.0

    assert new_gp.p_mutation == 0.5

    assert new_gp.p_crossover == 0.5

    assert new_gp.prunning_ratio == 0.5


def test_gp_params_setter():
    new_gp = gp.GP()

    try:
        new_gp.p_reproduction = 'a'
    except:
        new_gp.p_reproduction = 0.75

    try:
        new_gp.p_reproduction = -1
    except:
        new_gp.p_reproduction = 0.75

    assert new_gp.p_reproduction == 0.75

    try:
        new_gp.p_mutation = 'b'
    except:
        new_gp.p_mutation = 0.5

    try:
        new_gp.p_mutation = -1
    except:
        new_gp.p_mutation = 0.5

    assert new_gp.p_mutation == 0.5

    try:
        new_gp.p_crossover = 'c'
    except:
        new_gp.p_crossover = 0.25

    try:
        new_gp.p_crossover = -1
    except:
        new_gp.p_crossover = 0.25

    assert new_gp.p_crossover == 0.25

    try:
        new_gp.prunning_ratio = 'd'
    except:
        new_gp.prunning_ratio = 0.25

    try:
        new_gp.prunning_ratio = -1
    except:
        new_gp.prunning_ratio = 0.25

    assert new_gp.prunning_ratio == 0.25


def test_gp_prune_nodes():
    new_gp = gp.GP()

    n_nodes = new_gp._prune_nodes(10)

    assert n_nodes == 10

    new_gp.prunning_ratio = 1

    n_nodes = new_gp._prune_nodes(10)

    assert n_nodes == 2


def test_gp_reproduction():
    new_gp = gp.GP()

    tree_space = tree.TreeSpace(n_agents=10, n_terminals=2, n_variables=1,
                                min_depth=1, max_depth=5,
                                functions=['SUM'], lower_bound=[0], upper_bound=[10])

    new_gp._reproduction(tree_space)


def test_gp_mutation():
    new_gp = gp.GP()

    tree_space = tree.TreeSpace(n_agents=10, n_terminals=2, n_variables=1,
                                min_depth=1, max_depth=5,
                                functions=['SUM'], lower_bound=[0], upper_bound=[10])

    new_gp._mutation(tree_space)


def test_gp_mutate():
    new_gp = gp.GP()

    tree_space = tree.TreeSpace(n_agents=10, n_terminals=2, n_variables=1,
                                min_depth=1, max_depth=5,
                                functions=['SUM'], lower_bound=[0], upper_bound=[10])

    new_gp._mutate(tree_space, tree_space.trees[0], 1)


def test_gp_crossover():
    new_gp = gp.GP()

    tree_space = tree.TreeSpace(n_agents=10, n_terminals=2, n_variables=1,
                                min_depth=1, max_depth=5,
                                functions=['SUM'], lower_bound=[0], upper_bound=[10])

    new_gp._crossover(tree_space)


def test_gp_cross():
    new_gp = gp.GP()

    tree_space = tree.TreeSpace(n_agents=10, n_terminals=2, n_variables=1,
                                min_depth=1, max_depth=5,
                                functions=['SUM'], lower_bound=[0], upper_bound=[10])

    new_gp._cross(tree_space.trees[0], tree_space.trees[1], 1, 1)


def test_gp_evaluate():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    tree_space = tree.TreeSpace(n_agents=1000, n_terminals=2, n_variables=1,
                                min_depth=1, max_depth=5,
                                functions=['SUM'], lower_bound=[0], upper_bound=[10])

    new_gp = gp.GP()

    new_gp.evaluate(tree_space, new_function)

    assert tree_space.best_agent.fit < sys.float_info.max


def test_gp_update():
    new_gp = gp.GP()

    tree_space = tree.TreeSpace(n_agents=10, n_terminals=2, n_variables=1,
                                min_depth=1, max_depth=2,
                                functions=['SUM', 'SUB', 'MUL', 'DIV'], lower_bound=[0], upper_bound=[10])

    new_gp.update(tree_space)
