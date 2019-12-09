import sys

import numpy as np

from opytimizer.core import function
from opytimizer.optimizers import gp
from opytimizer.spaces import tree
from opytimizer.utils import constants


def test_gp_hyperparams():
    hyperparams = {
        'p_reproduction': 1.0,
        'p_mutation': 0.5,
        'p_crossover': 0.5,
        'prunning_ratio': 0.5
    }

    new_gp = gp.GP(hyperparams=hyperparams)

    assert new_gp.p_reproduction == 1.0

    assert new_gp.p_mutation == 0.5

    assert new_gp.p_crossover == 0.5

    assert new_gp.prunning_ratio == 0.5


def test_gp_hyperparams_setter():
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


def test_gp_build():
    new_gp = gp.GP()

    assert new_gp.built == True


def test_gp_prune_nodes():
    new_gp = gp.GP()

    n_nodes = new_gp._prune_nodes(10)

    assert n_nodes == 10

    new_gp.prunning_ratio = 1

    n_nodes = new_gp._prune_nodes(10)

    assert n_nodes == 2


def test_gp_reproduction():
    new_gp = gp.GP()


def test_gp_mutation():
    new_gp = gp.GP()


def test_gp_mutate():
    new_gp = gp.GP()


def test_gp_crossover():
    new_gp = gp.GP()


def test_gp_cross():
    new_gp = gp.GP()


def test_gp_evaluate():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    tree_space = tree.TreeSpace(n_trees=1000, n_terminals=2, n_variables=1,
                                n_iterations=10, min_depth=1, max_depth=5,
                                functions=['SUM'], lower_bound=[0], upper_bound=[10])

    new_gp = gp.GP()

    new_gp._evaluate(tree_space, new_function)

    for t in tree_space.trees:
        print(t)

    assert tree_space.best_agent.fit < sys.float_info.max


def test_gp_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_gp = gp.GP()

    tree_space = tree.TreeSpace(n_trees=10, n_terminals=2, n_variables=1,
                                n_iterations=500, min_depth=1, max_depth=2,
                                functions=['SUM', 'SUB', 'MUL', 'DIV'], lower_bound=[0], upper_bound=[10])

    history = new_gp.run(tree_space, new_function, pre_evaluation_hook=hook)

    print(tree_space.best_tree)
    print(tree_space.best_tree.post_order)

    tree_space = tree.TreeSpace(n_trees=10, n_terminals=2, n_variables=1,
                                n_iterations=500, min_depth=2, max_depth=3,
                                functions=['EXP', 'LOG', 'SQRT', 'ABS', 'COS', 'SIN'], lower_bound=[0], upper_bound=[10])

    history = new_gp.run(tree_space, new_function, pre_evaluation_hook=hook)

    print(tree_space.best_tree)
    print(tree_space.best_tree.post_order)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm gp failed to converge.'
