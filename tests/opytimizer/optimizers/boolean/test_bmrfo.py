import numpy as np
from opytimark.markers.boolean import Knapsack

import opytimizer.math.random as r
from opytimizer.core import function
from opytimizer.optimizers.boolean import bmrfo
from opytimizer.spaces import boolean
from opytimizer.utils import constants

np.random.seed(0)


def test_bmrfo_hyperparams():
    hyperparams = {
        'S': r.generate_binary_random_number(size=(1, 1))
    }

    new_bmrfo = bmrfo.BMRFO(hyperparams=hyperparams)

    assert new_bmrfo.S == 0 or new_bmrfo.S == 1


def test_bmrfo_hyperparams_setter():
    new_bmrfo = bmrfo.BMRFO()

    try:
        new_bmrfo.S = 'a'
    except:
        new_bmrfo.S = r.generate_binary_random_number(size=(1, 1))

    assert new_bmrfo.S == 0 or new_bmrfo.S == 1


def test_bmrfo_build():
    new_bmrfo = bmrfo.BMRFO()

    assert new_bmrfo.built == True


def test_bmrfo_cyclone_foraging():
    new_bmrfo = bmrfo.BMRFO()

    boolean_space = boolean.BooleanSpace(
        n_agents=5, n_iterations=20, n_variables=2)

    cyclone = new_bmrfo._cyclone_foraging(
        boolean_space.agents, boolean_space.best_agent.position, 1, 1, 20)

    assert cyclone[0] == False or cyclone[0] == True


def test_bmrfo_chain_foraging():
    new_bmrfo = bmrfo.BMRFO()

    boolean_space = boolean.BooleanSpace(
        n_agents=5, n_iterations=20, n_variables=2)

    chain = new_bmrfo._chain_foraging(
        boolean_space.agents, boolean_space.best_agent.position, 1)

    assert chain[0] == False or chain[0] == True


def test_bmrfo_somersault_foraging():
    new_bmrfo = bmrfo.BMRFO()

    boolean_space = boolean.BooleanSpace(
        n_agents=5, n_iterations=20, n_variables=2)

    somersault = new_bmrfo._somersault_foraging(
        boolean_space.agents[0].position, boolean_space.best_agent.position)

    assert somersault[0] == False or somersault[0] == True


def test_bmrfo_run():
    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=Knapsack(
        values=(55, 10, 47, 5, 4), weights=(95, 4, 60, 32, 23), max_capacity=100))

    new_bmrfo = bmrfo.BMRFO()

    boolean_space = boolean.BooleanSpace(
        n_agents=10, n_iterations=20, n_variables=5)

    history = new_bmrfo.run(boolean_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm bmrfo failed to converge.'
