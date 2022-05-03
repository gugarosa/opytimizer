import numpy as np
from opytimark.markers.boolean import Knapsack

import opytimizer.math.random as r
from opytimizer.core import function
from opytimizer.optimizers.boolean import bmrfo
from opytimizer.spaces import boolean


def test_bmrfo_params():
    params = {"S": r.generate_binary_random_number(size=(1, 1))}

    new_bmrfo = bmrfo.BMRFO(params=params)

    assert new_bmrfo.S == 0 or new_bmrfo.S == 1


def test_bmrfo_params_setter():
    new_bmrfo = bmrfo.BMRFO()

    try:
        new_bmrfo.S = "a"
    except:
        new_bmrfo.S = r.generate_binary_random_number(size=(1, 1))

    assert new_bmrfo.S == 0 or new_bmrfo.S == 1


def test_bmrfo_cyclone_foraging():
    new_bmrfo = bmrfo.BMRFO()

    boolean_space = boolean.BooleanSpace(n_agents=100, n_variables=2)

    cyclone = new_bmrfo._cyclone_foraging(
        boolean_space.agents, boolean_space.best_agent.position, 0, 1, 100
    )

    assert cyclone[0] is False or cyclone[0] is True

    cyclone = new_bmrfo._cyclone_foraging(
        boolean_space.agents, boolean_space.best_agent.position, 1, 1, 100
    )

    assert cyclone[0] is False or cyclone[0] is True

    cyclone = new_bmrfo._cyclone_foraging(
        boolean_space.agents, boolean_space.best_agent.position, 0, 1, 1
    )

    assert cyclone[0] is False or cyclone[0] is True

    cyclone = new_bmrfo._cyclone_foraging(
        boolean_space.agents, boolean_space.best_agent.position, 1, 1, 1
    )

    assert cyclone[0] is False or cyclone[0] is True


def test_bmrfo_chain_foraging():
    new_bmrfo = bmrfo.BMRFO()

    boolean_space = boolean.BooleanSpace(n_agents=100, n_variables=2)

    chain = new_bmrfo._chain_foraging(
        boolean_space.agents, boolean_space.best_agent.position, 0
    )

    assert chain[0] is False or chain[0] is True


def test_bmrfo_somersault_foraging():
    new_bmrfo = bmrfo.BMRFO()

    boolean_space = boolean.BooleanSpace(n_agents=100, n_variables=2)

    somersault = new_bmrfo._somersault_foraging(
        boolean_space.agents[0].position, boolean_space.best_agent.position
    )

    assert somersault[0] is False or somersault[0] is True


def test_bmrfo_update():
    new_function = function.Function(
        pointer=Knapsack(
            values=(55, 10, 47, 5, 4), weights=(95, 4, 60, 32, 23), max_capacity=100
        )
    )

    new_bmrfo = bmrfo.BMRFO()

    boolean_space = boolean.BooleanSpace(n_agents=100, n_variables=5)

    new_bmrfo.update(boolean_space, new_function, 1, 20)
