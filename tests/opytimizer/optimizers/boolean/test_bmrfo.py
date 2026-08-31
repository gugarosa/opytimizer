from opytimizer.optimizers.boolean import bmrfo
from opytimizer.spaces import boolean


def test_bmrfo_params():
    params = {"S": 1}

    new_bmrfo = bmrfo.BMRFO(params=params)

    assert new_bmrfo.S == 0 or new_bmrfo.S == 1


def test_bmrfo_cyclone_foraging():
    new_bmrfo = bmrfo.BMRFO()

    boolean_space = boolean.BooleanSpace(n_agents=100, n_variables=2)

    cyclone = new_bmrfo._cyclone_foraging(
        boolean_space.agents, boolean_space.best_agent.position, 0, 1, 100
    )

    assert cyclone[0].item() is False or cyclone[0].item() is True

    cyclone = new_bmrfo._cyclone_foraging(
        boolean_space.agents, boolean_space.best_agent.position, 1, 1, 100
    )

    assert cyclone[0].item() is False or cyclone[0].item() is True

    cyclone = new_bmrfo._cyclone_foraging(
        boolean_space.agents, boolean_space.best_agent.position, 0, 1, 1
    )

    assert cyclone[0].item() is False or cyclone[0].item() is True

    cyclone = new_bmrfo._cyclone_foraging(
        boolean_space.agents, boolean_space.best_agent.position, 1, 1, 1
    )

    assert cyclone[0].item() is False or cyclone[0].item() is True


def test_bmrfo_chain_foraging():
    new_bmrfo = bmrfo.BMRFO()

    boolean_space = boolean.BooleanSpace(n_agents=100, n_variables=2)

    chain = new_bmrfo._chain_foraging(
        boolean_space.agents, boolean_space.best_agent.position, 0
    )

    assert chain[0].item() is False or chain[0].item() is True


def test_bmrfo_somersault_foraging():
    new_bmrfo = bmrfo.BMRFO()

    boolean_space = boolean.BooleanSpace(n_agents=100, n_variables=2)

    somersault = new_bmrfo._somersault_foraging(
        boolean_space.agents[0].position, boolean_space.best_agent.position
    )

    assert somersault[0].item() is False or somersault[0].item() is True


def test_bmrfo_update():
    new_bmrfo = bmrfo.BMRFO()

    boolean_space = boolean.BooleanSpace(n_agents=100, n_variables=5)

    new_bmrfo.update(boolean_space, lambda x: x.sum(), 1, 20)
