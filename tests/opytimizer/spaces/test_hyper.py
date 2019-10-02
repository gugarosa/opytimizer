import pytest

from opytimizer.spaces import hyper


def test_hyper_initialize_agents():
    new_hyper_space = hyper.HyperSpace()

    assert new_hyper_space.agents[0].position[0][0] > 0


def test_hyper_check_limits():
    new_hyper_space = hyper.HyperSpace()

    new_hyper_space.agents[0].position[0][0] = 20

    new_hyper_space.check_limits()

    assert new_hyper_space.agents[0].position[0][0] != 20
