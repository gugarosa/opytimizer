import numpy as np

from opytimizer.spaces import grid


def test_grid_space_step():
    new_grid_space = grid.GridSpace(1, 0.1, 0, 1)

    assert new_grid_space.step == 0.1


def test_grid_space_step_setter():
    new_grid_space = grid.GridSpace(1, 0.1, 0, 1)

    try:
        new_grid_space.step = "a"
    except:
        new_grid_space.step = np.array([0.1])

    assert new_grid_space.step == 0.1

    try:
        new_grid_space.step = np.array([0.1, 0.1])
    except:
        new_grid_space.step = np.array([0.1])

    assert new_grid_space.step == 0.1


def test_grid_space_grid():
    new_grid_space = grid.GridSpace(1, 0.1, 0, 1)

    assert len(new_grid_space.grid) == 11


def test_grid_space_terminals_setter():
    try:
        new_grid_space = grid.GridSpace(1, 0.1, 0, 1)
        new_grid_space.grid = "a"
    except:
        new_grid_space = grid.GridSpace(1, 0.1, 0, 1)
        new_grid_space.grid = np.array([1, 1])

    assert len(new_grid_space.grid) == 2


def test_grid_create_grid():
    new_grid_space = grid.GridSpace(1, 0.1, 0, 1)

    new_grid_space._create_grid()

    assert len(new_grid_space.grid) == 11


def test_grid_initialize_agents():
    new_grid_space = grid.GridSpace(1, 0.1, 0, 1)

    assert new_grid_space.agents[0].position[0] != 1
