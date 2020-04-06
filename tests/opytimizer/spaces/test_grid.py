import numpy as np
import pytest

from opytimizer.spaces import grid


def test_grid_space_step():
    new_grid_space = grid.GridSpace()

    assert new_grid_space.step == 0.1


def test_grid_space_step_setter():
    new_grid_space = grid.GridSpace()

    try:
        new_grid_space.step = 'a'
    except:
        new_grid_space.step = 0.1

    assert new_grid_space.step == 0.1

    try:
        new_grid_space.step = 0
    except:
        new_grid_space.step = 0.1

    assert new_grid_space.step == 0.1


def test_grid_space_grid():
    new_grid_space = grid.GridSpace()

    assert len(new_grid_space.grid) == 10


def test_grid_space_terminals_setter():
    try:
        new_grid_space = grid.GridSpace()
        new_grid_space.grid = 'a'
    except:
        new_grid_space = grid.GridSpace()
        new_grid_space.grid = np.array((1, 1))

    assert len(new_grid_space.grid) == 2


def test_grid_create_grid():
    new_grid_space = grid.GridSpace()

    new_grid_space._create_grid(0.1, [1, 1], [2, 2])

    assert len(new_grid_space.grid) == 100


def test_grid_initialize_agents():
    new_grid_space = grid.GridSpace()

    assert new_grid_space.agents[0].position[0] != 1
