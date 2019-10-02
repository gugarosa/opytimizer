import pytest

from opytimizer.math import general


def test_pairwise():
    list = [1, 2, 3, 4]

    pairs = general.pairwise(list)

    for p in pairs:
        pass

    assert type(pairs).__name__ == 'callable_iterator' or 'generator'


def test_tournament_selection():
    fitness = [1, 2, 3, 4]

    selected = general.tournament_selection(fitness, 2)

    assert len(selected) == 2
