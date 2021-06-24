import numpy as np

from opytimizer.math import general


def test_euclidean_distance():
    x = np.array([0, 1, 2, 3])

    y = np.array([1, 2, 3, 4])

    assert general.euclidean_distance(x, y) == 2.0


def test_kmeans():
    x = np.random.uniform(0, 1, (20, 2, 1))

    y = general.kmeans(x, n_clusters=3)

    assert y.shape[0] == 20


def test_n_wise():
    list = [1, 2, 3, 4]

    pairs = general.n_wise(list)

    for _ in pairs:
        pass

    assert type(pairs).__name__ == 'callable_iterator' or 'generator'


def test_tournament_selection():
    fitness = [1, 2, 3, 4]

    selected = general.tournament_selection(fitness, 2)

    assert len(selected) == 2


def test_weighted_wheel_selection():
    weights = [1, 2, 3, 4, 5, 6, 7, 8]

    assert general.weighted_wheel_selection(weights) >= 0

    weights = [0, 0, 0, 0, 0, 0, 0, 0]

    assert general.weighted_wheel_selection(weights) is None
