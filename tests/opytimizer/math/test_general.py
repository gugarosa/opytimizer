import numpy as np

from opytimizer.math import general


def test_kmeans_groups_nearest_samples(monkeypatch):
    indexes = iter([0, 2])
    monkeypatch.setattr(np.random, "randint", lambda low, high: next(indexes))

    samples = np.array([[[0.0]], [[0.1]], [[10.0]], [[10.1]]])

    labels = general.kmeans(samples, n_clusters=2)

    np.testing.assert_array_equal(labels, [0, 0, 1, 1])


def test_n_wise_keeps_the_final_partial_group():
    groups = list(general.n_wise([1, 2, 3, 4, 5]))

    assert groups == [(1, 2), (3, 4), (5,)]


def test_tournament_selection_returns_the_best_drawn_indexes(monkeypatch):
    draws = iter([np.array([3, 2]), np.array([4, 4])])
    monkeypatch.setattr(np.random, "choice", lambda fitness, size: next(draws))

    selected = general.tournament_selection([1, 2, 3, 4], 2)

    assert selected == [1, 3]


def test_weighted_wheel_selection_returns_threshold_bucket(monkeypatch):
    monkeypatch.setattr(np.random, "uniform", lambda: 0.5)

    assert general.weighted_wheel_selection([1, 2, 7]) == 2


def test_weighted_wheel_selection_returns_none_without_weight():
    assert general.weighted_wheel_selection([0, 0]) is None
