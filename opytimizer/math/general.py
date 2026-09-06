# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""General-based mathematical functions.

"""

from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any

import numpy as np


def kmeans(
    x: np.ndarray,
    n_clusters: int = 1,
    max_iterations: int = 100,
    tol: float = 1e-4,
) -> np.ndarray:
    """Performs the K-Means clustering over the input data.

    Args:
        x: Input array with a shape equal to (n_samples, n_variables, n_dimensions).
        n_clusters: Number of clusters.
        max_iterations: Maximum number of clustering iterations.
        tol: Tolerance value to stop the clustering.

    Returns:
        An array holding the assigned cluster per input sample.

    """

    n_samples, n_variables, n_dimensions = x.shape[0], x.shape[1], x.shape[2]

    centroids = np.zeros((n_clusters, n_variables, n_dimensions))
    labels = np.zeros(n_samples)

    for i in range(n_clusters):
        idx = np.random.randint(0, n_samples)
        centroids[i] = x[idx]

    for _ in range(max_iterations):
        dists = np.array([np.linalg.norm(x - c, axis=(1, 2)) for c in centroids])
        updated_labels = np.argmin(dists, axis=0)

        ratio = np.sum(labels != updated_labels) / n_samples
        if ratio <= tol:
            break

        labels = updated_labels

        for i in range(n_clusters):
            centroid_samples = x[labels == i]
            if centroid_samples.shape[0] > 0:
                centroids[i] = np.mean(centroid_samples, axis=0)

    return labels


def n_wise(x: Iterable[Any], size: int = 2) -> Iterator[tuple[Any, ...]]:
    """Consume an iterable lazily in consecutive groups of up to ``size`` values.

    Args:
        x: Values to be iterated over.
        size: Amount of samples per iteration.

    Returns:
        Iterator of tuples, including a final shorter group when values remain.

    """

    iterator = iter(x)

    return iter(lambda: tuple(islice(iterator, size)), ())


def tournament_selection(fitness: list[float], n: int, size: int = 2) -> list[int]:
    """Selects n-individuals based on a tournament selection.

    Args:
        fitness: List of individuals fitness.
        n: Number of individuals to be selected.
        size: Tournament size.

    Returns:
        Indexes of selected individuals.

    """

    return [np.where(np.min(np.random.choice(fitness, size)) == fitness)[0][0] for _ in range(n)]


def weighted_wheel_selection(weights: list[float]) -> int | None:
    """Selects an individual from a weight-based roulette.

    Args:
        weights: List of individuals weights.

    Returns:
        Selected index, or ``None`` when no cumulative weight exceeds the sampled threshold.

    """

    cumulative_sum = np.cumsum(weights)
    prob = np.random.uniform() * cumulative_sum[-1]

    return next((i for i, c_sum in enumerate(cumulative_sum) if c_sum > prob), None)
