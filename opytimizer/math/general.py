"""General-based mathematical functions.
"""

from itertools import islice
from typing import Any, Iterable, List, Optional

import numpy as np

import opytimizer.math.random as r


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates the Euclidean distance between two n-dimensional points.

    Args:
        x: N-dimensional point.
        y: N-dimensional point.

    Returns:
        (float): Euclidean distance between `x` and `y`.

    """

    distance = np.linalg.norm(x - y)

    return distance


def kmeans(
    x: np.ndarray,
    n_clusters: Optional[int] = 1,
    max_iterations: Optional[int] = 100,
    tol: Optional[float] = 1e-4,
) -> np.ndarray:
    """Performs the K-Means clustering over the input data.

    Args:
        x: Input array with a shape equal to (n_samples, n_variables, n_dimensions).
        n_clusters: Number of clusters.
        max_iterations: Maximum number of clustering iterations.
        tol: Tolerance value to stop the clustering.

    Returns:
        (np.ndarray): An array holding the assigned cluster per input sample.

    """

    n_samples, n_variables, n_dimensions = x.shape[0], x.shape[1], x.shape[2]

    centroids = np.zeros((n_clusters, n_variables, n_dimensions))
    labels = np.zeros(n_samples)

    for i in range(n_clusters):
        idx = r.generate_integer_random_number(0, n_samples)
        centroids[i] = x[idx]

    for _ in range(max_iterations):
        dists = np.squeeze(np.array([np.linalg.norm(x - c, axis=1) for c in centroids]))
        updated_labels = np.squeeze(np.array(np.argmin(dists, axis=0)))

        ratio = np.sum(labels != updated_labels) / n_samples
        if ratio <= tol:
            break

        labels = updated_labels

        for i in range(n_clusters):
            centroid_samples = x[labels == i]
            if centroid_samples.shape[0] > 0:
                centroids[i] = np.mean(centroid_samples, axis=0)

    return labels


def n_wise(x: List[Any], size: Optional[int] = 2) -> Iterable:
    """Iterates over an iterator and returns n-wise samples from it.

    Args:
        x (list): Values to be iterated over.
        size: Amount of samples per iteration.

    Returns:
        (Iterable): N-wise samples from the iterator.

    """

    iterator = iter(x)

    return iter(lambda: tuple(islice(iterator, size)), ())


def tournament_selection(
    fitness: List[float], n: int, size: Optional[int] = 2
) -> np.array:
    """Selects n-individuals based on a tournament selection.

    Args:
        fitness (list): List of individuals fitness.
        n: Number of individuals to be selected.
        size: Tournament size.

    Returns:
        (np.array): Indexes of selected individuals.

    """

    selected = []
    for _ in range(n):
        step = [np.random.choice(fitness) for _ in range(size)]
        selected.append(np.where(min(step) == fitness)[0][0])

    return selected


def weighted_wheel_selection(weights: List[float]) -> int:
    """Selects an individual from a weight-based roulette.

    Args:
        weights: List of individuals weights.

    Returns:
        (int): Weight-based roulette individual.

    """

    cumulative_sum = np.cumsum(weights)
    prob = r.generate_uniform_random_number() * cumulative_sum[-1]

    for i, c_sum in enumerate(cumulative_sum):
        if c_sum > prob:
            return i

    return None
