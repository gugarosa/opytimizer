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

    # Gathers the corresponding dimensions
    n_samples, n_variables, n_dimensions = x.shape[0], x.shape[1], x.shape[2]

    # Creates an array of centroids and labels
    centroids = np.zeros((n_clusters, n_variables, n_dimensions))
    labels = np.zeros(n_samples)

    for i in range(n_clusters):
        # Chooses a random sample to compose the centroid
        idx = r.generate_integer_random_number(0, n_samples)
        centroids[i] = x[idx]

    for _ in range(max_iterations):
        # Calculates the euclidean distance between samples and each centroid
        dists = np.squeeze(np.array([np.linalg.norm(x - c, axis=1) for c in centroids]))

        # Gathers the minimum distance as the cluster that conquers the sample
        updated_labels = np.squeeze(np.array(np.argmin(dists, axis=0)))

        # Calculates the difference ratio between old and new labels
        ratio = np.sum(labels != updated_labels) / n_samples

        if ratio <= tol:
            break

        # Updates the old labels with the new ones
        labels = updated_labels

        for i in range(n_clusters):
            # Gathers the samples that belongs to current centroid
            centroid_samples = x[labels == i]

            # If there are samples that belongs to the centroid
            if centroid_samples.shape[0] > 0:
                # Updates the centroid position
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

    # Creates a list to append selected individuals
    selected = []

    for _ in range(n):
        # For every tournament round, we select `size` individuals
        step = [np.random.choice(fitness) for _ in range(size)]

        # Selects the individual with the minimum fitness
        selected.append(np.where(min(step) == fitness)[0][0])

    return selected


def weighted_wheel_selection(weights: List[float]) -> int:
    """Selects an individual from a weight-based roulette.

    Args:
        weights: List of individuals weights.

    Returns:
        (int): Weight-based roulette individual.

    """

    # Gathers the cumulative summatory
    cumulative_sum = np.cumsum(weights)

    # Defines the selection probability
    prob = r.generate_uniform_random_number() * cumulative_sum[-1]

    for i, c_sum in enumerate(cumulative_sum):
        # If individual's cumulative sum is bigger than selection probability
        if c_sum > prob:
            return i

    return None
