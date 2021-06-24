"""General-based mathematical functions.
"""

from itertools import islice

import numpy as np

import opytimizer.math.random as r


def euclidean_distance(x, y):
    """Calculates the Euclidean distance between two n-dimensional points.

    Args:
        x (np.array): N-dimensional point.
        y (np.array): N-dimensional point.

    Returns:
        Euclidean distance between `x` and `y`.

    """

    # Calculates the Euclidean distance
    distance = np.linalg.norm(x - y)

    return distance


def kmeans(x, n_clusters=1, max_iterations=100):
    """Performs the K-Means clustering over the input data.

    Args:
        x (np.array): Input array with a shape equal to (n_samples, n_variables, n_dimensions).
        n_clusters (int): Number of clusters.
        max_iterations (int): Maximum number of clustering iterations.

    Returns:
        An array holding the assigned cluster per input sample.

    """

    # Gathers the corresponding dimensions
    n_samples, n_variables, n_dimensions = x.shape[0], x.shape[1], x.shape[2]

    # Creates an array of centroids and labels
    centroids = np.zeros((n_clusters, n_variables, n_dimensions))
    labels = np.zeros(n_samples)

    # Iterates through all possible clusters
    for i in range(n_clusters):
        # Chooses a random sample to compose the centroid
        idx = r.generate_integer_random_number(0, n_samples)
        centroids[i] = x[idx]

    # Iterates till the maximum amount of possible iterations
    for _ in range(max_iterations):
        #
        dists = np.array(
            [np.linalg.norm(x - c, axis=1) for c in centroids])


        print(dists.shape)


        #
        _labels = np.array(np.argmin(dists, axis=0))
        _labels = np.squeeze(_labels)

        # print(_labels)

        if (labels == _labels).all():
            break
        else:
            labels = _labels
            for i in range(n_clusters):
                if len(x[labels==i]) > 0:
                    centroids[i] = np.mean(x[labels == i], axis=0)

    return labels


def n_wise(x, size=2):
    """Iterates over an iterator and returns n-wise samples from it.

    Args:
        x (list): Values to be iterated over.
        size (int): Amount of samples per iteration.

    Returns:
        N-wise samples from the iterator.

    """

    # Creates an iterator from `x`
    iterator = iter(x)

    return iter(lambda: tuple(islice(iterator, size)), ())


def tournament_selection(fitness, n, size=2):
    """Selects n-individuals based on a tournament selection.

    Args:
        fitness (list): List of individuals fitness.
        n (int): Number of individuals to be selected.
        size (int): Tournament size.

    Returns:
        Indexes of selected individuals.

    """

    # Creates a list to append selected individuals
    selected = []

    # For every n-individual to be selected
    for _ in range(n):
        # For every tournament round, we select `size` individuals
        step = [np.random.choice(fitness) for _ in range(size)]

        # Selects the individual with the minimum fitness
        selected.append(np.where(min(step) == fitness)[0][0])

    return selected


def weighted_wheel_selection(weights):
    """Selects an individual from a weight-based roulette.

    Args:
        weights (list): List of individuals weights.

    Returns:
        Weight-based roulette individual.

    """

    # Gathers the cumulative summatory
    cumulative_sum = np.cumsum(weights)

    # Defines the selection probability
    prob = r.generate_uniform_random_number() * cumulative_sum[-1]

    # For every individual
    for i, c_sum in enumerate(cumulative_sum):
        # If individual's cumulative sum is bigger than selection probability
        if c_sum > prob:
            return i

    return None
