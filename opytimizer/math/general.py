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
    distance = np.linalg.norm(x - y) ** 2

    return distance


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
