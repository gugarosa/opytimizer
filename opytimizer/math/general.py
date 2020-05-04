from itertools import islice

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constants as c


def euclidean_distance(x, y):
    """Calculates the euclidean distance between two n-dimensional points.

    Args:
        x (np.array): First n-dimensional point.
        y (np.array): Second n-dimensional point.

    Returns:
        The euclidean distance between x and y.

    """

    # Calculates the euclidean distance
    distance = np.linalg.norm(x - y) ** 2

    return distance


def pairwise(x):
    """Iterates over an iterator and returns pairwise samples from it.

    Args:
        x (list): Values to be iterated over.

    Returns:
        Pairwise samples from the iterator.

    """

    # Creats an iterator from `x`
    iterator = iter(x)

    # Splits into pairs and returns a new iterator
    return iter(lambda: tuple(islice(iterator, 2)), ())


def tournament_selection(fitness, n):
    """Selects `n` individuals based on a tournament selection algorithm.

    Args:
        fitness (list): List of individuals fitness.
        n (int): Number of individuals to be selected.

    Returns:
        A list with the indexes of the selected individuals.

    """

    # Creating a list to append selected individuals
    selected = []

    # For every `n` individual to be selected
    for _ in range(n):
        # For every tournament round, we select `TOURNAMENT_SIZE` individuals
        step = [np.random.choice(fitness) for _ in range(c.TOURNAMENT_SIZE)]

        # Selects the individual with the minimum fitness
        selected.append(np.where(min(step) == fitness)[0][0])

    return selected


def weighted_wheel_selection(weights):
    """Selects an individual from a weight-based roulette.

    Args:
        weights (list): List of individuals weights.

    Returns:
        A roulette selected individual.

    """

    # Gathers the cumulative summatory
    cumulative_sum = np.cumsum(weights)

    # Defines the selection probability
    prob = r.generate_uniform_random_number() * cumulative_sum[-1]

    # For every individual
    for i, c_sum in enumerate(cumulative_sum):
        # If individual's cumulative sum is bigger than selection probability
        if c_sum > prob:
            # Returns the individual
            return i
