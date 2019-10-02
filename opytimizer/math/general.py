import numpy as np
from itertools import islice

import opytimizer.utils.constants as c


def pairwise(values):
    """Iterates over an iterator and returns pairwise samples from it.

    Args:
        values (list): Values to be iterated over.

    Returns:
        Pairwise samples from the iterator.

    """

    # Creats an iterator from values
    iterator = iter(values)

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
