import numpy as np

import opytimizer.utils.constants as c


def pairwise(iterator):
    """Iterates over an iterator and returns pairwise samples from it.

    Args:
        iterator (list): An iterator to be iterated.

    Returns:
        Pairwise samples from the iterator.

    """

    # Forces the creation of an iterator
    iterator = iter(iterator)

    # A permanent while loop
    while True:
        # Yields the next two samples from the iterator
        yield next(iterator), next(iterator)


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
