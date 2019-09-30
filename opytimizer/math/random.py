import numpy as np

import opytimizer.utils.constants as c


def generate_uniform_random_number(low=0.0, high=1.0, size=1):
    """Generates a random number or array based on an uniform distribution.

    Args:
        low (float): Lower interval.
        high (float): Higher interval.
        size (int): Size of array.

    Returns:
        An uniform random number or array.

    """

    # Generates a random uniform number or array
    uniform_array = np.random.uniform(low, high, size)

    return uniform_array


def generate_gaussian_random_number(mean=0.0, variance=1.0, size=1):
    """Generates a random number or array based on a gaussian distribution.

    Args:
        mean (float): Gaussian's mean value.
        variance (float): Gaussian's variance value.
        size (int): Size of array.

    Returns:
        A gaussian random number or array.

    """

    # Generates a random gaussian number or array
    gaussian_array = np.random.normal(mean, variance, size)

    return gaussian_array


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
