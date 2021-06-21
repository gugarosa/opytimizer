"""Distribution-based mathematical generators.
"""

from math import gamma, pi, sin

import numpy as np

import opytimizer.math.random as r


def generate_bernoulli_distribution(prob=0.0, size=1):
    """Generates a Bernoulli distribution based on an input probability.

    Args:
        prob (float): Probability of distribution.
        size (int): Size of array.

    Returns:
        Bernoulli distribution n-dimensional array.

    """

    # Creates the bernoulli array
    bernoulli_array = np.zeros(size)

    # Generates a random number
    r1 = r.generate_uniform_random_number(0, 1, size)

    # Masks the array based on input probability
    bernoulli_array[r1 < prob] = 1

    return bernoulli_array


def generate_choice_distribution(n=1, probs=None, size=1):
    """Generates a random choice distribution based on probabilities.

    Args:
        n (int): Amount of values to be picked from.
        probs (np.array): Array of probabilities.
        size (int): Size of array.

    Returns:
        Choice distribution array.

    """

    # Performs the random choice based on input probabilities
    choice_array = np.random.choice(n, size, p=probs, replace=False)

    return choice_array


def generate_levy_distribution(beta=0.1, size=1):
    """Generates a n-dimensional array based on a Lévy distribution.

    References:
        X.-S. Yang and S. Deb. Computers & Operations Research.
        Multiobjective Cuckoo Search for Design Optimization (2013).

    Args:
        beta (float): Skewness parameter.
        size (int): Size of array.

    Returns:
        Lévy distribution n-dimensional array.

    """

    # Calculates the equation's numerator and denominator
    num = gamma(1 + beta) * sin(pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))

    # Calculates `sigma`
    sigma = (num / den) ** (1 / beta)

    # Calculates 'u' and `v` distributions
    u = r.generate_gaussian_random_number(0, sigma ** 2, size=size)
    v = r.generate_gaussian_random_number(size=size)

    # Calculates the Lévy distribution
    levy_array = u / np.fabs(v) ** (1 / beta)

    return levy_array
