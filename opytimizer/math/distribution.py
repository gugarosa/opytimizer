"""Distribution-based mathematical generators.
"""

from math import gamma, pi, sin
from typing import Optional

import numpy as np

import opytimizer.math.random as r


def generate_bernoulli_distribution(prob: float = 0.0, size: int = 1) -> np.ndarray:
    """Generates a Bernoulli distribution based on an input probability.

    Args:
        prob: Probability of distribution.
        size: Size of array.

    Returns:
        (np.ndarray): Bernoulli distribution n-dimensional array.

    """

    bernoulli_array = np.zeros(size)

    r1 = r.generate_uniform_random_number(0, 1, size)
    bernoulli_array[r1 < prob] = 1

    return bernoulli_array


def generate_choice_distribution(
    n: int = 1, probs: Optional[np.ndarray] = None, size: int = 1
) -> np.ndarray:
    """Generates a random choice distribution based on probabilities.

    Args:
        n: Amount of values to be picked from.
        probs: Array of probabilities.
        size: Size of array.

    Returns:
        (np.ndarray): Choice distribution array.

    """

    choice_array = np.random.choice(n, size, p=probs, replace=False)

    return choice_array


def generate_levy_distribution(beta: float = 0.1, size: int = 1) -> np.ndarray:
    """Generates a n-dimensional array based on a Lévy distribution.

    References:
        X.-S. Yang and S. Deb. Computers & Operations Research.
        Multiobjective Cuckoo Search for Design Optimization (2013).

    Args:
        beta: Skewness parameter.
        size: Size of array.

    Returns:
        (np.ndarray): Lévy distribution n-dimensional array.

    """

    num = gamma(1 + beta) * sin(pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))

    sigma = (num / den) ** (1 / beta)

    u = r.generate_gaussian_random_number(0, sigma**2, size=size)
    v = r.generate_gaussian_random_number(size=size)

    levy_array = u / np.fabs(v) ** (1 / beta)

    return levy_array
