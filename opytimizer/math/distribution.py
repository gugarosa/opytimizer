from math import gamma, pi, sin

import numpy as np


def generate_levy_distribution(beta=0.0, size=1):
    """Generates a n-dimensional array based on a Lévy distribution.

    References:
        X.-S. Yang and S. Deb. Computers & Operations Research. Multiobjective Cuckoo Search for Design Optimization (2013).

    Args:
        beta (float): skewness parameter.
        size (int): size of array.

    Returns:
        A Lévy distribution or array.

    """

    # Calculates the equation's numerator
    num = gamma(1+beta) * sin(pi*beta/2)

    # Calculates the equation's denominator
    den = gamma((1+beta)/2) * beta * (2 ** ((beta-1)/2))

    # Calculates the sigma for further distribution generation
    sigma_u = (num/den) ** (1/beta)

    # Calculates the 'u' distribution
    u = np.random.normal(0, sigma_u ** 2, size)

    # Calculates the 'v' distribution
    v = np.random.normal(0, 1, size)

    # Finally, we can calculate the Lévy distribution
    step = u / np.fabs(v) ** (1 / beta)

    return step
