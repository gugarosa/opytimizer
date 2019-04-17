import numpy as np


def exponential(x):
    """Exponential's function.

    It can be used with 'n' variables and has minimum at -1.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = -exp(-0.5 * sum(x^2))

    """

    # Calculating Sphere's function
    s = sphere(x)

    return -np.exp(-0.5 * s)


def sphere(x):
    """Sphere's function.

    It can be used with 'n' variables and has minimum at 0.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = sum(x^2)

    """

    # Declaring Sphere's function
    y = x ** 2

    return np.sum(y)
