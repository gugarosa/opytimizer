import numpy as np


def generate_binary_random_number(size=1):
    """Generates a binary random number or array based on an uniform distribution.

    Args:
        size (int): Size of array.

    Returns:
        An binary random number or array.

    """

    # Generates a random binary number or array
    binary_array = np.round(np.random.uniform(0, 1, size))

    return binary_array


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
