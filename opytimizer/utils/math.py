""" A generic math module.
    Some of the mathematical functions used by opytimizer are defined in here.

    # Methods
    norm(vector): Calculates the norm over a vector.
    span(vector, lower_bound, upper_bound): Spans a value between lower and upper bound.
"""

import numpy as np


def norm(vector=None):
    """ Calculates the norm over a vector.

        # Arguments
        vector: vector to calculate norm.

        # Returns
        total: Total value of norm function over the chosen variable.
    """
    somatory = 0.0
    for i in range(vector.size):
        somatory += (vector[i] ** 2)
    total = np.sqrt(somatory)
    return total


def span(vector=None, lower_bound=None, upper_bound=None):
    """ Spans a value between lower and upper bound.

        # Arguments
        vector: vector to span value.
        lower_bound: lower bound value.
        upper_bound: upper bound value.

        # Returns
        value: Spanned value between lower and upper bounds.
    """
    vector_norm = norm(vector)
    value = (upper_bound - lower_bound) * \
        (vector_norm / np.sqrt(vector.size)) + lower_bound
    return value


def check_limit(vector=None, lower_bound=None, upper_bound=None):
    """ Check if vector is between lower and upper bounds.

        # Arguments
        vector: vector containing the posicion values.
        lower_bound: lower bound value.
        upper_bound: upper bound value.

        # Returns
        vector: vector with values between lower and upper bounds.
    """
    # Iterate through all vector dimensions
    for i in range(vector.size):
        # If value is lower than lower bound, update it to lower's bound value
        if vector[i] < lower_bound[i]:
            vector[i] = lower_bound[i]
            # If value is greater than upper bound, update it to upper's bound value
        elif vector[i] > upper_bound[i]:
            vector[i] = upper_bound[i]
    return vector
