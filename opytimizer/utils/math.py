""" A generic math module.
    Some of the mathematical functions used by opytimizer are defined in here.

    # Methods
    norm(vector): Calculates the norm over a vector.
    span(vector, lower_bound, upper_bound): Spans the vector to a value between lower and upper bound.
    check_unitary(vector): Check if vector is between 0 and 1.
    check_bounds(vector, lower_bound, upper_bound): Check if vector is between lower and upper bounds.
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
    # Iterate through all vector
    for i in range(vector.size):
        # Adds to somatory its squared element
        somatory += (vector[i] ** 2)
    # Apply square root to all somatory variable
    total = np.sqrt(somatory)
    return total


def span(vector=None, lower_bound=None, upper_bound=None):
    """ Spans the vector to a value between lower and upper bound.

        # Arguments
        vector: vector to span value.
        lower_bound: lower bound value.
        upper_bound: upper bound value.

        # Returns
        value: Spanned value between lower and upper bounds.
    """
    # Calculate the vector's norm
    vector_norm = norm(vector)
    # Span its value between lower and upper bounds
    value = (upper_bound - lower_bound) * \
        (vector_norm / np.sqrt(vector.size)) + lower_bound
    return value


def check_unitary(vector=None):
    """ Check if vector is between 0 and 1.

        # Arguments
        vector: vector containing the position values.

        # Returns
        vector: vector with values between 0 and 1.
    """
    # Iterate through all vector dimensions
    for i in range(vector.size):
        # If value is lower than 0, update it to 0
        if vector[i] < 0:
            vector[i] = 0
            # If value is greater than 1, update it to 1
        elif vector[i] > 1:
            vector[i] = 1
    return vector


def check_bounds(vector=None, lower_bound=None, upper_bound=None):
    """ Check if vector is between lower and upper bounds.

        # Arguments
        vector: vector containing the position values.
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
