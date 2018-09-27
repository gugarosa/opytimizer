import numpy as np


def norm(vector=None):
    """Calculates the norm over a vector.

    Args:
        vector (array): vector to calculate norm.

    Returns:
        Total value of norm function over the chosen variable.

    """

    somatory = 0.0

    # Iterate through all vector
    for i in range(vector.size):
        # Adds to somatory its squared element
        somatory += (vector[i] ** 2)

    # Apply square root to all somatory variable
    total = np.sqrt(somatory)

    return total


# def span(vector=None, lower_bound=None, upper_bound=None):
#     """Spans the vector to a value between lower and upper bound.

#     Args:
#         vector (array): vector to span value.
#         lower_bound (float): lower bound value.
#         upper_bound (float): upper bound value.

#     Returns:
#         Spanned value between lower and upper bounds.

#     """

#     # Calculate the vector's norm
#     vector_norm = norm(vector)

#     # Span its value between lower and upper bounds
#     value = (upper_bound - lower_bound) * \
#         (vector_norm / np.sqrt(vector.size)) + lower_bound

#     return value


def check_unitary(vector=None):
    """Check if vector is between 0 and 1.

    Args:
        vector (array): vector containing the position values.

    Returns:
        A vector with values between 0 and 1.

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
    """Check if vector is between lower and upper bounds.

    Args:
        vector (array): vector containing the position values.
        lower_bound (float): lower bound value.
        upper_bound (float): upper bound value.

    Returns:
        Vector with values between lower and upper bounds.

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
