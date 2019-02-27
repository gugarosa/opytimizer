import numpy as np

def norm(array):
    """
    """

    # Calculating the norm over the hypercomplex numbers
    array_norm = np.linalg.norm(array, axis=1)

    return array_norm

def span(array, lb, ub):
    """
    """

    # We need to force lower bound to be an array
    lb = np.array(lb)

    # Also the same thing goes for upper bound
    ub = np.array(ub)

    # Calculating span function
    array_span = (ub - lb) * (norm(array) / np.sqrt(array.shape[1])) + lb

    return array_span

