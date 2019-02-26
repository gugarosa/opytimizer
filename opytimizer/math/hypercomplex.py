import numpy as np

def norm(array):
    """
    """

    array_norm = np.linalg.norm(array, axis=1)

    return array_norm

def span(array, lb, ub):
    """
    """

    array_span = (ub - lb) * (norm(array) / np.sqrt(array.shape[1])) + lb

    return array_span

