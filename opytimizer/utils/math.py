""" A generic math module.
    Some of the mathematical functions used by opytimizer are defined in here.
"""

import numpy as np

def norm(agent, variable_index, lower_bound, upper_bound):
    """ Calculates the norm over a chosen variable.

        # Arguments
        agent: Agent object to be used.
        variable_index: Index identifier of chosen variable.
        lower_bound: Variable's lower bound.
        upper_bound: Variable's upper bound.

        # Returns
        total: Total value of norm function over the chosen variable.
    """
    # Initialize sum variable with 0
    somatory = 0.0
    # Iterate through all components (n_dimensions) from a chosen variable
    for i in range(len(agent.position[variable_index])):
        # Add to sum the corresponding value
        somatory += (agent.position[variable_index][i] ** 2)
    # Apply square root to get the partial value
    partial = np.sqrt(somatory)
    # If agent's number of dimensions is 1, returns only the norm
    if agent.n_dimensions == 1:
        return partial
    # If agent's number of dimensions is greater than 1,
    # span the value between lower and upper bounds
    if agent.n_dimensions > 1:
        total = (upper_bound[variable_index] - lower_bound[variable_index]) * (
            partial / np.sqrt(agent.n_dimensions)) + lower_bound[variable_index]
        return total
