""" This is the agent's structure and its basic functions module.
"""

import numpy as np


class Agent(object):
    """ An agent class for all meta-heuristic optimization techniques.

        # Arguments
            n_variables: number of decision variables.
            n_dimensions: dimension of search space.

        # Properties
            n_variables: number of decision variables.
            n_dimensions: dimension of search space.
            position: n_variables x n_dimensions matrix of position values.
            fit: agent's fitness value.

        # Methods
            check_limits(lower_bound, upper_bound): Check if vector 'position'
            is between lower and upper bounds.
            norm(variable_index): Calculates the norm over a chosen variable.
    """

    def __init__(self, **kwargs):
        # These properties should be set by the user via keyword arguments.
        allowed_kwargs = {'n_variables',
                          'n_dimensions',
                          }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        # Iterate through all properties and set the remaining ones.
        if 'n_variables' in kwargs and 'n_dimensions' in kwargs:
            n_variables = kwargs['n_variables']
            n_dimensions = kwargs['n_dimensions']
            self.n_variables = n_variables
            self.n_dimensions = n_dimensions

            # Create the position vector based on number of variables and dimensions.
            self.position = np.zeros((n_variables, n_dimensions))

            # Fitness value is initialized with zero.
            self.fit = 0

    def check_limits(self, lower_bound, upper_bound):
        """ Check if array 'position' is between lower and upper bounds.

            # Arguments
            lower_bound: array of lower bound values.
            upper_bound: array of upper bound values.
        """
        # Iterate through all dimensions, i for number of variables and j for number of dimensions
        for i in range(self.n_variables):
            # If agent's dimension is equal to 1, variables must be within lower and upper bounds
            if self.n_dimensions == 1:
                for j in range(self.n_dimensions):
                    # Check if current position is smaller than lower bound
                    if self.position[i][j] < lower_bound[i]:
                        self.position[i][j] = lower_bound[i]
                    # Else, check if current position is bigger than upper bound
                    elif self.position[i][j] > upper_bound[i]:
                        self.position[i][j] = upper_bound[i]
            # If agent's dimension is greater than 1, variables' components must be within 0 and 1
            if self.n_dimensions > 1:
                for j in range(self.n_dimensions):
                    # Check if current position is smaller than 0
                    if self.position[i][j] < 0:
                        self.position[i][j] = 0
                    # Else, check if current position is bigger than 1
                    elif self.position[i][j] > 1:
                        self.position[i][j] = 1

    def norm(self, variable_index, lower_bound, upper_bound):
        """ Calculates the norm over a chosen variable.

            # Arguments
            variable_index: Index identifier of chosen variable.
            lower_bound: Variable's lower bound.
            upper_bound: Variable's upper bound.

            # Returns
            total: Total value of norm function over the chosen variable.
        """
        # Initialize sum variable with 0
        somatory = 0.0
        # Iterate through all components (n_dimensions) from a chosen variable
        for i in range(len(self.position[variable_index])):
            # Add to sum the corresponding value
            somatory += (self.position[variable_index][i] ** 2)
        # Apply square root to get the partial value
        partial = np.sqrt(somatory)
        # If agent's number of dimensions is 1, returns only the norm
        if self.n_dimensions == 1:
            return partial
        # If agent's number of dimensions is greater than 1, span the value between lower and upper bounds
        if self.n_dimensions > 1:
            total = (upper_bound[variable_index] - lower_bound[variable_index]) * (
                partial / np.sqrt(self.n_dimensions)) + lower_bound[variable_index]
            return total
