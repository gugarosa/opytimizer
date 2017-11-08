"""This is the agent's structure and its basic functions module.
"""

import numpy as np


class Agent(object):
    """A agent class for all meta-heuristic optimization techniques.

        # Properties
            n_variables: number of decision variables.
            n_dimensions: dimension of search space.
            position: n_variables-dimensional array of position values.
            fit: agent's fitness value.

        # Methods
            check_limits(position, lower_bound, upper_bound): Check if vector 'position' is between lower and upper bounds.
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
        """Check if array 'position' is between lower and upper bounds.

            # Arguments
            position: n_variables-dimensional array of position values.
            lower_bound: array of lower bound values.
            upper_bound: array of upper bound values.
        """
        # Iterate through all dimensions, i for number of variables and j for number of dimensions
        for i in range(self.n_variables):
            for j in range(self.n_dimensions):
                # Check if current position is smaller than lower bound
                if self.position[i][j] < lower_bound[i]:
                    self.position[i][j] = lower_bound[i]
                # Else, check if current position is bigger than upper bound
                elif self.position[i][j] > upper_bound[i]:
                    self.position[i][j] = upper_bound[i]
