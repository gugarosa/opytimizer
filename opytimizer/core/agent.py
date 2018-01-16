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
            position: [n_variables x n_dimensions] matrix of position values.
            fit: agent's fitness value.

        # Methods
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
