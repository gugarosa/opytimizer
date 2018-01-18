""" This is the agent's structure and its basic functions module.
"""

import numpy as np

from opytimizer.utils.exception import ArgumentException


class Agent(object):
    """ An agent class for all meta-heuristic optimization techniques.

        # Arguments
            n_variables: number of decision variables.
            n_dimensions: dimension of search space.

        # Properties
            n_variables: number of decision variables.
            n_dimensions: dimension of search space.
            position: [n_variables x n_dimensions matrix of position values.
            fit: agent's fitness value.
    """

    def __init__(self, **kwargs):
        # These properties should be set by the user via keyword arguments.
        allowed_kwargs = {'n_variables',
                          'n_dimensions',
                         }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        # Define all class variables as 'None'
        self.n_variables = None
        self.n_dimensions = None
        self.position = None
        self.fit = None

        # Check if arguments are supplied
        if 'n_variables' not in kwargs:
            raise ArgumentException('n_variables')
        if 'n_dimensions' not in kwargs:
            raise ArgumentException('n_dimensions')

        # Apply arguments to class variables
        self.n_variables = kwargs['n_variables']
        self.n_dimensions = kwargs['n_dimensions']

        # Create the position vector based on number of variables and dimensions.
        self.position = np.zeros((self.n_variables, self.n_dimensions))

        # Fitness value is initialized with zero.
        self.fit = 0
