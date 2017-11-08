import numpy as np

class Agent(object):
    """A agent class for all meta-heuristic optimization techniques.

        # Properties
        	n_variables: number of decision variables
            n_dimensions: dimension of search space (1: normal | 4: quaternion | 8: octonion)
        	x: n-dimensional array of position values
        	fit: agent's fitness value

        # Methods
        check_limits(x, LB, UB): Check if vector 'x' is between lower and upper bounds
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
            self.x = np.zeros((n_variables,n_dimensions))

            # Fitness value is initialized with zero.
            self.fit = 0

    def check_limits(self, LB, UB):
        # Iterate through all dimensions, i for number of variables and j for number of dimensions
        for i in range(self.n_variables):
            for j in range(self.n_dimensions):
                # Check if current position is smaller than lower bound
                if self.x[i][j] < LB[i]:
                    self.x[i][j] = LB[i]
                # Else, check if current position is bigger than upper bound
                elif self.x[i][j] > UB[i]:
                    self.x[i][j] = UB[i]
