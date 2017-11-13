"""This is the optimizer's structure and its basic functions module.
"""

class Optimizer(object):
    """A optimizer class for all meta-heuristic optimization techniques.

        # Arguments
            hyperparams: Optimizer-related hyperparams.
            
        # Properties
            hyperparams: Optimizer-related hyperparams.

        # Methods
    """

    def __init__(self, **kwargs):
        # These properties should be set by the user via keyword arguments.
        allowed_kwargs = {'hyperparams',
                         }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        # Iterate through all properties and set the remaining ones.
        self.hyperparams = None

        # Check if arguments are supplied
        if 'hyperparams' in kwargs:
            hyperparams = kwargs['hyperparams']
            self.hyperparams = hyperparams
