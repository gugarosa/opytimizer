"""This is the optimizer's structure and its basic functions module.
"""

import json


class Optimizer(object):
    """A optimizer class for all meta-heuristic optimization techniques.

        # Properties
            algorithm: Optimization algorithm.
            hyperparams: JSON object containing hyperparams from hyperparam_path.

        # Methods
    """

    def __init__(self, **kwargs):
        # These properties should be set by the user via keyword arguments.
        allowed_kwargs = {'algorithm',
                          'hyperparams_path',
                          }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        # Iterate through all properties and set the remaining ones.
        self.algorithm = None
        self.hyperparams = None

        # Check if arguments are supplied
        if 'algorithm' in kwargs:
            algorithm = kwargs['algorithm']
            self.algorithm = algorithm
        if 'hyperparams_path' in kwargs:
            hyperparams_path = kwargs['hyperparams_path']
            # Loads hyperparams JSON
            with open(hyperparams_path) as json_file:
                hyperparams = json.load(json_file)
            self.hyperparams = hyperparams
