"""This is the optimizer's structure and its basic functions module.
"""

import json


class Optimizer(object):
    """A optimizer class for all meta-heuristic optimization techniques.

        # Properties
            algorithm: Optimization algorithm.
            hyperparams_path: JSON file containing hyperparams.

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
        if 'algorithm' in kwargs and 'hyperparams_path' in kwargs:
            algorithm = kwargs['algorithm']
            hyperparams_path = kwargs['hyperparams_path']
            self.algorithm = algorithm

            # Loads hyperparams JSON
            with open(hyperparams_path) as json_file:
                hyperparams = json.load(json_file)
            self.hyperparams = hyperparams
