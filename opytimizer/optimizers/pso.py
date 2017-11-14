""" This is PSO's structure and its basic functions module.
"""

from opytimizer.core.optimizer import Optimizer


class PSO(Optimizer):
    """ A particle swarm optimization class.

        # Arguments:
            hyperparams: PSO-related hyperparams.

        # Properties
            algorithm: Algorithm identifier (PSO).
            hyperparams: PSO-related hyperparams.

        # Methods
    """

    def __init__(self, **kwargs):
        super(PSO, self).__init__(**kwargs)
        self.algorithm = 'PSO'
