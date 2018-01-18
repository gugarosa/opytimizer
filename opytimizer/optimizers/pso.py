""" This is PSO's structure and its basic functions module.
"""

from opytimizer.core.optimizer import Optimizer
from opytimizer.utils.exception import ParameterException


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
        if 'w' not in self.hyperparams:
            raise ParameterException('w')
        self.w = self.hyperparams['w']

    def updateVelocity(self, vector=None):
        for i in range(vector.size):
            vector[i] = self.w

    def call(self, n_agents=None, agent=None):
        for i in range(n_agents):
            self.updateVelocity(vector=agent[i].position)
            agent[i].fit = self.function.evaluate(data_type=self.data_type, position=self.agent[i].position)
