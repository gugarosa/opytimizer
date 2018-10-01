import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class PSO(Optimizer):

    def __init__(self, hyperparams=None):

        logger.info('Overriding Optimizer with PSO ...')

        # Override its parent class with the receiving hyperparams
        super(PSO, self).__init__(hyperparams=hyperparams)

        # Define its algorithm attribute based on whatever it is
        self.algorithm = 'PSO'

        # Default algorithm hyperparams
        self.w = 2.0

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if self.hyperparams:
            if 'w' in self.hyperparams:
                self.w = self.hyperparams['w']

        # We will log some important information
        logger.info('PSO created.')
