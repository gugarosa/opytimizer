import copy

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class GP(Optimizer):
    """A GP class, inherited from Optimizer.

    This will be the designed class to define GP-related
    variables and methods.

    References:
        J. Koza. Genetic programming: On the programming of computers by means of natural selection (1992).

    """

    def __init__(self, algorithm='GP', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): A string holding optimizer's algorithm name.
            hyperparams (dict): An hyperparams dictionary containing key-value
                parameters to meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> GP.')

        # Override its parent class with the receiving hyperparams
        super(GP, self).__init__(algorithm=algorithm)

        # Probability of reproduction
        self.reproduction = 0.3

        # Probability of mutation
        self.mutation = 0.4

        # Probability of crossover
        self.crossover = 0.4

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def reproduction(self):
        """float: Probability of reproduction.

        """

        return self._reproduction

    @reproduction.setter
    def reproduction(self, reproduction):
        self._reproduction = reproduction

    @property
    def mutation(self):
        """float: Probability of mutation.

        """

        return self._mutation

    @mutation.setter
    def mutation(self, mutation):
        self._mutation = mutation

    @property
    def crossover(self):
        """float: Probability of crossover.

        """

        return self._crossover

    @crossover.setter
    def crossover(self, crossover):
        self._crossover = crossover

    def _build(self, hyperparams):
        """This method will serve as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
                parameters to meta-heuristics.

        """

        logger.debug('Running private method: build().')

        # We need to save the hyperparams object for faster looking up
        self.hyperparams = hyperparams

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if hyperparams:
            if 'reproduction' in hyperparams:
                self.reproduction = hyperparams['reproduction']
            if 'mutation' in hyperparams:
                self.mutation = hyperparams['mutation']
            if 'crossover' in hyperparams:
                self.crossover = hyperparams['crossover']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: reproduction = {self.reproduction}, mutation = {self.mutation}, crossover = {self.crossover} | Built: {self.built}.')
