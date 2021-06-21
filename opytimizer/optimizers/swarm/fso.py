"""Flying Squirrel Optimizer.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core import Optimizer

logger = l.get_logger(__name__)


class FSO(Optimizer):
    """A FSO class, inherited from Optimizer.

    This is the designed class to define FSO-related
    variables and methods.

    References:
        G. Azizyan et al.
        Flying Squirrel Optimizer (FSO): A novel SI-based optimization algorithm for engineering problems.
        Iranian Journal of Optimization (2019).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> FSO.')

        # Overrides its parent class with the receiving params
        super(FSO, self).__init__()

        # Lévy distribution parameter
        self.beta = 1.5

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    @property
    def beta(self):
        """float: Lévy distribution parameter.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        if not isinstance(beta, (float, int)):
            raise e.TypeError('`beta` should be a float or integer')
        if beta <= 0 or beta > 2:
            raise e.ValueError('`beta` should be between 0 and 2')

        self._beta = beta

    def update(self, space, iteration, n_iterations):
        # Calculates the Sigma Reduction Factor of current iteration (eq. 5)
        SRF = np.sqrt((-np.log(1 - (1 / np.sqrt(iteration + 2)))) ** 2)

        # Calculates the Beta Expansion Factor
        beta = self.beta + (2 - self.beta) * ((iteration + 1) / n_iterations)

        # Iterates through all agents
        for agent in space.agents:
            # Updates the agent's position with a random walk (eq. 2 and 3)
            agent.position += SRF * r.generate_gaussian_random_number()

            # Updates the agent's position with a Lévy flight (eq. 6 to 18)
            agent.position += d.generate_levy_distribution(beta)
