"""Firefly Algorithm.
"""

import copy

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class FA(Optimizer):
    """A FA class, inherited from Optimizer.

    This is the designed class to define FA-related
    variables and methods.

    References:
        X.-S. Yang. Firefly algorithms for multimodal optimization.
        International symposium on stochastic algorithms (2009).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> FA.')

        # Override its parent class with the receiving params
        super(FA, self).__init__()

        # Arguments that should be used in this optimizer
        # Note they must be properties from Opytimizer class
        args = {
            'evaluate': ['space', 'function'],
            'update': ['space.agents', 'n_iterations'],
            'history': {
                'agents': 'space.agents',
                'best_agent': 'space.best_agent'
            }
        }

        # Randomization parameter
        self.alpha = 0.5

        # Attractiveness
        self.beta = 0.2

        # Light absorption coefficient
        self.gamma = 1.0

        # Builds the class
        self.build(params, args)

        logger.info('Class overrided.')

    @property
    def alpha(self):
        """float: Randomization parameter.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (float, int)):
            raise e.TypeError('`alpha` should be a float or integer')
        if alpha < 0:
            raise e.ValueError('`alpha` should be >= 0')

        self._alpha = alpha

    @property
    def beta(self):
        """float: Attractiveness parameter.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        if not isinstance(beta, (float, int)):
            raise e.TypeError('`beta` should be a float or integer')
        if beta < 0:
            raise e.ValueError('`beta` should be >= 0')

        self._beta = beta

    @property
    def gamma(self):
        """float: Light absorption coefficient.

        """

        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        if not isinstance(gamma, (float, int)):
            raise e.TypeError('`gamma` should be a float or integer')
        if gamma < 0:
            raise e.ValueError('`gamma` should be >= 0')

        self._gamma = gamma

    def update(self, agents, n_iterations):
        """Method that wraps Firefly Algorithm over all agents and variables (eq. 3-9).

        Args:
            agents (list): List of agents.
            n_iterations (int): Maximum number of iterations.

        """

        # Calculating current iteration delta
        delta = 1 - ((10e-4) / 0.9) ** (1 / n_iterations)

        # Applying update to alpha parameter
        self.alpha *= (1 - delta)

        # We copy a temporary list for iterating purposes
        temp_agents = copy.deepcopy(agents)

        # Iterating through 'i' agents
        for agent in agents:
            # Iterating through 'j' agents
            for temp in temp_agents:
                # Distance is calculated by an euclidean distance between 'i' and 'j' (eq. 8)
                distance = g.euclidean_distance(agent.position, temp.position)

                # If 'i' fit is bigger than 'j' fit
                if agent.fit > temp.fit:
                    # Recalculate the attractiveness (eq. 6)
                    beta = self.beta * np.exp(-self.gamma * distance)

                    # Generates a random uniform distribution
                    r1 = r.generate_uniform_random_number()

                    # Updates agent's position (eq. 9)
                    agent.position = beta * \
                        (temp.position + agent.position) + \
                        self.alpha * (r1 - 0.5)
