"""Owl Search Algorithm.
"""

import copy

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class OSA(Optimizer):
    """An OSA class, inherited from Optimizer.

    This is the designed class to define OSA-related
    variables and methods.

    References:
        M. Jain, S. Maurya, A. Rani and V. Singh.
        Owl search algorithm: A novelnature-inspired heuristic paradigm for global optimization.
        Journal of Intelligent & Fuzzy Systems (2018).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> OSA.')

        # Overrides its parent class with the receiving params
        super(OSA, self).__init__()

        # Exploration intensity
        self.beta = 1.9

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    @property
    def beta(self):
        """float: Exploration intensity.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        if not isinstance(beta, (float, int)):
            raise e.TypeError('`beta` should be a float or integer')
        if beta < 0:
            raise e.ValueError('`beta` should be >= 0')

        self._beta = beta

    def update(self, space, iteration, n_iterations):
        """Wraps Owl Search Algorithm over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # Gathers best and worst agents (eq. 5 and 6)
        best = copy.deepcopy(space.agents[0])
        worst = copy.deepcopy(space.agents[-1])

        # Linearly decreases the `beta` coefficient
        beta = self.beta - ((iteration + 1) / n_iterations) * self.beta

        # Iterates through all agents
        for agent in space.agents:
            # Calculates the normalized intensity (eq. 4)
            intensity = (agent.fit - best.fit) / (worst.fit - best.fit + c.EPSILON)

            # Calculates the distance between owl and prey (eq. 7)
            distance = g.euclidean_distance(agent.position, best.position)

            # Obtains the change in intensity (eq. 8)
            noise = r.generate_uniform_random_number()
            intensity_change = intensity / (distance ** 2 + c.EPSILON) + noise

            # print(agent.fit, worst.fit, best.fit, intensity, intensity_change)

            # Generates the probability of vole movement and random `alpha`
            p_vm = r.generate_uniform_random_number()
            alpha = r.generate_uniform_random_number(high=0.5)

            # If probability of vole movement is smaller than 0.5
            if p_vm < 0.5:
                # Updates current's owl position (eq. 9 - top)
                agent.position += beta * intensity_change * \
                    np.fabs(alpha * best.position - agent.position)

            # If probability is bigger or equal to 0.5
            else:
                # Updates current's owl position (eq. 9 - bottom)
                agent.position -= beta * intensity_change * \
                    np.fabs(alpha * best.position - agent.position)
