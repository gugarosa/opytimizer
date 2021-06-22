"""Algorithm of the Innovative Gunner.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core import Optimizer

logger = l.get_logger(__name__)


class AIG(Optimizer):
    """An AIG class, inherited from Optimizer.

    This is the designed class to define AIG-related
    variables and methods.

    References:
        P. Pijarski and P. Kacejko.
        A new metaheuristic optimization method: the algorithm of the innovative gunner (AIG).
        Engineering Optimization (2019).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> AIG.')

        # Overrides its parent class with the receiving params
        super(AIG, self).__init__()

        # First correction angle
        self.alpha = np.pi

        # Second correction angle
        self.beta = np.pi

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    def update(self, space, function):
        # Calculating the maximum correction angles (eq. 18)
        a = r.generate_uniform_random_number()
        alpha_max = self.alpha * a
        beta_max = self.beta * a

        # Iterates through all agents
        for agent in space.agents:
            # Makes a deep copy of current agent
            a = copy.deepcopy(agent)

            # Sampling correction angles
            alpha = r.generate_gaussian_random_number(0, alpha_max/3, (agent.n_variables, agent.n_dimensions))
            beta = r.generate_gaussian_random_number(0, beta_max/3, (agent.n_variables, agent.n_dimensions))

            # Calculating correction functions (eq. 16 and 17)
            g_alpha = np.where(alpha < 0, np.cos(alpha), 1 / np.cos(alpha))
            g_beta = np.where(beta < 0, np.cos(beta), 1 / np.cos(beta))

            # Updating temporary agent's position
            a.position *= g_alpha * g_beta

            # Checks agent's limits
            a.clip_by_bound()

            # Re-evaluates the temporary agent
            a.fit = function(a.position)

            # If temporary agent's fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Replace its position and fitness
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)
