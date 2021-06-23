"""Water Evaporation Optimization.
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


class WEO(Optimizer):
    """A WEO class, inherited from Optimizer.

    This is the designed class to define WEO-related
    variables and methods.

    References:
        A. Kaveh and T. Bakhshpoori.
        Water Evaporation Optimization: A novel physically inspired optimization algorithm.
        Computers & Structures (2016).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> WEO.')

        # Overrides its parent class with the receiving params
        super(WEO, self).__init__()

        # Minimum substrate energy
        self.E_min = -3.5

        # Maximum substrate energy
        self.E_max = -0.5

        # Minimum contact angle
        self.theta_min = -np.pi / 3.6

        # Maximum contact angle
        self.theta_max = -np.pi / 9

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    def _evaporation_flux(self, theta):
        """
        """

        # (eq. 7)
        J = (1 / 2.6) * ((2 / 3 + np.cos(theta) ** 3 / 3 - np.cos(theta)) ** (-2 / 3)) * (1 - np.cos(theta))

        return J

    def update(self, space, iteration, n_iterations):

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # Gathers best and worst agents
        best, worst = space.agents[0], space.agents[-1]

        # Iterates through all agents
        for agent in space.agents:
            # Checks whether it is the first half of iterations
            if int(iteration <= n_iterations / 2):
                # (eq. 5)
                E_sub = ((self.E_max - self.E_min) * (agent.fit - best.fit)) / (worst.fit - best.fit) + self.E_min

                # (eq. 6)
                r1 = r.generate_uniform_random_number(size=(agent.n_variables, agent.n_dimensions))
                MEP = np.where(r1 < np.exp(E_sub), 1, 0)

                # (eq. 10)
                r2 = r.generate_uniform_random_number()
                i = r.generate_integer_random_number(0, space.n_agents)
                j = r.generate_integer_random_number(0, space.n_agents, i)
                S = r2 * (space.agents[i].position - space.agents[j].position)

                # (eq. 11)
                agent.position += S * MEP

            # If it is the second half of iterations
            else:
                pass
                # (eq. 8)
                theta = ((self.theta_max - self.theta_min) * (agent.fit - best.fit)) / (worst.fit - best.fit) + self.theta_min

                # (eq. 9)
                r1 = r.generate_uniform_random_number(size=(agent.n_variables, agent.n_dimensions))
                DEP = np.where(r1 < self._evaporation_flux(theta), 1, 0)

                # (eq. 10)
                r2 = r.generate_uniform_random_number()
                i = r.generate_integer_random_number(0, space.n_agents)
                j = r.generate_integer_random_number(0, space.n_agents, i)
                S = r2 * (space.agents[i].position - space.agents[j].position)

                # (eq. 11)
                agent.position += S * DEP
