"""Sooty Tern Optimization Algorithm.
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


class STOA(Optimizer):
    """An STOA class, inherited from Optimizer.

    This is the designed class to define STOA-related
    variables and methods.

    References:
        G. Dhiman and A. Kaur. STOA: A bio-inspired based optimization algorithm for industrial engineering problems.
        Engineering Applications of Artificial Intelligence (2019).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> STOA.')

        # Overrides its parent class with the receiving params
        super(STOA, self).__init__()

        # Controlling variable
        self.Cf = 2

        #
        self.u = 1

        #
        self.v = 1

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    def update(self, space, iteration, n_iterations):
        """
        """

        # (eq. 2)
        Sa = self.Cf - (iteration * (self.Cf / n_iterations))

        # (eq. 4)
        Cb = 0.5 * r.generate_uniform_random_number()

        #
        for agent in space.agents:
            # (eq. 1)
            C = Sa * agent.position

            # (eq. 3)
            M = Cb * (space.best_agent.position - agent.position)

            # (eq. 5)
            D = C + M

            # (eq. 9)
            k = r.generate_uniform_random_number(0, 2*np.pi)
            R = self.u * np.exp(k * self.v)

            # (eq. 6, 7 and 8)
            i = r.generate_uniform_random_number(0, k)
            x = R * np.sin(i)
            y = R * np.cos(i)
            z = R * i

            # (eq. 10)
            agent.position = (D * (x + y + z)) * space.best_agent.position
