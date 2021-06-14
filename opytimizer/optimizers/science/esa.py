"""Electro-Search Algorithm.
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


class ESA(Optimizer):
    """An ESA class, inherited from Optimizer.

    This is the designed class to define ES-related
    variables and methods.

    References:
        A. Tabari and A. Ahmad. A new optimization method: Electro-Search algorithm.
        Computers & Chemical Engineering (2017).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> ESA.')

        # Overrides its parent class with the receiving params
        super(ESA, self).__init__()

        #
        self.n_electrons = 5

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    def compile(self, space):
        """
        """

        self.electron = np.zeros((space.n_agents, self.n_electrons, space.n_variables, space.n_dimensions))

    def update(self, space, function):
        """
        """

        D = r.generate_uniform_random_number()

        for i, agent in enumerate(space.agents):
            a = copy.deepcopy(agent)
            electrons = [copy.deepcopy(agent) for _ in range(self.n_electrons)]

            for electron in electrons:
                #
                r1 = r.generate_uniform_random_number()
                n = r.generate_integer_random_number(2, 6)

                # (eq. 3)
                electron.position += (2 * r1 - 1) * (1 - 1 / n ** 2) / D

                #
                electron.clip_by_bound()

                #
                electron.fit = function(electron.position)

            electrons.sort(key=lambda x: x.fit)

            D = (electrons[0].position - space.best_agent.position) + 0.5 * (1 / space.best_agent.position ** 2 - 1 / a.position ** 2)
            a.position += 0.5 * D
            a.clip_by_bound()
            a.fit = function(a.position)

            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)
            