"""Artificial Flora.
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


class AF(Optimizer):
    """An AF class, inherited from Optimizer.

    This is the designed class to define AF-related
    variables and methods.

    References:
        L. Cheng, W. Xue-han and Y. Wang. Artificial flora (AF) optimization algorithm.
        Applied Sciences (2018).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> AF.')

        # Overrides its parent class with the receiving params
        super(AF, self).__init__()

        # First learning coefficient
        self.c1 = 0.75

        # Second learning coefficient
        self.c2 = 1.25

        # Amount of branches
        self.m = 10

        # Selective probability
        self.Q = 0.75

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    def compile(self, space):
        self.p_distance = r.generate_uniform_random_number(size=space.n_agents)
        self.g_distance = r.generate_uniform_random_number(size=space.n_agents)

    def update(self, space, function):

        #
        space.agents.sort(key=lambda x: x.fit)

        #
        new_agents = []

        for i, agent in enumerate(space.agents):
            #
            for _ in range(self.m):
                #
                a = copy.deepcopy(agent)

                #
                r1 = r.generate_uniform_random_number()
                r2 = r.generate_uniform_random_number()

                #
                distance = self.g_distance[i] * r1 * \
                    self.c1 + self.p_distance[i] * r2 * self.c2

                # print(distance)

                #
                D = r.generate_gaussian_random_number(variance=distance)

                a.position += D
                a.clip_by_bound()

                #
                a.fit = function(a.position)

                # print(a.fit)

                # print(D)

                #
                p = np.fabs(np.sqrt(a.fit / space.agents[-1].fit)) * self.Q

                #
                r3 = r.generate_uniform_random_number()

                if r3 < p:
                    new_agents.append(a)

            #
            self.g_distance[i] = self.p_distance[i]
            self.p_distance[i] = np.sqrt(
                np.sum((agent.position - a.position) ** 2) / agent.n_variables)

            print(self.g_distance[i], self.p_distance[i])

        space.agents += new_agents
        space.agents.sort(key=lambda x: x.fit)
        space.agents = space.agents[:space.n_agents]
