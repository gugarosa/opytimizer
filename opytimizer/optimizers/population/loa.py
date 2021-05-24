"""Lion Optimization Algorithm.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class LOA(Optimizer):
    """An LOA class, inherited from Optimizer.

    This is the designed class to define LOA-related
    variables and methods.

    References:
        M. Yazdani and F. Jolai. Lion Optimization Algorithm (LOA): A nature-inspired metaheuristic algorithm.
        Journal of Computational Design and Engineering (2016).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> LOA.')

        # Overrides its parent class with the receiving params
        super(LOA, self).__init__()

        #
        self.N = 0.5

        #
        self.P = 2

        #
        self.S = 0.75

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    def compile(self, space):
        self.n_nomad = int(self.N * space.n_agents)
        self.n_pride = (space.n_agents - self.n_nomad) // self.P

        self.gender = [1] * space.n_agents

    def _get_nomad_lions(self, agents):
        return agents[:self.n_nomad], self.gender[:self.n_nomad]

    def _get_pride_lions(self, agents):
        prides, genders = [], []
        for i in range(self.P):
            start, end = i * self.n_pride, (i + 1) * self.n_pride
            prides.append(agents[start:end])
            genders.append(self.gender[start:end])
        return prides, genders

    def _hunting(self, prides, genders):
        for pride, gender in zip(prides, genders):
            for p, g in zip(pride, gender):
                if g == 0:
                    pass


    def update(self, space):
        p, g = self._get_pride_lions(space.agents)

        self._hunting(p, g)

