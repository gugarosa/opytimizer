"""Passing Vehicle Search.
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


class PVS(Optimizer):
    """A PVS class, inherited from Optimizer.

    This is the designed class to define PVS-related
    variables and methods.

    References:
        P. Savsani and V. Savsani. Passing vehicle search (PVS): A novel metaheuristic algorithm.
        Applied Mathematical Modelling (2016).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> PVS.')

        # Overrides its parent class with the receiving params
        super(PVS, self).__init__()

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    def update(self, space, function):
        #
        space.agents.sort(key=lambda x: x.fit)

        #
        for i, agent in enumerate(space.agents):
            a = copy.deepcopy(agent)

            #
            R = [0, 0]

            #
            while R[0] == R[1]:
                #
                R = r.generate_integer_random_number(0, space.n_agents, i, 2)

            #
            D1 = 1 / space.n_agents * agent.fit
            D2 = 1 / space.n_agents * space.agents[R[0]].fit
            D3 = 1 / space.n_agents * space.agents[R[1]].fit

            #
            V1 = r.generate_uniform_random_number() * (1 - D1)
            V2 = r.generate_uniform_random_number() * (1 - D2)
            V3 = r.generate_uniform_random_number() * (1 - D3)

            #
            x = np.fabs(D3 - D1)
            y = np.fabs(D3 - D2)
            x1 = (V3 * x) / (V1 - V3)
            y1 = (V2 * x) / (V1 - V3)

            r1 = r.generate_uniform_random_number()

            #
            if V3 < V1:
                #
                if (y - y1) > x1:
                    Vco = V1 / (V1 - V3)
                    a.position += Vco * r1 * (a.position - space.agents[R[1]].position)
                else:
                    a.position += r1 * (a.position - space.agents[R[0]].position)
            else:
                a.position += r1 * (space.agents[R[1]].position - a.position)
            
            a.clip_by_bound()
            a.fit = function(a.position)

            if a.fit < agent.fit:
                agent.fit = copy.deepcopy(a.fit)
                agent.position = copy.deepcopy(a.position)