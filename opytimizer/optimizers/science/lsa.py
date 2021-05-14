"""Lightning Search Algorithm.
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


class LSA(Optimizer):
    """An LSA class, inherited from Optimizer.

    This is the designed class to define LSA-related
    variables and methods.

    References:
        H. Shareef, A. Ibrahim and A. Mutlag. Lightning search algorithm.
        Applied Soft Computing (2015).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> LSA.')

        # Overrides its parent class with the receiving params
        super(LSA, self).__init__()

        #
        self.max_time = 10

        #
        self.init_energy = 2.05

        #
        self.p_fork = 0.01

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    def create_additional_attrs(self, space):
        self.time = 0
        self.direction = np.sign(r.generate_uniform_random_number(-1, 1, space.n_variables))

        # print(self.direction)



    def update(self, space, function, iteration, n_iterations):
        """Wraps Lightning Search Algorithm over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            iteration (int): Current iteration.

        """

        #
        self.time += 1

        if self.time >= self.max_time:
            space.agents.sort(key=lambda x: x.fit)
            space.agents[-1] = copy.deepcopy(space.agents[0])
            self.time = 0

        space.agents.sort(key=lambda x: x.fit)

        best = space.agents[0].fit

        energy = self.init_energy - 2 * np.exp(-5 * (n_iterations - iteration) / n_iterations)

        for j in range(space.n_variables):
            direction = space.agents[0].position[j] + self.direction[j] * 0.005 * (space.ub[j] - space.lb[j])
            direction_fit = function(direction)
            if direction_fit > best:
                self.direction[j] *= -1

        for agent in space.agents:
            temp = copy.deepcopy(agent)
            distance = agent.position - space.agents[0].position
            for j in range(agent.n_variables):
                if agent.position[j] == space.agents[0].position[j]:
                    temp.position[j] += self.direction[j] * np.fabs(r.generate_gaussian_random_number(0, energy))
                else:
                    if distance[j] < 0:
                        temp.position[j] += np.fabs(distance[j])
                    else:
                        temp.position[j] -= distance[j]

            temp.clip_by_bound()

            temp.fit = function(temp.position)

            if temp.fit < agent.fit:
                agent.position = copy.deepcopy(temp.position)
                agent.fit = copy.deepcopy(temp.fit)

                r1 = r.generate_uniform_random_number()

                if r1 < self.p_fork:
                    temp = copy.deepcopy(agent)
                    for j in range(agent.n_variables):
                        temp.position[j] = temp.ub[j] + temp.lb[j] - temp.position[j]
                    temp.fit = function(temp.position)
        
                    if temp.fit < agent.fit:
                        agent.position = copy.deepcopy(temp.position)
                        agent.fit = copy.deepcopy(temp.fit)