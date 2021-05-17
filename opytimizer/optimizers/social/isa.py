"""Interactive Search Algorithm.
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


class ISA(Optimizer):
    """An ISA class, inherited from Optimizer.

    This is the designed class to define ISA-related
    variables and methods.

    References:
        A. Mortazavi, V. Toğan and A. Nuhoğlu.
        Interactive search algorithm: A new hybrid metaheuristic optimization algorithm.
        Engineering Applications of Artificial Intelligence (2018).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> ISA.')

        # Overrides its parent class with the receiving params
        super(ISA, self).__init__()

        self.w = 0.7

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    def create_additional_attrs(self, space):
        """Creates additional attributes that are used by this optimizer.

        Args:
            space (Space): A Space object containing meta-information.

        """

        # Arrays of local positions, velocities and masses
        self.local_position = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))
        self.velocity = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def update(self, space, function):
        #
        space.agents.sort(key=lambda x: x.fit)

        #
        best, worst = space.agents[0], space.agents[-1]

        # (eq. 1.2)
        coef = [(best.fit - agent.fit) / (best.fit - worst.fit + c.EPSILON) for agent in space.agents]

        # (eq. 1.1 - bottom)
        w_coef = [c / np.sum(coef) for c in coef]

        # (eq. 1.1 - top)
        w_position = np.sum([c * agent.position for c, agent in zip(w_coef, space.agents)], axis=0)
        w_fit = function(w_position)

        for i, agent in enumerate(space.agents):
            tendency = r.generate_uniform_random_number()
            idx = r.generate_integer_random_number(high=space.n_agents, exclude_value=i)

            if tendency >= 0.3:
                phi3 = r.generate_uniform_random_number()
                phi2 = 2 * r.generate_uniform_random_number()
                phi1 = -(phi2 + phi3) * r.generate_uniform_random_number()
                self.velocity[i] = self.w * self.velocity[i] + (phi1 * (self.local_position[idx] - agent.position) + phi2 * (space.best_agent.position - self.local_position[idx]) + phi3 * (w_position - self.local_position[idx]))
            else:
                r1 = r.generate_uniform_random_number()

                if agent.fit < space.agents[idx].fit:
                    self.velocity[i] = r1 * (agent.position - space.agents[idx].position)
                else:
                    self.velocity[i] = r1 * (space.agents[idx].position - agent.position)

            agent.position += self.velocity[i]

            agent.fit = function(agent.position)
            local_fit = function(self.local_position[i])

            if w_fit < local_fit:
                self.local_position[i] = w_position

            if agent.fit < w_fit:
                self.local_position[i] = agent.position



