"""Magnetic Optimization Algorithm.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class MOA(Optimizer):
    """An MOA class, inherited from Optimizer.

    This is the designed class to define MOA-related
    variables and methods.

    References:
        M.-H. Tayarani and M.-R. Akbarzadeh. Magnetic-inspired optimization algorithms: Operators and structures.
        Swarm and Evolutionary Computation (2014).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> MOA.')

        # Overrides its parent class with the receiving params
        super(MOA, self).__init__()

        #
        self.alpha = 1

        #
        self.rho = 2

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    def create_additional_attrs(self, space):
        #
        if not np.sqrt(space.n_agents).is_integer():
            raise e.SizeError('`n_agents` should have a perfect square')

        self.velocity = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def update(self, space):
        #
        space.agents.sort(key=lambda x: x.fit)

        # (eq. 2)
        best, worst = space.agents[0], space.agents[-1]
        fitness = [(agent.fit - best.fit) / (worst.fit - best.fit + c.EPSILON) for agent in space.agents]

        print([agent.fit for agent in space.agents])

        # (eq. 3)
        mass = [self.alpha + self.rho * fit for fit in fitness]

        #
        for i, agent in enumerate(space.agents):
            # (eq. 4)
            root = np.sqrt(space.n_agents)
            north = int((i - root) % space.n_agents)
            south = int((i + root) % space.n_agents)
            west = int((i - 1) + ((i + root - 1) % root) // (root - 1) * root)
            east = int((i + 1) - (i % root) // (root - 1) * root)

            #
            neighbours = [north, south, west, east]

            #
            force = 0

            #
            for n in neighbours:
                # (eq. 7)
                distance = g.euclidean_distance(agent.position, space.agents[n].position)

                # (eq. 5)
                force += (space.agents[n].position - agent.position) * fitness[n] / (distance + c.EPSILON)

            # print(force, mass[i])

            # (eq. 11)
            r1 = r.generate_uniform_random_number(-10, 10, size=(space.n_variables, space.n_dimensions))
            acceleration = (force / mass[i]) * r1

            # (eq. 12 - top)
            self.velocity[i] += acceleration

            # print(force, mass, velocity)

            # (eq. 12 - bottom)
            agent.position += self.velocity[i]
