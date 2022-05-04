"""Magnetic Optimization Algorithm.
"""

from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class MOA(Optimizer):
    """An MOA class, inherited from Optimizer.

    This is the designed class to define MOA-related
    variables and methods.

    References:
        M.-H. Tayarani and M.-R. Akbarzadeh. Magnetic-inspired optimization algorithms: Operators and structures.
        Swarm and Evolutionary Computation (2014).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> MOA.")

        # Overrides its parent class with the receiving params
        super(MOA, self).__init__()

        # Particle moviment first constant
        self.alpha = 1.0

        # Particle moviment second constant
        self.rho = 2.0

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """Particle moviment first constant."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        if alpha < 0:
            raise e.ValueError("`alpha` should be >= 0")

        self._alpha = alpha

    @property
    def rho(self) -> float:
        """Particle moviment second constant."""

        return self._rho

    @rho.setter
    def rho(self, rho: float) -> None:
        if not isinstance(rho, (float, int)):
            raise e.TypeError("`rho` should be a float or integer")
        if rho < 0:
            raise e.ValueError("`rho` should be >= 0")

        self._rho = rho

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Checks if supplied number of agents has a perfect square
        if not np.sqrt(space.n_agents).is_integer():
            raise e.SizeError("`n_agents` should have a perfect square")

    def update(self, space: Space) -> None:
        """Wraps Magnetic Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # Gathers the best and worst agents and calculates a list of normalized fitness (eq. 2)
        best, worst = space.agents[0], space.agents[-1]
        fitness = [
            (agent.fit - best.fit) / (worst.fit - best.fit + c.EPSILON)
            for agent in space.agents
        ]

        # Calculates the masses (eq. 3)
        mass = [self.alpha + self.rho * fit for fit in fitness]

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Gathers the agents neighbours (eq. 4)
            root = np.sqrt(space.n_agents)
            north = int((i - root) % space.n_agents)
            south = int((i + root) % space.n_agents)
            west = int((i - 1) + ((i + root - 1) % root) // (root - 1) * root)
            east = int((i + 1) - (i % root) // (root - 1) * root)
            neighbours = [north, south, west, east]

            # Initializes the force as a zero value
            force = 0

            # Iterates through all neighbours
            for n in neighbours:
                # Calculates the distance between current agent and neighbour (eq. 7)
                distance = g.euclidean_distance(
                    agent.position, space.agents[n].position
                )

                # Calculates the force between agents (eq. 5)
                force += (
                    (space.agents[n].position - agent.position)
                    * fitness[n]
                    / (distance + c.EPSILON)
                )

            # Calculates the force's mean
            # This increases the performance of algorithm by eliminating addition biases
            force = np.mean(force)

            # Updates the agent's velocity(eq. 9)
            r1 = r.generate_uniform_random_number()
            velocity = force / mass[i] * r1

            # Updates the agent's position (eq. 10)
            agent.position += velocity

            # Clips the agent's limits
            agent.clip_by_bound()
