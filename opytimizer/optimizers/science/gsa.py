"""Gravitational Search Algorithm.
"""

from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class GSA(Optimizer):
    """A GSA class, inherited from Optimizer.

    This is the designed class to define GSA-related
    variables and methods.

    References:
        E. Rashedi, H. Nezamabadi-Pour and S. Saryazdi. GSA: a gravitational search algorithm.
        Information Sciences (2009).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> GSA.")

        # Overrides its parent class with the receiving params
        super(GSA, self).__init__()

        # Initial gravity value
        self.G = 2.467

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def G(self) -> float:
        """Initial gravity."""

        return self._G

    @G.setter
    def G(self, G: float) -> None:
        if not isinstance(G, (float, int)):
            raise e.TypeError("`G` should be a float or integer")
        if G < 0:
            raise e.ValueError("`G` should be >= 0")

        self._G = G

    @property
    def velocity(self) -> np.ndarray:
        """Array of velocities."""

        return self._velocity

    @velocity.setter
    def velocity(self, velocity: np.ndarray) -> None:
        if not isinstance(velocity, np.ndarray):
            raise e.TypeError("`velocity` should be a numpy array")

        self._velocity = velocity

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Arrays of velocities
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def _calculate_mass(self, agents: List[Agent]) -> float:
        """Calculates agents' mass (eq. 16).

        Args:
            agents: List of agents.

        Returns:
            (float): The agents' mass.

        """

        # Gathers the best and worst agents
        best, worst = agents[0].fit, agents[-1].fit

        # Calculates agents' masses using equation 15
        mass = [(agent.fit - worst) / (best - worst + c.EPSILON) for agent in agents]

        # Normalizes agents' masses
        norm_mass = mass / (np.sum(mass) + c.EPSILON)

        return norm_mass

    def _calculate_force(
        self, agents: List[Agent], mass: np.ndarray, gravity: float
    ) -> float:
        """Calculates agents' force (eq. 7-9).

        Args:
            agents: List of agents.
            mass: An array of agents' mass.
            gravity: Current gravity value.

        Returns:
            (float): The attraction force between all agents.

        """

        # Calculates the force
        force = [
            [
                gravity
                * (mass[i] * mass[j])
                / (
                    g.euclidean_distance(agents[i].position, agents[j].position)
                    + c.EPSILON
                )
                * (agents[j].position - agents[i].position)
                for j in range(len(agents))
            ]
            for i in range(len(agents))
        ]

        # Transforms the force into an array
        force = np.asarray(force)

        # Applies a stochastic trait to the force
        force = np.sum(r.generate_uniform_random_number() * force, axis=1)

        return force

    def update(self, space: Space, iteration: int) -> None:
        """Wraps Gravitational Search Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.

        """

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # Calculates the current gravity
        gravity = self.G / (iteration + 1)

        # Calculates agents' mass
        mass = self._calculate_mass(space.agents)

        # Calculates agents' attraction force
        force = self._calculate_force(space.agents, mass, gravity)

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Calculates the acceleration (eq. 10)
            acceleration = force[i] / (mass[i] + c.EPSILON)

            # Updates current agent velocity (eq. 11)
            r1 = r.generate_uniform_random_number()
            self.velocity[i] = r1 * self.velocity[i] + acceleration

            # Updates current agent position (eq. 12)
            agent.position += self.velocity[i]
