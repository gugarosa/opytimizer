"""Gravitational Search Algorithm."""

from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


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

        super(GSA, self).__init__()

        self.G = 2.467

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

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

        best, worst = agents[0].fit, agents[-1].fit

        # Calculates agents' masses (eq. 15)
        mass = [(agent.fit - worst) / (best - worst + c.EPSILON) for agent in agents]
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

        force = [
            [
                gravity
                * (mass[i] * mass[j])
                / (np.linalg.norm(agents[i].position - agents[j].position) + c.EPSILON)
                * (agents[j].position - agents[i].position)
                for j in range(len(agents))
            ]
            for i in range(len(agents))
        ]

        force = np.asarray(force)
        force = np.sum(np.random.uniform(0.0, 1.0, 1) * force, axis=1)

        return force

    def update(self, space: Space, iteration: int) -> None:
        """Wraps Gravitational Search Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.

        """

        space.agents.sort(key=lambda x: x.fit)

        gravity = self.G / (iteration + 1)
        mass = self._calculate_mass(space.agents)
        force = self._calculate_force(space.agents, mass, gravity)

        for i, agent in enumerate(space.agents):
            # Calculates the acceleration (eq. 10)
            acceleration = force[i] / (mass[i] + c.EPSILON)

            # Updates current agent velocity (eq. 11)
            r1 = np.random.uniform(0.0, 1.0, 1)
            self.velocity[i] = r1 * self.velocity[i] + acceleration

            # Updates current agent position (eq. 12)
            agent.position += self.velocity[i]
