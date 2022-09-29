"""Atom Search Optimization.
"""

from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class ASO(Optimizer):
    """An ASO class, inherited from Optimizer.

    This is the designed class to define ASO-related
    variables and methods.

    References:
        W. Zhao, L. Wang and Z. Zhang.
        A novel atom search optimization for dispersion coefficient estimation in groundwater.
        Future Generation Computer Systems (2019).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> ASO.")

        super(ASO, self).__init__()

        self.alpha = 50.0
        self.beta = 0.2

        self.build(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """Depth weight."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")

        self._alpha = alpha

    @property
    def beta(self) -> float:
        """Multiplier weight."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        if beta < 0 or beta > 1:
            raise e.ValueError("`beta` should be between 0 and 1")

        self._beta = beta

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

        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def _calculate_mass(self, agents: List[Agent]) -> List[float]:
        """Calculates the atoms' masses (eq. 17 and 18).

        Args:
            agents: List of agents.

        Returns:
            (List[float]): A list holding the atoms' masses.

        """

        agents.sort(key=lambda x: x.fit)

        worst = agents[-1].fit
        best = agents[0].fit

        total_fit = np.sum(
            [
                np.exp(-(agent.fit - best) / (worst - best + c.EPSILON))
                for agent in agents
            ]
        )

        mass = [
            np.exp(-(agent.fit - best) / (worst - best + c.EPSILON)) / total_fit
            for agent in agents
        ]

        return mass

    def _calculate_potential(
        self,
        agent: Agent,
        K_agent: Agent,
        average: np.ndarray,
        iteration: int,
        n_iterations: int,
    ) -> None:
        """Calculates the potential of an agent based on its neighbour and average positioning.

        Args:
            agent: Agent to have its potential calculated.
            K_agent: Neighbour agent.
            average: Array of average positions.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        distance = np.linalg.norm(agent.position - average)
        radius = np.linalg.norm(agent.position - K_agent.position)

        rsmin = 1.1 + 0.1 * np.sin((iteration + 1) / n_iterations * np.pi / 2)
        rsmax = 1.24

        if radius / (distance + c.EPSILON) < rsmin:
            rs = rsmin
        else:
            if radius / (distance + c.EPSILON) > rsmax:
                rs = rsmax
            else:
                rs = radius / (distance + c.EPSILON)

        r1 = r.generate_uniform_random_number()

        coef = (1 - iteration / n_iterations) ** 3
        potential = (
            coef
            * (12 * (-rs) ** (-13) - 6 * (-rs) ** (-7))
            * r1
            * ((K_agent.position - agent.position) / (radius + c.EPSILON))
        )

        return potential

    def _calculate_acceleration(
        self,
        agents: List[Agent],
        best_agent: Agent,
        mass: np.ndarray,
        iteration: int,
        n_iterations: int,
    ) -> np.ndarray:
        """Calculates the atoms' acceleration.

        Args:
            agents: List of agents.
            best_agent: Global best agent.
            mass: Array of masses.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        Returns:
            (np.ndarray): An array holding the atoms' acceleration.

        """

        acceleration = np.zeros(
            (len(agents), best_agent.n_variables, best_agent.n_dimensions)
        )

        G = np.exp(-20.0 * iteration / n_iterations)

        K = int(len(agents) - (len(agents) - 2) * np.sqrt(iteration / n_iterations))
        K_agents, _ = map(
            list, zip(*sorted(zip(agents, mass), key=lambda x: x[1], reverse=True)[:K])
        )

        average = np.mean([agent.position for agent in K_agents])

        for i, agent in enumerate(agents):
            total_potential = np.zeros((agent.n_variables, agent.n_dimensions))

            for K_agent in K_agents:
                total_potential += self._calculate_potential(
                    agent, K_agent, average, iteration, n_iterations
                )

            # Finally, calculates the acceleration (eq. 16)
            acceleration[i] = (
                G * self.alpha * total_potential
                + self.beta * (best_agent.position - agent.position) / mass[i]
            )

        return acceleration

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Atom Search Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Calculates the masses (eq. 17 and 18)
        mass = self._calculate_mass(space.agents)

        # Calculates the acceleration (eq. 16)
        acceleration = self._calculate_acceleration(
            space.agents, space.best_agent, mass, iteration, n_iterations
        )

        for i, agent in enumerate(space.agents):
            # Updates current agent's velocity (eq. 21)
            r1 = r.generate_uniform_random_number()
            self.velocity[i] = r1 * self.velocity[i] + acceleration[i]

            # Updates current agent's position (eq. 22)
            agent.position += self.velocity[i]
