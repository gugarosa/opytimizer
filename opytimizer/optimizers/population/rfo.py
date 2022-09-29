"""Red Fox Optimization.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class RFO(Optimizer):
    """A RFO class, inherited from Optimizer.

    This is the designed class to define RFO-related
    variables and methods.

    References:
        D. Polap and M. Woźniak. Red fox optimization algorithm.
        Expert Systems with Applications (2021).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> RFO.")

        super(RFO, self).__init__()

        self.phi = r.generate_uniform_random_number(0, 2 * np.pi)[0]
        self.theta = r.generate_uniform_random_number()[0]
        self.p_replacement = 0.05

        self.build(params)

        logger.info("Class overrided.")

    @property
    def phi(self) -> float:
        """Observation angle."""

        return self._phi

    @phi.setter
    def phi(self, phi: float) -> None:
        if not isinstance(phi, (float, int)):
            raise e.TypeError("`phi` should be a float or integer")
        if phi < 0 or phi > 2 * np.pi:
            raise e.ValueError("`phi` should be between 0 and 2PI")

        self._phi = phi

    @property
    def theta(self) -> float:
        """Weather condition."""

        return self._theta

    @theta.setter
    def theta(self, theta: float) -> None:
        if not isinstance(theta, (float, int)):
            raise e.TypeError("`theta` should be a float or integer")
        if theta < 0 or theta > 1:
            raise e.ValueError("`theta` should be between 0 and 1")

        self._theta = theta

    @property
    def p_replacement(self) -> float:
        """Percentual of foxes replacement."""

        return self._p_replacement

    @p_replacement.setter
    def p_replacement(self, p_replacement: float) -> None:
        if not isinstance(p_replacement, (float, int)):
            raise e.TypeError("`p_replacement` should be a float or integer")
        if p_replacement < 0 or p_replacement > 1:
            raise e.ValueError("`p_replacement` should be between 0 and 1")

        self._p_replacement = p_replacement

    @property
    def n_replacement(self) -> int:
        """Number of foxes to be replaced."""

        return self._n_replacement

    @n_replacement.setter
    def n_replacement(self, n_replacement: int) -> None:
        if not isinstance(n_replacement, int):
            raise e.TypeError("`n_replacement` should be an integer")
        if n_replacement < 0:
            raise e.ValueError("`n_replacement` should be >= 0")

        self._n_replacement = n_replacement

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.n_replacement = int(self.p_replacement * space.n_agents)

    def _rellocation(self, agent: Agent, best_agent: Agent, function: Function) -> None:
        """Performs the fox rellocation procedure.

        Args:
            agent: Current agent.
            best_agent: Best agent.
            function: A Function object that will be used as the objective function.

        """

        temp = copy.deepcopy(agent)

        # Calculates the square root of euclidean distance between agent and best agent (eq. 1)
        distance = np.sqrt(g.euclidean_distance(temp.position, best_agent.position))

        # Calculates individual reallocation (eq. 2)
        alpha = r.generate_uniform_random_number(0, distance)
        temp.position += alpha * np.sign(best_agent.position - temp.position)
        temp.clip_by_bound()

        temp.fit = function(temp.position)
        if temp.fit < agent.fit:
            agent.position = copy.deepcopy(temp.position)
            agent.fit = copy.deepcopy(temp.fit)

    def _noticing(self, agent: Agent, function: Function, alpha: float) -> None:
        """Performs the fox noticing procedure.

        Args:
            agent: Current agent.
            function: A Function object that will be used as the objective function.
            alpha: Scaling parameter.

        """

        mu = r.generate_uniform_random_number()
        if mu > 0.75:
            if self.phi != 0:
                # Calculates fox observation radius (eq. 4 - top)
                radius = alpha * np.sin(self.phi) / self.phi
            else:
                # Calculates fox observation radius (eq. 4 - bottom)
                radius = self.theta

            phi = r.generate_uniform_random_number(0, 2 * np.pi, agent.n_variables)

            for j in range(agent.n_variables):
                total_sum = 0

                for k in range(j):
                    total_sum += np.sin(phi[k])

                # Updates the corresponding position (eq. 5)
                agent.position[j] += alpha * radius * (total_sum + np.cos(phi[j]))
            agent.clip_by_bound()

            agent.fit = function(agent.position)

    def update(self, space: Space, function: Function) -> None:
        """Wraps Red Fox Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        alpha = r.generate_uniform_random_number(0, 0.2)

        for agent in space.agents:
            self._rellocation(agent, space.best_agent, function)
            self._noticing(agent, function, alpha)

        space.agents.sort(key=lambda x: x.fit)

        # Calculates the habitat's center and diameter (eq. 6 and 7)
        habitat_center = (space.agents[0].position + space.agents[1].position) / 2
        habitat_diameter = np.sqrt(
            g.euclidean_distance(space.agents[0].position, space.agents[1].position)
        )

        k = r.generate_uniform_random_number()

        for agent in space.agents[-self.n_replacement :]:
            # If sampled number is bigger than 0.45 (eq. 8 - top)
            if k >= 0.45:
                agent.fill_with_uniform()
                agent.position += habitat_center + habitat_diameter / 2

            # If sampled number is smaller than 0.45 (eq. 8 - bottom)
            else:
                # Reproduces parents into a new position (eq. 9)
                agent.position = (
                    k * (space.agents[0].position + space.agents[1].position) / 2
                )

            agent.clip_by_bound()
