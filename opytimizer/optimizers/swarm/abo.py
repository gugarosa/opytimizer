"""Artificial Butterfly Optimization.
"""

import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class ABO(Optimizer):
    """An ABO class, inherited from Optimizer.

    This is the designed class to define ABO-related
    variables and methods.

    References:
        X. Qi, Y. Zhu and H. Zhang. A new meta-heuristic butterfly-inspired algorithm.
        Journal of Computational Science (2017).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> ABO.")

        super(ABO, self).__init__()

        self.sunspot_ratio = 0.9
        self.a = 2.0

        self.build(params)

        logger.info("Class overrided.")

    @property
    def sunspot_ratio(self) -> float:
        """Ratio of sunspot butterflies."""

        return self._sunspot_ratio

    @sunspot_ratio.setter
    def sunspot_ratio(self, sunspot_ratio: float) -> None:
        if not isinstance(sunspot_ratio, (float, int)):
            raise e.TypeError("`sunspot_ratio` should be a float or integer")
        if sunspot_ratio < 0 or sunspot_ratio > 1:
            raise e.ValueError("`sunspot_ratio` should be between 0 and 1")

        self._sunspot_ratio = sunspot_ratio

    @property
    def a(self) -> float:
        """Free flight constant."""

        return self._a

    @a.setter
    def a(self, a: float) -> None:
        if not isinstance(a, (float, int)):
            raise e.TypeError("`a` should be a float or integer")
        if a < 0:
            raise e.ValueError("`a` should be >= 0")

        self._a = a

    def _flight_mode(
        self, agent: Agent, neighbour: Agent, function: Function
    ) -> Tuple[Agent, bool]:
        """Flies to a new location according to the flight mode (eq. 1).

        Args:
            agent: Current agent.
            neighbour: Selected neigbour.
            function: A Function object that will be used as the objective function.

        Returns:
            (Tuple[Agent, bool]): Current agent or an agent with updated position, along with a boolean that indicates whether
            agent is better or not than current one.

        """

        j = r.generate_integer_random_number(0, agent.n_variables)
        r1 = r.generate_uniform_random_number(-1, 1)

        temp = copy.deepcopy(agent)

        # Updates temporary agent's position (eq. 1)
        temp.position[j] = (
            agent.position[j] + (agent.position[j] - neighbour.position[j]) * r1
        )
        temp.clip_by_bound()

        temp.fit = function(temp.position)
        if temp.fit < agent.fit:
            return temp.position, temp.fit, True

        return agent.position, agent.fit, False

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Artificial Butterfly Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        space.agents.sort(key=lambda x: x.fit)

        n_sunspots = int(self.sunspot_ratio * len(space.agents))
        for agent in space.agents[:n_sunspots]:
            k = r.generate_integer_random_number(0, len(space.agents))

            # Performs a flight mode using sunspot butterflies (eq. 1)
            agent.position, agent.fit, _ = self._flight_mode(
                agent, space.agents[k], function
            )

        for agent in space.agents[n_sunspots:]:
            k = r.generate_integer_random_number(0, len(space.agents) - n_sunspots)

            # Performs a flight mode using canopy butterflies (eq. 1)
            agent.position, agent.fit, is_better = self._flight_mode(
                agent, space.agents[k], function
            )

            if not is_better:
                k = r.generate_integer_random_number(0, len(space.agents))
                r1 = r.generate_uniform_random_number()

                # Calculates `D` (eq. 4)
                D = np.fabs(2 * r1 * space.agents[k].position - agent.position)

                r2 = r.generate_uniform_random_number()

                # Updates the agent's position (eq. 3)
                a = self.a - self.a * (iteration / n_iterations)
                agent.position = space.agents[k].position - 2 * a * r2 - a * D
                agent.clip_by_bound()

                agent.fit = function(agent.position)
