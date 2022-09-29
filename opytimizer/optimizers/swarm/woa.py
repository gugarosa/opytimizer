"""Whale Optimization Algorithm.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class WOA(Optimizer):
    """A WOA class, inherited from Optimizer.

    This is the designed class to define WOA-related
    variables and methods.

    References:
        S. Mirjalli and A. Lewis. The Whale Optimization Algorithm.
        Advances in Engineering Software (2016).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(WOA, self).__init__()

        self.b = 1

        self.build(params)

        logger.info("Class overrided.")

    @property
    def b(self) -> float:
        """Logarithmic spiral."""

        return self._b

    @b.setter
    def b(self, b: float) -> None:
        if not isinstance(b, (float, int)):
            raise e.TypeError("`b` should be a float or integer")

        self._b = b

    def _generate_random_agent(self, agent: Agent) -> Agent:
        """Generates a new random-based agent.

        Args:
            agent: Agent to be copied.

        Returns:
            (Agent): Random-based agent.

        """

        a = copy.deepcopy(agent)
        a.fill_with_uniform()

        return a

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Whale Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations (int): Maximum number of iterations

        """

        coefficient = 2 - 2 * iteration / (n_iterations - 1)

        for agent in space.agents:
            r1 = r.generate_uniform_random_number()

            A = 2 * coefficient * r1 - coefficient
            C = 2 * r1

            p = r.generate_uniform_random_number()
            if p < 0.5:
                if np.fabs(A) < 1:
                    D = np.fabs(C * space.best_agent.position - agent.position)
                    agent.position = space.best_agent.position - A * D
                else:
                    a = self._generate_random_agent(agent)
                    D = np.fabs(C * a.position - agent.position)
                    agent.position = a.position - A * D
            else:
                l = r.generate_gaussian_random_number()
                D = np.fabs(space.best_agent.position - agent.position)
                agent.position = (
                    D * np.exp(self.b * l) * np.cos(2 * np.pi * l)
                    + space.best_agent.position
                )
