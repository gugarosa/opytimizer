"""Whale Optimization Algorithm."""

import copy
from typing import Any, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


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
            r1 = np.random.uniform(0.0, 1.0, 1)

            A = 2 * coefficient * r1 - coefficient
            C = 2 * r1

            p = np.random.uniform(0.0, 1.0, 1)
            if p < 0.5:
                if np.fabs(A) < 1:
                    D = np.fabs(C * space.best_agent.position - agent.position)
                    agent.position = space.best_agent.position - A * D
                else:
                    a = self._generate_random_agent(agent)
                    D = np.fabs(C * a.position - agent.position)
                    agent.position = a.position - A * D
            else:
                l = np.random.normal(0.0, 1.0, 1)
                D = np.fabs(space.best_agent.position - agent.position)
                agent.position = (
                    D * np.exp(self.b * l) * np.cos(2 * np.pi * l)
                    + space.best_agent.position
                )
