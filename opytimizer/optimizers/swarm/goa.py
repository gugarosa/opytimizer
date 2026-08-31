"""Grasshopper Optimization Algorithm."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class GOA(Optimizer):
    """A GOA class, inherited from Optimizer.

    This is the designed class to define GOA-related
    variables and methods.

    References:
        S. Saremi, S. Mirjalili and A. Lewis. Grasshopper Optimisation Algorithm: Theory and application.
        Advances in Engineering Software (2017).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(GOA, self).__init__()

        self.c_min = 0.00001
        self.c_max = 1

        self.f = 0.5
        self.l = 1.5

        self.build(params)

    def _social_force(self, r: np.ndarray) -> np.ndarray:
        """Calculates the social force based on an input value.

        Args:
            r: Array of values.

        Returns:
            (np.ndarray): The social force based on the input value.

        """

        # Calculates the social force (eq. 2.3)
        s = self.f * np.exp(-r / self.l) - np.exp(-r)

        return s

    def update(
        self, space: Space, function: Callable, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Grasshopper Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Calculates the comfort coefficient (eq. 2.8)
        comfort = self.c_max - iteration * ((self.c_max - self.c_min) / n_iterations)

        temp_agents = copy.deepcopy(space.agents)

        for agent in space.agents:
            total_comfort = np.zeros((agent.n_variables, agent.n_dimensions))

            for temp in temp_agents:
                distance = np.linalg.norm(agent.position - temp.position)
                unit = (temp.position - agent.position) / (distance + c.EPSILON)

                s = self._social_force(2 + np.fmod(distance, 2))

                ub = np.expand_dims(agent.ub, -1)
                lb = np.expand_dims(agent.lb, -1)

                total_comfort += comfort * ((ub - lb) / 2) * s * unit

            # Updates the agent's position (eq. 2.7)
            agent.position = comfort * total_comfort + space.best_agent.position
            agent.clip_by_bound()

            agent.fit = function(agent.position)
