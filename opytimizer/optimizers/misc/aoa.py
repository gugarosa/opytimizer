"""Arithmetic Optimization Algorithm."""

from typing import Any, Dict, Optional

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class AOA(Optimizer):
    """An AOA class, inherited from Optimizer.

    This is the designed class to define AOA-related
    variables and methods.

    References:
        L. Abualigah et al. The Arithmetic Optimization Algorithm.
        Computer Methods in Applied Mechanics and Engineering (2021).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(AOA, self).__init__()

        self.a_min = 0.2
        self.a_max = 1.0

        self.alpha = 5.0
        self.mu = 0.499

        self.build(params)

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Arithmetic Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Calculates math optimizer accelarated coefficient (eq. 2)
        MOA = self.a_min + iteration * ((self.a_max - self.a_min) / n_iterations)

        # Calculates math optimizer probability (eq. 4)
        MOP = 1 - (iteration ** (1 / self.alpha) / n_iterations ** (1 / self.alpha))

        for agent in space.agents:
            for j in range(agent.n_variables):
                search_partition = (agent.ub[j] - agent.lb[j]) * self.mu + agent.lb[j]

                r1 = np.random.uniform(0.0, 1.0, 1)
                if r1 > MOA:
                    r2 = np.random.uniform(0.0, 1.0, 1)
                    if r2 > 0.5:
                        # Updates position with (eq. 3 - top)
                        agent.position[j] = (
                            space.best_agent.position[j]
                            / (MOP + c.EPSILON)
                            * search_partition
                        )
                    else:
                        # Updates position with (eq. 3 - bottom)
                        agent.position[j] = (
                            space.best_agent.position[j] * MOP * search_partition
                        )
                else:
                    r3 = np.random.uniform(0.0, 1.0, 1)
                    if r3 > 0.5:
                        # Updates position with (eq. 5 - top)
                        agent.position[j] = (
                            space.best_agent.position[j] - MOP * search_partition
                        )
                    else:
                        # Updates position with (eq. 5 - bottom)
                        agent.position[j] = (
                            space.best_agent.position[j] + MOP * search_partition
                        )
