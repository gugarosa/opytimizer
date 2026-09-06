# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Sooty Tern Optimization Algorithm.

References:
    G. Dhiman and A. Kaur. STOA: A bio-inspired based optimization algorithm for industrial engineering problems.
    Engineering Applications of Artificial Intelligence (2019).

"""

from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class STOA(Optimizer):
    """Search through sooty tern collision avoidance and spiral attack movement.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure sooty tern migration and spiral geometry.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``Cf`` (2.0) is the initial collision-avoidance coefficient, decreased linearly over iterations.
            ``u`` (1.0) scales the spiral radius, and ``v`` (1.0) controls its exponential growth.

        """

        super(STOA, self).__init__()

        self.Cf = 2.0
        self.u = 1.0
        self.v = 1.0

        self.build(params)

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        # Calculates the movement of search space (eq. 2)
        Sa = self.Cf - (iteration * (self.Cf / n_iterations))

        # Calculates the exploration variable (eq. 4)
        Cb = 0.5 * np.random.uniform(0.0, 1.0, 1)

        for agent in space.agents:
            # Calculates the collision avoidance (eq. 1)
            C = Sa * agent.position

            # Calculates the convergence towards the best agent (eq. 3)
            M = Cb * (space.best_agent.position - agent.position)

            # Calculates the gap between agent and best agent (eq. 5)
            D = C + M

            # Defines the spiral radius (eq. 9)
            k = np.random.uniform(0, 2 * np.pi, 1)
            R = self.u * np.exp(k * self.v)

            # Calculates the spiral movement (eq. 6, 7 and 8)
            i = np.random.uniform(0, k, 1)
            x = R * np.sin(i)
            y = R * np.cos(i)
            z = R * i

            # Updates the agent's position (eq. 10)
            agent.position = (D * (x + y + z)) * space.best_agent.position
