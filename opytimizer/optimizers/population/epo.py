# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Emperor Penguin Optimizer.

References:
    G. Dhiman and V. Kumar. Emperor penguin optimizer: A bio-inspired algorithm for engineering problems.
    Knowledge-Based Systems (2018).

"""

from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class EPO(Optimizer):
    """Implement Emperor Penguin Optimizer.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure emperor-penguin movement.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``f`` (2.0) scales the exponential social-force term and ``l`` (1.5)
            controls its decay with the iteration count.

        """

        super(EPO, self).__init__()

        self.f = 2.0
        self.l = 1.5

        self.build(params)

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        for agent in space.agents:
            R = np.random.uniform(0.0, 1.0, 1)
            if R >= 0.5:
                T = 0
            else:
                T = 1

            # Calculates the temperature profile (eq. 7)
            T_p = T - n_iterations / (iteration - n_iterations)

            # Calculates the polygon grid accuracy (eq. 10)
            P_grid = np.fabs(space.best_agent.position - agent.position)

            r1 = np.random.uniform(0.0, 1.0, 1)
            C = np.random.uniform(0.0, 1.0, (agent.n_variables, 1))

            # Calculates the avoidance coefficient (eq. 9)
            A = 2 * (T_p + P_grid) * r1 - T_p

            # Calculates the social forces of emperor penguin (eq. 12)
            S = (np.fabs(self.f * np.exp(-iteration / self.l) - np.exp(-iteration))) ** 2

            # Calculates the distance between current agent and emperor penguin (eq. 8)
            D_ep = np.fabs(S * space.best_agent.position - C * agent.position)

            # Updates current agent's position (eq. 13)
            agent.position = space.best_agent.position - A * D_ep
