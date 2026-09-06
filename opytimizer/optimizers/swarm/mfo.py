# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Moth-Flame Optimization.

References:
    S. Mirjalili. Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm.
    Knowledge-Based Systems (2015).

"""

import copy
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class MFO(Optimizer):
    """Move moths along logarithmic spirals around ranked flames.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure moth-flame spiral movement.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``b`` (1) controls the logarithmic spiral's exponential growth.

        """

        super(MFO, self).__init__()

        self.b = 1

        self.build(params)

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        flames = copy.deepcopy(space.agents)
        flames.sort(key=lambda x: x.fit)

        # Calculates the number of flames (eq. 3.14)
        n_flames = int(len(flames) - iteration * ((len(flames) - 1) / n_iterations)) - 1

        r = -1 + iteration * (-1 / n_iterations)

        for i, agent in enumerate(space.agents):
            for j in range(agent.n_variables):
                t = np.random.uniform(r, 1, 1)

                if i < n_flames:
                    # Calculates the distance (eq. 3.13)
                    D = np.fabs(flames[i].position[j] - agent.position[j])

                    # Updates current agent's position (eq. 3.12)
                    agent.position[j] = D * np.exp(self.b * t) * np.cos(2 * np.pi * t) + flames[i].position[j]
                else:
                    # Calculates the distance (eq. 3.13)
                    D = np.fabs(flames[0].position[j] - agent.position[j])

                    # Updates current agent's position (eq. 3.12)
                    agent.position[j] = D * np.exp(self.b * t) * np.cos(2 * np.pi * t) + flames[0].position[j]
