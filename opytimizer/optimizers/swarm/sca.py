# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Sine Cosine Algorithm.

References:
    S. Mirjalili. SCA: A Sine Cosine Algorithm for solving optimization problems.
    Knowledge-Based Systems (2016).

"""

from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class SCA(Optimizer):
    """Move agents with alternating sine and cosine displacements.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure sine-cosine movement scales.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``r_min`` (0) and ``r_max`` (2) bound the random weight applied to the best position.
            ``a`` (3) sets the initial movement amplitude, which decreases linearly over iterations.
            Each update shares an amplitude, angle, target weight, and sine-or-cosine choice across agents.

        """

        super(SCA, self).__init__()

        self.r_min = 0
        self.r_max = 2

        self.a = 3

        self.build(params)

    def _update_position(
        self,
        agent_position: np.ndarray,
        best_position: np.ndarray,
        r1: float,
        r2: float,
        r3: float,
        r4: float,
    ) -> np.ndarray:
        # Sine-cosine movement toward the weighted best position (eq. 3.3)
        if r4 < 0.5:
            new_position = agent_position + r1 * np.sin(r2) * np.fabs(r3 * best_position - agent_position)
        else:
            new_position = agent_position + r1 * np.cos(r2) * np.fabs(r3 * best_position - agent_position)

        return new_position

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        r1 = self.a - (iteration * self.a / n_iterations)
        r2 = np.random.uniform(0, 2 * np.pi, 1)
        r3 = np.random.uniform(self.r_min, self.r_max, 1)
        r4 = np.random.uniform(0.0, 1.0, 1)

        for agent in space.agents:
            agent.position = self._update_position(agent.position, space.best_agent.position, r1, r2, r3, r4)
