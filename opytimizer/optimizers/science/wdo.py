# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Wind Driven Optimization.

References:
    Z. Bayraktar et al. The wind driven optimization technique and its application in electromagnetics.
    IEEE transactions on antennas and propagation (2013).

"""

from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class WDO(Optimizer):
    """Implement Wind Driven Optimization.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure wind velocity and force coefficients.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``v_max`` (0.3) bounds each velocity component and ``alpha`` (0.8) is the friction coefficient.
            ``g`` (0.6) scales gravitational pull, ``c`` (1.0) scales velocity coupling,
            and ``RT`` (1.5) scales attraction toward the best agent.
            Compilation allocates zero velocities for the population.

        """

        super(WDO, self).__init__()

        self.v_max = 0.3
        self.alpha = 0.8
        self.g = 0.6
        self.c = 1.0
        self.RT = 1.5

        self.build(params)

    def compile(self, space: Space) -> None:
        self.velocity = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def update(self, space: Space, function: Callable) -> None:
        for i, agent in enumerate(space.agents):
            index = np.random.randint(0, len(space.agents), None)

            # Updates velocity (eq. 15)
            self.velocity[i] = (
                (1 - self.alpha) * self.velocity[i]
                - self.g * agent.position
                + (self.RT * np.abs(1 / (index + 1) - 1) * (space.best_agent.position - agent.position))
                + (self.c * self.velocity[index] / (index + 1))
            )

            self.velocity = np.clip(self.velocity, -self.v_max, self.v_max)

            # Updates agent's position (eq. 16)
            agent.position += self.velocity[i]
            agent.clip_by_bound()

            agent.fit = function(agent.position)
