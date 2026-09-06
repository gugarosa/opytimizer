# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Arithmetic Optimization Algorithm.

Updates balance multiplicative exploration and additive exploitation around the best agent.

References:
    L. Abualigah et al. The Arithmetic Optimization Algorithm.
    Computer Methods in Applied Mechanics and Engineering (2021).

"""

from typing import Any

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class AOA(Optimizer):
    """Optimize a population with arithmetic exploration and exploitation.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize arithmetic acceleration and search controls.

        Args:
            params: Overrides for the supported optimizer parameters.

        Notes:
            Supported keys are ``a_min`` (initial acceleration, 0.2), ``a_max``
            (final acceleration, 1.0), ``alpha`` (probability schedule exponent, 5.0),
            and ``mu`` (search partition fraction, 0.499).

        """

        super(AOA, self).__init__()

        self.a_min = 0.2
        self.a_max = 1.0

        self.alpha = 5.0
        self.mu = 0.499

        self.build(params)

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        # Equation 2
        MOA = self.a_min + iteration * ((self.a_max - self.a_min) / n_iterations)

        # Equation 4
        MOP = 1 - (iteration ** (1 / self.alpha) / n_iterations ** (1 / self.alpha))

        for agent in space.agents:
            for j in range(agent.n_variables):
                search_partition = (agent.ub[j] - agent.lb[j]) * self.mu + agent.lb[j]

                r1 = np.random.uniform(0.0, 1.0, 1)
                if r1 > MOA:
                    r2 = np.random.uniform(0.0, 1.0, 1)
                    if r2 > 0.5:
                        # Equation 3, top
                        agent.position[j] = space.best_agent.position[j] / (MOP + c.EPSILON) * search_partition
                    else:
                        # Equation 3, bottom
                        agent.position[j] = space.best_agent.position[j] * MOP * search_partition
                else:
                    r3 = np.random.uniform(0.0, 1.0, 1)
                    if r3 > 0.5:
                        # Equation 5, top
                        agent.position[j] = space.best_agent.position[j] - MOP * search_partition
                    else:
                        # Equation 5, bottom
                        agent.position[j] = space.best_agent.position[j] + MOP * search_partition
