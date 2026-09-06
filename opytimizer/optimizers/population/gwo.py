# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Grey Wolf Optimizer.

References:
    S. Mirjalili, S. Mirjalili and A. Lewis. Grey Wolf Optimizer.
    Advances in Engineering Software (2014).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class GWO(Optimizer):
    """Implement Grey Wolf Optimizer.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize grey-wolf hunting.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            This optimizer has no algorithm-specific configuration keys.

        """

        super(GWO, self).__init__()

        self.build(params)

    def _calculate_coefficients(self, a: float) -> tuple[float, float]:
        r1 = np.random.uniform(0.0, 1.0, 1)
        r2 = np.random.uniform(0.0, 1.0, 1)

        # Calculates the `A` coefficient (eq. 3.3)
        A = 2 * a * r1 - a

        # Calculates the `C` coefficient (eq. 3.4)
        C = 2 * r2

        return A, C

    def update(self, space: Space, function: Callable, iteration: int, n_iterations: int) -> None:
        space.agents.sort(key=lambda x: x.fit)
        alpha, beta, delta = copy.deepcopy(space.agents[:3])

        a = 2 - 2 * iteration / (n_iterations - 1)

        for agent in space.agents:
            X = copy.deepcopy(agent)

            A_1, C_1 = self._calculate_coefficients(a)
            A_2, C_2 = self._calculate_coefficients(a)
            A_3, C_3 = self._calculate_coefficients(a)

            # Simulates hunting behavior (Eqs. 3.5 and 3.6)
            X_1 = alpha.position - A_1 * np.fabs(C_1 * alpha.position - agent.position)
            X_2 = beta.position - A_2 * np.fabs(C_2 * beta.position - agent.position)
            X_3 = delta.position - A_3 * np.fabs(C_3 * delta.position - agent.position)

            # Calculates the temporary agent (eq. 3.7)
            X.position = (X_1 + X_2 + X_3) / 3
            X.clip_by_bound()

            X.fit = function(X.position)
            if X.fit < agent.fit:
                agent.position = copy.deepcopy(X.position)
                agent.fit = copy.deepcopy(X.fit)
