# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Algorithm of the Innovative Gunner.

References:
    P. Pijarski and P. Kacejko.
    A new metaheuristic optimization method: the algorithm of the innovative gunner (AIG).
    Engineering Optimization (2019).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class AIG(Optimizer):
    """Implement Algorithm of the Innovative Gunner.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure the gunner's angular search scales.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``alpha`` and ``beta`` (both ``pi``) scale the two angular limits.
            Each update multiplies both limits by one shared uniform draw,
            then uses one third of each limit as its Gaussian sampling deviation.

        """

        super(AIG, self).__init__()

        self.alpha = np.pi
        self.beta = np.pi

        self.build(params)

    def update(self, space: Space, function: Callable) -> None:
        # Calculates the maximum correction angles (eq. 18)
        a = np.random.uniform(0.0, 1.0, 1)
        alpha_max = self.alpha * a
        beta_max = self.beta * a

        for agent in space.agents:
            a = copy.deepcopy(agent)

            alpha = np.random.normal(0, alpha_max / 3, (agent.n_variables, agent.n_dimensions))
            beta = np.random.normal(0, beta_max / 3, (agent.n_variables, agent.n_dimensions))

            # Calculates correction functions (eq. 16 and 17)
            g_alpha = np.where(alpha < 0, np.cos(alpha), 1 / np.cos(alpha))
            g_beta = np.where(beta < 0, np.cos(beta), 1 / np.cos(beta))

            # Updates temporary agent's position (eq. 15)
            a.position *= g_alpha * g_beta
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)
