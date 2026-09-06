# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Flying Squirrel Optimizer.

References:
    G. Azizyan et al.
    Flying Squirrel Optimizer (FSO): A novel SI-based optimization algorithm for engineering problems.
    Iranian Journal of Optimization (2019).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.math.distribution as d
from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class FSO(Optimizer):
    """Search with population-centered random walks and expanding Lévy flights.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure flying squirrel Lévy expansion.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``beta`` (0.5) is the initial Lévy exponent, expanded toward two over the iteration budget.

        """

        super(FSO, self).__init__()

        self.beta = 0.5

        self.build(params)

    def update(self, space: Space, function: Callable, iteration: int, n_iterations: int) -> None:
        mean_position = np.mean([agent.position for agent in space.agents], axis=0)

        # Calculates the Sigma Reduction Factor (eq. 5)
        SRF = (-np.log(1 - (1 / np.sqrt(iteration + 2)))) ** 2

        BEF = self.beta + (2 - self.beta) * ((iteration + 1) / n_iterations)

        for agent in space.agents:
            a = copy.deepcopy(agent)

            for j in range(agent.n_variables):
                # Calculates the random walk (eq. 2 and 3)
                random_step = np.random.normal(mean_position[j], SRF, 1)

                # Calculates the Lévy flight (eq. 6 to 18)
                levy_step = d.generate_levy_distribution(BEF)

                a.position[j] += random_step * levy_step * (agent.position[j] - space.best_agent.position[j])
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)
